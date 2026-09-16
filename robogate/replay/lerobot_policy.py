"""python-policy adapter backed by the lerobot package (optional extra)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from robogate.replay.base import Adapter, AdapterInfo, StepOut
from robogate.scenario import Target
from robogate.slice import Slice

ACTION_TOL = 1e-6


class LeRobotPolicyAdapter(Adapter):
    def __init__(
        self,
        *,
        drop_camera: str | None = None,
        state_noise: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.info = AdapterInfo(name="lerobot")
        self._policy: Any = None
        self._preprocessor: Any = None
        self._postprocessor: Any = None
        self._dataset: Any = None
        self._offset = 0
        self._device = "cpu"
        self._input_keys: list[str] = []
        self._drop_camera = drop_camera
        self._state_noise = float(state_noise)
        self._rng = np.random.default_rng(seed)

    def load(self, target: Target, slice_: Slice, *, device: str) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "lerobot extra is required: pip install -e '.[lerobot]'"
            ) from exc

        checkpoint = target.checkpoint
        if not checkpoint:
            raise ValueError("target.checkpoint is required for the lerobot adapter")
        revision = target.revision or None
        self._device = device
        self._offset = slice_.meta.frame_from

        _isolate_lerobot_imports()
        policy = _load_policy(checkpoint, revision=revision, device=device)
        self._input_keys = list(getattr(policy.config, "input_features", {}) or [])

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        _patch_video_decoder()
        kwargs: dict[str, Any] = {
            "repo_id": slice_.meta.dataset,
            "episodes": [slice_.meta.episode_index],
            "video_backend": _video_backend(),
        }
        if slice_.meta.dataset_revision:
            kwargs["revision"] = slice_.meta.dataset_revision
        local_root = _dataset_root(slice_.meta.dataset)
        if local_root is not None:
            kwargs["root"] = local_root
            kwargs["download_videos"] = not _videos_present(local_root)
        try:
            dataset = LeRobotDataset(**kwargs)
        except TypeError:
            kwargs.pop("revision", None)
            kwargs.pop("video_backend", None)
            kwargs.pop("download_videos", None)
            dataset = LeRobotDataset(**kwargs)
        self._dataset = dataset
        _check_recorded_actions(dataset, slice_, self._offset)

        preprocessor, postprocessor, processor_source = resolve_processors(
            checkpoint,
            revision=revision,
            device=device,
            dataset=dataset,
            policy=policy,
        )
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor

        versions = {
            "lerobot": _pkg_version("lerobot"),
            "torch": torch.__version__,
        }
        config = {
            "chunk_size": getattr(policy.config, "chunk_size", None),
            "n_action_steps": getattr(policy.config, "n_action_steps", None),
            "input_features": list(self._input_keys),
            "processor_source": processor_source,
        }
        self.info = AdapterInfo(
            name="lerobot",
            target_version=checkpoint,
            target_revision=revision,
            versions=versions,
            policy_config={k: v for k, v in config.items() if v is not None},
        )
        if hasattr(policy, "reset"):
            policy.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(0)
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()

    def step(self, obs: dict[str, Any], frame_index: int) -> StepOut:
        del obs
        if self._policy is None or self._dataset is None:
            raise RuntimeError("adapter not loaded")
        row = self._dataset[self._offset + frame_index]
        batch = _select_inputs(row, self._input_keys)
        batch = _perturb_inputs(
            batch,
            drop_camera=self._drop_camera,
            state_noise=self._state_noise,
            rng=self._rng,
        )
        t0 = time.perf_counter()
        processed = self._preprocessor(batch) if self._preprocessor is not None else batch
        processed = _to_device(processed, self._device)
        action = self._policy.select_action(processed)
        if self._postprocessor is not None:
            action = self._postprocessor(action)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return StepOut(
            action=_as_numpy(action),
            latency_ms=latency_ms,
            confidence=None,
            images=_batch_images(batch),
        )


def _isolate_lerobot_imports() -> None:
    """Keep ACT/dataset imports from pulling broken extras (Groot) and robot stacks."""
    import sys
    import types
    from enum import Enum
    from pathlib import Path

    import lerobot

    def namespace(name: str, rel: str) -> None:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(Path(lerobot.__file__).parent / rel)]
        pkg.__package__ = name
        sys.modules[name] = pkg

    def stub(name: str, **attrs: Any) -> None:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod

    namespace("lerobot.policies", "policies")

    class TrainPipelineConfig:  # noqa: N801
        pass

    stub("lerobot.configs.train", TrainPipelineConfig=TrainPipelineConfig)

    class TeleopEvents(Enum):
        SUCCESS = "success"
        FAILURE = "failure"
        RERECORD_EPISODE = "rerecord_episode"
        IS_INTERVENTION = "is_intervention"
        TERMINATE_EPISODE = "terminate_episode"

    stub("lerobot.teleoperators")
    stub("lerobot.teleoperators.utils", TeleopEvents=TeleopEvents)


def _ensure_local_act_type(checkpoint: str) -> None:
    from pathlib import Path

    path = Path(checkpoint) / "config.json"
    if not path.is_file():
        return
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type"):
        return
    data["type"] = "act"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _load_policy(checkpoint: str, *, revision: str | None, device: str) -> Any:
    _isolate_lerobot_imports()
    from lerobot.policies.act.modeling_act import ACTPolicy

    _ensure_local_act_type(checkpoint)
    kwargs: dict[str, Any] = {}
    if revision:
        kwargs["revision"] = revision
    try:
        policy = ACTPolicy.from_pretrained(checkpoint, **kwargs)
    except Exception as exc:
        if not _unexpected_weight_keys(exc):
            raise
        try:
            policy = ACTPolicy.from_pretrained(checkpoint, strict=False, **kwargs)
        except TypeError:
            raise exc from exc
    if hasattr(policy, "config") and hasattr(policy.config, "device"):
        policy.config.device = device
    if hasattr(policy, "to"):
        policy.to(device)
    if hasattr(policy, "eval"):
        policy.eval()
    return policy


def _load_processors(checkpoint: str, *, revision: str | None, device: str) -> tuple[Any, Any]:
    """Load ACT processors from the checkpoint. Avoid policies.factory (imports Groot)."""
    from lerobot.processor.converters import (
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    _ensure_local_act_type(checkpoint)
    kwargs: dict[str, Any] = {}
    if revision:
        kwargs["revision"] = revision
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=checkpoint,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        **kwargs,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=checkpoint,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
        **kwargs,
    )
    _set_processor_device(preprocessor, device)
    _set_processor_device(postprocessor, "cpu")
    return preprocessor, postprocessor


def resolve_processors(
    checkpoint: str,
    *,
    revision: str | None,
    device: str,
    dataset: Any,
    policy: Any,
) -> tuple[Any, Any, str]:
    """Load processors from the checkpoint, or build them from dataset.meta.stats."""
    if checkpoint_has_processors(checkpoint, revision):
        return (*_load_processors(checkpoint, revision=revision, device=device), "checkpoint")
    return (*processors_from_dataset_stats(dataset, policy, device), "dataset_stats")


def checkpoint_has_processors(checkpoint: str, revision: str | None = None) -> bool:
    from pathlib import Path

    root = Path(checkpoint)
    if (root / "policy_preprocessor.json").is_file():
        return True
    if root.is_dir():
        return False
    try:
        from huggingface_hub import hf_hub_download

        kwargs: dict[str, Any] = {
            "repo_id": checkpoint,
            "filename": "policy_preprocessor.json",
        }
        if revision:
            kwargs["revision"] = revision
        hf_hub_download(**kwargs)
        return True
    except Exception:  # noqa: BLE001 — missing file or offline Hub
        return False


def dataset_stats(dataset: Any) -> dict[str, Any]:
    meta = getattr(dataset, "meta", None)
    stats = getattr(meta, "stats", None) if meta is not None else None
    if stats is None and isinstance(dataset, dict):
        stats = dataset.get("stats")
    return dict(stats) if stats else {}


def _unexpected_weight_keys(exc: BaseException) -> bool:
    msg = str(exc)
    return "Unexpected key" in msg or "Missing key" in msg or "size mismatch" in msg.lower()


def processors_from_dataset_stats(dataset: Any, policy: Any, device: str) -> tuple[Any, Any]:
    """Build ACT pre/post processors from dataset.meta.stats (old Hub checkpoints)."""
    _isolate_lerobot_imports()
    from lerobot.processor.batch_processor import AddBatchDimensionProcessorStep
    from lerobot.processor.converters import (
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from lerobot.processor.device_processor import DeviceProcessorStep
    from lerobot.processor.normalize_processor import (
        NormalizerProcessorStep,
        UnnormalizerProcessorStep,
    )
    from lerobot.processor.pipeline import DataProcessorPipeline
    from lerobot.processor.rename_processor import RenameObservationsProcessorStep

    stats = dataset_stats(dataset)
    if not stats:
        raise RuntimeError("dataset.meta.stats missing; cannot build processors")
    features, norm_map = _policy_features(policy)
    normalizer = NormalizerProcessorStep(
        features=features,
        norm_map=norm_map,
        stats=stats,
        device=device,
    )
    action_features = {
        name: feat
        for name, feat in features.items()
        if name == "action" or getattr(feat, "type", None) == "ACTION"
    }
    unnormalizer = UnnormalizerProcessorStep(
        features=action_features or features,
        norm_map=norm_map,
        stats=stats,
        device="cpu",
    )
    preprocessor = DataProcessorPipeline(
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=device),
            normalizer,
        ],
        name="policy_preprocessor",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = DataProcessorPipeline(
        steps=[
            unnormalizer,
            DeviceProcessorStep(device="cpu"),
        ],
        name="policy_postprocessor",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def _policy_features(policy: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    config = getattr(policy, "config", policy)
    features: dict[str, Any] = {}
    for group in ("input_features", "output_features"):
        mapping = getattr(config, group, None) or {}
        if isinstance(mapping, dict):
            features.update(mapping)
    norm_map = getattr(config, "normalization_mapping", None) or {
        "VISUAL": "MEAN_STD",
        "STATE": "MEAN_STD",
        "ACTION": "MEAN_STD",
    }
    return features, dict(norm_map)


def _set_processor_device(pipeline: Any, device: str) -> None:
    steps = getattr(pipeline, "steps", None) or []
    for step in steps:
        if hasattr(step, "device"):
            step.device = device
            post_init = getattr(step, "__post_init__", None)
            if callable(post_init):
                post_init()


def _to_device(value: Any, device: str) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _video_backend() -> str:
    try:
        import torchcodec  # noqa: F401

        return "torchcodec"
    except ImportError:
        return "pyav"


def _patch_video_decoder() -> None:
    """torchvision 0.26+ dropped VideoReader; decode AV1 with PyAV instead."""
    import sys

    decoder = _SequentialPyAVDecoder()
    import lerobot.datasets.video_utils as video_utils

    video_utils.decode_video_frames = decoder  # type: ignore[assignment]
    loaded = sys.modules.get("lerobot.datasets.lerobot_dataset")
    if loaded is not None:
        loaded.decode_video_frames = decoder


class _SequentialPyAVDecoder:
    """Decode one timestamp at a time without reopening the mp4 every frame."""

    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}

    def __call__(
        self,
        video_path: Any,
        timestamps: list[float],
        tolerance_s: float,
        backend: str | None = None,
    ) -> Any:
        del backend
        import torch

        path = str(video_path)
        state = self._open(path)
        query = [float(ts) for ts in timestamps]
        first_ts = min(query)
        last_ts = max(query)
        if state["last_ts"] is None or first_ts + 1e-3 < float(state["last_ts"]):
            self._seek(state, first_ts)
        reopened = False
        while not state["buf"] or state["buf"][-1][0] < last_ts:
            if self._read_one(state):
                continue
            if reopened or state["buf"]:
                break
            state = self._reopen(path)
            self._seek(state, first_ts)
            reopened = True
        if not state["buf"]:
            raise RuntimeError(f"no frames decoded from {path} at {query}")
        loaded_ts = torch.tensor([item[0] for item in state["buf"]], dtype=torch.float64)
        query_ts = torch.tensor(query, dtype=torch.float64)
        dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
        min_dist, argmin = dist.min(1)
        limit = max(float(tolerance_s), 1.0 / max(state["fps"], 1.0))
        if bool((min_dist > limit).any()):
            raise RuntimeError(
                f"video timestamps miss query by {min_dist.max().item():.4f}s "
                f"(tol={limit}) in {path}"
            )
        frames = torch.stack([state["buf"][int(i)][1] for i in argmin.tolist()])
        cutoff = last_ts - 1.0
        state["buf"] = [item for item in state["buf"] if item[0] >= cutoff]
        return frames.float() / 255.0

    def _open(self, path: str) -> dict[str, Any]:
        import av

        state = self._states.get(path)
        if state is not None:
            return state
        container = av.open(path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate) if stream.average_rate else 50.0
        state = {
            "container": container,
            "stream": stream,
            "fps": fps,
            "buf": [],
            "last_ts": None,
            "path": path,
        }
        self._states[path] = state
        return state

    def _reopen(self, path: str) -> dict[str, Any]:
        state = self._states.pop(path, None)
        if state is not None:
            try:
                state["container"].close()
            except Exception:  # noqa: BLE001
                pass
        return self._open(path)

    def _seek(self, state: dict[str, Any], timestamp: float) -> None:
        container = state["container"]
        stream = state["stream"]
        offset = max(timestamp - 1.0, 0.0)
        try:
            container.seek(int(offset / float(stream.time_base)), stream=stream, backward=True)
        except Exception:  # noqa: BLE001
            container.seek(0)
        state["buf"] = []
        state["last_ts"] = None

    def _read_one(self, state: dict[str, Any]) -> bool:
        import torch

        try:
            frame = next(state["container"].decode(state["stream"]))
        except Exception as exc:  # noqa: BLE001 — PyAV raises EOFError, not only StopIteration
            if type(exc).__name__ in {"EOFError", "StopIteration", "InvalidDataError"}:
                return False
            raise
        ts = float(frame.time) if frame.time is not None else (
            float(frame.pts) * float(state["stream"].time_base)
        )
        img = frame.to_ndarray(format="rgb24")
        tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        state["buf"].append((ts, tensor))
        state["last_ts"] = ts
        return True


def _check_recorded_actions(dataset: Any, slice_: Slice, offset: int) -> None:
    recorded = np.asarray(slice_.recorded_action().get_column("value").to_list(), dtype=np.float64)
    actions = _tabular_actions(dataset, offset, len(recorded))
    if actions is None:
        actions = _sampled_actions(dataset, offset, len(recorded))
    if actions.shape != recorded.shape:
        raise ValueError(
            f"dataset action shape {actions.shape} != slice recorded {recorded.shape}"
        )
    worst = float(np.max(np.abs(actions - recorded)))
    if worst >= ACTION_TOL:
        raise ValueError(
            f"dataset action diverges from slice: max-abs-diff={worst} (tol={ACTION_TOL})"
        )


def _tabular_actions(dataset: Any, offset: int, n: int) -> np.ndarray | None:
    hf = getattr(dataset, "hf_dataset", None)
    if hf is None:
        return None
    try:
        length = len(hf)
        start = offset if offset + n <= length else 0
        rows = hf[start : start + n]
        return np.asarray(rows["action"], dtype=np.float64)
    except Exception:  # noqa: BLE001
        return None


def _sampled_actions(dataset: Any, offset: int, n: int) -> np.ndarray:
    out = []
    for i in range(n):
        row = dataset[offset + i]
        out.append(_as_numpy(row.get("action")))
    return np.asarray(out, dtype=np.float64)


def _maybe_local_root(dataset: str) -> str | None:
    from pathlib import Path

    path = Path(dataset)
    if (path / "meta" / "info.json").is_file():
        return str(path.resolve())
    return None


def _dataset_root(dataset: str) -> str | None:
    import os
    from pathlib import Path

    local = _maybe_local_root(dataset)
    if local is not None:
        return local
    home = os.environ.get("HF_LEROBOT_HOME")
    if not home:
        return None
    candidate = Path(home) / dataset
    if (candidate / "meta" / "info.json").is_file():
        return str(candidate)
    return None


def _videos_present(root: str) -> bool:
    from pathlib import Path

    videos = Path(root) / "videos"
    return videos.is_dir() and any(videos.rglob("*.mp4"))


def _perturb_inputs(
    batch: dict[str, Any],
    *,
    drop_camera: str | None,
    state_noise: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    out = dict(batch)
    if drop_camera:
        for key, value in list(out.items()):
            if _camera_match(key, drop_camera):
                out[key] = _zero_like(value)
    if state_noise:
        for key, value in list(out.items()):
            if key == "observation.state" or key.endswith(".state"):
                out[key] = _add_noise(value, state_noise, rng)
    return out


def _camera_match(key: str, spec: str) -> bool:
    return key == spec or key.endswith(spec) or spec in key.split(".")


def _zero_like(value: Any) -> Any:
    if hasattr(value, "zero_"):
        return value * 0
    return np.zeros_like(np.asarray(value))


def _add_noise(value: Any, sigma: float, rng: np.random.Generator) -> Any:
    noise = rng.normal(0.0, sigma, size=np.asarray(value).shape)
    if hasattr(value, "cpu"):
        import torch

        return value + torch.as_tensor(noise, device=value.device, dtype=value.dtype)
    return np.asarray(value, dtype=np.float64) + noise


def _batch_images(batch: dict[str, Any]) -> dict[str, Any] | None:
    images: dict[str, Any] = {}
    for key, value in batch.items():
        if not isinstance(key, str) or not key.startswith("observation.images."):
            continue
        images[key.removeprefix("observation.images.")] = value
    return images or None


def _select_inputs(row: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    if not keys:
        return {
            key: value
            for key, value in row.items()
            if isinstance(key, str)
            and (key.startswith("observation.") or key.startswith("cam_"))
        }
    out: dict[str, Any] = {}
    missing: list[str] = []
    for key in keys:
        resolved = _resolve_key(row, key)
        if resolved is None:
            missing.append(key)
        else:
            out[key] = resolved
    if missing:
        raise KeyError(f"dataset frame missing policy inputs: {missing}")
    return out


def _resolve_key(row: dict[str, Any], key: str) -> Any:
    candidates = [
        key,
        f"observation.images.{key}",
        f"observation.{key}",
        key.removeprefix("observation.images."),
        key.removeprefix("observation."),
    ]
    for candidate in candidates:
        if candidate in row:
            return row[candidate]
    return None


def _as_numpy(value: Any) -> np.ndarray:
    import torch

    if isinstance(value, dict):
        for key in ("action", "value"):
            if key in value:
                return _as_numpy(value[key])
        raise TypeError(f"cannot coerce dict to action: {list(value)}")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim > 1 and value.shape[0] == 1:
            value = value.squeeze(0)
        return np.asarray(value, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr.reshape(arr.shape[1:])
    return arr


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return "unknown"
