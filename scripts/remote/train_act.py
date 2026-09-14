#!/usr/bin/env python3
"""Minimal ACT training loop. Avoids lerobot-train import chain (Groot / wandb)."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


def _ensure_act_type(dest: Path) -> None:
    path = dest / "config.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type"):
        return
    data["type"] = "act"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _save_steps() -> list[int]:
    raw = os.environ.get("ACT_SAVE_STEPS", "2000,10000,30000")
    steps = sorted({int(item) for item in raw.split(",") if item.strip()})
    if not steps:
        raise ValueError("ACT_SAVE_STEPS is empty")
    return steps


def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in items[0]:
        values = [item[key] for item in items]
        first = values[0]
        if isinstance(first, torch.Tensor):
            out[key] = torch.stack(values)
        else:
            out[key] = values
    return out


def _to_device(value: Any, device: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _forward_loss(policy: Any, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    out = policy(batch)
    if isinstance(out, tuple) and out:
        loss = out[0]
        extras = out[1] if len(out) > 1 and isinstance(out[1], dict) else {}
        return loss, extras
    if isinstance(out, dict) and "loss" in out:
        return out["loss"], out
    if isinstance(out, torch.Tensor):
        return out, {}
    raise TypeError(f"unexpected policy output type {type(out)}")


def _align_action_stats(pipe: Any, action_stats: dict[str, Any], device: str) -> None:
    """Replace checkpoint action mean/std with the dataset's, then refresh tensors."""
    if pipe is None or not action_stats:
        return
    for step in getattr(pipe, "steps", None) or []:
        stats = getattr(step, "stats", None)
        if not isinstance(stats, dict) or "action" not in stats:
            continue
        stats["action"] = {key: np.asarray(value) for key, value in action_stats.items()}
        if hasattr(step, "to"):
            step.to(device)


def _save_processors(preprocessor: Any, postprocessor: Any, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for pipe in (preprocessor, postprocessor):
        if pipe is not None and hasattr(pipe, "save_pretrained"):
            pipe.save_pretrained(dest)


def _episode_ranges(dataset: Any, episodes: list[int]) -> list[tuple[int, int]]:
    hf = getattr(dataset, "hf_dataset", None)
    if hf is not None and "episode_index" in getattr(hf, "column_names", []):
        idx = np.asarray(hf["episode_index"])
        ranges: list[tuple[int, int]] = []
        for ep in episodes:
            where = np.flatnonzero(idx == ep)
            if len(where):
                ranges.append((int(where[0]), int(where[-1]) + 1))
        if ranges:
            return ranges
    n = len(dataset)
    if not episodes:
        return [(0, n)]
    size = max(n // len(episodes), 1)
    return [(i * size, min((i + 1) * size, n)) for i in range(len(episodes))]


def _patch_worker_runtime(_worker_id: int = 0) -> None:
    """Spawn workers do not inherit the main-process PyAV decoder patch."""
    from robogate.replay.lerobot_policy import (
        _isolate_lerobot_imports,
        _patch_video_decoder,
        _video_backend,
    )

    _isolate_lerobot_imports()
    if _video_backend() != "torchcodec":
        _patch_video_decoder()


class BufferedEpisodeIterable(IterableDataset):
    """Shuffle episode order, read each episode sequentially, pop from a buffer."""

    def __init__(
        self,
        dataset: Any,
        ranges: list[tuple[int, int]],
        *,
        buffer_size: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.ranges = ranges
        self.buffer_size = buffer_size
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, Any]]:
        _patch_worker_runtime()
        info = torch.utils.data.get_worker_info()
        ranges = list(self.ranges)
        if info is not None:
            ranges = ranges[info.id :: info.num_workers]
            rng = np.random.default_rng(self.seed + info.id)
        else:
            rng = np.random.default_rng(self.seed)
        rng.shuffle(ranges)
        buf: list[dict[str, Any]] = []
        while True:
            for start, stop in ranges:
                for idx in range(start, stop):
                    buf.append(self.dataset[idx])
                    if len(buf) == 1 or len(buf) % 50 == 0:
                        wid = info.id if info is not None else 0
                        print(
                            f"[train] worker {wid} buffer {len(buf)}/{self.buffer_size}",
                            flush=True,
                        )
                    if len(buf) >= self.buffer_size:
                        j = int(rng.integers(0, len(buf)))
                        yield buf.pop(j)
            rng.shuffle(ranges)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim > 1 and value.shape[0] == 1:
            value = value.squeeze(0)
        return np.asarray(value, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _probe_open_loop(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    dataset: Any,
    device: str,
    *,
    n: int = 200,
) -> dict[str, float]:
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()
    preds: list[np.ndarray] = []
    refs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(min(n, len(dataset))):
            row = dataset[i]
            batch = {
                key: value
                for key, value in row.items()
                if isinstance(key, str)
                and (key.startswith("observation.") or key.startswith("action"))
            }
            if preprocessor is not None:
                batch = preprocessor(batch)
            batch = _to_device(batch, device)
            action = policy.select_action(batch)
            if postprocessor is not None:
                action = postprocessor(action)
            pred_vec = _as_numpy(action).reshape(-1)
            ref_raw = _as_numpy(row["action"]).reshape(-1)
            if ref_raw.size != pred_vec.size:
                ref_raw = ref_raw.reshape(-1, pred_vec.size)[0]
            preds.append(pred_vec)
            refs.append(ref_raw)
    policy.train()
    if hasattr(policy, "reset"):
        policy.reset()
    pred = np.stack(preds)
    ref = np.stack(refs)
    return {
        "probe_l2": float(np.mean(np.linalg.norm(pred - ref, axis=1))),
        "probe_std": float(pred.std()),
        "probe_ref_std": float(ref.std()),
    }


def main() -> None:
    from robogate.replay.lerobot_policy import (
        _dataset_root,
        _ensure_local_act_type,
        _isolate_lerobot_imports,
        _load_processors,
        _patch_video_decoder,
        _video_backend,
    )

    save_at = _save_steps()
    max_steps = max(save_at)
    out_root = Path(os.environ.get("ACT_OUT", "/root/autodl-tmp/ckpts/v2"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id = os.environ.get("ACT_REPO", "lerobot/aloha_static_coffee")
    base_ckpt = os.environ.get("ACT_BASE_CKPT", "gozdebaydogmus/act-aloha-static-coffee-test")
    base_rev = os.environ.get("ACT_BASE_REV", "646846823c8473712f689f59bdd03f798a688f6a")
    batch_size = int(os.environ.get("ACT_BATCH", "8"))
    workers = int(os.environ.get("ACT_WORKERS", "4"))
    # ~4000 samples in total: each worker keeps its own sequential buffer.
    default_buf = "1000" if workers > 0 else "4000"
    buffer_size = int(os.environ.get("ACT_BUFFER", default_buf))
    lr = float(os.environ.get("ACT_LR", "1e-5"))
    train_from = int(os.environ.get("ACT_EP_FROM", "10"))
    train_to = int(os.environ.get("ACT_EP_TO", "49"))
    episodes = list(range(train_from, train_to + 1))

    print(
        f"[train] device={device} steps={max_steps} save={save_at} "
        f"eps={train_from}-{train_to} batch={batch_size} workers={workers} buffer={buffer_size}"
    )
    if workers > 0:
        import torch.multiprocessing as mp

        try:
            mp.set_start_method("spawn", force=True)
            print("[train] multiprocessing start_method=spawn", flush=True)
        except RuntimeError as exc:
            print(f"[train] spawn not set: {exc}", flush=True)
    _isolate_lerobot_imports()
    if _video_backend() != "torchcodec":
        _patch_video_decoder()

    from lerobot.policies.act.modeling_act import ACTPolicy

    resume = os.environ.get("ACT_RESUME")
    start_step = int(os.environ.get("ACT_START_STEP", "0"))
    if resume:
        print(f"[train] resume {resume} from step {start_step}")
        _ensure_local_act_type(resume)
        policy = ACTPolicy.from_pretrained(resume)
        config = policy.config
    else:
        print(f"[train] loading config from {base_ckpt}@{base_rev}")
        template = ACTPolicy.from_pretrained(base_ckpt, revision=base_rev)
        config = template.config
        policy = ACTPolicy(config)
        del template
    policy.to(device)
    policy.train()

    preprocessor, postprocessor = _load_processors(base_ckpt, revision=base_rev, device=device)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    fps = 50.0
    chunk = int(getattr(config, "chunk_size", 100) or 100)
    delta = {"action": [i / fps for i in range(chunk)]}
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "episodes": episodes,
        "delta_timestamps": delta,
        "video_backend": _video_backend(),
    }
    root = _dataset_root(repo_id)
    if root is not None:
        kwargs["root"] = root
        kwargs["download_videos"] = False
    try:
        dataset = LeRobotDataset(**kwargs)
    except TypeError:
        kwargs.pop("video_backend", None)
        kwargs.pop("download_videos", None)
        dataset = LeRobotDataset(**kwargs)
    ranges = _episode_ranges(dataset, episodes)
    print(f"[train] dataset n={len(dataset)} ranges={len(ranges)}")
    meta_stats = getattr(getattr(dataset, "meta", None), "stats", None) or {}
    action_stats = meta_stats.get("action")
    if action_stats:
        _align_action_stats(preprocessor, action_stats, device)
        _align_action_stats(postprocessor, action_stats, device)
        mean = np.asarray(action_stats.get("mean"))
        print(f"[train] aligned action mean={mean.reshape(-1).round(4).tolist()}", flush=True)

    probe_kwargs = dict(kwargs)
    probe_kwargs["episodes"] = [0]
    try:
        probe_ds = LeRobotDataset(**probe_kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[train] probe dataset unavailable: {exc}")
        probe_ds = None

    print("[train] building dataloader", flush=True)
    iterable = BufferedEpisodeIterable(
        dataset, ranges, buffer_size=buffer_size, seed=0
    )
    loader = DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=_collate,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=1 if workers > 0 else None,
        worker_init_fn=_patch_worker_runtime if workers > 0 else None,
    )
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    print("[train] waiting for first batch (filling shuffle buffers)", flush=True)
    iterator = iter(loader)
    saved: list[int] = []
    t0 = time.time()
    for step in range(start_step + 1, max_steps + 1):
        try:
            batch = next(iterator)
        except Exception as exc:  # noqa: BLE001
            print(f"[train] skip step {step}: {type(exc).__name__}: {exc}", flush=True)
            iterator = iter(loader)
            continue
        if preprocessor is not None:
            batch = preprocessor(batch)
        batch = _to_device(batch, device)
        loss, extras = _forward_loss(policy, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 20 == 0 or step in save_at or step <= 5:
            extra = " ".join(f"{k}={v:.4f}" for k, v in extras.items() if hasattr(v, "item"))
            print(
                f"[train] step {step}/{max_steps} loss={float(loss.detach().cpu()):.6f} "
                f"elapsed={time.time() - t0:.0f}s {extra}",
                flush=True,
            )
        if probe_ds is not None and (step == 1 or step % 500 == 0 or step in save_at):
            try:
                stats = _probe_open_loop(
                    policy, preprocessor, postprocessor, probe_ds, device
                )
                print(
                    f"[train] probe step {step} l2={stats['probe_l2']:.4f} "
                    f"pred_std={stats['probe_std']:.4f} ref_std={stats['probe_ref_std']:.4f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[train] probe failed: {type(exc).__name__}: {exc}", flush=True)
        if step in save_at:
            dest = out_root / f"act-coffee-{step}"
            dest.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(dest)
            _ensure_act_type(dest)
            _save_processors(preprocessor, postprocessor, dest)
            (dest / "train_state.json").write_text(
                json.dumps({"step": step, "loss": float(loss.detach().cpu())}, indent=2) + "\n",
                encoding="utf-8",
            )
            saved.append(step)
            print(f"[train] saved {dest}", flush=True)
    print(f"[train] finished saved={saved} elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
