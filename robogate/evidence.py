"""Failure evidence: worst-frame PNGs, action plot, and evidence.json."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robogate.asserts import AssertResult, overall_ok, run_assertions
from robogate.asserts.action_deviation import vectors_from_frame
from robogate.run import Run
from robogate.scenario import ActionBounds, ActionDeviation, Scenario
from robogate.slice import Slice

DEFAULT_TOP_K = 5
EVIDENCE_MODES = ("fail", "always", "never")


@dataclass(order=True)
class _HeapItem:
    l2: float
    index: int
    seq: int
    images: dict[str, np.ndarray] | None = field(compare=False)


@dataclass
class WorstFrame:
    index: int
    l2: float
    images: dict[str, np.ndarray] | None = None
    files: list[str] = field(default_factory=list)


class TopKFrames:
    """Keep the k highest-L2 frames; convert images only when a frame enters."""

    def __init__(self, k: int = DEFAULT_TOP_K, min_separation: int = 0) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.min_separation = max(int(min_separation), 0)
        self._items: list[_HeapItem] = []
        self._seq = 0

    def _min_item(self) -> _HeapItem | None:
        if not self._items:
            return None
        return min(self._items, key=lambda item: (item.l2, -item.index))

    def _neighbor(self, index: int) -> _HeapItem | None:
        if self.min_separation <= 0:
            return None
        for item in self._items:
            if abs(item.index - index) < self.min_separation:
                return item
        return None

    def _would_enter(self, l2: float) -> bool:
        lowest = self._min_item()
        return lowest is None or len(self._items) < self.k or float(l2) > lowest.l2

    def _convert(self, images: dict[str, Any] | None) -> dict[str, np.ndarray] | None:
        if not images:
            return None
        converted: dict[str, np.ndarray] = {}
        try:
            for name, image in images.items():
                converted[str(name)] = _to_uint8_hwc(image)
        except Exception:  # noqa: BLE001 — evidence is fail-soft
            return None
        return converted

    def feed(self, index: int, l2: float, images: dict[str, Any] | None) -> None:
        score = float(l2)
        if not math.isfinite(score):
            return
        neighbor = self._neighbor(index)
        if neighbor is not None:
            if score <= neighbor.l2:
                return
            converted = self._convert(images)
            self._items.remove(neighbor)
            self._items.append(
                _HeapItem(l2=score, index=int(index), seq=self._seq, images=converted)
            )
            self._seq += 1
            return
        if not self._would_enter(score):
            return
        converted = self._convert(images)
        self._items.append(_HeapItem(l2=score, index=int(index), seq=self._seq, images=converted))
        self._seq += 1
        if len(self._items) > self.k:
            lowest = self._min_item()
            if lowest is not None:
                self._items.remove(lowest)

    def ranked(self) -> list[WorstFrame]:
        items = sorted(self._items, key=lambda item: (-item.l2, item.index))
        return [
            WorstFrame(index=item.index, l2=item.l2, images=item.images) for item in items
        ]


def nms_worst_indexes(l2: np.ndarray, k: int, min_separation: int) -> list[int]:
    scores = np.asarray(l2, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    selected: list[int] = []
    sep = max(int(min_separation), 0)
    for idx in order.tolist():
        if not math.isfinite(float(scores[idx])):
            continue
        if sep and any(abs(int(idx) - other) < sep for other in selected):
            continue
        selected.append(int(idx))
        if len(selected) >= k:
            break
    return selected


def min_separation_from_fps(fps: float) -> int:
    if fps <= 0:
        return 0
    return max(int(round(fps)), 0)


def per_frame_l2(pred: np.ndarray, rec: np.ndarray) -> np.ndarray:
    left = np.asarray(pred, dtype=np.float64)
    right = np.asarray(rec, dtype=np.float64)
    n = min(len(left), len(right))
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("action arrays must be 2-D (T, dim)")
    dim = min(left.shape[1], right.shape[1])
    return np.linalg.norm(left[:n, :dim] - right[:n, :dim], axis=1)


def bounds_violations(pred: np.ndarray, spec: ActionBounds) -> np.ndarray:
    predicted = np.asarray(pred, dtype=np.float64)
    low = np.asarray(spec.min, dtype=np.float64)
    high = np.asarray(spec.max, dtype=np.float64)
    if low.shape != high.shape:
        raise ValueError(f"bounds dim mismatch: min {low.shape} vs max {high.shape}")
    if predicted.ndim != 2:
        raise ValueError("predicted actions must be 2-D (T, dim)")
    if predicted.shape[1] != low.shape[0]:
        raise ValueError(
            f"action dim mismatch: predicted {predicted.shape[1]} vs bounds {low.shape[0]}"
        )
    return predicted - np.clip(predicted, low, high)


def should_write(results: list[AssertResult], mode: str) -> bool:
    if mode not in EVIDENCE_MODES:
        raise ValueError(f"unknown evidence mode {mode!r}; expected {list(EVIDENCE_MODES)}")
    if mode == "never":
        return False
    if mode == "always":
        return True
    return not overall_ok(results)


def write_evidence(
    run_dir: Path,
    scenario: Scenario,
    results: list[AssertResult],
    worst_frames: list[WorstFrame],
    pred: np.ndarray,
    rec: np.ndarray,
    t_ns: list[int] | np.ndarray,
    *,
    frames_skip_reason: str | None = None,
) -> Path:
    dest = Path(run_dir) / "evidence"
    dest.mkdir(parents=True, exist_ok=True)
    pred_arr = np.asarray(pred, dtype=np.float64)
    rec_arr = np.asarray(rec, dtype=np.float64)
    times = [int(item) for item in list(t_ns)]
    l2 = per_frame_l2(pred_arr, rec_arr)
    skipped: dict[str, str] = {}

    if frames_skip_reason:
        skipped["frames"] = frames_skip_reason
    try:
        written = _write_frame_pngs(dest, worst_frames)
    except Exception as exc:  # noqa: BLE001
        written = False
        skipped.setdefault("frames", f"{type(exc).__name__}: {exc}")
    if not written and "frames" not in skipped:
        skipped["frames"] = (
            "no images on adapter"
            if not any(item.images for item in worst_frames)
            else "png write failed"
        )

    bounds_spec = _bounds_spec(scenario)
    tracking = tracking_stats(pred_arr, rec_arr)
    try:
        _write_action_plot(
            dest / "actions.png",
            pred_arr,
            rec_arr,
            worst_frames,
            bounds_spec,
            max_l2=_deviation_max_l2(scenario),
            l2=l2,
            tracking=tracking,
        )
    except Exception as exc:  # noqa: BLE001
        skipped["plot"] = f"{type(exc).__name__}: {exc}"

    payload = {
        "scenario_id": scenario.id,
        "run_id": Path(run_dir).name,
        "mean_l2": float(np.mean(l2)) if len(l2) else 0.0,
        "worst_frames": [
            _worst_frame_payload(item, pred_arr, rec_arr, times) for item in worst_frames
        ],
        "bounds": _bounds_summary(pred_arr, bounds_spec),
        "tracking": tracking,
        "assertions": [item.to_dict() for item in results],
        "skipped": skipped,
    }
    (dest / "evidence.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dest


def rebuild_evidence(
    scenario: Scenario,
    run: Run,
    *,
    slice_: Slice | None = None,
    top_k: int = DEFAULT_TOP_K,
    force: bool = False,
) -> Path:
    dest = run.path / "evidence"
    marker = dest / "evidence.json"
    if marker.is_file() and not force:
        raise FileExistsError(f"evidence exists: {marker}; pass --force to overwrite")
    if dest.is_dir() and force:
        shutil.rmtree(dest)

    pred, rec, t_ns = load_action_arrays(run, scenario)
    l2 = per_frame_l2(pred, rec)
    sep = min_separation_from_fps(slice_.meta.fps) if slice_ is not None else 0
    indexes = nms_worst_indexes(l2, top_k, sep)
    decoded: dict[int, dict[str, Any]] | None = None
    frames_reason: str | None = None
    if slice_ is not None:
        decoded, frames_reason = try_decode_worst_frames(
            slice_,
            indexes,
            dataset_hint=_dataset_hint(scenario),
        )
    else:
        frames_reason = "video not local"

    worst: list[WorstFrame] = []
    for index in indexes:
        images = decoded.get(index) if decoded is not None else None
        converted = None
        if images:
            try:
                converted = {name: _to_uint8_hwc(image) for name, image in images.items()}
            except Exception:  # noqa: BLE001
                converted = None
        worst.append(WorstFrame(index=index, l2=float(l2[index]), images=converted))
    results = run_assertions(scenario, run)
    return write_evidence(
        run.path,
        scenario,
        results,
        worst,
        pred,
        rec,
        t_ns,
        frames_skip_reason=frames_reason,
    )


def load_action_arrays(
    run: Run, scenario: Scenario
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    topic = _action_topic(scenario)
    predicted = run.output(topic)
    recorded = run.output(topic, recorded=True)
    pred = vectors_from_frame(predicted)
    rec = vectors_from_frame(recorded)
    if "t_ns" not in predicted.columns:
        raise ValueError("action parquet must have a t_ns column")
    t_ns = [int(item) for item in predicted.get_column("t_ns").to_list()]
    return pred, rec, t_ns


def try_decode_worst_frames(
    slice_: Slice,
    indexes: list[int],
    *,
    dataset_hint: str | None = None,
) -> tuple[dict[int, dict[str, Any]] | None, str | None]:
    if not slice_.meta.videos:
        return None, "no videos in slice"
    root = _video_root(slice_, dataset_hint)
    if root is None:
        return None, "video not local"
    missing = [
        ref.path for ref in slice_.meta.videos.values() if not (root / ref.path).is_file()
    ]
    if missing:
        return None, "video not local"
    try:
        import av  # noqa: F401
    except ImportError:
        return None, "pyav not available"
    try:
        from robogate.replay.lerobot_policy import _SequentialPyAVDecoder
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"

    decoder = _SequentialPyAVDecoder()
    fps = float(slice_.meta.fps)
    if fps <= 0:
        return None, "invalid slice fps"
    out: dict[int, dict[str, Any]] = {}
    try:
        for index in indexes:
            images: dict[str, Any] = {}
            for cam, ref in slice_.meta.videos.items():
                timestamp = float(ref.from_timestamp) + float(index) / fps
                frames = decoder(root / ref.path, [timestamp], tolerance_s=max(1.0 / fps, 1e-3))
                images[_camera_slug(cam)] = frames[0]
            out[int(index)] = images
    except Exception as exc:  # noqa: BLE001
        return (out or None), f"{type(exc).__name__}: {exc}"
    return out, None


def _write_frame_pngs(dest: Path, worst_frames: list[WorstFrame]) -> bool:
    if not any(item.images for item in worst_frames):
        return False
    image_cls = _load_pil()
    wrote = False
    for item in worst_frames:
        item.files = []
        if not item.images:
            continue
        for cam, image in item.images.items():
            name = f"frame_{item.index:04d}_{_camera_slug(cam)}.png"
            try:
                image_cls.fromarray(np.asarray(image)).save(dest / name)
            except Exception:  # noqa: BLE001
                continue
            item.files.append(name)
            wrote = True
    if not wrote:
        raise RuntimeError("png write failed")
    return True


def tracking_stats(pred: np.ndarray, rec: np.ndarray) -> dict[str, Any]:
    left = np.asarray(pred, dtype=np.float64)
    right = np.asarray(rec, dtype=np.float64)
    n = min(len(left), len(right))
    dim = min(left.shape[1], right.shape[1]) if n and left.ndim == 2 and right.ndim == 2 else 0
    if n == 0 or dim == 0:
        return {
            "per_dim_corr": [],
            "median_corr": 0.0,
            "std_pred": [],
            "std_rec": [],
            "std_ratio": [],
            "mean_abs_diff_per_dim": [],
        }
    left, right = left[:n, :dim], right[:n, :dim]
    corrs: list[float] = []
    for d in range(dim):
        if left[:, d].std() < 1e-8 or right[:, d].std() < 1e-8:
            corrs.append(0.0)
            continue
        corrs.append(float(np.corrcoef(left[:, d], right[:, d])[0, 1]))
    std_pred = left.std(axis=0)
    std_rec = right.std(axis=0)
    ratio = [
        float(p / r) if r > 1e-12 else 0.0
        for p, r in zip(std_pred.tolist(), std_rec.tolist(), strict=True)
    ]
    return {
        "per_dim_corr": corrs,
        "median_corr": float(np.median(corrs)),
        "std_pred": [float(v) for v in std_pred.tolist()],
        "std_rec": [float(v) for v in std_rec.tolist()],
        "std_ratio": ratio,
        "mean_abs_diff_per_dim": np.mean(np.abs(left - right), axis=0).astype(float).tolist(),
    }


def _write_action_plot(
    dest: Path,
    pred: np.ndarray,
    rec: np.ndarray,
    worst_frames: list[WorstFrame],
    bounds: ActionBounds | None,
    *,
    max_l2: float | None = None,
    l2: np.ndarray | None = None,
    tracking: dict[str, Any] | None = None,
) -> None:
    plt = _load_mpl()
    n = min(pred.shape[1] if pred.ndim == 2 else 0, rec.shape[1] if rec.ndim == 2 else 0)
    if n == 0:
        raise ValueError("no action dimensions to plot")
    length = min(len(pred), len(rec))
    xs = np.arange(length)
    cols = 2 if n > 1 else 1
    rows = int(math.ceil(n / cols))
    fig = plt.figure(figsize=(10, 2.2 + 2.0 * rows), layout="constrained")
    grid = fig.add_gridspec(rows + 1, cols, height_ratios=[1.3] + [1.0] * rows)
    scores = np.asarray(l2 if l2 is not None else per_frame_l2(pred, rec), dtype=np.float64)
    ax_l2 = fig.add_subplot(grid[0, :])
    ax_l2.plot(xs, scores[:length], color="C1", linewidth=1.0, label="frame L2")
    if length:
        ax_l2.axhline(
            float(np.mean(scores[:length])),
            color="0.35",
            linestyle=":",
            linewidth=0.8,
            label="mean L2",
        )
    if max_l2 is not None:
        ax_l2.axhline(float(max_l2), color="C3", linestyle="--", linewidth=0.8, label="max_l2")
    marks = [item.index for item in worst_frames if 0 <= item.index < length]
    for index in marks:
        ax_l2.axvline(index, color="C3", linewidth=0.8, alpha=0.7)
    ax_l2.set_ylabel("L2")
    ax_l2.grid(True, alpha=0.25)
    ax_l2.legend(loc="upper right", fontsize=8)

    stats = tracking or tracking_stats(pred, rec)
    corrs = list(stats.get("per_dim_corr") or [])
    mean_abs = list(stats.get("mean_abs_diff_per_dim") or [])
    overflow = None
    if bounds is not None:
        try:
            overflow = bounds_violations(pred[:length], bounds)
        except Exception:  # noqa: BLE001
            overflow = None
    low = np.asarray(bounds.min, dtype=np.float64) if bounds is not None else None
    high = np.asarray(bounds.max, dtype=np.float64) if bounds is not None else None
    for dim in range(n):
        ax = fig.add_subplot(grid[1 + dim // cols, dim % cols])
        ax.plot(xs, rec[:length, dim], color="0.45", linewidth=1.0, label="recorded")
        ax.plot(xs, pred[:length, dim], color="C0", linewidth=1.0, label="pred")
        if low is not None and high is not None and dim < len(low):
            ax.axhline(float(low[dim]), color="0.5", linestyle="--", linewidth=0.7)
            ax.axhline(float(high[dim]), color="0.5", linestyle="--", linewidth=0.7)
        if overflow is not None and dim < overflow.shape[1]:
            hit = np.abs(overflow[:, dim]) > 1e-12
            if bool(np.any(hit)):
                ax.scatter(xs[hit], pred[:length, dim][hit], s=8, c="C3", zorder=3)
        for index in marks:
            ax.axvline(index, color="C3", linewidth=0.8, alpha=0.7)
        corr = corrs[dim] if dim < len(corrs) else 0.0
        delta = mean_abs[dim] if dim < len(mean_abs) else 0.0
        ax.set_ylabel(f"d{dim}  corr={corr:.2f}  mean|Δ|={delta:.2f}")
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("pred vs recorded")
    fig.savefig(dest, dpi=120)
    plt.close(fig)


def _worst_frame_payload(
    item: WorstFrame,
    pred: np.ndarray,
    rec: np.ndarray,
    t_ns: list[int],
) -> dict[str, Any]:
    stamp = t_ns[item.index] if 0 <= item.index < len(t_ns) else None
    diff: list[float] = []
    if 0 <= item.index < len(pred) and item.index < len(rec):
        dim = min(pred.shape[1], rec.shape[1])
        diff = np.abs(pred[item.index, :dim] - rec[item.index, :dim]).astype(float).tolist()
    return {
        "index": item.index,
        "t_ns": stamp,
        "l2": item.l2,
        "per_dim_abs_diff": diff,
        "files": list(item.files),
    }


def _bounds_summary(pred: np.ndarray, spec: ActionBounds | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    try:
        overflow = bounds_violations(pred, spec)
    except Exception:  # noqa: BLE001
        return None
    norms = np.linalg.norm(overflow, axis=1)
    out_frames = np.nonzero(np.any(np.abs(overflow) > 1e-12, axis=1))[0]
    out_dims = np.nonzero(np.any(np.abs(overflow) > 1e-12, axis=0))[0]
    worst = int(np.argmax(norms)) if len(norms) else None
    return {
        "n_frames_out": int(len(out_frames)),
        "dims_out": [int(i) for i in out_dims.tolist()],
        "worst_frame": worst,
    }


def _bounds_spec(scenario: Scenario) -> ActionBounds | None:
    for spec in scenario.expected:
        if isinstance(spec, ActionBounds):
            return spec
    return None


def _deviation_max_l2(scenario: Scenario) -> float | None:
    for spec in scenario.expected:
        if isinstance(spec, ActionDeviation):
            return float(spec.max_l2)
    return None


def _action_topic(scenario: Scenario) -> str:
    for spec in scenario.expected:
        topic = getattr(spec, "topic", None)
        if spec.type in {"action_deviation", "action_bounds"} and topic:
            return str(topic)
    return "action"


def _dataset_hint(scenario: Scenario) -> str | None:
    source = scenario.source
    return getattr(source, "dataset", None)


def _video_root(slice_: Slice, dataset_hint: str | None) -> Path | None:
    from robogate.replay.lerobot_policy import _dataset_root

    for name in (dataset_hint, slice_.meta.dataset):
        if not name:
            continue
        resolved = _dataset_root(name)
        if resolved is not None:
            return Path(resolved)
        path = Path(name)
        if (path / "meta" / "info.json").is_file():
            return path.resolve()
    return None


def _camera_slug(name: str) -> str:
    slug = name.removeprefix("observation.images.")
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in slug)


def _to_uint8_hwc(image: Any) -> np.ndarray:
    arr: Any = image
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    while arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"expected CHW or HWC image, got shape {arr.shape}")
    if arr.shape[0] in {1, 3} and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        peak = float(np.max(arr)) if arr.size else 1.0
        if peak <= 1.0 + 1e-6:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
        arr = np.rint(arr).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.ascontiguousarray(arr)


def _load_pil() -> Any:
    from PIL import Image

    return Image


def _load_mpl() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
