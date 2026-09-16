"""Derive mutant runs from a baseline run directory (CPU, no GPU)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from robogate.run import Run

ACTION_STATS_KEY = "action"
STATE_STATS_KEY = "observation.state"


def copy_run(src: str | Path, dest: str | Path) -> Path:
    dest_path = Path(dest)
    if dest_path.exists():
        shutil.rmtree(dest_path)
    shutil.copytree(src, dest_path)
    return dest_path


def apply_dim_bias(arr: np.ndarray, dim: int, value: float) -> np.ndarray:
    if dim < 0 or dim >= arr.shape[1]:
        raise ValueError(f"dim {dim} out of range for action width {arr.shape[1]}")
    out = np.asarray(arr, dtype=np.float64).copy()
    out[:, dim] += float(value)
    return out


def apply_gain(arr: np.ndarray, gain: float) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    mean = values.mean(axis=0, keepdims=True)
    return mean + (values - mean) * float(gain)


def apply_constant(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    if len(values) == 0:
        return values.copy()
    return np.broadcast_to(values[0], values.shape).copy()


def apply_stats_swap(
    arr: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    mean_a = np.asarray(action_mean, dtype=np.float64).reshape(-1)
    std_a = np.asarray(action_std, dtype=np.float64).reshape(-1)
    mean_s = np.asarray(state_mean, dtype=np.float64).reshape(-1)
    std_s = np.asarray(state_std, dtype=np.float64).reshape(-1)
    width = values.shape[1]
    if any(vec.shape != (width,) for vec in (mean_a, std_a, mean_s, std_s)):
        raise ValueError("stats_swap vectors must match action width")
    safe = np.where(np.abs(std_a) < 1e-12, 1.0, std_a)
    return (values - mean_a) / safe * std_s + mean_s


def load_swap_stats(
    stats: dict[str, Any] | None = None,
    stats_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = stats
    if payload is None:
        if stats_path is None:
            raise ValueError("stats_swap needs stats= or stats_path=")
        payload = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    if ACTION_STATS_KEY not in payload or STATE_STATS_KEY not in payload:
        raise ValueError("stats_swap needs 'action' and 'observation.state' blocks")
    action = payload[ACTION_STATS_KEY]
    state = payload[STATE_STATS_KEY]
    return (
        _stats_vector(action, "mean"),
        _stats_vector(action, "std"),
        _stats_vector(state, "mean"),
        _stats_vector(state, "std"),
    )


def synth_run(
    src: str | Path,
    dest: str | Path,
    *,
    kind: str,
    value: float | int | None = None,
    seed: int = 0,
    dim: int | None = None,
    stats: dict[str, Any] | None = None,
    stats_path: str | Path | None = None,
) -> Path:
    dest_path = copy_run(src, dest)
    if kind == "bias":
        _map_action(dest_path, lambda arr: arr + float(value or 0.0))
    elif kind == "dim_bias":
        if dim is None:
            raise ValueError("dim_bias requires dim=")
        _map_action(dest_path, lambda arr: apply_dim_bias(arr, dim, float(value or 0.0)))
    elif kind == "gain":
        if value is None:
            raise ValueError("gain requires value=")
        _map_action(dest_path, lambda arr: apply_gain(arr, float(value)))
    elif kind == "stats_swap":
        mean_a, std_a, mean_s, std_s = load_swap_stats(stats=stats, stats_path=stats_path)
        _map_action(dest_path, lambda arr: apply_stats_swap(arr, mean_a, std_a, mean_s, std_s))
    elif kind == "constant":
        _map_action(dest_path, apply_constant)
    elif kind == "noise":
        rng = np.random.default_rng(seed)
        sigma = float(value or 0.0)
        _map_action(dest_path, lambda arr: arr + rng.normal(0.0, sigma, size=arr.shape))
    elif kind == "lag":
        _map_action(dest_path, lambda arr: _lag(arr, int(value or 0)))
    elif kind == "drop_recorded":
        path = dest_path / "outputs" / "action.recorded.parquet"
        if path.is_file():
            path.unlink()
    elif kind == "dim13":
        _map_action(dest_path, lambda arr: arr[:, :13] if arr.shape[1] >= 13 else arr)
    elif kind == "wrong_id":
        _patch_meta(dest_path, scenario_id="mutant-wrong-id")
    elif kind == "wrong_hash":
        _patch_meta(dest_path, scenario_hash="sha256:deadbeef")
    else:
        raise ValueError(f"unknown synth kind {kind!r}")
    return dest_path


def corrupt_eval_hashes(src: str | Path, dest: str | Path, digest: str = "sha256:deadbeef") -> Path:
    table = pl.read_parquet(src).with_columns(pl.lit(digest).alias("scenario_hash"))
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(dest_path)
    return dest_path


def _stats_vector(block: Any, key: str) -> np.ndarray:
    if not isinstance(block, dict) or key not in block:
        raise ValueError(f"stats block missing {key}")
    return np.asarray(block[key], dtype=np.float64).reshape(-1)


def _map_action(run_dir: Path, fn: Any) -> None:
    path = run_dir / "outputs" / "action.parquet"
    frame = pl.read_parquet(path)
    values = np.asarray(frame.get_column("value").to_list(), dtype=np.float64)
    updated = fn(values)
    pl.DataFrame({"t_ns": frame.get_column("t_ns"), "value": updated.tolist()}).write_parquet(path)


def _lag(arr: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 0 or len(arr) == 0:
        return arr
    out = np.empty_like(arr)
    out[:steps] = arr[0]
    out[steps:] = arr[:-steps]
    return out


def _patch_meta(run_dir: Path, **fields: Any) -> None:
    run = Run.load(run_dir)
    payload = json.loads(run.meta.model_dump_json(exclude_none=True))
    payload.update(fields)
    (run_dir / "meta.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
