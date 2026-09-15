"""Derive mutant runs from a baseline run directory (CPU, no GPU)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from robogate.run import Run


def copy_run(src: str | Path, dest: str | Path) -> Path:
    dest_path = Path(dest)
    if dest_path.exists():
        shutil.rmtree(dest_path)
    shutil.copytree(src, dest_path)
    return dest_path


def synth_run(
    src: str | Path,
    dest: str | Path,
    *,
    kind: str,
    value: float | int | None = None,
    seed: int = 0,
) -> Path:
    dest_path = copy_run(src, dest)
    if kind == "bias":
        _map_action(dest_path, lambda arr: arr + float(value or 0.0))
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
