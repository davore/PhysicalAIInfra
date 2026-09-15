#!/usr/bin/env python3
"""Compare B unnormalizer stats to dataset actions; optional cross-run check."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _stats_from_processor(pipe: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for step in getattr(pipe, "steps", None) or []:
        name = type(step).__name__
        for attr in ("mean", "std", "stats", "norm_map"):
            value = getattr(step, attr, None)
            if value is None:
                continue
            out[f"{name}.{attr}"] = _jsonable(value)
    return out


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(type(value))


def _dataset_action_stats(repo_id: str) -> dict[str, Any]:
    import polars as pl

    from robogate.replay.lerobot_policy import _dataset_root

    root = _dataset_root(repo_id)
    if root is None:
        return {"error": "dataset root not found"}
    files = sorted(Path(root).glob("data/**/*.parquet"))
    if not files:
        return {"error": f"no parquet under {root}"}
    table = pl.concat([pl.read_parquet(path, columns=["action"]) for path in files[:8]])
    arr = np.asarray(table.get_column("action").to_list(), dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean": arr.mean(0).tolist(),
        "std": arr.std(0).tolist(),
        "min": arr.min(0).tolist(),
        "max": arr.max(0).tolist(),
    }


def _pick_run(runs_root: Path, scenario_id: str, *, needle: str) -> Path | None:
    from robogate.run import Run

    matches: list[Path] = []
    for meta in Path(runs_root).glob("*/meta.json"):
        try:
            run = Run.load(meta.parent)
        except Exception:  # noqa: BLE001
            continue
        version = run.meta.target_version or ""
        if run.meta.scenario_id != scenario_id:
            continue
        if run.meta.perturbations:
            continue
        if needle not in version:
            continue
        matches.append(meta.parent)
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _cross_run(runs_root: Path, *, needle: str = "act-aloha-static-coffee-test") -> dict[str, Any]:
    from robogate.run import Run

    path_a = _pick_run(runs_root, "aloha-static-coffee-ep0", needle=needle)
    path_b = _pick_run(runs_root, "aloha-static-coffee-ep1", needle=needle)
    if path_a is None or path_b is None:
        return {"error": f"no unperturbed runs matching {needle}"}
    a = np.asarray(
        Run.load(path_a).output("action").get_column("value").to_list(),
        dtype=np.float64,
    )
    b = np.asarray(
        Run.load(path_b).output("action").get_column("value").to_list(),
        dtype=np.float64,
    )
    n = min(len(a), len(b))
    return {
        "run_a": path_a.name,
        "run_b": path_b.name,
        "mean_abs_diff": float(np.mean(np.abs(a[:n] - b[:n]))),
        "max_abs_diff": float(np.max(np.abs(a[:n] - b[:n]))),
        "obs_reaches_model": bool(np.mean(np.abs(a[:n] - b[:n])) > 1e-4),
    }


def main() -> None:
    runs_root = Path(os.environ.get("RUNS", "runs"))
    needle = os.environ.get("BASE_RUN_NEEDLE", "act-aloha-static-coffee-test")
    if os.environ.get("DIAGNOSE_CROSS_ONLY") == "1":
        print(json.dumps(_cross_run(runs_root, needle=needle), indent=2))
        return

    from robogate.replay.lerobot_policy import _isolate_lerobot_imports, _load_processors

    repo_id = os.environ.get("ACT_REPO", "lerobot/aloha_static_coffee")
    ckpt = os.environ.get("ACT_BASE_CKPT", "gozdebaydogmus/act-aloha-static-coffee-test")
    rev = os.environ.get("ACT_BASE_REV", "646846823c8473712f689f59bdd03f798a688f6a")
    _isolate_lerobot_imports()
    pre, post = _load_processors(ckpt, revision=rev, device="cpu")
    payload = {
        "checkpoint": f"{ckpt}@{rev}",
        "preprocessor": _stats_from_processor(pre),
        "postprocessor": _stats_from_processor(post),
        "dataset_action": _dataset_action_stats(repo_id),
        "cross_run_ep0_ep1": _cross_run(runs_root, needle=needle),
    }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
