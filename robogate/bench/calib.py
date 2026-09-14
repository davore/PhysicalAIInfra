"""Calibrate assertion thresholds from a baseline eval + runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from robogate.run import Run
from robogate.scenario import Scenario
from robogate.slice import Slice
from robogate.suite import load_suite


def inter_demo_l2(slice_root: Path, scenario_ids: list[str]) -> dict[str, Any]:
    actions: dict[str, np.ndarray] = {}
    for sid in scenario_ids:
        recorded = Slice.load(Path(slice_root) / sid).recorded_action()
        actions[sid] = np.asarray(recorded.get_column("value").to_list(), dtype=np.float64)
    pairs: list[dict[str, Any]] = []
    ids = list(actions)
    for i, left_id in enumerate(ids):
        for right_id in ids[i + 1 :]:
            left, right = actions[left_id], actions[right_id]
            n = min(len(left), len(right))
            if n == 0:
                continue
            measured = float(np.mean(np.linalg.norm(left[:n] - right[:n], axis=1)))
            pairs.append({"a": left_id, "b": right_id, "mean_l2": measured})
    values = [item["mean_l2"] for item in pairs]
    return {
        "n_pairs": len(pairs),
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "pairs": pairs,
    }


def chunk_drift(run_dir: Path, *, chunk_size: int = 100) -> dict[str, Any]:
    run = Run.load(run_dir)
    pred = np.asarray(run.output("action").get_column("value").to_list(), dtype=np.float64)
    recorded = run.output("action", recorded=True).get_column("value").to_list()
    ref = np.asarray(recorded, dtype=np.float64)
    n = min(len(pred), len(ref))
    per_frame = np.linalg.norm(pred[:n] - ref[:n], axis=1)
    buckets = {int(i): [] for i in range(chunk_size)}
    for i, value in enumerate(per_frame.tolist()):
        buckets[i % chunk_size].append(value)
    means = [float(np.mean(buckets[i])) if buckets[i] else None for i in range(chunk_size)]
    present = [item for item in means if item is not None]
    rising = all(present[i] <= present[i + 1] + 1e-9 for i in range(len(present) - 1))
    return {"chunk_size": chunk_size, "mean_l2_by_offset": means, "monotonic_rising": rising}


def measured_by_type(table: pl.DataFrame, type_name: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if table.height == 0:
        return out
    for row in table.filter(pl.col("type") == type_name).iter_rows(named=True):
        if row["measured"] is None:
            continue
        out[row["scenario_id"]] = float(row["measured"])
    return out


def rule_mean_plus_3sigma(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) else 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    p95 = float(np.quantile(arr, 0.95)) if len(arr) else 0.0
    return {
        "mean": mean,
        "std": std,
        "p95": p95,
        "mean_plus_3sigma": mean + 3.0 * std,
    }


def predicted_envelope(
    runs_root: Path,
    scenario_ids: list[str],
    *,
    pad: float = 0.10,
) -> tuple[list[float], list[float]]:
    blocks: list[np.ndarray] = []
    for sid in scenario_ids:
        matches = []
        for meta in Path(runs_root).glob("*/meta.json"):
            try:
                run = Run.load(meta.parent)
            except Exception:  # noqa: BLE001
                continue
            if run.meta.scenario_id == sid:
                matches.append(meta.parent)
        if not matches:
            continue
        latest = max(matches, key=lambda path: path.stat().st_mtime)
        values = Run.load(latest).output("action").get_column("value").to_list()
        pred = np.asarray(values, dtype=np.float64)
        blocks.append(pred)
    if not blocks:
        raise FileNotFoundError(f"no predicted actions under {runs_root}")
    stacked = np.concatenate(blocks, axis=0)
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    return (lo - pad * span).tolist(), (hi + pad * span).tolist()


def thresholds_from_eval(
    parquet: Path,
    runs_root: Path,
    suite: Path,
) -> dict[str, Any]:
    table = pl.read_parquet(parquet)
    items = load_suite(suite)
    ids = [scenario.id for _, scenario in items]
    l2 = measured_by_type(table, "action_deviation")
    delta = measured_by_type(table, "action_smoothness")
    l2_rule = rule_mean_plus_3sigma(list(l2.values()))
    delta_rule = rule_mean_plus_3sigma(list(delta.values()))
    mins, maxs = predicted_envelope(runs_root, ids)
    return {
        "max_l2": max(l2_rule["mean_plus_3sigma"], 1e-6),
        "max_delta": max(delta_rule["mean_plus_3sigma"], 1e-6),
        "bounds_min": mins,
        "bounds_max": maxs,
        "l2_rule": l2_rule,
        "delta_rule": delta_rule,
        "n_scenarios": len(ids),
    }


def write_suite_thresholds(
    suite: Path,
    thresholds: dict[str, Any],
    *,
    eval_id: str,
) -> list[Path]:
    written: list[Path] = []
    for path, scenario in load_suite(suite):
        updated = Scenario.model_validate(
            {
                **scenario.model_dump(mode="json"),
                "expected": _apply_thresholds(scenario, thresholds),
            }
        )
        header = (
            f"# thresholds from eval_id={eval_id}\n"
            f"# max_l2=mean+3σ={thresholds['max_l2']}\n"
            f"# max_delta=mean+3σ={thresholds['max_delta']}\n"
        )
        body = yaml.safe_dump(
            updated.model_dump(mode="json", exclude_none=True),
            sort_keys=False,
            allow_unicode=True,
        )
        path.write_text(header + body, encoding="utf-8")
        written.append(path)
    return written


def _apply_thresholds(scenario: Scenario, thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    expected = []
    for item in scenario.expected:
        data = item.model_dump(mode="json")
        if item.type == "action_deviation":
            data["max_l2"] = float(thresholds["max_l2"])
        elif item.type == "action_smoothness":
            data["max_delta"] = float(thresholds["max_delta"])
        elif item.type == "action_bounds":
            data["min"] = [float(v) for v in thresholds["bounds_min"]]
            data["max"] = [float(v) for v in thresholds["bounds_max"]]
        expected.append(data)
    return expected
