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


def recorded_envelope(
    slice_root: Path,
    scenario_ids: list[str],
    *,
    pad: float = 0.10,
) -> tuple[list[float], list[float]]:
    blocks: list[np.ndarray] = []
    for sid in scenario_ids:
        recorded = Slice.load(Path(slice_root) / sid).recorded_action()
        blocks.append(np.asarray(recorded.get_column("value").to_list(), dtype=np.float64))
    if not blocks:
        raise FileNotFoundError(f"no recorded actions under {slice_root}")
    stacked = np.concatenate(blocks, axis=0)
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    return (lo - pad * span).tolist(), (hi + pad * span).tolist()


def predicted_envelope(
    runs_root: Path,
    scenario_ids: list[str],
    *,
    pad: float = 0.10,
) -> tuple[list[float], list[float]]:
    """Deprecated alias kept for call sites that still pass runs."""
    del pad
    raise RuntimeError("use recorded_envelope(slice_root, ids)")


def lag_from_runs(
    runs_root: Path,
    scenario_ids: list[str],
    *,
    search_frames: int = 50,
    version_contains: str | None = None,
) -> dict[str, Any]:
    from robogate.asserts.action_lag import estimate_lag
    from robogate.eval import find_run

    lags: dict[str, int] = {}
    for sid in scenario_ids:
        try:
            run = Run.load(
                find_run(
                    runs_root,
                    sid,
                    version_contains=version_contains,
                    unperturbed=True,
                )
            )
            pred = np.asarray(run.output("action").get_column("value").to_list(), dtype=np.float64)
            rec = np.asarray(
                run.output("action", recorded=True).get_column("value").to_list(),
                dtype=np.float64,
            )
            lag, _, _ = estimate_lag(pred, rec, search_frames=search_frames)
        except Exception:  # noqa: BLE001
            continue
        lags[sid] = int(lag)
    abs_lags = [abs(v) for v in lags.values()]
    return {
        "by_scenario": lags,
        "max_abs": max(abs_lags) if abs_lags else 0,
        "max_lag_frames": (max(abs_lags) if abs_lags else 0) + 2,
    }


def thresholds_from_eval(
    parquet: Path,
    runs_root: Path,
    suite: Path,
    *,
    slice_root: Path = Path("slices"),
    bounds_pad: float = 0.10,
    version_contains: str | None = None,
) -> dict[str, Any]:
    table = pl.read_parquet(parquet)
    items = load_suite(suite)
    ids = [scenario.id for _, scenario in items]
    l2 = measured_by_type(table, "action_deviation")
    delta = measured_by_type(table, "action_smoothness")
    l2_rule = rule_mean_plus_3sigma(list(l2.values()))
    delta_rule = rule_mean_plus_3sigma(list(delta.values()))
    mins, maxs = recorded_envelope(slice_root, ids, pad=bounds_pad)
    lag = lag_from_runs(runs_root, ids, version_contains=version_contains)
    pct = int(round(bounds_pad * 100))
    return {
        "max_l2": max(l2_rule["mean_plus_3sigma"], 1e-6),
        "max_delta": max(delta_rule["mean_plus_3sigma"], 1e-6),
        "bounds_min": mins,
        "bounds_max": maxs,
        "max_lag_frames": int(lag["max_lag_frames"]),
        "l2_rule": l2_rule,
        "delta_rule": delta_rule,
        "lag": lag,
        "n_scenarios": len(ids),
        "bounds_source": f"recorded+{pct}%",
        "bounds_pad": bounds_pad,
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
            f"# bounds={thresholds.get('bounds_source', 'recorded+10%')}\n"
            f"# max_lag_frames={thresholds.get('max_lag_frames', 2)}\n"
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
        elif item.type == "action_lag" and "max_lag_frames" in thresholds:
            data["max_lag_frames"] = int(thresholds["max_lag_frames"])
        expected.append(data)
    has_lag = any(item.get("type") == "action_lag" for item in expected)
    if "max_lag_frames" in thresholds and not has_lag:
        expected.append(
            {
                "type": "action_lag",
                "topic": "action",
                "reference": "recorded",
                "max_lag_frames": int(thresholds["max_lag_frames"]),
                "search_frames": 50,
            }
        )
    return expected
