"""Compare two eval parquets scene-by-scene."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from robogate.eval import fold_eval


def diff_evals(
    path_a: str | Path,
    path_b: str | Path,
    *,
    runs_a: Path | None = None,
    runs_b: Path | None = None,
    search_frames: int = 50,
) -> dict[str, Any]:
    table_a = pl.read_parquet(path_a)
    table_b = pl.read_parquet(path_b)
    fold_a = fold_eval(table_a)
    fold_b = fold_eval(table_b)
    ids = sorted(set(fold_a) | set(fold_b))
    regressions: list[dict[str, str]] = []
    fixes: list[dict[str, str]] = []
    for sid in ids:
        left = fold_a.get(sid, "missing")
        right = fold_b.get(sid, "missing")
        if left == "pass" and right in {"fail", "error"}:
            regressions.append({"scenario_id": sid, "from": left, "to": right})
        if left in {"fail", "error"} and right == "pass":
            fixes.append({"scenario_id": sid, "from": left, "to": right})
    payload = {
        "n_scenarios": len(ids),
        "n_regressions": len(regressions),
        "n_fixes": len(fixes),
        "regressions": regressions,
        "fixes": fixes,
        "l2_deltas": _l2_deltas(table_a, table_b),
        "status_a": fold_a,
        "status_b": fold_b,
    }
    if runs_a is not None and runs_b is not None:
        payload["pred_shift"] = pred_shifts(runs_a, runs_b, ids, search_frames=search_frames)
    return payload


def pred_shifts(
    runs_a: Path,
    runs_b: Path,
    scenario_ids: list[str],
    *,
    search_frames: int = 50,
    version_contains_a: str | None = None,
    version_contains_b: str | None = None,
) -> list[dict[str, Any]]:
    from robogate.asserts.action_deviation import vectors_from_frame
    from robogate.asserts.action_lag import estimate_lag
    from robogate.eval import find_run
    from robogate.run import Run

    out: list[dict[str, Any]] = []
    for sid in scenario_ids:
        pred_a = vectors_from_frame(
            Run.load(
                find_run(runs_a, sid, version_contains=version_contains_a, unperturbed=True)
            ).output("action")
        )
        pred_b = vectors_from_frame(
            Run.load(find_run(runs_b, sid, version_contains=version_contains_b)).output("action")
        )
        lag, best_l2, zero_l2 = estimate_lag(pred_b, pred_a, search_frames=search_frames)
        out.append(
            {
                "scenario_id": sid,
                "shift": lag,
                "l2_aligned": best_l2,
                "l2_zero": zero_l2,
            }
        )
    return out


def _l2_deltas(table_a: pl.DataFrame, table_b: pl.DataFrame) -> list[dict[str, Any]]:
    def pick(table: pl.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        if table.height == 0:
            return out
        match = table.filter(pl.col("type") == "action_deviation")
        for row in match.iter_rows(named=True):
            if row["measured"] is None:
                continue
            out[row["scenario_id"]] = float(row["measured"])
        return out

    left = pick(table_a)
    right = pick(table_b)
    deltas = []
    for sid in sorted(set(left) | set(right)):
        if sid not in left or sid not in right:
            continue
        deltas.append(
            {
                "scenario_id": sid,
                "measured_a": left[sid],
                "measured_b": right[sid],
                "delta": right[sid] - left[sid],
            }
        )
    return deltas
