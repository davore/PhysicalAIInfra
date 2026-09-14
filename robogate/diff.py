"""Compare two eval parquets scene-by-scene."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from robogate.eval import fold_eval


def diff_evals(path_a: str | Path, path_b: str | Path) -> dict[str, Any]:
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
    return {
        "n_scenarios": len(ids),
        "n_regressions": len(regressions),
        "n_fixes": len(fixes),
        "regressions": regressions,
        "fixes": fixes,
        "l2_deltas": _l2_deltas(table_a, table_b),
        "status_a": fold_a,
        "status_b": fold_b,
    }


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
