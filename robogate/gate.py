"""CI gate: baseline-pass must not fail; blocking must pass."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from robogate.eval import fold_eval
from robogate.scenario import content_hash
from robogate.suite import load_suite


def run_gate(
    suite: Path,
    baseline: Path,
    candidate: Path,
) -> dict[str, Any]:
    items = load_suite(suite)
    current = {scenario.id: (scenario, content_hash(scenario)) for _, scenario in items}
    table_a = pl.read_parquet(baseline)
    table_b = pl.read_parquet(candidate)
    fold_a = fold_eval(table_a)
    fold_b = fold_eval(table_b)
    hash_b = _hashes(table_b)

    reasons: list[str] = []
    for sid, (scenario, digest) in current.items():
        cand_hash = hash_b.get(sid)
        if cand_hash is not None and cand_hash != digest:
            reasons.append(f"{sid}: scenario_hash {cand_hash} != current {digest}")
            continue
        cand = fold_b.get(sid)
        if cand is None:
            reasons.append(f"{sid}: missing from candidate")
            continue
        if fold_a.get(sid) == "pass" and cand in {"fail", "error"}:
            reasons.append(f"{sid}: baseline pass became {cand}")
        if scenario.blocking and cand != "pass":
            reasons.append(f"{sid}: blocking is {cand}")

    return {
        "ok": not reasons,
        "n_scenarios": len(current),
        "reasons": reasons,
        "baseline": fold_a,
        "candidate": fold_b,
    }


def _hashes(table: pl.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    if table.height == 0:
        return out
    for row in table.iter_rows(named=True):
        out[row["scenario_id"]] = row["scenario_hash"]
    return out
