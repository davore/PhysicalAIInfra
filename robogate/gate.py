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
    *,
    runs_baseline: Path | None = None,
    runs_candidate: Path | None = None,
    max_shift: int | None = None,
    search_frames: int = 50,
    version_contains_baseline: str | None = None,
    version_contains_candidate: str | None = None,
) -> dict[str, Any]:
    items = load_suite(suite)
    current = {scenario.id: (scenario, content_hash(scenario)) for _, scenario in items}
    table_a = pl.read_parquet(baseline)
    table_b = pl.read_parquet(candidate)
    fold_a = fold_eval(table_a)
    fold_b = fold_eval(table_b)
    hash_b = _hashes(table_b)

    reasons: list[str] = []
    shifts: list[dict[str, Any]] = []
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

    if runs_baseline is not None and runs_candidate is not None and max_shift is not None:
        from robogate.diff import pred_shifts

        shifts = pred_shifts(
            runs_baseline,
            runs_candidate,
            list(current),
            search_frames=search_frames,
            version_contains_a=version_contains_baseline,
            version_contains_b=version_contains_candidate,
        )
        for item in shifts:
            if abs(int(item["shift"])) > max_shift:
                reasons.append(
                    f"{item['scenario_id']}: pred_shift {item['shift']} "
                    f"exceeds max_shift {max_shift}"
                )

    evidence = _candidate_evidence(
        fold_b,
        current,
        runs_candidate,
        version_contains=version_contains_candidate,
    )
    return {
        "ok": not reasons,
        "n_scenarios": len(current),
        "reasons": reasons,
        "baseline": fold_a,
        "candidate": fold_b,
        "pred_shift": shifts,
        "evidence": evidence,
    }


def _candidate_evidence(
    fold_b: dict[str, str],
    current: dict[str, Any],
    runs_candidate: Path | None,
    *,
    version_contains: str | None,
) -> list[dict[str, str]]:
    if runs_candidate is None:
        return []
    from robogate.eval import find_run

    items: list[dict[str, str]] = []
    for sid, status in fold_b.items():
        if status not in {"fail", "error"} or sid not in current:
            continue
        try:
            run_path = find_run(
                runs_candidate,
                sid,
                version_contains=version_contains,
            )
        except FileNotFoundError:
            continue
        ev_dir = run_path / "evidence"
        if not (ev_dir / "evidence.json").is_file():
            continue
        items.append(
            {
                "scenario_id": sid,
                "run_id": run_path.name,
                "path": str(ev_dir),
            }
        )
    return items


def _hashes(table: pl.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    if table.height == 0:
        return out
    for row in table.iter_rows(named=True):
        out[row["scenario_id"]] = row["scenario_hash"]
    return out
