"""Summarize gate-benchmark parquets into docs/notes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from robogate.diff import diff_evals
from robogate.eval import fold_eval
from robogate.gate import run_gate
from robogate.scenario import content_hash
from robogate.suite import load_suite


def gate_red_rate(suite: Path, baseline: Path, candidate: Path) -> dict[str, Any]:
    payload = run_gate(suite, baseline, candidate)
    n = max(payload["n_scenarios"], 1)
    red = 0 if payload["ok"] else len(payload["reasons"])
    fold = fold_eval(pl.read_parquet(candidate))
    n_red_scen = sum(1 for status in fold.values() if status in {"fail", "error"})
    return {
        "ok": payload["ok"],
        "reasons": payload["reasons"],
        "red_reason_rate": red / n,
        "candidate_red_rate": n_red_scen / max(len(fold), 1),
        "by_scenario": fold,
    }


def ranking_consistency(t2k: Path, t10k: Path, t30k: Path) -> dict[str, Any]:
    d_low = {item["scenario_id"]: item["delta"] for item in diff_evals(t10k, t2k)["l2_deltas"]}
    d_high = {item["scenario_id"]: item["delta"] for item in diff_evals(t30k, t10k)["l2_deltas"]}
    ids = sorted(set(d_low) & set(d_high))
    ok = [sid for sid in ids if d_low[sid] > 0 and d_high[sid] > 0]
    return {
        "n": len(ids),
        "n_consistent": len(ok),
        "rate": (len(ok) / len(ids)) if ids else 0.0,
        "consistent": ok,
    }


def threshold_coverage(baseline: Path) -> dict[str, Any]:
    table = pl.read_parquet(baseline)
    fold = fold_eval(table)
    n = len(fold)
    n_pass = sum(1 for status in fold.values() if status == "pass")
    return {"n": n, "n_pass": n_pass, "rate": (n_pass / n) if n else 0.0, "by_scenario": fold}


def write_report(path: Path, payload: dict[str, Any]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {payload.get('title', 'Robogate gate benchmark')}",
        "",
        payload.get("intro", "Known-answer test of eval / diff / gate."),
        "",
        "## Metrics",
        "",
        f"- false-alarm rate: {payload.get('false_alarm')}",
        f"- detection: {payload.get('detection')}",
        f"- ranking consistency: {payload.get('ranking')}",
        f"- threshold coverage: {payload.get('coverage')}",
        "",
        "## Details",
        "",
        "```",
        _pretty(payload.get("details") or {}),
        "```",
        "",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def suite_hashes(suite: Path) -> dict[str, str]:
    return {scenario.id: content_hash(scenario) for _, scenario in load_suite(suite)}


def _pretty(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, ensure_ascii=False, default=str)
