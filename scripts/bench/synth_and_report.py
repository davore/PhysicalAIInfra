#!/usr/bin/env python3
"""CPU: synthesize mutant runs from B, eval/gate them, write the bench note."""

from __future__ import annotations

import json
from pathlib import Path

from robogate.bench.calib import chunk_drift, inter_demo_l2
from robogate.bench.report import (
    gate_red_rate,
    ranking_consistency,
    threshold_coverage,
    write_report,
)
from robogate.bench.synth import corrupt_eval_hashes, synth_run
from robogate.eval import run_eval
from robogate.gate import run_gate
from robogate.run import Run
from robogate.suite import load_suite


def _latest_run(runs_root: Path, scenario_id: str) -> Path:
    matches = []
    for meta in runs_root.glob("*/meta.json"):
        try:
            run = Run.load(meta.parent)
        except Exception:  # noqa: BLE001
            continue
        version = run.meta.target_version or ""
        if run.meta.scenario_id != scenario_id:
            continue
        if run.meta.perturbations:
            continue
        if "act-aloha-static-coffee-test" not in version and "identity" not in version:
            continue
        matches.append(meta.parent)
    if not matches:
        raise FileNotFoundError(f"no clean run for {scenario_id}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    suite = root / "scenarios" / "real"
    runs = root / "runs"
    results = root / "results"
    slices = root / "slices"
    mutants = root / "runs" / "mutants"
    mutants.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    items = load_suite(suite)
    ids = [scenario.id for _, scenario in items]
    baseline = results / "B-calib.parquet"
    if not baseline.is_file():
        baseline = results / "B.parquet"
    if not baseline.is_file():
        raise FileNotFoundError("need results/B-calib.parquet or results/B.parquet")

    kinds = [
        ("bias-0.1", "bias", 0.1),
        ("bias-0.5", "bias", 0.5),
        ("noise-0.05", "noise", 0.05),
        ("noise-0.2", "noise", 0.2),
        ("lag-10", "lag", 10),
        ("drop-recorded", "drop_recorded", None),
        ("dim13", "dim13", None),
        ("wrong-id", "wrong_id", None),
    ]
    detection: dict[str, object] = {}
    for name, kind, value in kinds:
        dest_root = mutants / name
        dest_root.mkdir(parents=True, exist_ok=True)
        for sid in ids:
            src = _latest_run(runs, sid)
            synth_run(src, dest_root / src.name, kind=kind, value=value)
        parquet = run_eval(suite, runs_root=dest_root, out_root=results, eval_id=f"mutant-{name}")
        detection[name] = gate_red_rate(suite, baseline, parquet)

    hash_src = results / "B-calib.parquet"
    if not hash_src.is_file():
        hash_src = results / "B.parquet"
    corrupt = corrupt_eval_hashes(hash_src, results / "mutant-wrong-hash.parquet")
    detection["wrong-hash"] = run_gate(suite, baseline, corrupt)

    rerun = results / "B-rerun.parquet"
    false_alarm = None
    if rerun.is_file():
        false_alarm = gate_red_rate(suite, baseline, rerun)

    ranking = None
    if all((results / name).is_file() for name in ("T2k.parquet", "T10k.parquet", "T30k.parquet")):
        ranking = ranking_consistency(
            results / "T2k.parquet",
            results / "T10k.parquet",
            results / "T30k.parquet",
        )

    demo = inter_demo_l2(slices, ids) if (slices / ids[0]).is_dir() else {}
    drift = None
    try:
        drift = chunk_drift(_latest_run(runs, ids[0]))
    except Exception as exc:  # noqa: BLE001
        drift = {"error": str(exc)}

    coverage = threshold_coverage(baseline)
    note = root / "docs" / "notes" / "2026-09-14-gate-bench.md"
    write_report(
        note,
        {
            "title": "Robogate gate benchmark (2026-09-14)",
            "intro": (
                "Known-answer test of eval / diff / gate. "
                "Mutants are offline (bias/noise/lag/missing/wrong dim/id/hash). "
                "drop_camera is reported only; direction is not guaranteed."
            ),
            "false_alarm": false_alarm,
            "detection": {
                k: (v.get("ok") if isinstance(v, dict) else v) for k, v in detection.items()
            },
            "ranking": ranking,
            "coverage": coverage,
            "details": {
                "false_alarm": false_alarm,
                "detection": detection,
                "ranking": ranking,
                "coverage": coverage,
                "inter_demo_l2": {k: demo.get(k) for k in ("n_pairs", "mean", "std")},
                "chunk_drift": {
                    "monotonic_rising": None if drift is None else drift.get("monotonic_rising"),
                    "mean_l2_by_offset": None if drift is None else drift.get("mean_l2_by_offset"),
                    "error": None if drift is None else drift.get("error"),
                },
            },
        },
    )
    print(
        json.dumps(
            {"note": str(note), "coverage": coverage, "false_alarm": false_alarm},
            default=str,
        )
    )


if __name__ == "__main__":
    main()
