#!/usr/bin/env python3
"""CPU: synthesize mutant runs from B, eval/gate them, write the v2 bench note."""

from __future__ import annotations

import json
import os
from pathlib import Path

from robogate.bench.calib import chunk_drift, inter_demo_l2
from robogate.bench.report import (
    gate_red_rate,
    ranking_consistency,
    threshold_coverage,
    write_report,
)
from robogate.bench.synth import corrupt_eval_hashes, synth_run
from robogate.diff import pred_shifts
from robogate.eval import run_eval
from robogate.gate import run_gate
from robogate.run import Run
from robogate.suite import load_suite


def _latest_run(runs_root: Path, scenario_id: str, *, needle: str) -> Path:
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
        if needle not in version:
            continue
        matches.append(meta.parent)
    if not matches:
        raise FileNotFoundError(f"no clean run for {scenario_id} matching {needle}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    suite = Path(os.environ.get("SUITE", root / "scenarios" / "real"))
    runs = Path(os.environ.get("RUNS", root / "runs"))
    results = Path(os.environ.get("RESULTS", root / "results"))
    slices = Path(os.environ.get("SLICES", root / "slices"))
    mutants = root / "runs" / "mutants"
    mutants.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    needle = os.environ.get("BASE_RUN_NEEDLE", "act-aloha-static-coffee-test")
    note = Path(
        os.environ.get(
            "BENCH_NOTE",
            str(root / "docs" / "notes" / "2026-09-15-gate-bench-v2.md"),
        )
    )

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
    lag_runs: Path | None = None
    for name, kind, value in kinds:
        dest_root = mutants / name
        dest_root.mkdir(parents=True, exist_ok=True)
        for sid in ids:
            src = _latest_run(runs, sid, needle=needle)
            synth_run(src, dest_root / src.name, kind=kind, value=value)
        parquet = run_eval(suite, runs_root=dest_root, out_root=results, eval_id=f"mutant-{name}")
        detection[name] = gate_red_rate(suite, baseline, parquet)
        if name == "lag-10":
            lag_runs = dest_root

    hash_src = results / "B-calib.parquet"
    if not hash_src.is_file():
        hash_src = results / "B.parquet"
    corrupt = corrupt_eval_hashes(hash_src, results / "mutant-wrong-hash.parquet")
    detection["wrong-hash"] = run_gate(suite, baseline, corrupt)

    shift_payload = None
    if lag_runs is not None:
        shifts = pred_shifts(
            runs,
            lag_runs,
            ids,
            version_contains_a=needle,
        )
        shift_gate = run_gate(
            suite,
            baseline,
            results / "mutant-lag-10.parquet",
            runs_baseline=runs,
            runs_candidate=lag_runs,
            max_shift=2,
            version_contains_baseline=needle,
        )
        shift_payload = {
            "shifts": shifts,
            "unique_shifts": sorted({int(item["shift"]) for item in shifts}),
            "gate": shift_gate,
        }
        detection["lag-10-shift"] = shift_gate

    rerun = results / "B-rerun.parquet"
    false_alarm = None
    if rerun.is_file():
        false_alarm = gate_red_rate(suite, baseline, rerun)

    tracking = None
    tracking_path = results / "tracking.json"
    if tracking_path.is_file():
        tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
    else:
        try:
            import importlib.util

            judge_path = Path(__file__).with_name("tracking_judge.py")
            spec = importlib.util.spec_from_file_location("tracking_judge", judge_path)
            if spec is None or spec.loader is None:
                raise ImportError(str(judge_path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            tracking = module.judge(suite, runs, version_contains="act-coffee-30000")
        except Exception as exc:  # noqa: BLE001
            tracking = {"error": str(exc)}

    ranking = None
    have_t = all(
        (results / name).is_file() for name in ("T2k.parquet", "T10k.parquet", "T30k.parquet")
    )
    if tracking and tracking.get("ok") and have_t:
        ranking = ranking_consistency(
            results / "T2k.parquet",
            results / "T10k.parquet",
            results / "T30k.parquet",
        )

    demo = inter_demo_l2(slices, ids) if (slices / ids[0]).is_dir() else {}
    drift = None
    try:
        drift = chunk_drift(_latest_run(runs, ids[0], needle=needle))
    except Exception as exc:  # noqa: BLE001
        drift = {"error": str(exc)}

    coverage = threshold_coverage(baseline)
    details = {
        "false_alarm": false_alarm,
        "detection": detection,
        "ranking": ranking,
        "coverage": coverage,
        "tracking": tracking,
        "bounds_source": "recorded+10%",
        "pred_shift_lag10": shift_payload,
        "inter_demo_l2": {k: demo.get(k) for k in ("n_pairs", "mean", "std")},
        "chunk_drift": {
            "monotonic_rising": None if drift is None else drift.get("monotonic_rising"),
            "mean_l2_by_offset": None if drift is None else drift.get("mean_l2_by_offset"),
            "error": None if drift is None else drift.get("error"),
        },
    }
    write_report(
        note,
        {
            "title": "Robogate gate benchmark v2 (2026-09-15)",
            "intro": (
                "Second-round known-answer test. "
                "Bounds from demonstration envelope + 10%. "
                "lag-10 is scored by pred_shift as well as action_lag. "
                "Ranking is filled only if T30k tracks."
            ),
            "false_alarm": false_alarm,
            "detection": {
                k: (v.get("ok") if isinstance(v, dict) else v) for k, v in detection.items()
            },
            "ranking": ranking,
            "coverage": coverage,
            "details": details,
        },
    )
    summary = results / "bench-v2.json"
    summary.write_text(json.dumps(details, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"note": str(note), "summary": str(summary)}, default=str))


if __name__ == "__main__":
    main()
