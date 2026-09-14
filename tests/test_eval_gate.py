from __future__ import annotations

from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from robogate.bench.synth import corrupt_eval_hashes, synth_run
from robogate.cli import app
from robogate.eval import fold_eval, run_eval
from robogate.extract import extract_lerobot
from robogate.gate import run_gate
from tests.helpers import write_mini_lerobot_v3

runner = CliRunner()


def _suite(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset = write_mini_lerobot_v3(tmp_path / "ds")
    suite = tmp_path / "suite"
    slices = tmp_path / "slices"
    for i, blocking in enumerate((True, False, False)):
        extract_lerobot(
            str(dataset),
            episode_index=0,
            frame_from=0,
            frame_to=8,
            scenario_id=f"mini-ep{i}",
            scenario_out=suite,
            slice_root=slices,
            repo_id="local/mini",
            checkpoint="identity",
            entry="mock",
            blocking=blocking,
        )
    return suite, slices, tmp_path / "runs"


def test_eval_gate_known_answers(tmp_path: Path) -> None:
    suite, slices, runs = _suite(tmp_path)
    out = tmp_path / "results"
    baseline = run_eval(
        suite,
        runs_root=runs,
        out_root=out,
        replay=True,
        slice_root=slices,
        adapter="mock",
        eval_id="base",
    )
    assert fold_eval(pl.read_parquet(baseline)) == {
        "mini-ep0": "pass",
        "mini-ep1": "pass",
        "mini-ep2": "pass",
    }

    biased = run_eval(
        suite,
        runs_root=runs / "biased",
        out_root=out,
        replay=True,
        slice_root=slices,
        adapter="mock",
        perturbations=["action_bias=0.5"],
        eval_id="bias",
    )
    gate_bias = run_gate(suite, baseline, biased)
    assert gate_bias["ok"] is False
    assert any("baseline pass" in reason or "blocking" in reason for reason in gate_bias["reasons"])

    rerun = run_eval(
        suite,
        runs_root=runs / "rerun",
        out_root=out,
        replay=True,
        slice_root=slices,
        adapter="mock",
        eval_id="rerun",
    )
    assert run_gate(suite, baseline, rerun)["ok"] is True

    missing_root = runs / "missing"
    missing_root.mkdir()
    for run_dir in (runs / "rerun").iterdir():
        if run_dir.is_dir() and (run_dir / "meta.json").is_file():
            synth_run(run_dir, missing_root / run_dir.name, kind="drop_recorded")
    missing = run_eval(suite, runs_root=missing_root, out_root=out, eval_id="missing")
    gate_missing = run_gate(suite, baseline, missing)
    assert gate_missing["ok"] is False
    assert fold_eval(pl.read_parquet(missing))["mini-ep0"] == "error"

    bad_hash = corrupt_eval_hashes(baseline, out / "bad-hash.parquet")
    gate_hash = run_gate(suite, baseline, bad_hash)
    assert gate_hash["ok"] is False
    assert any("scenario_hash" in reason for reason in gate_hash["reasons"])

    cli = runner.invoke(
        app,
        [
            "gate",
            str(suite),
            "--baseline",
            str(baseline),
            "--candidate",
            str(biased),
            "--json",
        ],
    )
    assert cli.exit_code == 1
    compact = cli.output.replace(" ", "").lower()
    assert '"ok": false' in cli.output.lower() or '"ok":false' in compact


def test_gate_max_shift_is_red(tmp_path: Path) -> None:
    from robogate.bench.synth import synth_run
    from robogate.replay import build_adapter, run_replay
    from robogate.scenario import load_scenario
    from robogate.slice import Slice

    dataset = write_mini_lerobot_v3(tmp_path / "ds", n_frames=24)
    suite = tmp_path / "suite"
    slices = tmp_path / "slices"
    scenario_path, _ = extract_lerobot(
        str(dataset),
        episode_index=0,
        frame_from=0,
        frame_to=24,
        scenario_id="shift-ep0",
        scenario_out=suite,
        slice_root=slices,
        repo_id="local/mini",
        checkpoint="identity",
        entry="mock",
    )
    raw = scenario_path.read_text(encoding="utf-8")
    raw = raw.replace("max_lag_frames: 2", "max_lag_frames: 50")
    raw = raw.replace("max_l2: 0.05", "max_l2: 10.0")
    scenario_path.write_text(raw, encoding="utf-8")
    scenario = load_scenario(scenario_path)
    slice_obj = Slice.load(slices / scenario.id)
    base_run = run_replay(scenario, slice_obj, build_adapter("mock"), out_root=tmp_path / "runs-a")
    synth_run(base_run, tmp_path / "runs-b" / base_run.name, kind="lag", value=10)
    out = tmp_path / "results"
    baseline = run_eval(suite, runs_root=tmp_path / "runs-a", out_root=out, eval_id="a")
    candidate = run_eval(suite, runs_root=tmp_path / "runs-b", out_root=out, eval_id="b")
    green = run_gate(suite, baseline, candidate)
    assert green["ok"] is True
    red = run_gate(
        suite,
        baseline,
        candidate,
        runs_baseline=tmp_path / "runs-a",
        runs_candidate=tmp_path / "runs-b",
        max_shift=2,
    )
    assert red["ok"] is False
    assert any("pred_shift" in reason for reason in red["reasons"])
