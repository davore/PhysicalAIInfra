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
