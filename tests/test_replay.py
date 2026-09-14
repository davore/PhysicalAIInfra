from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from robogate.asserts import Status, overall_ok, run_assertions
from robogate.cli import app
from robogate.extract import extract_lerobot
from robogate.run import Run
from robogate.scenario import load_scenario
from tests.helpers import write_mini_lerobot_v3

runner = CliRunner()


def test_mock_extract_replay_assert(tmp_path: Path) -> None:
    dataset = write_mini_lerobot_v3(tmp_path / "ds")
    scenario_path, _ = extract_lerobot(
        str(dataset),
        episode_index=0,
        frame_from=0,
        frame_to=8,
        scenario_id="mini-ep0",
        scenario_out=tmp_path / "scenarios",
        slice_root=tmp_path / "slices",
        repo_id="local/mini",
        checkpoint="identity",
        entry="mock",
    )
    result = runner.invoke(
        app,
        [
            "replay",
            str(scenario_path),
            "--slice",
            str(tmp_path / "slices" / "mini-ep0"),
            "--out",
            str(tmp_path / "runs"),
            "--adapter",
            "mock",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ROBOGATE_REPLAY_DONE" in result.output
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    scenario = load_scenario(scenario_path)
    results = run_assertions(scenario, Run.load(runs[0]))
    by_type = {item.type: item.status for item in results}
    assert by_type["action_deviation"] == Status.PASS
    assert by_type["action_bounds"] == Status.PASS
    assert by_type["action_smoothness"] == Status.PASS
    assert by_type["latency"] == Status.PASS
    assert by_type["confidence_floor"] == Status.SKIPPED
    assert overall_ok(results)

    assert_cmd = runner.invoke(app, ["assert", str(scenario_path), str(runs[0])])
    assert assert_cmd.exit_code == 0, assert_cmd.output
    assert "PASS" in assert_cmd.output
