from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from robogate.cli import app
from robogate.scenario import content_hash, load_scenario

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "scenarios" / "examples"
RUNS = ROOT / "runs" / "examples"

runner = CliRunner()


def test_validate_examples() -> None:
    for name in (
        "lerobot-open-loop.yaml",
        "mcap-open-loop.yaml",
        "mixed-closed-loop.yaml",
    ):
        result = runner.invoke(app, ["validate", str(EXAMPLES / name)])
        assert result.exit_code == 0, result.output
        assert "ok " in result.output


def test_schema_exports_json() -> None:
    result = runner.invoke(app, ["schema"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["title"] == "Scenario"


def test_assert_pass_exit_0() -> None:
    result = runner.invoke(
        app,
        ["assert", str(EXAMPLES / "lerobot-open-loop.yaml"), str(RUNS / "pass")],
    )
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_assert_fail_exit_1() -> None:
    result = runner.invoke(
        app,
        ["assert", str(EXAMPLES / "lerobot-open-loop.yaml"), str(RUNS / "fail")],
    )
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output


def test_assert_skipped_closed_loop_exit_0() -> None:
    result = runner.invoke(
        app,
        [
            "assert",
            "--json",
            str(EXAMPLES / "mixed-closed-loop.yaml"),
            str(RUNS / "pass"),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    statuses = {item["type"]: item["status"] for item in payload["results"]}
    assert statuses["grasp_success"] == "skipped"
    assert statuses["no_collision"] == "skipped"
    expected = content_hash(load_scenario(EXAMPLES / "mixed-closed-loop.yaml"))
    assert payload["scenario_hash"] == expected


def test_placeholder_commands_exit_2() -> None:
    for command in ("extract", "replay", "eval", "diff", "gate"):
        result = runner.invoke(app, [command])
        assert result.exit_code == 2, command
        assert "not implemented in M0" in result.output
