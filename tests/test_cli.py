from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from robogate.cli import app

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


def test_assert_skipped_closed_loop_rejects_id_mismatch() -> None:
    result = runner.invoke(
        app,
        [
            "assert",
            "--json",
            str(EXAMPLES / "mixed-closed-loop.yaml"),
            str(RUNS / "pass"),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "scenario_id" in result.output


def test_assert_scenario_id_mismatch_exit_1() -> None:
    result = runner.invoke(
        app,
        ["assert", str(EXAMPLES / "mcap-open-loop.yaml"), str(RUNS / "pass")],
    )
    assert result.exit_code == 1, result.output
    assert "run scenario_id" in result.output


def test_eval_requires_suite() -> None:
    result = runner.invoke(app, ["eval"])
    assert result.exit_code != 0
