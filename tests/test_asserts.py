from __future__ import annotations

from pathlib import Path

from robogate.asserts import Status, overall_ok, run_assertions
from robogate.run import Run
from robogate.scenario import load_scenario

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "scenarios" / "examples"
RUNS = ROOT / "runs" / "examples"


def test_pass_run_is_green() -> None:
    scenario = load_scenario(EXAMPLES / "lerobot-open-loop.yaml")
    results = run_assertions(scenario, Run.load(RUNS / "pass"))
    assert overall_ok(results)
    assert {item.type: item.status for item in results} == {
        "action_deviation": Status.PASS,
        "latency": Status.PASS,
        "confidence_floor": Status.PASS,
    }
    deviation = next(item for item in results if item.type == "action_deviation")
    assert deviation.measured is not None
    assert deviation.measured < 0.05


def test_fail_run_is_red() -> None:
    scenario = load_scenario(EXAMPLES / "lerobot-open-loop.yaml")
    results = run_assertions(scenario, Run.load(RUNS / "fail"))
    assert not overall_ok(results)
    by_type = {item.type: item.status for item in results}
    assert by_type["action_deviation"] == Status.FAIL
    assert by_type["confidence_floor"] == Status.FAIL
    assert by_type["latency"] == Status.PASS


def test_latency_never_fails() -> None:
    scenario = load_scenario(EXAMPLES / "lerobot-open-loop.yaml")
    results = run_assertions(scenario, Run.load(RUNS / "fail"))
    latency = next(item for item in results if item.type == "latency")
    assert latency.status == Status.PASS
    assert "not blocking" in latency.message


def test_closed_loop_asserts_skipped_on_open_loop_run() -> None:
    scenario = load_scenario(EXAMPLES / "mixed-closed-loop.yaml")
    results = run_assertions(scenario, Run.load(RUNS / "pass"))
    assert overall_ok(results)
    by_type = {item.type: item.status for item in results}
    assert by_type["action_deviation"] == Status.PASS
    assert by_type["grasp_success"] == Status.SKIPPED
    assert by_type["no_collision"] == Status.SKIPPED


def test_mcap_scenario_against_pass_run() -> None:
    scenario = load_scenario(EXAMPLES / "mcap-open-loop.yaml")
    results = run_assertions(scenario, Run.load(RUNS / "pass"))
    assert overall_ok(results)
