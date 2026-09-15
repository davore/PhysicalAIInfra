from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from robogate.asserts import Status, overall_ok, run_assertions
from robogate.asserts.action_deviation import dtw_normalized, mean_l2
from robogate.run import Run, RunMeta, write_meta
from robogate.scenario import Scenario, load_scenario

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "scenarios" / "examples"
RUNS = ROOT / "runs" / "examples"


def _write_run(
    dest: Path,
    *,
    scenario_id: str,
    predicted: np.ndarray,
    recorded: np.ndarray | None = None,
    latency_ms: list[float] | None = None,
    confidence: list[float] | None = None,
    action_column: str = "value",
    include_action: bool = True,
) -> Run:
    outputs = dest / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    t_ns = (np.arange(len(predicted), dtype=np.int64) * 20_000_000).tolist()
    if include_action:
        pred_df = pl.DataFrame({"t_ns": t_ns, action_column: predicted.tolist()})
        pred_df.write_parquet(outputs / "action.parquet")
        ref = recorded if recorded is not None else predicted
        pl.DataFrame({"t_ns": t_ns[: len(ref)], "value": ref.tolist()}).write_parquet(
            outputs / "action.recorded.parquet"
        )
    if latency_ms is not None or confidence is not None:
        payload: dict[str, object] = {"t_ns": t_ns}
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        if confidence is not None:
            payload["confidence"] = confidence
        pl.DataFrame(payload).write_parquet(outputs / "policy_action.parquet")
    write_meta(
        dest,
        RunMeta(
            scenario_id=scenario_id,
            scenario_hash="sha256:test",
            adapter="python-policy",
            mode="open_loop",
        ),
    )
    (dest / "events.jsonl").write_text("", encoding="utf-8")
    return Run.load(dest)


def _open_loop_scenario(**updates: object) -> Scenario:
    payload = load_scenario(EXAMPLES / "lerobot-open-loop.yaml").model_dump(mode="json")
    payload.update(updates)
    return Scenario.model_validate(payload)


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


def test_missing_value_column_is_error_not_crash(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path / "run",
        scenario_id="lerobot-aloha-transfer-cube-ep0",
        predicted=np.ones((4, 4)),
        action_column="action",
    )
    scenario = load_scenario(EXAMPLES / "lerobot-open-loop.yaml")
    results = run_assertions(scenario, run)
    by_type = {item.type: item for item in results}
    assert by_type["action_deviation"].status == Status.ERROR
    assert "value" in by_type["action_deviation"].message
    assert not overall_ok(results)


def test_missing_confidence_is_skipped(tmp_path: Path) -> None:
    predicted = np.zeros((4, 4))
    run = _write_run(
        tmp_path / "run",
        scenario_id="lerobot-aloha-transfer-cube-ep0",
        predicted=predicted,
        recorded=predicted,
        latency_ms=[10.0, 11.0, 12.0, 13.0],
    )
    scenario = load_scenario(EXAMPLES / "lerobot-open-loop.yaml")
    results = run_assertions(scenario, run)
    by_type = {item.type: item.status for item in results}
    assert by_type["action_deviation"] == Status.PASS
    assert by_type["latency"] == Status.PASS
    assert by_type["confidence_floor"] == Status.SKIPPED
    assert overall_ok(results)


def test_missing_policy_output_skips_latency_and_confidence(tmp_path: Path) -> None:
    predicted = np.zeros((4, 4))
    run = _write_run(
        tmp_path / "run",
        scenario_id="lerobot-aloha-transfer-cube-ep0",
        predicted=predicted,
        recorded=predicted,
    )
    scenario = load_scenario(EXAMPLES / "lerobot-open-loop.yaml")
    results = run_assertions(scenario, run)
    by_type = {item.type: item.status for item in results}
    assert by_type["latency"] == Status.SKIPPED
    assert by_type["confidence_floor"] == Status.SKIPPED
    assert overall_ok(results)


def test_action_bounds_pass_and_fail(tmp_path: Path) -> None:
    predicted = np.array([[0.1, 0.2], [0.15, 0.25]], dtype=np.float64)
    over = predicted + 2.0
    scenario = _open_loop_scenario(
        expected=[
            {"type": "action_bounds", "topic": "action", "min": [0.0, 0.0], "max": [1.0, 1.0]}
        ]
    )
    green = run_assertions(
        scenario,
        _write_run(tmp_path / "ok", scenario_id=scenario.id, predicted=predicted),
    )
    red = run_assertions(
        scenario,
        _write_run(tmp_path / "bad", scenario_id=scenario.id, predicted=over),
    )
    assert green[0].status == Status.PASS
    assert red[0].status == Status.FAIL
    assert overall_ok(green)
    assert not overall_ok(red)


def test_action_smoothness_pass_and_fail(tmp_path: Path) -> None:
    smooth = np.array([[0.0, 0.0], [0.01, 0.01], [0.02, 0.02]], dtype=np.float64)
    jerky = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    scenario = _open_loop_scenario(
        expected=[{"type": "action_smoothness", "topic": "action", "max_delta": 0.05}]
    )
    green = run_assertions(
        scenario,
        _write_run(tmp_path / "ok", scenario_id=scenario.id, predicted=smooth),
    )
    red = run_assertions(
        scenario,
        _write_run(tmp_path / "bad", scenario_id=scenario.id, predicted=jerky),
    )
    assert green[0].status == Status.PASS
    assert red[0].status == Status.FAIL


def test_dtw_metric_pass_and_fail(tmp_path: Path) -> None:
    recorded = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], dtype=np.float64)
    close = recorded + 0.01
    far = recorded + 1.0
    scenario = _open_loop_scenario(
        expected=[
            {
                "type": "action_deviation",
                "topic": "action",
                "reference": "recorded",
                "max_l2": 0.05,
                "metric": "dtw",
            }
        ]
    )
    green = run_assertions(
        scenario,
        _write_run(
            tmp_path / "ok",
            scenario_id=scenario.id,
            predicted=close,
            recorded=recorded,
        ),
    )
    red = run_assertions(
        scenario,
        _write_run(
            tmp_path / "bad",
            scenario_id=scenario.id,
            predicted=far,
            recorded=recorded,
        ),
    )
    assert green[0].status == Status.PASS
    assert red[0].status == Status.FAIL
    assert green[0].measured is not None
    assert green[0].measured == pytest.approx(dtw_normalized(close, recorded))


def test_dtw_matches_l2_on_identical_series() -> None:
    series = np.array([[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]], dtype=np.float64)
    assert dtw_normalized(series, series) == pytest.approx(0.0)
    assert mean_l2(series, series) == pytest.approx(0.0)
