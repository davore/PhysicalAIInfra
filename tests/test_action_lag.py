from __future__ import annotations

from pathlib import Path

from robogate.asserts import Status, run_assertions
from robogate.extract import extract_lerobot
from robogate.replay import build_adapter, run_replay
from robogate.run import Run
from robogate.scenario import load_scenario
from robogate.slice import Slice
from tests.helpers import write_mini_lerobot_v3


def _extract(tmp_path: Path, n_frames: int = 24) -> tuple[Path, Path]:
    dataset = write_mini_lerobot_v3(tmp_path / "ds", n_frames=n_frames)
    return extract_lerobot(
        str(dataset),
        episode_index=0,
        frame_from=0,
        frame_to=n_frames,
        scenario_id="lag-mini",
        scenario_out=tmp_path / "scenarios",
        slice_root=tmp_path / "slices",
        repo_id="local/mini",
        checkpoint="identity",
        entry="mock",
    )


def _replay(scenario_path: Path, tmp_path: Path, perturbations: list[str] | None = None) -> Run:
    scenario = load_scenario(scenario_path)
    slice_obj = Slice.load(tmp_path / "slices" / scenario.id)
    dest = run_replay(
        scenario,
        slice_obj,
        build_adapter("mock", perturbations=perturbations),
        out_root=tmp_path / "runs",
        perturbations=perturbations,
    )
    return Run.load(dest)


def test_identity_lag_is_zero(tmp_path: Path) -> None:
    scenario_path, _ = _extract(tmp_path)
    run = _replay(scenario_path, tmp_path)
    results = {item.type: item for item in run_assertions(load_scenario(scenario_path), run)}
    assert results["action_lag"].status == Status.PASS
    assert results["action_lag"].measured == 0


def test_action_lag_ten_is_detected(tmp_path: Path) -> None:
    scenario_path, _ = _extract(tmp_path)
    run = _replay(scenario_path, tmp_path, ["action_lag=10"])
    results = {item.type: item for item in run_assertions(load_scenario(scenario_path), run)}
    assert results["action_lag"].status == Status.FAIL
    assert results["action_lag"].measured == 10


def test_constant_prediction_is_unmeasurable(tmp_path: Path) -> None:
    scenario_path, _ = _extract(tmp_path)
    run = _replay(scenario_path, tmp_path, ["action_scale=0"])
    results = {item.type: item for item in run_assertions(load_scenario(scenario_path), run)}
    assert results["action_lag"].status == Status.PASS
    assert "too weak" in results["action_lag"].message
