from __future__ import annotations

from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from robogate.cli import app
from robogate.extract import extract_lerobot
from robogate.scenario import load_scenario
from robogate.slice import Slice, period_ns
from tests.helpers import write_mini_lerobot_v3

runner = CliRunner()


def test_extract_mini_v3_writes_slice_and_scenario(tmp_path: Path) -> None:
    dataset = write_mini_lerobot_v3(tmp_path / "ds", n_frames=8, fps=50.0)
    scenario_path, slice_path = extract_lerobot(
        str(dataset),
        episode_index=0,
        frame_from=0,
        frame_to=8,
        scenario_id="mini-ep0",
        scenario_out=tmp_path / "scenarios",
        slice_root=tmp_path / "slices",
        repo_id="local/mini",
        checkpoint="local/mock",
        entry="mock",
    )
    scenario = load_scenario(scenario_path)
    assert scenario.id == "mini-ep0"
    assert scenario.source.kind == "lerobot"
    assert scenario.source.dataset == "local/mini"
    assert scenario.source.frame_to == 8
    assert "observation.state" in scenario.inputs
    assert "observation.images.cam_high" in scenario.inputs
    assert "action" not in scenario.inputs
    assert scenario.target.entry == "mock"

    slice_obj = Slice.load(slice_path)
    assert slice_obj.meta.n_frames == 8
    assert slice_obj.meta.action_dim == 2
    assert slice_obj.meta.period_ns == period_ns(50.0)
    recorded = slice_obj.recorded_action()
    assert recorded.height == 8
    assert recorded["t_ns"].to_list()[0] == 0
    assert recorded["t_ns"].to_list()[1] == 20_000_000
    assert recorded["t_ns"].to_list()[-1] == 7 * 20_000_000
    state = slice_obj.input_frame("observation.state")
    assert state is not None
    assert state.height == 8
    assert "observation.images.cam_high" in slice_obj.meta.videos


def test_extract_cli_mini_v3(tmp_path: Path) -> None:
    dataset = write_mini_lerobot_v3(tmp_path / "ds")
    result = runner.invoke(
        app,
        [
            "extract",
            "lerobot",
            str(dataset),
            "--episode",
            "0",
            "--from",
            "0",
            "--to",
            "8",
            "--id",
            "cli-mini",
            "--out",
            str(tmp_path / "scenarios"),
            "--slice-dir",
            str(tmp_path / "slices"),
            "--repo-id",
            "local/mini",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ok cli-mini" in result.output
    assert (tmp_path / "slices" / "cli-mini" / "slice.json").is_file()


def test_extract_episodes_range(tmp_path: Path) -> None:
    dataset = write_mini_lerobot_v3(tmp_path / "ds", n_episodes=3)
    result = runner.invoke(
        app,
        [
            "extract",
            "lerobot",
            str(dataset),
            "--episodes",
            "0-2",
            "--from",
            "0",
            "--to",
            "8",
            "--out",
            str(tmp_path / "scenarios"),
            "--slice-dir",
            str(tmp_path / "slices"),
            "--repo-id",
            "local/mini",
            "--entry",
            "mock",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ok mini-ep0" in result.output
    assert "ok mini-ep2" in result.output
    assert (tmp_path / "slices" / "mini-ep1" / "slice.json").is_file()


def test_extract_subrange_t_ns_uses_frame_index(tmp_path: Path) -> None:
    dataset = write_mini_lerobot_v3(tmp_path / "ds", n_frames=8)
    _, slice_path = extract_lerobot(
        str(dataset),
        episode_index=0,
        frame_from=2,
        frame_to=5,
        scenario_id="mini-mid",
        scenario_out=tmp_path / "scenarios",
        slice_root=tmp_path / "slices",
        repo_id="local/mini",
        entry="mock",
    )
    recorded = pl.read_parquet(Path(slice_path) / "recorded" / "action.parquet")
    assert recorded.height == 3
    assert recorded["t_ns"].to_list() == [2 * 20_000_000, 3 * 20_000_000, 4 * 20_000_000]
