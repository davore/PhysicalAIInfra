from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from robogate.asserts import AssertResult, Status, run_assertions
from robogate.cli import app
from robogate.eval import run_eval
from robogate.evidence import (
    TopKFrames,
    WorstFrame,
    should_write,
    write_evidence,
)
from robogate.extract import extract_lerobot
from robogate.gate import run_gate
from robogate.replay import build_adapter, run_replay
from robogate.run import Run
from robogate.scenario import load_scenario
from robogate.slice import Slice
from tests.helpers import write_mini_lerobot_v3

runner = CliRunner()


def _extract(tmp_path: Path, *, n_frames: int = 8) -> tuple[Path, Path]:
    dataset = write_mini_lerobot_v3(tmp_path / "ds", n_frames=n_frames)
    return extract_lerobot(
        str(dataset),
        episode_index=0,
        frame_from=0,
        frame_to=n_frames,
        scenario_id="mini-ev",
        scenario_out=tmp_path / "scenarios",
        slice_root=tmp_path / "slices",
        repo_id="local/mini",
        checkpoint="identity",
        entry="mock",
    )


def test_noise_replay_writes_ranked_evidence(tmp_path: Path) -> None:
    scenario_path, slice_path = _extract(tmp_path)
    result = runner.invoke(
        app,
        [
            "replay",
            str(scenario_path),
            "--slice",
            str(slice_path),
            "--out",
            str(tmp_path / "runs"),
            "--adapter",
            "mock",
            "--noise",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    dest = next((tmp_path / "runs").iterdir())
    payload = json.loads((dest / "evidence" / "evidence.json").read_text(encoding="utf-8"))
    frames = payload["worst_frames"]
    assert frames
    assert [item["l2"] for item in frames] == sorted((item["l2"] for item in frames), reverse=True)
    assert payload["skipped"]["frames"]
    results = run_assertions(load_scenario(scenario_path), Run.load(dest))
    assert any(item.status == Status.FAIL and item.type == "action_deviation" for item in results)


def test_clean_replay_skips_evidence_unless_always(tmp_path: Path) -> None:
    scenario_path, _ = _extract(tmp_path)
    scenario = load_scenario(scenario_path)
    slice_obj = Slice.load(tmp_path / "slices" / scenario.id)
    clean = run_replay(
        scenario,
        slice_obj,
        build_adapter("mock"),
        out_root=tmp_path / "runs-clean",
    )
    assert not (clean / "evidence").exists()
    forced = run_replay(
        scenario,
        slice_obj,
        build_adapter("mock"),
        out_root=tmp_path / "runs-always",
        evidence="always",
    )
    assert (forced / "evidence" / "evidence.json").is_file()


def test_topk_keeps_five_and_skips_conversion(monkeypatch) -> None:
    calls: list[object] = []

    def stub(image: object) -> np.ndarray:
        calls.append(image)
        return np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr("robogate.evidence._to_uint8_hwc", stub)
    topk = TopKFrames(k=5)
    marker = object()
    for i in range(20):
        l2 = 10.0 - i if i < 5 else 0.1
        topk.feed(i, l2, {"cam": marker})
    assert len(calls) == 5
    ranked = topk.ranked()
    assert [item.index for item in ranked] == [0, 1, 2, 3, 4]
    assert [item.l2 for item in ranked] == [10.0, 9.0, 8.0, 7.0, 6.0]


def test_evidence_command_rebuilds_and_force_overwrites(tmp_path: Path) -> None:
    scenario_path, slice_path = _extract(tmp_path)
    scenario = load_scenario(scenario_path)
    dest = run_replay(
        scenario,
        Slice.load(slice_path),
        build_adapter("mock", noise=0.5),
        out_root=tmp_path / "runs",
        evidence="never",
    )
    assert not (dest / "evidence").exists()
    first = runner.invoke(
        app,
        ["evidence", str(scenario_path), str(dest), "--slice", str(slice_path)],
    )
    assert first.exit_code == 0, first.output
    marker = dest / "evidence" / "evidence.json"
    assert marker.is_file()
    first_text = marker.read_text(encoding="utf-8")
    blocked = runner.invoke(
        app,
        ["evidence", str(scenario_path), str(dest), "--slice", str(slice_path)],
    )
    assert blocked.exit_code == 1
    assert "force" in blocked.output.lower()
    marker.write_text("stale\n", encoding="utf-8")
    again = runner.invoke(
        app,
        ["evidence", str(scenario_path), str(dest), "--slice", str(slice_path), "--force"],
    )
    assert again.exit_code == 0, again.output
    rewritten = marker.read_text(encoding="utf-8")
    assert rewritten != "stale\n"
    assert json.loads(rewritten)["worst_frames"]
    assert json.loads(first_text)["worst_frames"]


def test_gate_payload_includes_evidence_path(tmp_path: Path) -> None:
    scenario_path, slice_path = _extract(tmp_path)
    suite = tmp_path / "scenarios"
    slices = tmp_path / "slices"
    baseline = run_eval(
        suite,
        runs_root=tmp_path / "runs-a",
        out_root=tmp_path / "results",
        replay=True,
        slice_root=slices,
        adapter="mock",
        eval_id="base",
    )
    candidate = run_eval(
        suite,
        runs_root=tmp_path / "runs-b",
        out_root=tmp_path / "results",
        replay=True,
        slice_root=slices,
        adapter="mock",
        noise=0.5,
        eval_id="noisy",
    )
    payload = run_gate(suite, baseline, candidate, runs_candidate=tmp_path / "runs-b")
    assert payload["ok"] is False
    assert payload["evidence"]
    item = payload["evidence"][0]
    assert item["scenario_id"] == "mini-ev"
    assert (Path(item["path"]) / "evidence.json").is_file()
    text = runner.invoke(
        app,
        [
            "gate",
            str(suite),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--runs-candidate",
            str(tmp_path / "runs-b"),
        ],
    )
    assert text.exit_code == 1
    assert "evidence:" in text.output


def test_missing_plot_libs_do_not_raise(tmp_path: Path, monkeypatch) -> None:
    scenario_path, slice_path = _extract(tmp_path)
    scenario = load_scenario(scenario_path)
    dest = run_replay(
        scenario,
        Slice.load(slice_path),
        build_adapter("mock"),
        out_root=tmp_path / "runs",
        evidence="never",
    )

    def boom() -> None:
        raise ImportError("missing extra")

    monkeypatch.setattr("robogate.evidence._load_pil", boom)
    monkeypatch.setattr("robogate.evidence._load_mpl", boom)
    results = [
        AssertResult(type="action_deviation", status=Status.FAIL, message="forced"),
    ]
    pred = np.zeros((4, 2), dtype=np.float64)
    rec = np.ones((4, 2), dtype=np.float64)
    frames = [
        WorstFrame(
            index=1,
            l2=1.4,
            images={"cam": np.zeros((2, 2, 3), dtype=np.uint8)},
        )
    ]
    out = write_evidence(dest, scenario, results, frames, pred, rec, [0, 1, 2, 3])
    payload = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert "frames" in payload["skipped"]
    assert "plot" in payload["skipped"]
    assert not (out / "actions.png").exists()


def test_should_write_modes() -> None:
    failed = [AssertResult(type="action_deviation", status=Status.FAIL)]
    passed = [AssertResult(type="action_deviation", status=Status.PASS)]
    assert should_write(failed, "fail") is True
    assert should_write(passed, "fail") is False
    assert should_write(passed, "always") is True
    assert should_write(failed, "never") is False
