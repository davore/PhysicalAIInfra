from __future__ import annotations

from pathlib import Path

from robogate.bench.calib import inter_demo_l2, rule_mean_plus_3sigma
from robogate.extract import extract_lerobot
from tests.helpers import write_mini_lerobot_v3


def test_inter_demo_and_threshold_rule(tmp_path: Path) -> None:
    dataset = write_mini_lerobot_v3(tmp_path / "ds", n_episodes=3)
    ids = []
    for i in range(3):
        extract_lerobot(
            str(dataset),
            episode_index=i,
            frame_from=0,
            frame_to=8,
            scenario_id=f"mini-ep{i}",
            scenario_out=tmp_path / "suite",
            slice_root=tmp_path / "slices",
            repo_id="local/mini",
            entry="mock",
        )
        ids.append(f"mini-ep{i}")
    demo = inter_demo_l2(tmp_path / "slices", ids)
    assert demo["n_pairs"] == 3
    assert demo["mean"] is not None
    assert demo["mean"] > 0
    rule = rule_mean_plus_3sigma([0.2, 0.3, 0.4])
    assert rule["mean"] == 0.3
    assert rule["mean_plus_3sigma"] > rule["mean"]
