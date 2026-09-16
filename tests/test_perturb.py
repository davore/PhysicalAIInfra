from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from robogate.replay import build_adapter
from robogate.replay.mock import MockAdapter
from robogate.replay.perturb import PerturbedAdapter, parse_perturbations
from robogate.scenario import Target
from robogate.slice import Slice, SliceMeta, write_slice_meta


def _slice(tmp_path: Path, actions: np.ndarray) -> Slice:
    dest = tmp_path / "slice"
    (dest / "recorded").mkdir(parents=True)
    t_ns = list(range(len(actions)))
    pl.DataFrame({"t_ns": t_ns, "value": actions.tolist()}).write_parquet(
        dest / "recorded" / "action.parquet"
    )
    write_slice_meta(
        dest,
        SliceMeta(
            scenario_id="p",
            dataset="local",
            episode_index=0,
            frame_from=0,
            frame_to=len(actions),
            fps=50.0,
            n_frames=len(actions),
            action_dim=actions.shape[1],
            period_ns=20_000_000,
        ),
    )
    return Slice.load(dest)


def _run(adapter, slice_obj: Slice) -> np.ndarray:
    adapter.load(Target(adapter="python-policy", entry="mock"), slice_obj, device="cpu")
    adapter.reset()
    out = []
    for i in range(slice_obj.meta.n_frames):
        out.append(adapter.step({}, i).action)
    return np.asarray(out, dtype=np.float64)


def test_same_seed_is_deterministic(tmp_path: Path) -> None:
    actions = np.linspace(0.0, 1.0, 8 * 2).reshape(8, 2)
    slice_obj = _slice(tmp_path, actions)
    a = PerturbedAdapter(MockAdapter(), {"action_noise": "0.2"}, seed=3)
    b = PerturbedAdapter(MockAdapter(), {"action_noise": "0.2"}, seed=3)
    np.testing.assert_allclose(_run(a, slice_obj), _run(b, slice_obj))


def test_l2_grows_with_bias_and_noise(tmp_path: Path) -> None:
    actions = np.linspace(0.0, 1.0, 8 * 2).reshape(8, 2)
    slice_obj = _slice(tmp_path, actions)
    ref = _run(MockAdapter(), slice_obj)

    def l2(specs: dict[str, str]) -> float:
        pred = _run(PerturbedAdapter(MockAdapter(), specs, seed=0), slice_obj)
        return float(np.mean(np.linalg.norm(pred - ref, axis=1)))

    assert l2({}) == 0.0
    assert l2({"action_bias": "0.1"}) < l2({"action_bias": "0.5"})
    assert l2({"action_noise": "0.0"}) <= l2({"action_noise": "0.2"})
    assert l2({"action_lag": "0"}) <= l2({"action_lag": "3"})
    assert l2({"action_scale": "1.0"}) <= l2({"action_scale": "1.5"})


def test_parse_action_stats_dataset_only() -> None:
    assert parse_perturbations(["action_stats=dataset"])["action_stats"] == "dataset"
    with pytest.raises(ValueError, match="action_stats"):
        parse_perturbations(["action_stats=checkpoint"])


def test_action_stats_requires_lerobot() -> None:
    with pytest.raises(ValueError, match="action_stats"):
        build_adapter("mock", perturbations=["action_stats=dataset"])
