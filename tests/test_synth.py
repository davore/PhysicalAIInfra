from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from robogate.bench.synth import synth_run
from robogate.run import RunMeta, write_meta


def _write_run(tmp_path: Path, predicted: np.ndarray) -> Path:
    dest = tmp_path / "src"
    outputs = dest / "outputs"
    outputs.mkdir(parents=True)
    t_ns = (np.arange(len(predicted), dtype=np.int64) * 50_000_000).tolist()
    frame = pl.DataFrame({"t_ns": t_ns, "value": predicted.tolist()})
    frame.write_parquet(outputs / "action.parquet")
    frame.write_parquet(outputs / "action.recorded.parquet")
    write_meta(
        dest,
        RunMeta(
            scenario_id="synth-mini",
            scenario_hash="sha256:deadbeef",
            adapter="python-policy",
            mode="open_loop",
            target_version="act-coffee-30000",
        ),
    )
    return dest


def _pred(path: Path) -> np.ndarray:
    frame = pl.read_parquet(path / "outputs" / "action.parquet")
    return np.asarray(frame.get_column("value").to_list(), dtype=np.float64)


def test_dim_bias_moves_one_dimension(tmp_path: Path) -> None:
    src = np.array([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]], dtype=np.float64)
    dest = synth_run(_write_run(tmp_path, src), tmp_path / "dim", kind="dim_bias", value=0.2, dim=0)
    out = _pred(dest)
    np.testing.assert_allclose(out[:, 0], src[:, 0] + 0.2)
    np.testing.assert_allclose(out[:, 1], src[:, 1])


def test_gain_keeps_mean_and_scales_std(tmp_path: Path) -> None:
    src = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float64)
    dest = synth_run(_write_run(tmp_path, src), tmp_path / "gain", kind="gain", value=0.5)
    out = _pred(dest)
    np.testing.assert_allclose(out.mean(axis=0), src.mean(axis=0))
    np.testing.assert_allclose(out.std(axis=0), 0.5 * src.std(axis=0))


def test_stats_swap_applies_affine(tmp_path: Path) -> None:
    src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    stats = {
        "action": {"mean": [0.0, 0.0], "std": [1.0, 2.0]},
        "observation.state": {"mean": [10.0, 20.0], "std": [2.0, 4.0]},
    }
    dest = synth_run(_write_run(tmp_path, src), tmp_path / "swap", kind="stats_swap", stats=stats)
    expected = np.array([[12.0, 24.0], [16.0, 28.0]], dtype=np.float64)
    np.testing.assert_allclose(_pred(dest), expected)


def test_constant_has_zero_std(tmp_path: Path) -> None:
    src = np.array([[0.3, -0.1], [1.0, 2.0], [4.0, -3.0]], dtype=np.float64)
    dest = synth_run(_write_run(tmp_path, src), tmp_path / "const", kind="constant")
    out = _pred(dest)
    np.testing.assert_allclose(out, np.broadcast_to(src[0], src.shape))
    np.testing.assert_allclose(out.std(axis=0), 0.0, atol=1e-12)
