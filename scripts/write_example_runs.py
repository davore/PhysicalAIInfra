"""Write hand-crafted pass/fail run fixtures used by M0 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from robogate.run import RunMeta, write_meta
from robogate.scenario import content_hash, load_scenario

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "examples" / "lerobot-open-loop.yaml"


def _action_frame(values: np.ndarray) -> pl.DataFrame:
    t_ns = (np.arange(len(values), dtype=np.int64) * 50_000_000).tolist()
    return pl.DataFrame({"t_ns": t_ns, "value": values.tolist()})


def _policy_frame(latency_ms: list[float], confidence: list[float]) -> pl.DataFrame:
    t_ns = (np.arange(len(latency_ms), dtype=np.int64) * 50_000_000).tolist()
    return pl.DataFrame(
        {"t_ns": t_ns, "latency_ms": latency_ms, "confidence": confidence}
    )


def _write_run(
    name: str,
    *,
    predicted: np.ndarray,
    recorded: np.ndarray,
    latency_ms: list[float],
    confidence: list[float],
    scenario_hash: str,
) -> Path:
    dest = ROOT / "runs" / "examples" / name
    outputs = dest / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    _action_frame(predicted).write_parquet(outputs / "action.parquet")
    _action_frame(recorded).write_parquet(outputs / "action.recorded.parquet")
    _policy_frame(latency_ms, confidence).write_parquet(outputs / "policy_action.parquet")
    (dest / "events.jsonl").write_text("", encoding="utf-8")
    write_meta(
        dest,
        RunMeta(
            scenario_id="lerobot-aloha-transfer-cube-ep0",
            scenario_hash=scenario_hash,
            adapter="python-policy",
            mode="open_loop",
            target_version="act-example",
            git_sha="deadbeef",
            host="local",
        ),
    )
    return dest


def main() -> None:
    digest = content_hash(load_scenario(SCENARIO))
    recorded = np.array(
        [
            [0.10, 0.20, 0.00, 0.50],
            [0.11, 0.21, 0.01, 0.50],
            [0.12, 0.22, 0.02, 0.51],
            [0.13, 0.23, 0.03, 0.51],
            [0.14, 0.24, 0.04, 0.52],
            [0.15, 0.25, 0.05, 0.52],
            [0.16, 0.26, 0.06, 0.53],
            [0.17, 0.27, 0.07, 0.53],
        ],
        dtype=np.float64,
    )
    close = recorded + 0.01
    far = recorded + 0.40
    _write_run(
        "pass",
        predicted=close,
        recorded=recorded,
        latency_ms=[40, 42, 45, 48, 50, 52, 55, 60],
        confidence=[0.91, 0.90, 0.88, 0.87, 0.92, 0.89, 0.93, 0.90],
        scenario_hash=digest,
    )
    _write_run(
        "fail",
        predicted=far,
        recorded=recorded,
        latency_ms=[40, 42, 45, 48, 50, 52, 55, 60],
        confidence=[0.10, 0.12, 0.11, 0.09, 0.15, 0.08, 0.13, 0.10],
        scenario_hash=digest,
    )
    print(f"wrote runs/examples/pass and runs/examples/fail ({digest})")


if __name__ == "__main__":
    main()
