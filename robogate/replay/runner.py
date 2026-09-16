"""Drive an adapter over a slice and write a run directory."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from robogate.evidence import DEFAULT_TOP_K, TopKFrames, should_write, write_evidence
from robogate.replay.base import Adapter, AdapterInfo
from robogate.run import Run, RunMeta, RunMode, write_meta
from robogate.scenario import Scenario, content_hash
from robogate.slice import Slice


def run_replay(
    scenario: Scenario,
    slice_: Slice,
    adapter: Adapter,
    *,
    out_root: Path,
    device: str = "cpu",
    host: str | None = None,
    perturbations: list[str] | None = None,
    scenario_hash: str | None = None,
    evidence: str = "fail",
) -> Path:
    adapter.load(scenario.target, slice_, device=device)
    adapter.reset()
    recorded = slice_.recorded_action()
    t_ns = recorded.get_column("t_ns").to_list()
    recorded_values = recorded.get_column("value").to_list()
    n = len(t_ns)
    predicted: list[list[float]] = []
    latency: list[float] = []
    confidence: list[float | None] = []
    topk = TopKFrames(k=DEFAULT_TOP_K)
    for i in range(n):
        obs = _frame_obs(slice_, i)
        print(f"[replay] scenario 1/1 frame {i + 1}/{n}", flush=True)
        step = adapter.step(obs, i)
        pred = np.asarray(step.action, dtype=np.float64).reshape(-1)
        predicted.append(pred.tolist())
        latency.append(float(step.latency_ms))
        confidence.append(step.confidence)
        rec = np.asarray(recorded_values[i], dtype=np.float64).reshape(-1)
        dim = min(pred.size, rec.size)
        l2 = float(np.linalg.norm(pred[:dim] - rec[:dim])) if dim else 0.0
        topk.feed(i, l2, step.images)

    dest = out_root / _run_name(scenario.id, adapter.info, perturbations)
    outputs = dest / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"t_ns": t_ns, "value": predicted}).write_parquet(outputs / "action.parquet")
    pl.DataFrame({"t_ns": t_ns, "value": recorded_values}).write_parquet(
        outputs / "action.recorded.parquet"
    )
    policy: dict[str, Any] = {"t_ns": t_ns, "latency_ms": latency}
    if any(item is not None for item in confidence):
        policy["confidence"] = [item if item is not None else float("nan") for item in confidence]
    pl.DataFrame(policy).write_parquet(outputs / "policy_action.parquet")
    (dest / "events.jsonl").write_text("", encoding="utf-8")
    write_meta(
        dest,
        RunMeta(
            scenario_id=scenario.id,
            scenario_hash=scenario_hash or content_hash(scenario),
            adapter=scenario.target.adapter,
            mode=RunMode.OPEN_LOOP,
            target_version=adapter.info.target_version,
            target_revision=adapter.info.target_revision or scenario.target.revision,
            host=host,
            device=device,
            adapter_versions=adapter.info.versions or None,
            dataset_revision=slice_.meta.dataset_revision,
            policy_config=adapter.info.policy_config or None,
            perturbations=list(perturbations) if perturbations else None,
        ),
    )
    _maybe_write_evidence(
        dest,
        scenario,
        evidence=evidence,
        topk=topk,
        predicted=predicted,
        recorded_values=recorded_values,
        t_ns=t_ns,
    )
    return dest


def _maybe_write_evidence(
    dest: Path,
    scenario: Scenario,
    *,
    evidence: str,
    topk: TopKFrames,
    predicted: list[list[float]],
    recorded_values: list[Any],
    t_ns: list[int],
) -> None:
    if evidence == "never":
        return
    try:
        from robogate.asserts import run_assertions

        run = Run.load(dest)
        results = run_assertions(scenario, run)
        if not should_write(results, evidence):
            return
        write_evidence(
            dest,
            scenario,
            results,
            topk.ranked(),
            np.asarray(predicted, dtype=np.float64),
            np.asarray(recorded_values, dtype=np.float64),
            t_ns,
        )
    except Exception as exc:  # noqa: BLE001 — evidence must not redden replay
        print(f"[evidence] skipped: {type(exc).__name__}: {exc}", flush=True)


def _frame_obs(slice_: Slice, index: int) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    inputs = slice_.path / "inputs"
    if not inputs.is_dir():
        return obs
    for path in sorted(inputs.glob("*.parquet")):
        frame = pl.read_parquet(path)
        if "value" not in frame.columns or index >= frame.height:
            continue
        obs[path.stem] = frame.get_column("value")[index]
    return obs


def _run_name(
    scenario_id: str,
    info: AdapterInfo,
    perturbations: list[str] | None = None,
) -> str:
    ckpt = info.target_version or info.name
    short = Path(str(ckpt)).name.replace("/", "-")[:48]
    extra = ""
    if perturbations:
        extra = "__" + "-".join(item.replace("=", "") for item in perturbations)[:32]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{scenario_id}__{short}{extra}__{stamp}"
