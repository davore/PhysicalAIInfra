"""Open-loop: predicted action vs recorded action (mean L2)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from robogate.asserts.base import Assertion, AssertResult, Status, register
from robogate.run import Run, RunMode
from robogate.scenario import ActionDeviation, Expectation


def vectors_from_frame(df: pl.DataFrame) -> np.ndarray:
    if "value" not in df.columns:
        raise ValueError("action parquet must have a 'value' column")
    rows = df.get_column("value").to_list()
    if not rows:
        raise ValueError("action parquet has no rows")
    return np.asarray(rows, dtype=np.float64)


def mean_l2(predicted: np.ndarray, recorded: np.ndarray) -> float:
    n = min(len(predicted), len(recorded))
    if n == 0:
        raise ValueError("cannot compare empty action sequences")
    pred = predicted[:n]
    ref = recorded[:n]
    if pred.shape[1] != ref.shape[1]:
        raise ValueError(
            f"action dim mismatch: predicted {pred.shape[1]} vs recorded {ref.shape[1]}"
        )
    diffs = pred - ref
    return float(np.mean(np.linalg.norm(diffs, axis=1)))


class ActionDeviationAssertion(Assertion):
    type = "action_deviation"
    requires_mode = RunMode.OPEN_LOOP

    def evaluate(self, run: Run, spec: Expectation) -> AssertResult:
        if not isinstance(spec, ActionDeviation):
            raise TypeError("expected ActionDeviation spec")
        predicted = vectors_from_frame(run.output(spec.topic))
        recorded = vectors_from_frame(run.output(spec.topic, recorded=True))
        if spec.metric != "l2":
            return AssertResult(
                type=self.type,
                status=Status.FAIL,
                threshold=spec.max_l2,
                topic=spec.topic,
                message=f"metric {spec.metric} is not implemented in M0",
            )
        length_ratio = min(len(predicted), len(recorded)) / max(len(predicted), len(recorded))
        if length_ratio < 0.9:
            return AssertResult(
                type=self.type,
                status=Status.FAIL,
                threshold=spec.max_l2,
                topic=spec.topic,
                message=(
                    f"sequence length mismatch: predicted={len(predicted)} "
                    f"recorded={len(recorded)}"
                ),
            )
        measured = mean_l2(predicted, recorded)
        passed = measured <= spec.max_l2 and math.isfinite(measured)
        return AssertResult(
            type=self.type,
            status=Status.PASS if passed else Status.FAIL,
            measured=measured,
            threshold=spec.max_l2,
            topic=spec.topic,
            message=(
                f"mean L2={measured:.6f} (max_l2={spec.max_l2})"
                if passed
                else f"mean L2={measured:.6f} exceeds max_l2={spec.max_l2}"
            ),
        )


register(ActionDeviationAssertion())
