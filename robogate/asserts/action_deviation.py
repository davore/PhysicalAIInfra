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


def _require_same_dim(predicted: np.ndarray, recorded: np.ndarray) -> None:
    if predicted.ndim != 2 or recorded.ndim != 2:
        raise ValueError("action arrays must be 2-D (T, dim)")
    if predicted.shape[1] != recorded.shape[1]:
        raise ValueError(
            f"action dim mismatch: predicted {predicted.shape[1]} vs recorded {recorded.shape[1]}"
        )


def mean_l2(predicted: np.ndarray, recorded: np.ndarray) -> float:
    n = min(len(predicted), len(recorded))
    if n == 0:
        raise ValueError("cannot compare empty action sequences")
    pred = predicted[:n]
    ref = recorded[:n]
    _require_same_dim(pred, ref)
    diffs = pred - ref
    return float(np.mean(np.linalg.norm(diffs, axis=1)))


def dtw_normalized(predicted: np.ndarray, recorded: np.ndarray) -> float:
    """O(n²) DTW on per-frame L2, divided by the recovered path length."""
    if len(predicted) == 0 or len(recorded) == 0:
        raise ValueError("cannot compare empty action sequences")
    _require_same_dim(predicted, recorded)
    n, m = len(predicted), len(recorded)
    pairwise = np.linalg.norm(predicted[:, None, :] - recorded[None, :, :], axis=2)
    inf = np.inf
    cost = np.full((n + 1, m + 1), inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = pairwise[i - 1, j - 1] + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
            )
    i, j = n, m
    length = 0
    while i > 0 and j > 0:
        length += 1
        diag, up, left = cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]
        if diag <= up and diag <= left:
            i, j = i - 1, j - 1
        elif up <= left:
            i -= 1
        else:
            j -= 1
    if length == 0:
        raise ValueError("DTW path is empty")
    return float(cost[n, m] / length)


class ActionDeviationAssertion(Assertion):
    type = "action_deviation"
    requires_mode = RunMode.OPEN_LOOP

    def evaluate(self, run: Run, spec: Expectation) -> AssertResult:
        if not isinstance(spec, ActionDeviation):
            raise TypeError("expected ActionDeviation spec")
        predicted = vectors_from_frame(run.output(spec.topic))
        recorded = vectors_from_frame(run.output(spec.topic, recorded=True))
        if spec.metric == "l2":
            length_ratio = min(len(predicted), len(recorded)) / max(
                len(predicted), len(recorded)
            )
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
            label = "mean L2"
        elif spec.metric == "dtw":
            measured = dtw_normalized(predicted, recorded)
            label = "DTW"
        else:
            return AssertResult(
                type=self.type,
                status=Status.FAIL,
                threshold=spec.max_l2,
                topic=spec.topic,
                message=f"metric {spec.metric} is not implemented",
            )
        passed = measured <= spec.max_l2 and math.isfinite(measured)
        return AssertResult(
            type=self.type,
            status=Status.PASS if passed else Status.FAIL,
            measured=measured,
            threshold=spec.max_l2,
            topic=spec.topic,
            message=(
                f"{label}={measured:.6f} (max_l2={spec.max_l2})"
                if passed
                else f"{label}={measured:.6f} exceeds max_l2={spec.max_l2}"
            ),
        )


register(ActionDeviationAssertion())
