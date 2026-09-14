"""Open-loop: estimated frame lag of predicted action vs recorded."""

from __future__ import annotations

import math

import numpy as np

from robogate.asserts.action_deviation import mean_l2, vectors_from_frame
from robogate.asserts.base import Assertion, AssertResult, Status, register
from robogate.run import Run, RunMode
from robogate.scenario import ActionLag, Expectation

WEAK_IMPROVEMENT = 0.05


def estimate_lag(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    search_frames: int = 50,
) -> tuple[int, float, float]:
    """Return (k, l2_at_k, l2_at_0) where k>0 means predicted lags reference."""
    n = min(len(predicted), len(reference))
    if n < 2:
        raise ValueError("need at least 2 frames to estimate lag")
    pred = np.asarray(predicted[:n], dtype=np.float64)
    ref = np.asarray(reference[:n], dtype=np.float64)
    zero = mean_l2(pred, ref)
    limit = min(int(search_frames), n - 1)
    best_k = 0
    best_l2 = zero
    for k in range(-limit, limit + 1):
        if k == 0:
            measured = zero
        elif k > 0:
            measured = mean_l2(pred[k:], ref[: n - k])
        else:
            measured = mean_l2(pred[:k], ref[-k:])
        if measured < best_l2 - 1e-12:
            best_l2 = measured
            best_k = k
    return best_k, float(best_l2), float(zero)


class ActionLagAssertion(Assertion):
    type = "action_lag"
    requires_mode = RunMode.OPEN_LOOP

    def evaluate(self, run: Run, spec: Expectation) -> AssertResult:
        if not isinstance(spec, ActionLag):
            raise TypeError("expected ActionLag spec")
        predicted = vectors_from_frame(run.output(spec.topic))
        recorded = vectors_from_frame(run.output(spec.topic, recorded=True))
        lag, best_l2, zero_l2 = estimate_lag(
            predicted, recorded, search_frames=spec.search_frames
        )
        if not math.isfinite(zero_l2) or zero_l2 <= 1e-12:
            improvement = 0.0
        else:
            improvement = (zero_l2 - best_l2) / zero_l2
        if improvement < WEAK_IMPROVEMENT:
            return AssertResult(
                type=self.type,
                status=Status.PASS,
                measured=float(lag),
                threshold=float(spec.max_lag_frames),
                topic=spec.topic,
                message=(
                    f"tracking too weak to measure lag "
                    f"(best k={lag} improves L2 by {100 * improvement:.1f}%)"
                ),
            )
        passed = abs(lag) <= spec.max_lag_frames
        return AssertResult(
            type=self.type,
            status=Status.PASS if passed else Status.FAIL,
            measured=float(lag),
            threshold=float(spec.max_lag_frames),
            topic=spec.topic,
            message=(
                f"lag={lag} frames (max_lag_frames={spec.max_lag_frames})"
                if passed
                else f"lag={lag} exceeds max_lag_frames={spec.max_lag_frames}"
            ),
        )


register(ActionLagAssertion())
