"""Open-loop: every predicted action stays inside per-dimension limits."""

from __future__ import annotations

import numpy as np

from robogate.asserts.action_deviation import vectors_from_frame
from robogate.asserts.base import Assertion, AssertResult, Status, register
from robogate.run import Run, RunMode
from robogate.scenario import ActionBounds, Expectation


class ActionBoundsAssertion(Assertion):
    type = "action_bounds"
    requires_mode = RunMode.OPEN_LOOP

    def evaluate(self, run: Run, spec: Expectation) -> AssertResult:
        if not isinstance(spec, ActionBounds):
            raise TypeError("expected ActionBounds spec")
        predicted = vectors_from_frame(run.output(spec.topic))
        low = np.asarray(spec.min, dtype=np.float64)
        high = np.asarray(spec.max, dtype=np.float64)
        if low.shape != high.shape:
            raise ValueError(f"bounds dim mismatch: min {low.shape} vs max {high.shape}")
        if predicted.shape[1] != low.shape[0]:
            raise ValueError(
                f"action dim mismatch: predicted {predicted.shape[1]} vs bounds {low.shape[0]}"
            )
        overflow = predicted - np.clip(predicted, low, high)
        measured = float(np.max(np.linalg.norm(overflow, axis=1)))
        passed = bool(np.all(predicted >= low - 1e-12) and np.all(predicted <= high + 1e-12))
        return AssertResult(
            type=self.type,
            status=Status.PASS if passed else Status.FAIL,
            measured=measured,
            topic=spec.topic,
            message=(
                "all actions within bounds"
                if passed
                else f"actions leave bounds (max overflow L2={measured:.6f})"
            ),
        )


register(ActionBoundsAssertion())
