"""Open-loop: consecutive-frame action L2 stays under max_delta."""

from __future__ import annotations

import numpy as np

from robogate.asserts.action_deviation import vectors_from_frame
from robogate.asserts.base import Assertion, AssertResult, Status, register
from robogate.run import Run, RunMode
from robogate.scenario import ActionSmoothness, Expectation


class ActionSmoothnessAssertion(Assertion):
    type = "action_smoothness"
    requires_mode = RunMode.OPEN_LOOP

    def evaluate(self, run: Run, spec: Expectation) -> AssertResult:
        if not isinstance(spec, ActionSmoothness):
            raise TypeError("expected ActionSmoothness spec")
        predicted = vectors_from_frame(run.output(spec.topic))
        if len(predicted) < 2:
            raise ValueError("need at least two frames to measure smoothness")
        deltas = np.linalg.norm(np.diff(predicted, axis=0), axis=1)
        measured = float(np.max(deltas))
        passed = measured <= spec.max_delta
        return AssertResult(
            type=self.type,
            status=Status.PASS if passed else Status.FAIL,
            measured=measured,
            threshold=spec.max_delta,
            topic=spec.topic,
            message=(
                f"max frame delta L2={measured:.6f} (max_delta={spec.max_delta})"
                if passed
                else f"max frame delta L2={measured:.6f} exceeds max_delta={spec.max_delta}"
            ),
        )


register(ActionSmoothnessAssertion())
