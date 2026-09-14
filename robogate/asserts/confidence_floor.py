"""Open-loop: minimum reported confidence must stay above the floor."""

from __future__ import annotations

import numpy as np
import polars as pl

from robogate.asserts.base import Assertion, AssertResult, Status, register
from robogate.run import Run, RunMode
from robogate.scenario import ConfidenceFloor, Expectation


def min_confidence(df: pl.DataFrame) -> float:
    if "confidence" not in df.columns:
        raise ValueError("confidence parquet must have a 'confidence' column")
    values = np.asarray(df.get_column("confidence").to_list(), dtype=np.float64)
    if values.size == 0:
        raise ValueError("confidence parquet has no rows")
    return float(np.min(values))


class ConfidenceFloorAssertion(Assertion):
    type = "confidence_floor"
    requires_mode = RunMode.OPEN_LOOP

    def evaluate(self, run: Run, spec: Expectation) -> AssertResult:
        if not isinstance(spec, ConfidenceFloor):
            raise TypeError("expected ConfidenceFloor spec")
        if not run.has_output(spec.topic):
            return AssertResult(
                type=self.type,
                status=Status.SKIPPED,
                threshold=spec.min_confidence,
                topic=spec.topic,
                message="adapter did not record confidence",
            )
        df = run.output(spec.topic)
        if "confidence" not in df.columns:
            return AssertResult(
                type=self.type,
                status=Status.SKIPPED,
                threshold=spec.min_confidence,
                topic=spec.topic,
                message="adapter did not record confidence",
            )
        measured = min_confidence(df)
        passed = measured >= spec.min_confidence
        return AssertResult(
            type=self.type,
            status=Status.PASS if passed else Status.FAIL,
            measured=measured,
            threshold=spec.min_confidence,
            topic=spec.topic,
            message=(
                f"min confidence={measured:.4f} (floor={spec.min_confidence})"
                if passed
                else f"min confidence={measured:.4f} below floor={spec.min_confidence}"
            ),
        )


register(ConfidenceFloorAssertion())
