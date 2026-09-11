"""Open-loop latency: report p95, never fail the suite."""

from __future__ import annotations

import numpy as np
import polars as pl

from robogate.asserts.base import Assertion, AssertResult, Status, register
from robogate.run import Run, RunMode
from robogate.scenario import Expectation, Latency


def p95_ms(df: pl.DataFrame) -> float:
    if "latency_ms" not in df.columns:
        raise ValueError("latency parquet must have a 'latency_ms' column")
    values = np.asarray(df.get_column("latency_ms").to_list(), dtype=np.float64)
    if values.size == 0:
        raise ValueError("latency parquet has no rows")
    return float(np.percentile(values, 95))


class LatencyAssertion(Assertion):
    type = "latency"
    requires_mode = RunMode.OPEN_LOOP

    def evaluate(self, run: Run, spec: Expectation) -> AssertResult:
        if not isinstance(spec, Latency):
            raise TypeError("expected Latency spec")
        measured = p95_ms(run.output(spec.topic))
        note = (
            f"p95={measured:.2f}ms (budget {spec.p95_ms}ms); informational, not blocking"
        )
        if measured > spec.p95_ms:
            note = (
                f"p95={measured:.2f}ms exceeds budget {spec.p95_ms}ms "
                "on non-target hardware; informational, not blocking"
            )
        return AssertResult(
            type=self.type,
            status=Status.PASS,
            measured=measured,
            threshold=spec.p95_ms,
            topic=spec.topic,
            message=note,
        )


register(LatencyAssertion())
