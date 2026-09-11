"""Run registered assertions. Any fail is a suite failure; skipped is not."""

from __future__ import annotations

from robogate.asserts import action_deviation, closed_loop, confidence_floor, latency  # noqa: F401
from robogate.asserts.base import (
    Assertion,
    AssertResult,
    Status,
    get_assertion,
    maybe_skip,
    register,
    registered_types,
)
from robogate.run import Run
from robogate.scenario import Scenario


def run_assertions(scenario: Scenario, run: Run) -> list[AssertResult]:
    results: list[AssertResult] = []
    for spec in scenario.expected:
        assertion = get_assertion(spec.type)
        if assertion is None:
            results.append(
                AssertResult(
                    type=spec.type,
                    status=Status.FAIL,
                    topic=getattr(spec, "topic", None),
                    message=f"assertion type not implemented: {spec.type}",
                )
            )
            continue
        skipped = maybe_skip(assertion, run, spec)
        if skipped is not None:
            results.append(skipped)
            continue
        results.append(assertion.evaluate(run, spec))
    return results


def overall_ok(results: list[AssertResult]) -> bool:
    return all(item.status != Status.FAIL for item in results)


__all__ = [
    "AssertResult",
    "Assertion",
    "Status",
    "get_assertion",
    "overall_ok",
    "register",
    "registered_types",
    "run_assertions",
]
