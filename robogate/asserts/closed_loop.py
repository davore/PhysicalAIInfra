"""Closed-loop assertions. Skipped on open-loop runs (M0)."""

from __future__ import annotations

from robogate.asserts.base import Assertion, AssertResult, Status, register
from robogate.run import Run, RunMode
from robogate.scenario import Expectation, GraspSuccess


class GoalReachedAssertion(Assertion):
    type = "goal_reached"
    requires_mode = RunMode.CLOSED_LOOP

    def evaluate(self, run: Run, spec: Expectation) -> AssertResult:
        hit = any(event.get("type") == "goal_reached" for event in run.events)
        return AssertResult(
            type=self.type,
            status=Status.PASS if hit else Status.FAIL,
            topic=getattr(spec, "topic", None),
            message="goal_reached event present" if hit else "no goal_reached event",
        )


class GraspSuccessAssertion(Assertion):
    type = "grasp_success"
    requires_mode = RunMode.CLOSED_LOOP

    def evaluate(self, run: Run, spec: Expectation) -> AssertResult:
        if not isinstance(spec, GraspSuccess):
            raise TypeError("expected GraspSuccess spec")
        deadline_ns = int(spec.within_s * 1_000_000_000)
        for event in run.events:
            if event.get("type") != "grasp_success":
                continue
            t_ns = int(event.get("t_ns", 0))
            if t_ns <= deadline_ns and event.get("success") is True:
                return AssertResult(
                    type=self.type,
                    status=Status.PASS,
                    topic=spec.topic,
                    threshold=spec.within_s,
                    message=f"grasp succeeded at t_ns={t_ns}",
                )
        return AssertResult(
            type=self.type,
            status=Status.FAIL,
            topic=spec.topic,
            threshold=spec.within_s,
            message=f"no successful grasp within {spec.within_s}s",
        )


class NoCollisionAssertion(Assertion):
    type = "no_collision"
    requires_mode = RunMode.CLOSED_LOOP

    def evaluate(self, run: Run, spec: Expectation) -> AssertResult:
        hits = [event for event in run.events if event.get("type") == "collision"]
        return AssertResult(
            type=self.type,
            status=Status.FAIL if hits else Status.PASS,
            message=f"{len(hits)} collision event(s)" if hits else "no collision events",
        )


register(GoalReachedAssertion())
register(GraspSuccessAssertion())
register(NoCollisionAssertion())
