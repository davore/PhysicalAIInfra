"""Assertion registry, result types, and mode gating."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from robogate.run import Run, RunMode
from robogate.scenario import Expectation


class Status(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


class AssertResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    status: Status
    measured: float | None = None
    threshold: float | None = None
    message: str = ""
    topic: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class Assertion(ABC):
    type: str
    requires_mode: RunMode

    @abstractmethod
    def evaluate(self, run: Run, spec: Expectation) -> AssertResult:
        raise NotImplementedError


_REGISTRY: dict[str, Assertion] = {}


def register(assertion: Assertion) -> Assertion:
    _REGISTRY[assertion.type] = assertion
    return assertion


def get_assertion(type_name: str) -> Assertion | None:
    return _REGISTRY.get(type_name)


def registered_types() -> list[str]:
    return sorted(_REGISTRY)


def maybe_skip(assertion: Assertion, run: Run, spec: Expectation) -> AssertResult | None:
    if run.mode == assertion.requires_mode:
        return None
    topic = getattr(spec, "topic", None)
    return AssertResult(
        type=assertion.type,
        status=Status.SKIPPED,
        topic=topic,
        message=(
            f"{assertion.type} requires {assertion.requires_mode.value}, "
            f"run mode is {run.mode.value}"
        ),
    )
