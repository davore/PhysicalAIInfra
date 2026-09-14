"""Scenario schema v0: read-only definition + content hash."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 0


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LeRobotSource(_Strict):
    kind: Literal["lerobot"]
    dataset: str
    episode_index: int = Field(ge=0)
    frame_from: int = Field(ge=0)
    frame_to: int = Field(ge=0)
    revision: str | None = None
    foxglove_url: str | None = None

    @model_validator(mode="after")
    def _range(self) -> LeRobotSource:
        if self.frame_to <= self.frame_from:
            raise ValueError("frame_to must be greater than frame_from")
        return self


class McapSource(_Strict):
    kind: Literal["mcap"]
    mcap: str
    from_ns: int | None = Field(default=None, ge=0)
    to_ns: int | None = Field(default=None, ge=0)
    from_iso: str | None = None
    to_iso: str | None = None
    foxglove_url: str | None = None

    @model_validator(mode="after")
    def _range(self) -> McapSource:
        has_ns = self.from_ns is not None and self.to_ns is not None
        has_iso = self.from_iso is not None and self.to_iso is not None
        if not has_ns and not has_iso:
            raise ValueError("mcap source needs from_ns/to_ns or from_iso/to_iso")
        if has_ns and self.to_ns <= self.from_ns:  # type: ignore[operator]
            raise ValueError("to_ns must be greater than from_ns")
        if has_iso:
            _require_tz(self.from_iso, "from_iso")
            _require_tz(self.to_iso, "to_iso")
        return self


def _require_tz(value: str | None, field: str) -> None:
    if value is None:
        raise ValueError(f"{field} is required")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone offset")


Source = Annotated[LeRobotSource | McapSource, Field(discriminator="kind")]


class Target(_Strict):
    adapter: Literal["python-policy", "ros2-node", "isaac-lab"]
    entry: str | None = None
    checkpoint: str | None = None
    revision: str | None = None


class ActionDeviation(_Strict):
    type: Literal["action_deviation"]
    topic: str
    reference: Literal["recorded"] = "recorded"
    max_l2: float = Field(gt=0)
    metric: Literal["l2", "dtw"] = "l2"


class ActionBounds(_Strict):
    type: Literal["action_bounds"]
    topic: str
    min: list[float]
    max: list[float]


class ActionSmoothness(_Strict):
    type: Literal["action_smoothness"]
    topic: str
    max_delta: float = Field(gt=0)


class Latency(_Strict):
    type: Literal["latency"]
    topic: str
    p95_ms: float = Field(gt=0)


class ConfidenceFloor(_Strict):
    type: Literal["confidence_floor"]
    topic: str
    min_confidence: float = Field(ge=0, le=1)


class GoalReached(_Strict):
    type: Literal["goal_reached"]
    topic: str | None = None
    within_s: float | None = Field(default=None, gt=0)


class GraspSuccess(_Strict):
    type: Literal["grasp_success"]
    topic: str | None = None
    within_s: float = Field(gt=0)


class NoCollision(_Strict):
    type: Literal["no_collision"]


class SimSuccessRate(_Strict):
    type: Literal["sim_success_rate"]
    min_rate: float = Field(ge=0, le=1)


class RecoveryWithin(_Strict):
    type: Literal["recovery_within"]
    within_s: float = Field(gt=0)


Expectation = Annotated[
    ActionDeviation
    | ActionBounds
    | ActionSmoothness
    | Latency
    | ConfidenceFloor
    | GoalReached
    | GraspSuccess
    | NoCollision
    | SimSuccessRate
    | RecoveryWithin,
    Field(discriminator="type"),
]


class Scenario(_Strict):
    schema_version: Literal[0] = SCHEMA_VERSION
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    blocking: bool = False
    source: Source
    tags: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(min_length=1)
    target: Target
    expected: list[Expectation] = Field(min_length=1)


def load_scenario(path: str | Path) -> Scenario:
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("scenario file must be a YAML mapping")
    return Scenario.model_validate(data)


def canonical_dict(scenario: Scenario) -> dict[str, Any]:
    return scenario.model_dump(mode="json", exclude_none=True)


def content_hash(scenario: Scenario) -> str:
    payload = json.dumps(canonical_dict(scenario), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def json_schema() -> dict[str, Any]:
    return Scenario.model_json_schema()
