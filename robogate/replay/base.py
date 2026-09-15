"""Adapter protocol for open-loop replay."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robogate.scenario import Target
from robogate.slice import Slice


@dataclass
class StepOut:
    action: np.ndarray
    latency_ms: float
    confidence: float | None = None


@dataclass
class AdapterInfo:
    name: str
    target_version: str | None = None
    target_revision: str | None = None
    versions: dict[str, str] = field(default_factory=dict)
    policy_config: dict[str, Any] = field(default_factory=dict)


class Adapter(ABC):
    info: AdapterInfo

    @abstractmethod
    def load(self, target: Target, slice_: Slice, *, device: str) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        return None

    @abstractmethod
    def step(self, obs: dict[str, Any], frame_index: int) -> StepOut:
        raise NotImplementedError
