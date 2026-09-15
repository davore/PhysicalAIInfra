"""Identity / noisy replay adapter. No lerobot dependency."""

from __future__ import annotations

from typing import Any

import numpy as np

from robogate.replay.base import Adapter, AdapterInfo, StepOut
from robogate.scenario import Target
from robogate.slice import Slice


class MockAdapter(Adapter):
    def __init__(self, noise: float = 0.0, seed: int = 0) -> None:
        self.noise = noise
        self.seed = seed
        self._recorded: np.ndarray | None = None
        self._rng = np.random.default_rng(seed)
        self.info = AdapterInfo(name="mock")

    def load(self, target: Target, slice_: Slice, *, device: str) -> None:
        recorded = slice_.recorded_action()
        if "value" not in recorded.columns:
            raise ValueError("slice recorded/action.parquet must have a 'value' column")
        self._recorded = np.asarray(recorded.get_column("value").to_list(), dtype=np.float64)
        self.info = AdapterInfo(
            name="mock",
            target_version=target.checkpoint or "identity",
            versions={"adapter": "mock", "device": device},
        )

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def step(self, obs: dict[str, Any], frame_index: int) -> StepOut:
        del obs
        if self._recorded is None:
            raise RuntimeError("adapter not loaded")
        if frame_index >= len(self._recorded):
            raise IndexError(f"frame_index {frame_index} out of recorded range")
        action = self._recorded[frame_index].copy()
        if self.noise:
            action = action + self._rng.normal(0.0, self.noise, size=action.shape)
        return StepOut(action=action, latency_ms=0.0, confidence=None)
