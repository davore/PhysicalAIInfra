"""Action-side perturbations wrapped around any adapter."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from robogate.replay.base import Adapter, AdapterInfo, StepOut
from robogate.scenario import Target
from robogate.slice import Slice

ACTION_KEYS = ("action_noise", "action_bias", "action_lag", "action_scale")
INPUT_KEYS = ("drop_camera", "state_noise", "action_stats")
KNOWN_KEYS = ACTION_KEYS + INPUT_KEYS


def parse_perturbations(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"invalid --perturb {item!r}; expected key=value")
        out[key.strip()] = value.strip()
    unknown = [key for key in out if key not in KNOWN_KEYS]
    if unknown:
        raise ValueError(f"unknown perturbation keys: {unknown}; expected {list(KNOWN_KEYS)}")
    if "action_stats" in out and out["action_stats"] != "dataset":
        raise ValueError("action_stats must be 'dataset'")
    return out


class PerturbedAdapter(Adapter):
    def __init__(
        self,
        inner: Adapter,
        specs: dict[str, str],
        *,
        seed: int = 0,
    ) -> None:
        self.inner = inner
        self.specs = dict(specs)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._lag: deque[np.ndarray] = deque()
        inner_name = getattr(inner, "info", AdapterInfo(name="inner")).name
        self.info = AdapterInfo(name=f"perturb+{inner_name}")

    def load(self, target: Target, slice_: Slice, *, device: str) -> None:
        self.inner.load(target, slice_, device=device)
        self.info = AdapterInfo(
            name=f"perturb+{self.inner.info.name}",
            target_version=self.inner.info.target_version,
            target_revision=self.inner.info.target_revision,
            versions=dict(self.inner.info.versions),
            policy_config=dict(self.inner.info.policy_config),
        )
        self.reset()

    def reset(self) -> None:
        self.inner.reset()
        self._rng = np.random.default_rng(self.seed)
        self._lag = deque()

    def step(self, obs: dict[str, Any], frame_index: int) -> StepOut:
        out = self.inner.step(obs, frame_index)
        action = np.asarray(out.action, dtype=np.float64).reshape(-1).copy()
        if "action_scale" in self.specs:
            action = action * float(self.specs["action_scale"])
        if "action_bias" in self.specs:
            action = action + float(self.specs["action_bias"])
        if "action_noise" in self.specs:
            sigma = float(self.specs["action_noise"])
            action = action + self._rng.normal(0.0, sigma, size=action.shape)
        if "action_lag" in self.specs:
            lag = int(self.specs["action_lag"])
            self._lag.append(action)
            if len(self._lag) <= lag:
                action = self._lag[0].copy()
            else:
                action = self._lag.popleft()
        return StepOut(
            action=action,
            latency_ms=out.latency_ms,
            confidence=out.confidence,
            images=out.images,
        )
