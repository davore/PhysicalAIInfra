"""Replay adapters and runner."""

from robogate.replay.base import Adapter, AdapterInfo, StepOut
from robogate.replay.mock import MockAdapter
from robogate.replay.perturb import (
    ACTION_KEYS,
    PerturbedAdapter,
    parse_perturbations,
)
from robogate.replay.runner import run_replay

__all__ = [
    "Adapter",
    "AdapterInfo",
    "MockAdapter",
    "PerturbedAdapter",
    "StepOut",
    "run_replay",
    "build_adapter",
    "adapter_entry",
]


def adapter_entry(adapter: str, target_entry: str | None) -> str:
    if adapter == "mock":
        return "mock"
    if adapter == "lerobot":
        return "lerobot"
    if adapter != "auto":
        return adapter
    if target_entry in {None, "", "lerobot"}:
        return "lerobot"
    return target_entry


def build_adapter(
    entry: str | None,
    *,
    noise: float = 0.0,
    perturbations: list[str] | None = None,
) -> Adapter:
    specs = parse_perturbations(perturbations)
    name = (entry or "lerobot").strip()
    if name in {"mock", "identity"}:
        input_specs = {
            key: specs[key]
            for key in ("drop_camera", "state_noise", "action_stats")
            if key in specs
        }
        if input_specs:
            raise ValueError("drop_camera/state_noise/action_stats require the lerobot adapter")
        inner: Adapter = MockAdapter(noise=noise)
    elif name == "lerobot":
        from robogate.replay.lerobot_policy import LeRobotPolicyAdapter

        inner = LeRobotPolicyAdapter(
            drop_camera=specs.get("drop_camera"),
            state_noise=float(specs["state_noise"]) if "state_noise" in specs else 0.0,
            action_stats=specs.get("action_stats"),
        )
    elif ":" in name:
        inner = _load_callable_adapter(name)
    else:
        from robogate.replay.lerobot_policy import LeRobotPolicyAdapter

        inner = LeRobotPolicyAdapter(
            drop_camera=specs.get("drop_camera"),
            state_noise=float(specs["state_noise"]) if "state_noise" in specs else 0.0,
            action_stats=specs.get("action_stats"),
        )
    action_specs = {key: specs[key] for key in ACTION_KEYS if key in specs}
    if action_specs:
        return PerturbedAdapter(inner, action_specs)
    return inner


def _load_callable_adapter(entry: str) -> Adapter:
    module_name, _, attr = entry.partition(":")
    if not module_name or not attr:
        raise ValueError(f"invalid entry {entry!r}; expected module:callable")
    import importlib

    module = importlib.import_module(module_name)
    loaded = getattr(module, attr)
    if isinstance(loaded, type):
        candidate = loaded()
    else:
        candidate = loaded
    if not hasattr(candidate, "load") or not hasattr(candidate, "step"):
        raise TypeError(f"{entry} did not return an Adapter with load/step")
    return candidate  # type: ignore[return-value]
