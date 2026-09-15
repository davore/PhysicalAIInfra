"""Load a directory of scenario.yaml files."""

from __future__ import annotations

from pathlib import Path

from robogate.scenario import Scenario, load_scenario


def load_suite(suite: str | Path) -> list[tuple[Path, Scenario]]:
    root = Path(suite)
    if root.is_file():
        return [(root, load_scenario(root))]
    if not root.is_dir():
        raise FileNotFoundError(f"suite not found: {root}")
    paths = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
    if not paths:
        raise ValueError(f"no scenario yaml in {root}")
    return [(path, load_scenario(path)) for path in paths]


def parse_episodes(spec: str | None, episode: int) -> list[int]:
    if not spec:
        return [episode]
    out: list[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"invalid episode range {token}")
            out.extend(range(start, end + 1))
        else:
            out.append(int(token))
    if not out:
        raise ValueError(f"no episodes in {spec!r}")
    return out
