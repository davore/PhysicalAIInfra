"""Load a run directory: meta.json + outputs/*.parquet + events.jsonl."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict


class RunMode(StrEnum):
    OPEN_LOOP = "open_loop"
    CLOSED_LOOP = "closed_loop"


class RunMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    scenario_hash: str
    adapter: str
    mode: RunMode
    target_version: str | None = None
    git_sha: str | None = None
    host: str | None = None


class Run:
    def __init__(self, path: Path, meta: RunMeta, events: list[dict[str, Any]]) -> None:
        self.path = path
        self.meta = meta
        self.events = events

    @property
    def mode(self) -> RunMode:
        return self.meta.mode

    def output_path(self, topic: str, *, recorded: bool = False) -> Path:
        name = topic_filename(topic, recorded=recorded)
        return self.path / "outputs" / name

    def output(self, topic: str, *, recorded: bool = False) -> pl.DataFrame:
        path = self.output_path(topic, recorded=recorded)
        if not path.is_file():
            raise FileNotFoundError(f"run output not found: {path}")
        return pl.read_parquet(path)

    @classmethod
    def load(cls, path: str | Path) -> Run:
        root = Path(path)
        meta_path = root / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"run meta.json not found: {meta_path}")
        meta = RunMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        events_path = root / "events.jsonl"
        events: list[dict[str, Any]] = []
        if events_path.is_file():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))
        return cls(root, meta, events)


def topic_filename(topic: str, *, recorded: bool = False) -> str:
    stem = topic.strip("/").replace("/", "_")
    if not stem:
        raise ValueError("topic must not be empty")
    suffix = ".recorded.parquet" if recorded else ".parquet"
    return f"{stem}{suffix}"


def write_meta(path: str | Path, meta: RunMeta) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    (Path(path) / "meta.json").write_text(
        meta.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )
