"""On-disk slice produced by extract: recorded actions, tabular inputs, metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class VideoRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    from_timestamp: float
    to_timestamp: float
    chunk_index: int | None = None
    file_index: int | None = None


class SliceMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    dataset: str
    dataset_revision: str | None = None
    episode_index: int
    frame_from: int
    frame_to: int
    fps: float
    n_frames: int
    action_dim: int | None = None
    features: dict[str, Any] = Field(default_factory=dict)
    videos: dict[str, VideoRef] = Field(default_factory=dict)
    period_ns: int


class Slice:
    def __init__(self, path: Path, meta: SliceMeta) -> None:
        self.path = path
        self.meta = meta

    @property
    def recorded_action_path(self) -> Path:
        return self.path / "recorded" / "action.parquet"

    def recorded_action(self) -> pl.DataFrame:
        return pl.read_parquet(self.recorded_action_path)

    def input_path(self, key: str) -> Path:
        return self.path / "inputs" / f"{key}.parquet"

    def input_frame(self, key: str) -> pl.DataFrame | None:
        path = self.input_path(key)
        if not path.is_file():
            return None
        return pl.read_parquet(path)

    @classmethod
    def load(cls, path: str | Path) -> Slice:
        root = Path(path)
        meta_path = root / "slice.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"slice.json not found: {meta_path}")
        meta = SliceMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        return cls(root, meta)


def write_slice_meta(path: str | Path, meta: SliceMeta) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(meta.model_dump(mode="json", exclude_none=True), indent=2)
    (root / "slice.json").write_text(payload + "\n", encoding="utf-8")


def period_ns(fps: float) -> int:
    if fps <= 0:
        raise ValueError("fps must be positive")
    return int(round(1_000_000_000 / fps))


def frame_times_ns(frame_index: list[int] | range, fps: float) -> list[int]:
    step = period_ns(fps)
    return [int(idx) * step for idx in frame_index]
