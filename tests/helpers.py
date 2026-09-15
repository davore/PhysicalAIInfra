"""Shared test fixtures for extract / replay / assert."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl


def write_mini_lerobot_v3(
    root: Path,
    *,
    n_frames: int = 8,
    fps: float = 50.0,
    action_dim: int = 2,
    episode_index: int = 0,
    n_episodes: int = 1,
) -> Path:
    meta = root / "meta"
    (meta / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "test",
        "fps": fps,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [action_dim]},
            "observation.images.cam_high": {
                "dtype": "video",
                "shape": [8, 8, 3],
            },
            "action": {"dtype": "float32", "shape": [action_dim]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    states: list[list[float]] = []
    actions: list[list[float]] = []
    ep_idx: list[int] = []
    frame_idx: list[int] = []
    stamps: list[float] = []
    ep_rows: list[dict[str, object]] = []
    cursor = 0
    for offset in range(n_episodes):
        ep = episode_index + offset
        frames = np.arange(n_frames, dtype=np.int64)
        if n_episodes == 1:
            action = np.stack([frames * 0.01, np.ones(n_frames)], axis=1).astype(np.float32)
        else:
            action = np.stack(
                [frames * 0.01 + offset * 0.1, np.ones(n_frames) + offset * 0.05],
                axis=1,
            ).astype(np.float32)
        if action_dim != 2:
            pad = np.zeros((n_frames, action_dim - 2), dtype=np.float32)
            action = np.concatenate([action, pad], axis=1)
        state = action + 0.001
        states.extend(state.tolist())
        actions.extend(action.tolist())
        ep_idx.extend([ep] * n_frames)
        frame_idx.extend(frames.tolist())
        stamps.extend((frames / fps).astype(np.float32).tolist())
        ep_rows.append(
            {
                "episode_index": ep,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": cursor,
                "dataset_to_index": cursor + n_frames,
                "length": n_frames,
                "videos/observation.images.cam_high/chunk_index": 0,
                "videos/observation.images.cam_high/file_index": 0,
                "videos/observation.images.cam_high/from_timestamp": 0.0,
                "videos/observation.images.cam_high/to_timestamp": (n_frames - 1) / fps,
            }
        )
        cursor += n_frames
    pl.DataFrame(
        {
            "observation.state": states,
            "action": actions,
            "episode_index": ep_idx,
            "frame_index": frame_idx,
            "timestamp": stamps,
        }
    ).write_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pl.DataFrame(ep_rows).write_parquet(meta / "episodes" / "chunk-000" / "file-000.parquet")
    return root
