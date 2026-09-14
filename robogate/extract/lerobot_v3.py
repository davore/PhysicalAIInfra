"""Slice a LeRobot v3 dataset into robogate slice + scenario.yaml."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from robogate.scenario import Scenario, content_hash
from robogate.slice import Slice, SliceMeta, VideoRef, frame_times_ns, write_slice_meta

SKIP_FEATURES = {
    "action",
    "episode_index",
    "frame_index",
    "timestamp",
    "next.done",
    "index",
    "task_index",
}


def extract_lerobot(
    dataset: str,
    *,
    episode_index: int,
    frame_from: int,
    frame_to: int | None,
    scenario_id: str,
    scenario_out: Path,
    slice_root: Path,
    revision: str | None = None,
    repo_id: str | None = None,
    checkpoint: str | None = None,
    target_revision: str | None = None,
    blocking: bool = True,
    entry: str = "lerobot",
) -> tuple[Path, Path]:
    root = resolve_dataset_root(dataset, revision=revision)
    info = load_info(root)
    fps = float(info["fps"])
    episode = load_episode_row(root, episode_index)
    if episode.get("length") is not None:
        length = int(episode["length"])
    else:
        length = int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])
    end = length if frame_to is None else frame_to
    if end > length:
        raise ValueError(f"frame_to {end} exceeds episode length {length}")
    if end <= frame_from:
        raise ValueError("frame_to must be greater than frame_from")

    table = load_episode_frames(root, info, episode, frame_from, end)
    n_frames = table.height
    if n_frames == 0:
        raise ValueError("slice is empty")

    frame_idx = table.get_column("frame_index").to_list()
    t_ns = frame_times_ns([int(i) for i in frame_idx], fps)
    features: dict[str, Any] = info.get("features") or {}
    inputs = [name for name in features if name not in SKIP_FEATURES]
    videos = video_refs(info, episode, frame_from, end, fps)

    slice_dir = Path(slice_root) / scenario_id
    (slice_dir / "recorded").mkdir(parents=True, exist_ok=True)
    (slice_dir / "inputs").mkdir(parents=True, exist_ok=True)

    action_dim: int | None = None
    if "action" in table.columns:
        actions = table.get_column("action").to_list()
        action_dim = len(actions[0])
        pl.DataFrame({"t_ns": t_ns, "value": actions}).write_parquet(
            slice_dir / "recorded" / "action.parquet"
        )
    for key in inputs:
        if key not in table.columns:
            continue
        if _is_video(features.get(key)):
            continue
        pl.DataFrame({"t_ns": t_ns, "value": table.get_column(key).to_list()}).write_parquet(
            slice_dir / "inputs" / f"{key}.parquet"
        )

    dataset_name = repo_id or (dataset if not Path(dataset).exists() else dataset)
    meta = SliceMeta(
        scenario_id=scenario_id,
        dataset=dataset_name,
        dataset_revision=revision
        or (
            None
            if Path(dataset).exists()
            else lookup_revision(dataset_name, local_root=root)
        ),
        episode_index=episode_index,
        frame_from=frame_from,
        frame_to=end,
        fps=fps,
        n_frames=n_frames,
        action_dim=action_dim,
        features=features,
        videos=videos,
        period_ns=int(round(1_000_000_000 / fps)),
    )
    write_slice_meta(slice_dir, meta)

    expected: list[dict[str, Any]] = [
        {
            "type": "action_deviation",
            "topic": "action",
            "reference": "recorded",
            "max_l2": 0.05,
        },
        {"type": "latency", "topic": "/policy/action", "p95_ms": 80},
        {"type": "confidence_floor", "topic": "/policy/action", "min_confidence": 0.3},
    ]
    if action_dim and "action" in table.columns:
        actions = table.get_column("action").to_list()
        mins, maxs, max_delta = _bounds_and_smoothness(actions)
        expected.append({"type": "action_bounds", "topic": "action", "min": mins, "max": maxs})
        expected.append(
            {"type": "action_smoothness", "topic": "action", "max_delta": max_delta}
        )

    source: dict[str, Any] = {
        "kind": "lerobot",
        "dataset": dataset_name,
        "episode_index": episode_index,
        "frame_from": frame_from,
        "frame_to": end,
    }
    if meta.dataset_revision:
        source["revision"] = meta.dataset_revision

    target: dict[str, Any] = {"adapter": "python-policy", "entry": entry}
    if checkpoint:
        target["checkpoint"] = checkpoint
    if target_revision:
        target["revision"] = target_revision

    payload: dict[str, Any] = {
        "schema_version": 0,
        "id": scenario_id,
        "blocking": blocking,
        "source": source,
        "tags": ["lerobot", "open-loop"],
        "inputs": inputs,
        "target": target,
        "expected": expected,
    }
    scenario = Scenario.model_validate(payload)
    scenario_out = Path(scenario_out)
    scenario_out.mkdir(parents=True, exist_ok=True)
    dest = scenario_out / f"{scenario_id}.yaml"
    dest.write_text(
        yaml.safe_dump(
            scenario.model_dump(mode="json", exclude_none=True),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    # keep a pointer so callers can log the hash
    _ = content_hash(scenario)
    return dest, Slice.load(slice_dir).path


def resolve_dataset_root(dataset: str, *, revision: str | None = None) -> Path:
    local = Path(dataset)
    if (local / "meta" / "info.json").is_file():
        return local.resolve()
    return download_dataset(dataset, revision=revision)


def download_dataset(repo_id: str, *, revision: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download, list_repo_files

    dest = Path(".cache") / "hf" / repo_id.replace("/", "--")
    dest.mkdir(parents=True, exist_ok=True)
    files = list_repo_files(repo_id, repo_type="dataset", revision=revision)
    wanted = [
        name
        for name in files
        if name.startswith("meta/")
        or (name.startswith("data/") and name.endswith(".parquet"))
    ]
    for name in wanted:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=name,
            revision=revision,
            local_dir=str(dest),
        )
    return dest.resolve()


def lookup_revision(dataset: str, *, local_root: Path | None = None) -> str | None:
    if Path(dataset).exists():
        return None
    try:
        from huggingface_hub import dataset_info

        info = dataset_info(dataset, timeout=10)
        return info.sha
    except Exception:  # noqa: BLE001
        refs = local_root / ".cache" if local_root else None
        if refs is None:
            return None
        return None


def load_info(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"meta/info.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_episode_row(root: Path, episode_index: int) -> dict[str, Any]:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode parquet under {root / 'meta' / 'episodes'}")
    table = pl.concat([pl.read_parquet(path) for path in files], how="vertical_relaxed")
    match = table.filter(pl.col("episode_index") == episode_index)
    if match.height == 0:
        raise ValueError(f"episode_index {episode_index} not found")
    return match.row(0, named=True)


def load_episode_frames(
    root: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    frame_from: int,
    frame_to: int,
) -> pl.DataFrame:
    chunk = int(episode.get("data/chunk_index", 0))
    file_index = int(episode.get("data/file_index", 0))
    template = info.get("data_path") or "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    rel = template.format(chunk_index=chunk, file_index=file_index)
    path = root / rel
    if not path.is_file():
        raise FileNotFoundError(f"data parquet not found: {path}")
    table = pl.read_parquet(path)
    if "episode_index" in table.columns:
        table = table.filter(pl.col("episode_index") == episode["episode_index"])
    else:
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        table = table.slice(start, stop - start)
    if "frame_index" in table.columns:
        table = table.filter(
            (pl.col("frame_index") >= frame_from) & (pl.col("frame_index") < frame_to)
        ).sort("frame_index")
    else:
        table = table.slice(frame_from, frame_to - frame_from)
    return table


def video_refs(
    info: dict[str, Any],
    episode: dict[str, Any],
    frame_from: int,
    frame_to: int,
    fps: float,
) -> dict[str, VideoRef]:
    default_video = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    template = info.get("video_path") or default_video
    refs: dict[str, VideoRef] = {}
    features = info.get("features") or {}
    for key, spec in features.items():
        if not _is_video(spec):
            continue
        chunk = episode.get(f"videos/{key}/chunk_index")
        file_index = episode.get(f"videos/{key}/file_index")
        if chunk is None or file_index is None:
            continue
        ep_from = float(episode.get(f"videos/{key}/from_timestamp") or 0.0)
        from_ts = ep_from + frame_from / fps
        to_ts = ep_from + (frame_to - 1) / fps
        refs[key] = VideoRef(
            path=template.format(
                video_key=key, chunk_index=int(chunk), file_index=int(file_index)
            ),
            from_timestamp=from_ts,
            to_timestamp=to_ts,
            chunk_index=int(chunk),
            file_index=int(file_index),
        )
    return refs


def _is_video(spec: Any) -> bool:
    return isinstance(spec, dict) and spec.get("dtype") == "video"


def _bounds_and_smoothness(actions: list[Any]) -> tuple[list[float], list[float], float]:
    import numpy as np

    arr = np.asarray(actions, dtype=np.float64)
    mins = (arr.min(axis=0) - 1e-3).tolist()
    maxs = (arr.max(axis=0) + 1e-3).tolist()
    if len(arr) < 2:
        return mins, maxs, 1.0
    deltas = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    max_delta = float(np.max(deltas) * 1.5 + 1e-6)
    return mins, maxs, max_delta
