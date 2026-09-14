#!/usr/bin/env python3
"""Minimal ACT training loop. Avoids lerobot-train import chain (Groot / wandb)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


def _ensure_act_type(dest: Path) -> None:
    path = dest / "config.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type"):
        return
    data["type"] = "act"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _save_steps() -> list[int]:
    raw = os.environ.get("ACT_SAVE_STEPS", "2000,10000,30000")
    steps = sorted({int(item) for item in raw.split(",") if item.strip()})
    if not steps:
        raise ValueError("ACT_SAVE_STEPS is empty")
    return steps


def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in items[0]:
        values = [item[key] for item in items]
        first = values[0]
        if isinstance(first, torch.Tensor):
            out[key] = torch.stack(values)
        else:
            out[key] = values
    return out


def _to_device(value: Any, device: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _forward_loss(policy: Any, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    out = policy(batch)
    if isinstance(out, tuple) and out:
        loss = out[0]
        extras = out[1] if len(out) > 1 and isinstance(out[1], dict) else {}
        return loss, extras
    if isinstance(out, dict) and "loss" in out:
        return out["loss"], out
    if isinstance(out, torch.Tensor):
        return out, {}
    raise TypeError(f"unexpected policy output type {type(out)}")


def _save_processors(preprocessor: Any, postprocessor: Any, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for pipe in (preprocessor, postprocessor):
        if pipe is not None and hasattr(pipe, "save_pretrained"):
            pipe.save_pretrained(dest)


def main() -> None:
    from robogate.replay.lerobot_policy import (
        _dataset_root,
        _ensure_local_act_type,
        _isolate_lerobot_imports,
        _load_processors,
        _patch_video_decoder,
        _video_backend,
    )

    save_at = _save_steps()
    max_steps = max(save_at)
    out_root = Path(os.environ.get("ACT_OUT", "/root/autodl-tmp/ckpts"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id = os.environ.get("ACT_REPO", "lerobot/aloha_static_coffee")
    base_ckpt = os.environ.get("ACT_BASE_CKPT", "gozdebaydogmus/act-aloha-static-coffee-test")
    base_rev = os.environ.get("ACT_BASE_REV", "646846823c8473712f689f59bdd03f798a688f6a")
    batch_size = int(os.environ.get("ACT_BATCH", "2"))
    workers = int(os.environ.get("ACT_WORKERS", "2"))
    lr = float(os.environ.get("ACT_LR", "1e-5"))
    train_from = int(os.environ.get("ACT_EP_FROM", "10"))
    train_to = int(os.environ.get("ACT_EP_TO", "49"))
    episodes = list(range(train_from, train_to + 1))

    print(f"[train] device={device} steps={max_steps} save={save_at} eps={train_from}-{train_to}")
    _isolate_lerobot_imports()
    if _video_backend() != "torchcodec":
        _patch_video_decoder()

    from lerobot.policies.act.modeling_act import ACTPolicy

    resume = os.environ.get("ACT_RESUME")
    start_step = int(os.environ.get("ACT_START_STEP", "0"))
    if resume:
        print(f"[train] resume {resume} from step {start_step}")
        _ensure_local_act_type(resume)
        policy = ACTPolicy.from_pretrained(resume)
        config = policy.config
    else:
        print(f"[train] loading config from {base_ckpt}@{base_rev}")
        template = ACTPolicy.from_pretrained(base_ckpt, revision=base_rev)
        config = template.config
        policy = ACTPolicy(config)
        del template
    policy.to(device)
    policy.train()

    preprocessor, postprocessor = _load_processors(base_ckpt, revision=base_rev, device=device)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    fps = 50.0
    chunk = int(getattr(config, "chunk_size", 100) or 100)
    delta = {"action": [i / fps for i in range(chunk)]}
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "episodes": episodes,
        "delta_timestamps": delta,
        "video_backend": _video_backend(),
    }
    root = _dataset_root(repo_id)
    if root is not None:
        kwargs["root"] = root
        kwargs["download_videos"] = False
    try:
        dataset = LeRobotDataset(**kwargs)
    except TypeError:
        kwargs.pop("video_backend", None)
        kwargs.pop("download_videos", None)
        dataset = LeRobotDataset(**kwargs)
    print(f"[train] dataset n={len(dataset)} batch={batch_size} workers={workers}")

    # Sequential reads keep the mp4 decoder hot; full shuffle seeks every step.
    shuffle = os.environ.get("ACT_SHUFFLE", "0") == "1"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=_collate,
        pin_memory=device == "cuda",
        drop_last=True,
        persistent_workers=False,
        prefetch_factor=2 if workers > 0 else None,
    )
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    iterator = iter(loader)
    saved: list[int] = []
    t0 = time.time()
    for step in range(start_step + 1, max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        except Exception as exc:  # noqa: BLE001 — skip a corrupt video frame
            print(f"[train] skip step {step}: {type(exc).__name__}: {exc}", flush=True)
            iterator = iter(loader)
            continue
        if preprocessor is not None:
            batch = preprocessor(batch)
        batch = _to_device(batch, device)
        loss, extras = _forward_loss(policy, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 20 == 0 or step in save_at or step <= 5:
            extra = " ".join(f"{k}={v:.4f}" for k, v in extras.items() if hasattr(v, "item"))
            print(
                f"[train] step {step}/{max_steps} loss={float(loss.detach().cpu()):.6f} "
                f"elapsed={time.time() - t0:.0f}s {extra}",
                flush=True,
            )
        if step in save_at:
            dest = out_root / f"act-coffee-{step}"
            dest.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(dest)
            _ensure_act_type(dest)
            _save_processors(preprocessor, postprocessor, dest)
            (dest / "train_state.json").write_text(
                json.dumps({"step": step, "loss": float(loss.detach().cpu())}, indent=2) + "\n",
                encoding="utf-8",
            )
            saved.append(step)
            print(f"[train] saved {dest}", flush=True)
    print(f"[train] finished saved={saved} elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
