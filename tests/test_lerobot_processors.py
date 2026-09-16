from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robogate.replay.lerobot_policy import (
    align_action_stats,
    checkpoint_has_processors,
    dataset_stats,
    resolve_processors,
)


def test_checkpoint_has_processors_local(tmp_path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    assert checkpoint_has_processors(str(ckpt)) is False
    (ckpt / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    assert checkpoint_has_processors(str(ckpt)) is True


def test_dataset_stats_from_meta() -> None:
    dataset = SimpleNamespace(
        meta=SimpleNamespace(stats={"action": {"mean": [0.0], "std": [1.0]}})
    )
    assert dataset_stats(dataset)["action"]["mean"] == [0.0]
    assert dataset_stats(object()) == {}


def test_resolve_processors_falls_back_to_dataset_stats(tmp_path, monkeypatch) -> None:
    ckpt = tmp_path / "old-act"
    ckpt.mkdir()
    (ckpt / "config.json").write_text('{"type": "act"}\n', encoding="utf-8")
    called: dict[str, bool] = {}

    def fake_from_stats(dataset: object, policy: object, device: str) -> tuple[str, str]:
        called["fallback"] = True
        assert device == "cpu"
        assert dataset is not None
        return "pre-from-stats", "post-from-stats"

    monkeypatch.setattr(
        "robogate.replay.lerobot_policy.processors_from_dataset_stats",
        fake_from_stats,
    )
    pre, post, source = resolve_processors(
        str(ckpt),
        revision=None,
        device="cpu",
        dataset=SimpleNamespace(meta=SimpleNamespace(stats={"action": {}})),
        policy=SimpleNamespace(config=SimpleNamespace()),
    )
    assert source == "dataset_stats"
    assert called["fallback"] is True
    assert pre == "pre-from-stats"
    assert post == "post-from-stats"


def test_resolve_processors_uses_checkpoint_json(tmp_path, monkeypatch) -> None:
    ckpt = tmp_path / "new-act"
    ckpt.mkdir()
    (ckpt / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    called: dict[str, bool] = {}

    def fake_load(checkpoint: str, *, revision: str | None, device: str) -> tuple[str, str]:
        called["checkpoint"] = True
        assert checkpoint == str(ckpt)
        return "pre-ckpt", "post-ckpt"

    monkeypatch.setattr("robogate.replay.lerobot_policy._load_processors", fake_load)
    pre, post, source = resolve_processors(
        str(ckpt),
        revision=None,
        device="cpu",
        dataset=object(),
        policy=object(),
    )
    assert source == "checkpoint"
    assert called["checkpoint"] is True
    assert pre == "pre-ckpt"
    assert post == "post-ckpt"


def test_align_action_stats_replaces_mean_std() -> None:
    class Step:
        def __init__(self) -> None:
            self.stats = {"action": {"mean": np.array([0.0]), "std": np.array([1.0])}}
            self.device = None

        def to(self, device: str) -> None:
            self.device = device

    other = SimpleNamespace(stats={"state": {"mean": np.array([9.0])}})
    pipe = SimpleNamespace(steps=[Step(), other])
    align_action_stats(pipe, {"mean": [1.5, 2.5], "std": [0.2, 0.4]}, "cpu")
    np.testing.assert_allclose(pipe.steps[0].stats["action"]["mean"], [1.5, 2.5])
    np.testing.assert_allclose(pipe.steps[0].stats["action"]["std"], [0.2, 0.4])
    assert pipe.steps[0].device == "cpu"
    np.testing.assert_allclose(other.stats["state"]["mean"], [9.0])


def test_resolve_processors_aligns_checkpoint_action_stats(tmp_path, monkeypatch) -> None:
    ckpt = tmp_path / "new-act"
    ckpt.mkdir()
    (ckpt / "policy_preprocessor.json").write_text("{}", encoding="utf-8")
    aligned: list[tuple[object, object, str]] = []

    def fake_load(checkpoint: str, *, revision: str | None, device: str) -> tuple[str, str]:
        del checkpoint, revision, device
        return "pre-ckpt", "post-ckpt"

    def fake_align(pipe: object, stats: object, device: str) -> None:
        aligned.append((pipe, stats, device))

    monkeypatch.setattr("robogate.replay.lerobot_policy._load_processors", fake_load)
    monkeypatch.setattr("robogate.replay.lerobot_policy.align_action_stats", fake_align)
    dataset = SimpleNamespace(
        meta=SimpleNamespace(stats={"action": {"mean": [0.1], "std": [0.2]}})
    )
    pre, post, source = resolve_processors(
        str(ckpt),
        revision=None,
        device="cuda",
        dataset=dataset,
        policy=object(),
        action_stats="dataset",
    )
    assert source == "checkpoint+dataset_action"
    assert pre == "pre-ckpt"
    assert post == "post-ckpt"
    assert aligned == [
        ("pre-ckpt", {"mean": [0.1], "std": [0.2]}, "cuda"),
        ("post-ckpt", {"mean": [0.1], "std": [0.2]}, "cpu"),
    ]
