from __future__ import annotations

from types import SimpleNamespace

from robogate.replay.lerobot_policy import (
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
