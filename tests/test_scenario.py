from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from robogate.scenario import Scenario, content_hash, json_schema, load_scenario

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "scenarios" / "examples"


@pytest.mark.parametrize(
    "name",
    [
        "lerobot-open-loop.yaml",
        "mcap-open-loop.yaml",
        "mixed-closed-loop.yaml",
    ],
)
def test_example_scenarios_validate(name: str) -> None:
    scenario = load_scenario(EXAMPLES / name)
    assert scenario.schema_version == 0
    assert content_hash(scenario).startswith("sha256:")


def test_content_hash_is_stable() -> None:
    path = EXAMPLES / "lerobot-open-loop.yaml"
    first = content_hash(load_scenario(path))
    second = content_hash(load_scenario(path))
    assert first == second


def test_content_hash_changes_when_id_changes() -> None:
    scenario = load_scenario(EXAMPLES / "lerobot-open-loop.yaml")
    mutated = scenario.model_copy(update={"id": "lerobot-aloha-transfer-cube-ep1"})
    assert content_hash(scenario) != content_hash(mutated)


def test_unknown_field_is_rejected() -> None:
    scenario = load_scenario(EXAMPLES / "lerobot-open-loop.yaml")
    payload = scenario.model_dump(mode="json")
    payload["history"] = [{"model": "v37", "result": "fail"}]
    with pytest.raises(ValidationError):
        Scenario.model_validate(payload)


def test_mcap_requires_resolvable_time() -> None:
    payload = load_scenario(EXAMPLES / "mcap-open-loop.yaml").model_dump(mode="json")
    payload["source"] = {"kind": "mcap", "mcap": "bag.mcap"}
    with pytest.raises(ValidationError, match="from_ns/to_ns or from_iso/to_iso"):
        Scenario.model_validate(payload)


def test_mcap_iso_requires_timezone() -> None:
    payload = load_scenario(EXAMPLES / "mcap-open-loop.yaml").model_dump(mode="json")
    payload["source"] = {
        "kind": "mcap",
        "mcap": "bag.mcap",
        "from_iso": "2026-09-10T14:32:02.000",
        "to_iso": "2026-09-10T14:32:12.000",
    }
    with pytest.raises(ValidationError, match="timezone"):
        Scenario.model_validate(payload)


def test_lerobot_frame_range() -> None:
    payload = load_scenario(EXAMPLES / "lerobot-open-loop.yaml").model_dump(mode="json")
    payload["source"]["frame_from"] = 10
    payload["source"]["frame_to"] = 10
    with pytest.raises(ValidationError, match="frame_to"):
        Scenario.model_validate(payload)


def test_json_schema_exports_discriminator() -> None:
    schema = json_schema()
    assert schema["properties"]["schema_version"]["const"] == 0
    source = schema["$defs"]["LeRobotSource"]
    assert source["properties"]["kind"]["const"] == "lerobot"
