from __future__ import annotations

from pathlib import Path

import polars as pl

from robogate.diff import diff_evals


def _table(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_diff_lists_regressions_fixes_and_l2(tmp_path: Path) -> None:
    cols = {
        "scenario_hash": "h",
        "blocking": False,
        "run_id": "r",
        "target_version": "v",
        "target_revision": "",
        "perturbations": "",
        "threshold": 1.0,
    }
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    _table(
        [
            {
                "scenario_id": "s1",
                "type": "action_deviation",
                "status": "pass",
                "measured": 0.2,
                **cols,
            },
            {
                "scenario_id": "s2",
                "type": "action_deviation",
                "status": "fail",
                "measured": 2.0,
                **cols,
            },
        ]
    ).write_parquet(a)
    _table(
        [
            {
                "scenario_id": "s1",
                "type": "action_deviation",
                "status": "fail",
                "measured": 0.8,
                **cols,
            },
            {
                "scenario_id": "s2",
                "type": "action_deviation",
                "status": "pass",
                "measured": 0.4,
                **cols,
            },
        ]
    ).write_parquet(b)
    payload = diff_evals(a, b)
    assert payload["n_regressions"] == 1
    assert payload["regressions"][0]["scenario_id"] == "s1"
    assert payload["n_fixes"] == 1
    assert payload["fixes"][0]["scenario_id"] == "s2"
    deltas = {item["scenario_id"]: item["delta"] for item in payload["l2_deltas"]}
    assert abs(deltas["s1"] - 0.6) < 1e-9
    assert abs(deltas["s2"] - (-1.6)) < 1e-9
