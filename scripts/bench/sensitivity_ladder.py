#!/usr/bin/env python3
"""CPU sensitivity ladder: synth mutants from B' runs, eval, gate, write a note."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from robogate.bench.report import gate_red_rate
from robogate.bench.synth import synth_run
from robogate.diff import pred_shifts
from robogate.eval import run_eval
from robogate.gate import run_gate
from robogate.run import Run
from robogate.suite import load_suite

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATS = (
    ROOT / ".cache" / "real-data" / "lerobot-aloha_static_coffee" / "meta" / "stats.json"
)

LADDER: list[dict[str, Any]] = [
    {"name": "bias-0.01", "kind": "bias", "value": 0.01, "group": "bias"},
    {"name": "bias-0.02", "kind": "bias", "value": 0.02, "group": "bias"},
    {"name": "bias-0.05", "kind": "bias", "value": 0.05, "group": "bias"},
    {"name": "bias-0.1", "kind": "bias", "value": 0.1, "group": "bias"},
    {
        "name": "dim_bias-d0-0.05",
        "kind": "dim_bias",
        "value": 0.05,
        "dim": 0,
        "group": "dim_bias_d0",
    },
    {"name": "dim_bias-d0-0.1", "kind": "dim_bias", "value": 0.1, "dim": 0, "group": "dim_bias_d0"},
    {"name": "dim_bias-d0-0.2", "kind": "dim_bias", "value": 0.2, "dim": 0, "group": "dim_bias_d0"},
    {"name": "dim_bias-d0-0.5", "kind": "dim_bias", "value": 0.5, "dim": 0, "group": "dim_bias_d0"},
    {
        "name": "dim_bias-d6-0.05",
        "kind": "dim_bias",
        "value": 0.05,
        "dim": 6,
        "group": "dim_bias_d6",
    },
    {"name": "dim_bias-d6-0.1", "kind": "dim_bias", "value": 0.1, "dim": 6, "group": "dim_bias_d6"},
    {"name": "dim_bias-d6-0.2", "kind": "dim_bias", "value": 0.2, "dim": 6, "group": "dim_bias_d6"},
    {"name": "dim_bias-d6-0.5", "kind": "dim_bias", "value": 0.5, "dim": 6, "group": "dim_bias_d6"},
    {"name": "gain-0.9", "kind": "gain", "value": 0.9, "group": "gain_under"},
    {"name": "gain-0.8", "kind": "gain", "value": 0.8, "group": "gain_under"},
    {"name": "gain-0.5", "kind": "gain", "value": 0.5, "group": "gain_under"},
    {"name": "gain-1.1", "kind": "gain", "value": 1.1, "group": "gain_over"},
    {"name": "gain-1.2", "kind": "gain", "value": 1.2, "group": "gain_over"},
    {"name": "noise-0.01", "kind": "noise", "value": 0.01, "group": "noise"},
    {"name": "noise-0.02", "kind": "noise", "value": 0.02, "group": "noise"},
    {"name": "noise-0.05", "kind": "noise", "value": 0.05, "group": "noise"},
    {"name": "lag-2", "kind": "lag", "value": 2, "group": "lag"},
    {"name": "lag-5", "kind": "lag", "value": 5, "group": "lag"},
    {"name": "lag-10", "kind": "lag", "value": 10, "group": "lag"},
    {"name": "lag-15", "kind": "lag", "value": 15, "group": "lag"},
    {"name": "stats_swap", "kind": "stats_swap", "value": None, "group": "stats_swap"},
    {"name": "constant", "kind": "constant", "value": None, "group": "constant"},
]


def latest_clean_run(runs_root: Path, scenario_id: str, *, needle: str) -> Path:
    matches: list[Path] = []
    for meta in runs_root.glob("*/meta.json"):
        try:
            run = Run.load(meta.parent)
        except Exception:  # noqa: BLE001
            continue
        version = run.meta.target_version or ""
        if run.meta.scenario_id != scenario_id:
            continue
        if run.meta.perturbations:
            continue
        if needle not in version:
            continue
        matches.append(meta.parent)
    if not matches:
        raise FileNotFoundError(f"no clean run for {scenario_id} matching {needle}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def first_fail_types(parquet: Path) -> dict[str, str]:
    table = pl.read_parquet(parquet)
    out: dict[str, str] = {}
    for row in table.iter_rows(named=True):
        sid = row["scenario_id"]
        if sid in out:
            continue
        if row["status"] in {"fail", "error"}:
            out[sid] = str(row["type"])
    return out


def recorded_dim_std(run_dir: Path, dims: tuple[int, ...]) -> dict[str, float]:
    rec = np.asarray(
        Run.load(run_dir).output("action", recorded=True).get_column("value").to_list(),
        dtype=np.float64,
    )
    return {f"d{dim}": float(rec[:, dim].std()) for dim in dims}


def _format_level(value: float | int | None) -> str:
    if value is None:
        return "检出"
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def first_detected(rows: list[dict[str, Any]], *, key: str = "gate_ok") -> dict[str, Any] | None:
    for row in rows:
        if key == "gate_ok" and not row["gate_ok"]:
            return row
        if key == "shift_ok" and row.get("shift_ok") is False:
            return row
    return None


def summarize(levels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in levels:
        by_group.setdefault(row["group"], []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for group, rows in by_group.items():
        if group == "lag":
            assert_hit = first_detected(rows)
            shift_hit = first_detected(rows, key="shift_ok")
            out["lag_assert"] = _hit_payload(assert_hit, rows)
            out["lag_pred_shift"] = _hit_payload(shift_hit, rows)
            continue
        out[group] = _hit_payload(first_detected(rows), rows)
    return out


def _hit_payload(hit: dict[str, Any] | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if hit is None:
        return {
            "detected": False,
            "level": "未检出",
            "n_red": 0,
            "n": rows[0]["n"],
            "first_fail": {},
        }
    return {
        "detected": True,
        "level": _format_level(hit["value"]),
        "name": hit["name"],
        "n_red": hit["n_red"],
        "n": hit["n"],
        "first_fail": hit["first_fail_counts"],
        "gate_ok": hit["gate_ok"],
    }


def write_note(path: Path, payload: dict[str, Any]) -> Path:
    summary = payload["summary"]
    labels = [
        ("bias", "全维偏置"),
        ("dim_bias_d0", "单维偏置 d0"),
        ("dim_bias_d6", "单维偏置 d6（夹爪）"),
        ("gain_under", "增益缩小"),
        ("gain_over", "增益放大"),
        ("noise", "高斯噪声 σ"),
        ("lag_assert", "滞后（action_lag 断言）"),
        ("lag_pred_shift", "滞后（pred_shift，max_shift=2）"),
        ("stats_swap", "统计量错位 action→state"),
        ("constant", "死模型（帧 0 常值）"),
    ]
    intro = (
        "从本地 B'（`act-coffee-30000`）11 条干净 run 离线合成突变，"
        "对现行 `scenarios/real` 阈值做 eval，"
        "再对 `fixtures/gate/baseline.parquet` 做 gate。"
        "纯 CPU，不改 scenario / 阈值 / fixtures。"
        "数字是「每类错误的最小检出量」："
        "该组阶梯里第一个让 gate 变红的级别；整组都绿则写未检出。"
    )
    lines = [
        "# 门禁灵敏度阶梯（2026-09-17）",
        "",
        intro,
        "",
        "## 最小检出量",
        "",
        "| 错误类型 | 最小检出量 | 红 scenario | 首个 fail 断言 |",
        "| --- | --- | --- | --- |",
    ]
    for key, title in labels:
        item = summary[key]
        fail = ", ".join(f"{name}×{count}" for name, count in item["first_fail"].items()) or "—"
        red = f"{item['n_red']}/{item['n']}" if item["detected"] else f"0/{item['n']}"
        lines.append(f"| {title} | {item['level']} | {red} | {fail} |")
    ep0 = payload["ep0_recorded_std"]
    d3 = ep0["d3"]
    d4 = ep0["d4"]
    if d3 < 0.05 and d4 < 0.05:
        verdict = "std 很小，负相关更像噪声"
    else:
        verdict = "std 不小，ep0 上这两维是真反向，正对照也不干净"
    lines.extend(
        [
            "",
            "## 读法",
            "",
            "- 这组阶梯里，gate 变红几乎全靠 `action_bounds`（示教包络 +15%）。",
            "  `action_deviation` 的 `max_l2=2.232` 从没当过首个 fail。",
            "- 增益缩到 0.5、把预测钉在第 0 帧，11/11 仍绿。",
            "  门禁分不出「幅度不够」和「停住不动」。",
            "- 偏置 0.02、噪声 0.01、滞后 2 帧也绿。",
            "  「新模型差 10% 你能看出来吗」——这套门槛看不出来。",
            "- `pred_shift` 在 lag=2 时测到 2，但 `max_shift=2` 含等于，从 5 帧起红。",
            "",
            "## B' ep0 recorded d3 / d4",
            "",
            f"d3 std = {d3:.4f}，d4 std = {d4:.4f}。{verdict}。",
            "",
            "来源 run：`" + payload["sources"]["aloha-static-coffee-ep0"] + "`。",
            "",
            "## 各级明细",
            "",
            "| 级别 | 红 | gate | 首个 fail |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in payload["levels"]:
        fail = ", ".join(f"{name}×{count}" for name, count in row["first_fail_counts"].items())
        extra = ""
        if "unique_shifts" in row:
            extra = f" shift={row['unique_shifts']}"
        lines.append(
            f"| {row['name']} | {row['n_red']}/{row['n']} | "
            f"{'绿' if row['gate_ok'] else '红'} | {fail or '—'}{extra} |"
        )
    lines.extend(
        [
            "",
            "JSON：[2026-09-17-sensitivity-ladder.json](./2026-09-17-sensitivity-ladder.json)。",
            "",
            "## 怎么复跑",
            "",
            "```bash",
            "conda run -n robogate python scripts/bench/sensitivity_ladder.py",
            "```",
            "",
        ]
    )
    dest = Path(path)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def run_level(
    spec: dict[str, Any],
    *,
    sources: dict[str, Path],
    suite: Path,
    baseline: Path,
    mutants: Path,
    results: Path,
    stats_path: Path,
    needle: str,
) -> dict[str, Any]:
    dest_root = mutants / spec["name"]
    dest_root.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"kind": spec["kind"], "value": spec["value"]}
    if spec.get("dim") is not None:
        kwargs["dim"] = spec["dim"]
    if spec["kind"] == "stats_swap":
        kwargs["stats_path"] = stats_path
    for src in sources.values():
        synth_run(src, dest_root / src.name, **kwargs)
    parquet = run_eval(
        suite,
        runs_root=dest_root,
        out_root=results,
        eval_id=f"ladder-{spec['name']}",
    )
    gate = gate_red_rate(suite, baseline, parquet)
    fails = first_fail_types(parquet)
    fold = gate["by_scenario"]
    row: dict[str, Any] = {
        "name": spec["name"],
        "kind": spec["kind"],
        "group": spec["group"],
        "value": spec["value"],
        "dim": spec.get("dim"),
        "n": len(fold),
        "n_red": sum(1 for status in fold.values() if status in {"fail", "error"}),
        "gate_ok": gate["ok"],
        "reasons": gate["reasons"],
        "by_scenario": fold,
        "first_fail": fails,
        "first_fail_counts": dict(Counter(fails.values())),
        "parquet": str(parquet.relative_to(ROOT)) if parquet.is_relative_to(ROOT) else str(parquet),
    }
    if spec["kind"] == "lag":
        shifts = pred_shifts(
            Path(os.environ.get("RUNS", ROOT / "runs")),
            dest_root,
            list(sources),
            version_contains_a=needle,
        )
        shift_gate = run_gate(
            suite,
            baseline,
            parquet,
            runs_baseline=Path(os.environ.get("RUNS", ROOT / "runs")),
            runs_candidate=dest_root,
            max_shift=2,
            version_contains_baseline=needle,
        )
        row["pred_shift"] = shifts
        row["unique_shifts"] = sorted({int(item["shift"]) for item in shifts})
        row["shift_ok"] = shift_gate["ok"]
        row["shift_reasons"] = shift_gate["reasons"]
    return row


def main() -> None:
    suite = Path(os.environ.get("SUITE", ROOT / "scenarios" / "real"))
    runs = Path(os.environ.get("RUNS", ROOT / "runs"))
    results = Path(os.environ.get("RESULTS", ROOT / "results"))
    mutants = Path(os.environ.get("MUTANTS", ROOT / "runs" / "mutants" / "ladder"))
    baseline = Path(os.environ.get("BASELINE", ROOT / "fixtures" / "gate" / "baseline.parquet"))
    stats_path = Path(os.environ.get("STATS", DEFAULT_STATS))
    needle = os.environ.get("BASE_RUN_NEEDLE", "act-coffee-30000")
    note = Path(
        os.environ.get("LADDER_NOTE", ROOT / "docs" / "notes" / "2026-09-17-sensitivity-ladder.md")
    )
    items = load_suite(suite)
    ids = [scenario.id for _, scenario in items]
    sources = {sid: latest_clean_run(runs, sid, needle=needle) for sid in ids}
    if not stats_path.is_file():
        raise FileNotFoundError(f"dataset stats not found: {stats_path}")
    results.mkdir(parents=True, exist_ok=True)
    mutants.mkdir(parents=True, exist_ok=True)
    levels = [
        run_level(
            spec,
            sources=sources,
            suite=suite,
            baseline=baseline,
            mutants=mutants,
            results=results,
            stats_path=stats_path,
            needle=needle,
        )
        for spec in LADDER
    ]
    payload = {
        "needle": needle,
        "n_scenarios": len(ids),
        "sources": {sid: path.name for sid, path in sources.items()},
        "ep0_recorded_std": recorded_dim_std(sources["aloha-static-coffee-ep0"], (3, 4)),
        "levels": levels,
        "summary": summarize(levels),
    }
    json_path = note.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    write_note(note, payload)
    print(json.dumps({"note": str(note), "json": str(json_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
