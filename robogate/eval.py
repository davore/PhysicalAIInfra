"""Run a suite: optional replay, then assert, write parquet + json."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from robogate.asserts import overall_ok, run_assertions
from robogate.run import Run
from robogate.scenario import Scenario, content_hash
from robogate.suite import load_suite


def run_eval(
    suite: Path,
    *,
    runs_root: Path,
    out_root: Path,
    replay: bool = False,
    slice_root: Path | None = None,
    device: str = "cpu",
    adapter: str = "auto",
    checkpoint: str | None = None,
    revision: str | None = None,
    perturbations: list[str] | None = None,
    noise: float = 0.0,
    eval_id: str | None = None,
    version_contains: str | None = None,
    evidence: str = "fail",
) -> Path:
    items = load_suite(suite)
    rows: list[dict[str, Any]] = []
    n = len(items)
    passed = 0
    for i, (path, scenario) in enumerate(items, start=1):
        digest = content_hash(scenario)
        scenario = _override_target(scenario, checkpoint=checkpoint, revision=revision)
        try:
            run = _resolve_run(
                scenario,
                runs_root=runs_root,
                replay=replay,
                slice_root=slice_root or Path("slices"),
                device=device,
                adapter=adapter,
                perturbations=perturbations,
                noise=noise,
                scenario_hash=digest,
                version_contains=version_contains,
                evidence=evidence,
            )
            if run.meta.scenario_id != scenario.id:
                raise ValueError(
                    f"run scenario_id {run.meta.scenario_id} != scenario {scenario.id}"
                )
            results = run_assertions(scenario, run)
            ok = overall_ok(results)
            if ok:
                passed += 1
            for item in results:
                rows.append(
                    _row(
                        scenario,
                        digest,
                        run,
                        item.type,
                        item.status.value,
                        item.measured,
                        item.threshold,
                        perturbations,
                    )
                )
        except Exception as exc:  # noqa: BLE001 — suite continues; row records the error
            rows.append(
                _row(
                    scenario,
                    digest,
                    None,
                    "eval",
                    "error",
                    None,
                    None,
                    perturbations,
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
        pct = 100.0 * passed / i
        print(f"[eval] scenario {i}/{n} pass={pct:.0f}%", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    name = eval_id or _eval_name(Path(suite).name, checkpoint, perturbations)
    dest = out_root / f"{name}.parquet"
    table = pl.DataFrame(rows).cast(_empty_table().schema, strict=False) if rows else _empty_table()
    table.write_parquet(dest)
    summary = summarize_eval(table)
    summary["eval_id"] = name
    summary["suite"] = str(suite)
    (out_root / f"{name}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dest


def summarize_eval(table: pl.DataFrame) -> dict[str, Any]:
    statuses = table.get_column("status").to_list() if table.height else []
    counts = {
        "pass": statuses.count("pass"),
        "fail": statuses.count("fail"),
        "error": statuses.count("error"),
        "skipped": statuses.count("skipped"),
    }
    overall = fold_eval(table)
    n_scen = len(overall)
    n_ok = sum(1 for status in overall.values() if status == "pass")
    return {
        "n_scenarios": n_scen,
        "n_rows": table.height,
        "n_pass": n_ok,
        "overall": n_ok == n_scen and n_scen > 0,
        "counts": counts,
        "by_scenario": overall,
    }


def fold_eval(table: pl.DataFrame) -> dict[str, str]:
    """Worst status per scenario: error > fail > pass (skipped ignored)."""
    out: dict[str, str] = {}
    if table.height == 0:
        return out
    for row in table.iter_rows(named=True):
        sid = row["scenario_id"]
        status = row["status"]
        if status == "skipped":
            out.setdefault(sid, "pass")
            continue
        current = out.get(sid, "pass")
        out[sid] = _worse(current, status)
    return out


def _worse(left: str, right: str) -> str:
    rank = {"pass": 0, "fail": 1, "error": 2}
    return left if rank.get(left, 0) >= rank.get(right, 0) else right


def find_run(
    runs_root: Path,
    scenario_id: str,
    *,
    version_contains: str | None = None,
    unperturbed: bool = False,
) -> Path:
    matches: list[Path] = []
    root = Path(runs_root)
    if not root.is_dir():
        raise FileNotFoundError(f"runs dir not found: {root}")
    for meta_path in root.glob("*/meta.json"):
        try:
            run = Run.load(meta_path.parent)
        except Exception:  # noqa: BLE001
            continue
        if run.meta.scenario_id != scenario_id:
            continue
        version = run.meta.target_version or ""
        if version_contains and version_contains not in version:
            continue
        matches.append(meta_path.parent)
    if not matches:
        extra = f" matching {version_contains}" if version_contains else ""
        raise FileNotFoundError(f"no run for scenario {scenario_id}{extra} under {root}")
    if unperturbed:
        clean = [path for path in matches if not Run.load(path).meta.perturbations]
        if clean:
            matches = clean
    return max(matches, key=lambda path: path.stat().st_mtime)


def _resolve_run(
    scenario: Scenario,
    *,
    runs_root: Path,
    replay: bool,
    slice_root: Path,
    device: str,
    adapter: str,
    perturbations: list[str] | None,
    noise: float,
    scenario_hash: str | None = None,
    version_contains: str | None = None,
    evidence: str = "fail",
) -> Run:
    if replay:
        from robogate.replay import adapter_entry, build_adapter, run_replay
        from robogate.slice import Slice

        slice_obj = Slice.load(Path(slice_root) / scenario.id)
        entry = adapter_entry(adapter, scenario.target.entry)
        built = build_adapter(entry, noise=noise, perturbations=perturbations)
        dest = run_replay(
            scenario,
            slice_obj,
            built,
            out_root=runs_root,
            device=device,
            perturbations=perturbations,
            scenario_hash=scenario_hash,
            evidence=evidence,
        )
        return Run.load(dest)
    return Run.load(
        find_run(
            runs_root,
            scenario.id,
            version_contains=version_contains,
            unperturbed=True,
        )
    )


def _override_target(
    scenario: Scenario,
    *,
    checkpoint: str | None,
    revision: str | None,
) -> Scenario:
    if not checkpoint and not revision:
        return scenario
    update: dict[str, Any] = {}
    if checkpoint:
        update["checkpoint"] = checkpoint
        if revision is None and Path(checkpoint).exists():
            update["revision"] = None
    if revision is not None:
        update["revision"] = revision or None
    return scenario.model_copy(update={"target": scenario.target.model_copy(update=update)})


def _row(
    scenario: Scenario,
    digest: str,
    run: Run | None,
    type_name: str,
    status: str,
    measured: float | None,
    threshold: float | None,
    perturbations: list[str] | None,
    message: str = "",
) -> dict[str, Any]:
    del message
    return {
        "scenario_id": scenario.id,
        "scenario_hash": digest,
        "blocking": scenario.blocking,
        "run_id": run.path.name if run is not None else "",
        "target_version": (
            (run.meta.target_version if run is not None else None)
            or scenario.target.checkpoint
            or ""
        ),
        "target_revision": (
            (run.meta.target_revision if run is not None else None)
            or scenario.target.revision
            or ""
        ),
        "perturbations": ",".join(perturbations or []),
        "type": type_name,
        "status": status,
        "measured": measured,
        "threshold": threshold,
    }


def _empty_table() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "scenario_id": pl.Utf8,
            "scenario_hash": pl.Utf8,
            "blocking": pl.Boolean,
            "run_id": pl.Utf8,
            "target_version": pl.Utf8,
            "target_revision": pl.Utf8,
            "perturbations": pl.Utf8,
            "type": pl.Utf8,
            "status": pl.Utf8,
            "measured": pl.Float64,
            "threshold": pl.Float64,
        }
    )


def _eval_name(
    suite_name: str,
    checkpoint: str | None,
    perturbations: list[str] | None,
) -> str:
    ckpt = Path(checkpoint).name.replace("/", "-")[:32] if checkpoint else "eval"
    extra = ""
    if perturbations:
        extra = "__" + "-".join(item.replace("=", "") for item in perturbations)[:24]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = suite_name.replace("/", "-")
    return f"{stem}__{ckpt}{extra}__{stamp}"
