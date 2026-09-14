"""robogate CLI. validate / schema / assert / extract / replay / eval / diff / gate."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from pydantic import ValidationError

from robogate import __version__
from robogate.asserts import overall_ok, run_assertions
from robogate.run import Run
from robogate.scenario import content_hash, json_schema, load_scenario

app = typer.Typer(
    name="robogate",
    no_args_is_help=True,
    add_completion=False,
    help="Turn a field failure into a permanent regression test that blocks the release.",
)

extract_app = typer.Typer(
    name="extract",
    no_args_is_help=True,
    help="Slice a recording into a scenario.",
)
app.add_typer(extract_app, name="extract")


@app.callback()
def _root(version: bool = typer.Option(False, "--version", help="Show version and exit.")) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit(0)


@app.command()
def validate(scenario: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Validate a scenario.yaml against schema v0."""
    try:
        loaded = load_scenario(scenario)
    except (ValidationError, ValueError, yaml_error()) as exc:
        typer.echo(f"invalid: {exc}", err=True)
        raise typer.Exit(1) from exc
    digest = content_hash(loaded)
    typer.echo(f"ok {loaded.id} {digest}")


def yaml_error() -> type[Exception]:
    import yaml

    return yaml.YAMLError


@app.command("schema")
def schema_cmd() -> None:
    """Print scenario JSON Schema v0 to stdout."""
    typer.echo(json.dumps(json_schema(), indent=2, ensure_ascii=False))


@app.command("assert")
def assert_cmd(
    scenario: Path = typer.Argument(..., exists=True, readable=True),
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of a table."),
) -> None:
    """Judge a run against a scenario. Exit 0 if no fail; 1 if any fail."""
    try:
        loaded = load_scenario(scenario)
        run = Run.load(run_dir)
    except (ValidationError, ValueError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if run.meta.scenario_id != loaded.id:
        typer.echo(
            f"error: run scenario_id {run.meta.scenario_id} != scenario {loaded.id}",
            err=True,
        )
        raise typer.Exit(1)

    results = run_assertions(loaded, run)
    ok = overall_ok(results)
    payload = {
        "ok": ok,
        "scenario_id": loaded.id,
        "scenario_hash": content_hash(loaded),
        "run": str(run.path),
        "run_mode": run.mode.value,
        "results": [item.to_dict() for item in results],
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        typer.echo(f"scenario {loaded.id}  run={run.path.name}  mode={run.mode.value}")
        if run.meta.scenario_hash and run.meta.scenario_hash != payload["scenario_hash"]:
            typer.echo(
                f"warning: run scenario_hash {run.meta.scenario_hash} "
                f"!= current {payload['scenario_hash']}",
                err=True,
            )
        for item in results:
            measured = "" if item.measured is None else f" measured={item.measured}"
            threshold = "" if item.threshold is None else f" threshold={item.threshold}"
            typer.echo(
                f"  {item.status.value:7} {item.type}{measured}{threshold}  {item.message}"
            )
        typer.echo("PASS" if ok else "FAIL")
    raise typer.Exit(0 if ok else 1)


@extract_app.command("lerobot")
def extract_lerobot_cmd(
    dataset: str = typer.Argument(..., help="HF repo id or local dataset root."),
    episode: int = typer.Option(0, "--episode", min=0),
    frame_from: int = typer.Option(0, "--from", min=0),
    frame_to: int | None = typer.Option(None, "--to"),
    scenario_id: str | None = typer.Option(None, "--id"),
    out: Path = typer.Option(Path("scenarios/real"), "--out"),
    slice_dir: Path = typer.Option(Path("slices"), "--slice-dir"),
    revision: str | None = typer.Option(None, "--revision"),
    repo_id: str | None = typer.Option(None, "--repo-id"),
    checkpoint: str | None = typer.Option(None, "--checkpoint"),
    target_revision: str | None = typer.Option(None, "--target-revision"),
    blocking: bool = typer.Option(True, "--blocking/--no-blocking"),
    entry: str = typer.Option("lerobot", "--entry"),
    episodes: str | None = typer.Option(None, "--episodes", help="Range like 0-9 or 0,2,5."),
) -> None:
    """Slice a LeRobot v3 dataset into scenario.yaml + slices/<id>/."""
    from robogate.extract import extract_lerobot
    from robogate.slice import Slice
    from robogate.suite import parse_episodes

    try:
        indexes = parse_episodes(episodes, episode)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    for index in indexes:
        sid = (
            scenario_id
            if scenario_id and len(indexes) == 1
            else _default_scenario_id(repo_id or dataset, index)
        )
        try:
            scenario_path, slice_path = extract_lerobot(
                dataset,
                episode_index=index,
                frame_from=frame_from,
                frame_to=frame_to,
                scenario_id=sid,
                scenario_out=out,
                slice_root=slice_dir,
                revision=revision,
                repo_id=repo_id,
                checkpoint=checkpoint,
                target_revision=target_revision,
                blocking=blocking,
                entry=entry,
            )
        except (ValidationError, ValueError, OSError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1) from exc
        loaded = load_scenario(scenario_path)
        meta = Slice.load(slice_path).meta
        typer.echo(
            f"ok {loaded.id} {content_hash(loaded)} frames={meta.n_frames} slice={slice_path}"
        )


@app.command()
def replay(
    scenario: Path = typer.Argument(..., exists=True, readable=True),
    slice_dir: Path | None = typer.Option(None, "--slice"),
    device: str = typer.Option("cpu", "--device"),
    out: Path = typer.Option(Path("runs"), "--out"),
    checkpoint: str | None = typer.Option(None, "--checkpoint"),
    revision: str | None = typer.Option(None, "--revision"),
    adapter: str = typer.Option("auto", "--adapter", help="auto, lerobot, or mock."),
    noise: float = typer.Option(0.0, "--noise", help="Mock adapter Gaussian noise."),
    perturb: list[str] = typer.Option([], "--perturb", help="Repeatable key=value perturbation."),
) -> None:
    """Replay slice inputs against an adapter and write runs/<id>/."""
    from robogate.eval import _override_target
    from robogate.replay import adapter_entry, build_adapter, run_replay
    from robogate.slice import Slice

    try:
        loaded = _override_target(
            load_scenario(scenario),
            checkpoint=checkpoint,
            revision=revision,
        )
        slice_path = slice_dir or Path("slices") / loaded.id
        slice_obj = Slice.load(slice_path)
        entry = adapter_entry(adapter, loaded.target.entry)
        built = build_adapter(entry, noise=noise, perturbations=perturb or None)
        dest = run_replay(
            loaded,
            slice_obj,
            built,
            out_root=out,
            device=device,
            perturbations=perturb or None,
        )
    except (ValidationError, ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"ok {loaded.id} run={dest}")
    typer.echo("ROBOGATE_REPLAY_DONE")


def _default_scenario_id(dataset: str, episode: int) -> str:
    stem = Path(str(dataset).rstrip("/")).name.replace("_", "-").replace(".", "-").lower()
    stem = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in stem).strip("-")
    return f"{stem}-ep{episode}"


@app.command()
def eval(  # noqa: A001
    suite: Path = typer.Argument(..., exists=True),
    runs: Path = typer.Option(Path("runs"), "--runs"),
    out: Path = typer.Option(Path("results"), "--out"),
    replay: bool = typer.Option(False, "--replay"),
    slice_dir: Path = typer.Option(Path("slices"), "--slice-dir"),
    device: str = typer.Option("cpu", "--device"),
    adapter: str = typer.Option("auto", "--adapter"),
    checkpoint: str | None = typer.Option(None, "--checkpoint"),
    revision: str | None = typer.Option(None, "--revision"),
    perturb: list[str] = typer.Option([], "--perturb"),
    noise: float = typer.Option(0.0, "--noise"),
    eval_id: str | None = typer.Option(None, "--eval-id"),
) -> None:
    """Run a suite, write results/<eval_id>.parquet + .json."""
    from robogate.eval import run_eval, summarize_eval

    try:
        dest = run_eval(
            suite,
            runs_root=runs,
            out_root=out,
            replay=replay,
            slice_root=slice_dir,
            device=device,
            adapter=adapter,
            checkpoint=checkpoint,
            revision=revision,
            perturbations=perturb or None,
            noise=noise,
            eval_id=eval_id,
        )
    except (ValidationError, ValueError, OSError, RuntimeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    import polars as pl

    summary = summarize_eval(pl.read_parquet(dest))
    typer.echo(f"ok eval={dest} overall={summary['overall']} n={summary['n_scenarios']}")
    typer.echo("ROBOGATE_EVAL_DONE")


@app.command()
def diff(
    a: Path = typer.Argument(..., exists=True, readable=True),
    b: Path = typer.Argument(..., exists=True, readable=True),
    fail_on_regression: bool = typer.Option(False, "--fail-on-regression"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Compare two eval parquets scene-by-scene."""
    from robogate.diff import diff_evals

    try:
        payload = diff_evals(a, b)
    except (ValidationError, ValueError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        typer.echo(f"regressions={payload['n_regressions']} fixes={payload['n_fixes']}")
        for item in payload["regressions"]:
            typer.echo(f"  regress {item['scenario_id']} {item['from']}→{item['to']}")
        for item in payload["fixes"]:
            typer.echo(f"  fix {item['scenario_id']} {item['from']}→{item['to']}")
        for item in payload["l2_deltas"]:
            typer.echo(
                f"  l2 {item['scenario_id']} {item['measured_a']:.6f}→{item['measured_b']:.6f} "
                f"delta={item['delta']:.6f}"
            )
    if fail_on_regression and payload["n_regressions"]:
        raise typer.Exit(1)


@app.command()
def gate(
    suite: Path = typer.Argument(..., exists=True),
    baseline: Path = typer.Option(..., "--baseline", exists=True, readable=True),
    candidate: Path = typer.Option(..., "--candidate", exists=True, readable=True),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """CI gate: baseline-pass must not fail; blocking must pass; hash must match."""
    from robogate.gate import run_gate

    try:
        payload = run_gate(suite, baseline, candidate)
    except (ValidationError, ValueError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        typer.echo("PASS" if payload["ok"] else "FAIL")
        for reason in payload["reasons"]:
            typer.echo(f"  {reason}")
    raise typer.Exit(0 if payload["ok"] else 1)


def main() -> None:
    app()
