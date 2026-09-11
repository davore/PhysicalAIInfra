"""robogate CLI. M0 implements validate / schema / assert."""

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


def _not_implemented() -> None:
    typer.echo("not implemented in M0", err=True)
    raise typer.Exit(2)


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


@app.command()
def extract(
    source: Path | None = typer.Argument(None),
) -> None:
    """Slice a recording into a scenario (M1+)."""
    _not_implemented()


@app.command()
def replay(
    scenario: Path | None = typer.Argument(None),
) -> None:
    """Replay inputs against an adapter (M1+)."""
    _not_implemented()


@app.command()
def eval(  # noqa: A001
    suite: Path | None = typer.Argument(None),
) -> None:
    """Run a suite and write parquet + HTML (M1+)."""
    _not_implemented()


@app.command()
def diff(
    run_a: Path | None = typer.Argument(None),
    run_b: Path | None = typer.Argument(None),
) -> None:
    """Compare two runs scene-by-scene (M1+)."""
    _not_implemented()


@app.command()
def gate(
    suite: Path | None = typer.Argument(None),
) -> None:
    """CI gate: baseline-pass must not fail; blocking must pass (M2)."""
    _not_implemented()


def main() -> None:
    app()
