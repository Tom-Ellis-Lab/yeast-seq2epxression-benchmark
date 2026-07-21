"""``ybench modal`` — run the benchmark on Modal GPUs (no local NVIDIA GPU needed).

A thin transport over ``yeastbench.modal.app``: read the config locally, ship its
bytes to a remote container that runs the *unmodified* ``ybench`` CLI, and bring
the results tree back. The run/scoring logic is never duplicated.

``modal`` is imported lazily inside each command, so ``ybench modal --help`` and
the rest of the CLI work even when the ``modal`` extra isn't fully importable.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    add_completion=False, help="Run ybench on Modal GPUs (no local GPU needed)."
)
data_app = typer.Typer(
    add_completion=False, help="Seed the remote data/weights volume."
)
app.add_typer(data_app, name="data")

_CONFIG_OPT = typer.Option("--config", "-c", help="YAML run-spec path")
_MODEL_OPT = typer.Option("--model", "-m", help="Run only this model (filter)")
_TASK_OPT = typer.Option("--task", "-t", help="Run only this task (filter)")


def _echo(msg: str) -> None:
    typer.echo(msg)


def _download_results(out: Path) -> None:
    """Download the whole results volume into ``out`` (recreating
    ``out/<out_dir>/…``). ``out`` MUST be an existing directory first, or
    ``modal volume get`` collapses every file onto the single ``out`` path
    (modal/cli/_download.py: ``output_path = dest / rel`` only when ``dest`` is a
    dir), so we mkdir before the call."""
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", "--force", "ybench-results", "/", str(out)],
        check=True,
    )


@data_app.command("get")
def data_get(
    config: Annotated[Path, _CONFIG_OPT],
    model: Annotated[Optional[str], _MODEL_OPT] = None,
    task: Annotated[Optional[str], _TASK_OPT] = None,
) -> None:
    """One-time CPU seed: fetch this config's data + weights into the volume."""
    import modal

    from yeastbench.modal.app import app as modal_app, seed
    from yeastbench.modal.plan import build_plan

    plan = build_plan(config, model, task)
    _echo(f"seeding 'ybench-data' volume for {len(plan.pairs)} pair(s) from {config} …")
    with modal.enable_output(), modal_app.run():
        seed.remote(plan.config_bytes, plan.config_name, model=model, task=task)
    _echo("done — data + weights committed to the 'ybench-data' volume.")


@app.command("run")
def run(
    config: Annotated[Path, _CONFIG_OPT],
    model: Annotated[Optional[str], _MODEL_OPT] = None,
    task: Annotated[Optional[str], _TASK_OPT] = None,
    gpu: Annotated[
        str, typer.Option("--gpu", help="Modal GPU type, e.g. A10, L4, T4, A100, H100")
    ] = "A10",
    out: Annotated[
        Path, typer.Option("--out", help="local dir to download results into")
    ] = Path("results"),
    detach: Annotated[
        bool, typer.Option("--detach", help="keep running if the client disconnects")
    ] = False,
) -> None:
    """Seed (CPU), then run the benchmark on a Modal GPU and pull results back.

    The GPU *type* is chosen with ``--gpu`` (A10/L4/T4/…). The container's torch
    device is always ``cuda`` (its single GPU), independent of the config's
    ``device:`` field — so a config written for a local multi-GPU box still runs.
    """
    import modal

    from yeastbench.modal.app import app as modal_app, run_benchmark, seed
    from yeastbench.modal.plan import build_plan

    plan = build_plan(config, model, task)
    _echo(f"config:   {config}  [hash {plan.source_hash}]")
    _echo(f"gpu:      {gpu}")
    _echo(f"runs:     {len(plan.pairs)} pair(s) → {', '.join(plan.pair_dirs())}")

    with modal.enable_output(), modal_app.run(detach=detach):
        seed.remote(plan.config_bytes, plan.config_name, model=model, task=task)
        fn = run_benchmark.with_options(gpu=gpu)
        produced = fn.remote(
            plan.config_bytes,
            plan.config_name,
            out_dir=plan.out_dir,
            model=model,
            task=task,
        )

    _echo(f"\nremote run produced {len(produced)} pair dir(s): {', '.join(produced)}")
    _echo(f"downloading results → {out}/ …")
    _download_results(out)
    _echo(f"done — results under {out}/")


@app.command("pull")
def pull(
    out: Annotated[
        Path, typer.Option("--out", help="local dir to download results into")
    ] = Path("results"),
) -> None:
    """Download the results volume via the modal CLI (no run needed)."""
    _echo("downloading the 'ybench-results' volume …")
    _download_results(out)
    _echo(f"results written under {out}/")


@app.command("status")
def status() -> None:
    """List what's currently on the data and results volumes."""
    for vol in ("ybench-data", "ybench-results"):
        _echo(f"\n# {vol}")
        subprocess.run(["modal", "volume", "ls", vol], check=False)
