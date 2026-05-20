"""Unified CLI for yeastbench. Invoke via ``ybench`` (installed script) or
``python -m yeastbench.cli``.
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

import typer

from yeastbench.config import Config, load_config
from yeastbench.registry import MODELS, TASKS


app = typer.Typer(add_completion=False, help="yeast-seq2expression benchmark runner")


def _echo(msg: str) -> None:
    typer.echo(msg)


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _run_pair(cfg: Config, model_name: str, task_name: str, model_config: dict) -> None:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Known: {sorted(MODELS)}")
    if task_name not in TASKS:
        raise ValueError(f"Unknown task '{task_name}'. Known: {sorted(TASKS)}")

    task_config = cfg.tasks_config.get(task_name, {})
    out_dir = cfg.out_dir / f"{model_name}__{task_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _echo(f"\n→ {model_name} × {task_name} → {out_dir}")

    t0 = time.time()
    task = TASKS[task_name](**task_config)
    adapter = MODELS[model_name](task, device=cfg.device, **model_config)
    _echo(f"  ready in {time.time() - t0:.1f}s")

    t0 = time.time()
    results = task.evaluate(adapter)
    eval_s = time.time() - t0
    _echo(f"  evaluated in {eval_s:.1f}s")

    # Persist prediction arrays first so a plotting bug doesn't lose
    # the eval results (forcing a re-forward through the model).
    task.save_results(results, out_dir)

    t0 = time.time()
    task.plot(results, out_dir)
    _echo(f"  plots  written in {time.time() - t0:.1f}s")

    summary = {
        "model": model_name,
        "task": task_name,
        "task_version": task.info.version,
        **task.summary_dict(results),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "task": task_name,
                "task_version": task.info.version,
                "config_path": str(cfg.source_path),
                "config_hash": cfg.source_hash,
                "device": cfg.device,
                "model_config": model_config,
                "task_config": task_config,
                "git_commit": _git_commit(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_eval_s": eval_s,
            },
            indent=2,
        )
    )

    _echo(f"  {task.headline(results)}")


@app.command("run")
def run_cmd(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="YAML run-spec path")
    ],
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Run only this model (filter)"),
    ] = None,
    task: Annotated[
        Optional[str],
        typer.Option("--task", "-t", help="Run only this task (filter)"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="List planned runs and exit"),
    ] = False,
) -> None:
    """Execute (model, task) pairs defined by the config, with optional filters."""
    cfg = load_config(config).filtered(model, task)
    if not cfg.runs:
        raise typer.Exit(
            f"No runs match filters (model={model!r}, task={task!r}) in {config}"
        )

    _echo(f"config:        {cfg.source_path}  [hash {cfg.source_hash}]")
    _echo(f"out_dir:       {cfg.out_dir}")
    _echo(f"device:        {cfg.device}")
    pairs = [(r.model, t) for r in cfg.runs for t in r.tasks]
    _echo(f"planned runs:  {len(pairs)}")
    for m, t in pairs:
        _echo(f"  - {m} × {t}")

    if dry_run:
        raise typer.Exit(code=0)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for r in cfg.runs:
        for t in r.tasks:
            _run_pair(cfg, r.model, t, r.model_config)

    _echo("\nAll runs complete.")

    # Auto-trigger the cross-model comparison. Walks the FULL config's
    # `out_dir` (i.e. ignores --model / --task filters when looking for
    # peer results), so a run that just produced one of two models still
    # gets paired against the prior model's results on disk.
    full_cfg = load_config(config)
    _run_compare(full_cfg)


def _run_compare(cfg: Config) -> None:
    """Shared between `ybench run`'s auto-trigger and `ybench compare`.
    Silent no-op when there's nothing to compare."""
    from yeastbench.compare import compare

    result = compare(cfg)
    if result.empty:
        return
    _echo(f"\nCross-model comparison → {result.out_dir}")
    _echo(f"  tasks compared: {', '.join(result.tasks_compared)}")
    if result.tasks_skipped:
        _echo(
            "  tasks skipped (< 2 models): "
            + ", ".join(result.tasks_skipped)
        )
    if result.summary_csv:
        _echo(f"  summary.csv:    {result.summary_csv}")
    if result.summary_md:
        _echo(f"  summary.md:     {result.summary_md}")
    if result.overview_path:
        _echo(f"  overview.png:   {result.overview_path}")


@app.command("compare")
def compare_cmd(
    config: Annotated[
        Path, typer.Option("--config", "-c", help="YAML run-spec path")
    ],
) -> None:
    """Build cross-model comparison plots / tables from existing results.

    Walks the config's ``out_dir`` for ``<model>__<task>/summary.json``,
    groups by task, and for every task with ≥ 2 models writes a
    comparison plot + summary under ``out_dir/compare/per_task/<task>/``.
    Also emits the cross-task ``summary.csv`` / ``summary.md`` and an
    ``overview.png`` mosaic. Silent no-op if nothing is comparable."""
    cfg = load_config(config)
    _echo(f"config:   {cfg.source_path}  [hash {cfg.source_hash}]")
    _echo(f"out_dir:  {cfg.out_dir}")
    _run_compare(cfg)


@app.command("replot")
def replot_cmd(
    run_dir: Annotated[
        Path, typer.Argument(help="A run output directory (model__task/)")
    ],
    task: Annotated[
        Optional[str],
        typer.Option(
            "--task",
            help="Task name. Inferred from directory (…__<task>) if omitted.",
        ),
    ] = None,
    task_config: Annotated[
        Optional[Path],
        typer.Option(
            "--task-config",
            help="Optional YAML config to pull task_config from (for distribution_dir etc.)",
        ),
    ] = None,
) -> None:
    """Regenerate plots from saved results."""
    run_dir = run_dir.resolve()
    if task is None:
        name = run_dir.name
        if "__" not in name:
            raise typer.Exit(
                f"Cannot infer task from directory name {name!r}. Pass --task."
            )
        task = name.split("__", 1)[1]
    if task not in TASKS:
        raise typer.Exit(f"Unknown task '{task}'. Known: {sorted(TASKS)}")

    if task_config is None:
        cfg_kwargs: dict = {}
        meta = run_dir / "run_metadata.json"
        if meta.exists():
            cfg_kwargs = json.loads(meta.read_text()).get("task_config", {})
    else:
        cfg = load_config(task_config)
        cfg_kwargs = cfg.tasks_config.get(task, {})

    benchmark = TASKS[task](**cfg_kwargs)
    results = benchmark.load_results(run_dir)
    benchmark.plot(results, run_dir)
    _echo(f"replotted → {run_dir}")


@app.command("list")
def list_cmd() -> None:
    """List registered models and tasks."""
    _echo("models:")
    for name in sorted(MODELS):
        _echo(f"  - {name}")
    _echo("tasks:")
    for name in sorted(TASKS):
        _echo(f"  - {name}")


if __name__ == "__main__":
    app()
