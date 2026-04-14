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

import numpy as np
import pandas as pd
import typer

from yeastbench.benchmarks.eqtl import EQTLIterationResult, EQTLResults
from yeastbench.config import Config, load_config
from yeastbench.registry import MODELS, TASKS
from sklearn.metrics import average_precision_score, roc_auc_score


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
    results: EQTLResults = task.evaluate(adapter)
    eval_s = time.time() - t0
    _echo(f"  evaluated in {eval_s:.1f}s")

    t0 = time.time()
    task.plot(results, out_dir)
    _echo(f"  plots  written in {time.time() - t0:.1f}s")

    # Persist per-iteration raw scores + metadata
    for r in results.per_iter:
        np.save(out_dir / f"{r.name}_scores.npy", r.scores)
        np.save(out_dir / f"{r.name}_labels.npy", r.labels)
        r.pairs.to_csv(out_dir / f"{r.name}_pairs.tsv", sep="\t", index=False)

    # Per-iter and aggregate summary
    summary = {
        "model": model_name,
        "task": task_name,
        "task_version": task.info.version,
        "per_iteration": [
            {
                "name": r.name,
                "n_pairs": int(len(r.pairs)),
                "auroc_signed": r.auroc_signed,
                "auprc_signed": r.auprc_signed,
                "auroc_abs": r.auroc_abs,
                "auprc_abs": r.auprc_abs,
                "zero_frac": float((r.scores == 0).mean()),
            }
            for r in results.per_iter
        ],
        "auroc_abs_mean": results.mean_auroc,
        "auroc_abs_sem": results.sem_auroc,
        "auprc_abs_mean": results.mean_auprc,
        "auprc_abs_sem": results.sem_auprc,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Run metadata — pinpoints the config + code version that produced these numbers
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

    _echo(
        f"  |score| AUROC {results.mean_auroc:.4f} ± {results.sem_auroc:.4f}  "
        f"AUPRC {results.mean_auprc:.4f} ± {results.sem_auprc:.4f}"
    )


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
    """Regenerate plots from saved per-iteration scores/labels/pairs."""
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

    per_iter: list[EQTLIterationResult] = []
    score_files = sorted(run_dir.glob("*_scores.npy"))
    if not score_files:
        raise typer.Exit(f"No *_scores.npy files under {run_dir}")
    for sp in score_files:
        name = sp.name.replace("_scores.npy", "")
        labels = np.load(run_dir / f"{name}_labels.npy")
        scores = np.load(sp)
        pairs = pd.read_csv(run_dir / f"{name}_pairs.tsv", sep="\t")
        per_iter.append(
            EQTLIterationResult(
                name=name,
                scores=scores,
                labels=labels,
                pairs=pairs,
                auroc_signed=float(roc_auc_score(labels, scores)),
                auprc_signed=float(average_precision_score(labels, scores)),
                auroc_abs=float(roc_auc_score(labels, np.abs(scores))),
                auprc_abs=float(average_precision_score(labels, np.abs(scores))),
            )
        )
    results = EQTLResults(per_iter=per_iter)
    benchmark.plot(results, run_dir)
    _echo(f"replotted {len(per_iter)} iterations → {run_dir}")


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
