"""Cross-model comparison runner.

Scoped to the run config: for every ``(model, task)`` pair the config
declares, walks ``config.out_dir/<model>__<task>/summary.json``, and for
each task with **≥ 2 models** produces a per-task comparison plot under
``config.out_dir/compare/per_task/<task>/``. Then aggregates into a
cross-task `summary.csv` (long format) and `summary.md` (wide tables).

Discovery is restricted to the config's pairs, so stale results from
earlier runs of other models/tasks left on disk are ignored — a run
compares exactly the models the config names.

Each benchmark's plot shape is controlled by its `compare_plot` method
(see `yeastbench.benchmarks.base.Benchmark.compare_plot`). The default
implementation is a grouped bar chart of the benchmark's headline metrics;
benchmarks like Brooks override it for shared-cohort intersection + custom
plots.

Used by `ybench run` — auto-triggered at the end of every run; a silent
no-op when no task has ≥ 2 models with results.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from yeastbench.config import Config
from yeastbench.registry import TASKS

log = logging.getLogger(__name__)

# Directory names are written as `<model>__<task>` by cli.py:_run_pair.
# The model group is non-greedy so model names with single underscores
# (e.g. `codon_transformer`) survive the split; `__` (double underscore)
# is the canonical separator, and neither model nor task names contain
# `__` in the registry.
_DIR_RE = re.compile(r"^(?P<model>.+?)__(?P<task>.+)$")


@dataclass(frozen=True)
class CompareSummary:
    """Returned by `compare()` so callers can log / chain on the result."""
    out_dir: Path
    tasks_compared: list[str] = field(default_factory=list)
    tasks_skipped: list[str] = field(default_factory=list)  # < 2 models
    per_task_plots: dict[str, Path] = field(default_factory=dict)
    summary_csv: Path | None = None
    summary_md: Path | None = None

    @property
    def empty(self) -> bool:
        return not self.tasks_compared


def _discover_results(
    out_dir: Path,
    restrict_to_pairs: set[tuple[str, str]] | None = None,
) -> dict[str, dict[str, Path]]:
    """Walk ``out_dir/<model>__<task>/`` and group by registry task name.

    Returns ``{task_name: {model_name: result_dir}}`` where ``task_name``
    is the directory's ``<task>`` suffix. Skips directories whose name
    starts with ``compare`` to avoid eating our own output.

    When ``restrict_to_pairs`` is given, only ``(model, task)`` result
    directories in that set are included; ``None`` (the default) includes
    every result directory found. ``ybench run`` passes the config's pairs
    so the auto-comparison covers exactly those — never stale peer results
    left on disk by an earlier run of a different config."""
    by_task: dict[str, dict[str, Path]] = {}
    if not out_dir.exists():
        return by_task
    for entry in sorted(out_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("compare"):
            continue
        m = _DIR_RE.match(entry.name)
        if not m:
            continue
        if (
            restrict_to_pairs is not None
            and (m["model"], m["task"]) not in restrict_to_pairs
        ):
            continue
        if not (entry / "summary.json").exists():
            continue
        by_task.setdefault(m["task"], {})[m["model"]] = entry
    return by_task


def _flat_scalar_metrics(summary: dict) -> dict[str, float]:
    """Copy of the helper in base.py for runner-side use (the aggregate
    CSV/MD reuse this). Skips booleans, NaN/inf, and non-numeric values."""
    out: dict[str, float] = {}
    for key, value in summary.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if v != v or v in (float("inf"), float("-inf")):
                continue
            out[key] = v
    return out


def _build_csv(
    by_task: dict[str, dict[str, Path]], out_path: Path,
) -> Path:
    """Long-format CSV: task, model, metric, value. Includes every
    numeric scalar across summaries — even metrics one model didn't
    report (missing cells absent rather than NaN)."""
    rows: list[tuple[str, str, str, float]] = []
    for task in sorted(by_task):
        for model in sorted(by_task[task]):
            summary_path = by_task[task][model] / "summary.json"
            try:
                summary = json.loads(summary_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            for metric, value in _flat_scalar_metrics(summary).items():
                rows.append((task, model, metric, value))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["task", "model", "metric", "value"])
        for row in rows:
            writer.writerow(row)
    return out_path


def _build_md(
    by_task: dict[str, dict[str, Path]], out_path: Path,
) -> Path:
    """Per-task wide tables: metrics on rows, models on columns. One
    section per task, in alphabetical order. Models missing a metric
    show as a blank cell."""
    chunks: list[str] = [
        "# Cross-model comparison\n",
    ]
    for task in sorted(by_task):
        models = sorted(by_task[task])
        if len(models) < 2:
            continue
        per_model_metrics: dict[str, dict[str, float]] = {}
        for model in models:
            summary_path = by_task[task][model] / "summary.json"
            try:
                summary = json.loads(summary_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            per_model_metrics[model] = _flat_scalar_metrics(summary)
        if not per_model_metrics:
            continue
        metric_keys = sorted(
            set().union(*(set(m.keys()) for m in per_model_metrics.values()))
        )
        chunks.append(f"\n## {task}\n")
        header = "| metric | " + " | ".join(models) + " |"
        sep = "| --- |" + " --- |" * len(models)
        chunks.append(header)
        chunks.append(sep)
        for metric in metric_keys:
            row_values = []
            for model in models:
                v = per_model_metrics.get(model, {}).get(metric)
                row_values.append("" if v is None else f"{v:+.4f}")
            chunks.append(f"| {metric} | " + " | ".join(row_values) + " |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(chunks) + "\n")
    return out_path


def compare(config: Config) -> CompareSummary:
    """Run the cross-model comparison for *config*. Always safe to call —
    silent no-op when no task in the config has ≥ 2 models with results on
    disk. Discovery is scoped to the config's ``(model, task)`` pairs."""
    out_dir = Path(config.out_dir)
    pairs = {(r.model, t) for r in config.runs for t in r.tasks}
    by_task = _discover_results(out_dir, restrict_to_pairs=pairs)

    compare_root = out_dir / "compare"
    per_task_root = compare_root / "per_task"

    tasks_compared: list[str] = []
    tasks_skipped: list[str] = []
    per_task_plots: dict[str, Path] = {}

    for task_name in sorted(by_task):
        model_dirs = by_task[task_name]
        if len(model_dirs) < 2:
            tasks_skipped.append(task_name)
            continue
        plot_path: Path | None
        if task_name not in TASKS:
            from yeastbench.benchmarks.base import _default_compare_plot
            log.info(
                "compare: unknown task '%s' in results — using default plot",
                task_name,
            )
            plot_path = _default_compare_plot(
                model_dirs, per_task_root / task_name,
            )
        else:
            task_cfg = dict(config.tasks_config.get(task_name, {}))
            try:
                bench = TASKS[task_name](**task_cfg)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "compare: failed to instantiate task '%s' (%s) — "
                    "falling back to default plot",
                    task_name, exc,
                )
                from yeastbench.benchmarks.base import _default_compare_plot
                plot_path = _default_compare_plot(
                    model_dirs, per_task_root / task_name,
                )
            else:
                plot_path = bench.compare_plot(
                    model_dirs, per_task_root / task_name,
                )
        if plot_path is not None:
            per_task_plots[task_name] = plot_path
        tasks_compared.append(task_name)

    summary_csv: Path | None = None
    summary_md: Path | None = None
    if tasks_compared:
        compare_root.mkdir(parents=True, exist_ok=True)
        summary_csv = _build_csv(by_task, compare_root / "summary.csv")
        summary_md = _build_md(by_task, compare_root / "summary.md")

    return CompareSummary(
        out_dir=compare_root,
        tasks_compared=tasks_compared,
        tasks_skipped=tasks_skipped,
        per_task_plots=per_task_plots,
        summary_csv=summary_csv,
        summary_md=summary_md,
    )
