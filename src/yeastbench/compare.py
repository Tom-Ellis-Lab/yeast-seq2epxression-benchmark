"""Cross-model comparison runner.

Walks ``config.out_dir`` for ``<model>__<task>/summary.json`` files,
groups them by task, and for each task with **≥ 2 models** produces a
per-task comparison plot + a per-task summary directory under
``config.out_dir/compare/per_task/<task>/``. Then aggregates the
per-task summaries into a cross-task `summary.csv` (long format),
`summary.md` (wide tables), and `overview.png` (mosaic of every
per-task plot).

Each benchmark's plot shape is controlled by its `compare_plot` method
(see `yeastbench.benchmarks.base.Benchmark.compare_plot`). The default
implementation is a grouped bar chart of every numeric scalar in the
per-model summary; benchmarks like Brooks override it for shared-cohort
intersection + custom plots.

Used by:
- `ybench compare --config ...` — standalone invocation.
- `ybench run` — auto-trigger at end of every run; silent no-op if no
  task has ≥ 2 models.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from typing import Any, Mapping

from yeastbench.config import Config
from yeastbench.registry import TASKS

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# Directory names are written as `<model>__<task>` by cli.py:_run_pair.
_DIR_RE = re.compile(r"^(?P<model>[^_]+)__(?P<task>.+)$")


@dataclass(frozen=True)
class CompareSummary:
    """Returned by `compare()` so callers can log / chain on the result."""
    out_dir: Path
    tasks_compared: list[str] = field(default_factory=list)
    tasks_skipped: list[str] = field(default_factory=list)  # < 2 models
    per_task_plots: dict[str, Path] = field(default_factory=dict)
    overview_path: Path | None = None
    summary_csv: Path | None = None
    summary_md: Path | None = None

    @property
    def empty(self) -> bool:
        return not self.tasks_compared


def _discover_results(
    out_dir: Path,
) -> dict[str, dict[str, Path]]:
    """Walk ``out_dir/<model>__<task>/`` and group by registry task name.

    Returns ``{task_name: {model_name: result_dir}}`` where ``task_name``
    is the *registry* name (the directory's ``<task>`` suffix). The
    caller is responsible for any cross-task aliasing (e.g. via
    ``Benchmark.compare_task_name``). Skips directories whose name
    starts with ``compare`` to avoid eating our own output."""
    by_task: dict[str, dict[str, Path]] = {}
    if not out_dir.exists():
        return by_task
    for entry in sorted(out_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("compare"):
            continue
        m = _DIR_RE.match(entry.name)
        if not m:
            continue
        if not (entry / "summary.json").exists():
            continue
        by_task.setdefault(m["task"], {})[m["model"]] = entry
    return by_task


def _group_by_compare_task(
    by_task: dict[str, dict[str, Path]],
    tasks_config: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Path]]]:
    """Re-group by ``Benchmark.compare_task_name`` so registry entries
    that point at the same logical benchmark (e.g. ``brooks_scramble``
    + ``brooks_scramble_shorkie``) merge into one comparison group.

    Returns ``{group_name: {registry_task_name: {model_name: dir}}}``.
    The inner registry-task-name mapping is preserved so the runner
    can pick any one of them to construct a benchmark instance and
    can route plot output by the group name (not per registry key).

    Tasks the registry doesn't know about (older runs, etc.) fall
    through to a group named by the registry task name itself."""
    groups: dict[str, dict[str, dict[str, Path]]] = {}
    for task_name, model_dirs in by_task.items():
        group_name = task_name
        if task_name in TASKS:
            task_cfg = dict(tasks_config.get(task_name, {}))
            try:
                bench = TASKS[task_name](**task_cfg)
                group_name = bench.compare_task_name
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "compare: couldn't instantiate '%s' to read "
                    "compare_task_name (%s) — grouping under registry "
                    "name instead", task_name, exc,
                )
        groups.setdefault(group_name, {})[task_name] = model_dirs
    return groups


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


def _build_mosaic(
    per_task_plots: dict[str, Path], out_path: Path,
) -> Path | None:
    """Combine every per-task plot into one big PNG. One row per task
    (each per-task PNG loaded and displayed as an image)."""
    if not per_task_plots:
        return None
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = sorted(per_task_plots)
    n = len(tasks)
    fig, axes = plt.subplots(
        nrows=n, ncols=1, figsize=(10, 4.5 * n), squeeze=False,
    )
    for i, task in enumerate(tasks):
        ax = axes[i, 0]
        try:
            img = mpimg.imread(per_task_plots[task])
            ax.imshow(img)
        except (FileNotFoundError, ValueError, OSError):
            ax.text(0.5, 0.5, f"[plot not loadable]",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_title(task, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def compare(config: Config) -> CompareSummary:
    """Run the cross-model comparison for *config*. Always safe to call
    — silent no-op when no compare-task group has ≥ 2 models with
    results on disk."""
    out_dir = Path(config.out_dir)
    by_task = _discover_results(out_dir)
    groups = _group_by_compare_task(by_task, config.tasks_config)

    compare_root = out_dir / "compare"
    per_task_root = compare_root / "per_task"

    tasks_compared: list[str] = []
    tasks_skipped: list[str] = []
    per_task_plots: dict[str, Path] = {}
    # Flatten the group's per-registry-task model_dirs into one
    # {model: dir} dict per group — model names should be unique within
    # a comparison group (one model = one wrapper instance), but a
    # group may span multiple registry task entries.
    by_group_flat: dict[str, dict[str, Path]] = {}
    for group_name in sorted(groups):
        flat: dict[str, Path] = {}
        for reg_task_name, model_dirs in groups[group_name].items():
            for model, mdir in model_dirs.items():
                if model in flat:
                    log.warning(
                        "compare: group '%s' has model '%s' appearing in "
                        "multiple registry tasks (%s + %s); keeping the "
                        "first.", group_name, model, flat[model], mdir,
                    )
                    continue
                flat[model] = mdir
        by_group_flat[group_name] = flat

    for group_name in sorted(by_group_flat):
        model_dirs = by_group_flat[group_name]
        if len(model_dirs) < 2:
            tasks_skipped.append(group_name)
            continue
        # Pick the first registry task name in the group for benchmark
        # instantiation; compare_plot lives on the class so any one of
        # them is interchangeable.
        reg_task_name = sorted(groups[group_name].keys())[0]
        plot_path: Path | None
        if reg_task_name not in TASKS:
            from yeastbench.benchmarks.base import _default_compare_plot
            log.info(
                "compare: unknown task '%s' in results — using default plot",
                reg_task_name,
            )
            plot_path = _default_compare_plot(
                model_dirs, per_task_root / group_name,
            )
        else:
            task_cfg = dict(config.tasks_config.get(reg_task_name, {}))
            try:
                bench = TASKS[reg_task_name](**task_cfg)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "compare: failed to instantiate task '%s' (%s) — "
                    "falling back to default plot",
                    reg_task_name, exc,
                )
                from yeastbench.benchmarks.base import _default_compare_plot
                plot_path = _default_compare_plot(
                    model_dirs, per_task_root / group_name,
                )
            else:
                plot_path = bench.compare_plot(
                    model_dirs, per_task_root / group_name,
                )
        if plot_path is not None:
            per_task_plots[group_name] = plot_path
        tasks_compared.append(group_name)

    # Aggregate CSV / MD use the *group* view (one row per (group, model))
    # so callers get one table even when Brooks ships under two registry
    # keys. Pass `by_group_flat` instead of `by_task`.
    overview_path: Path | None = None
    summary_csv: Path | None = None
    summary_md: Path | None = None
    if tasks_compared:
        compare_root.mkdir(parents=True, exist_ok=True)
        summary_csv = _build_csv(by_group_flat, compare_root / "summary.csv")
        summary_md = _build_md(by_group_flat, compare_root / "summary.md")
        overview_path = _build_mosaic(
            per_task_plots, compare_root / "overview.png"
        )

    return CompareSummary(
        out_dir=compare_root,
        tasks_compared=tasks_compared,
        tasks_skipped=tasks_skipped,
        per_task_plots=per_task_plots,
        overview_path=overview_path,
        summary_csv=summary_csv,
        summary_md=summary_md,
    )
