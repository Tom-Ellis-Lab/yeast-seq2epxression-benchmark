from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, Mapping, TypeVar

AdapterT = TypeVar("AdapterT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class BenchmarkInfo:
    name: str
    version: str
    description: str
    distribution_uri: str


class Benchmark(ABC, Generic[AdapterT, ResultT]):
    info: BenchmarkInfo
    adapter_protocol: ClassVar[type]

    @abstractmethod
    def evaluate(self, adapter: AdapterT) -> ResultT: ...

    @abstractmethod
    def plot(self, results: ResultT, out_dir: Path) -> None: ...

    @abstractmethod
    def save_results(self, results: ResultT, out_dir: Path) -> None:
        """Persist raw results (scores, labels, metadata) to *out_dir*."""
        ...

    @abstractmethod
    def load_results(self, out_dir: Path) -> ResultT:
        """Reconstruct results from files previously written by *save_results*."""
        ...

    @abstractmethod
    def summary_dict(self, results: ResultT) -> dict[str, Any]:
        """Return a JSON-serialisable summary of *results*."""
        ...

    @abstractmethod
    def headline(self, results: ResultT) -> str:
        """One-line summary for CLI output after evaluation."""
        ...

    # ── Cross-model comparison ───────────────────────────────────────────
    #
    # The compare runner (`yeastbench.compare`) calls `compare_plot` once
    # per task whose `out_dir/<model>__<task>/` directories cover at least
    # two models. The default implementation reads each model's
    # `summary.json` and draws a grouped bar chart of every numeric scalar
    # — works for every benchmark out of the box without any per-benchmark
    # state. Benchmarks that need richer comparisons (e.g. Brooks's
    # shared-cohort sample-id intersection) override this method.

    def compare_plot(
        self,
        model_dirs: Mapping[str, Path],
        out_dir: Path,
    ) -> Path | None:
        """Draw a per-model comparison plot for this task. Default reads
        ``summary.json`` per model and produces a grouped bar chart of
        every numeric scalar; returns the path of the resulting PNG (or
        ``None`` if there is nothing plottable). Override for richer
        per-protocol comparisons.

        ``model_dirs`` maps model name → result directory (the
        ``<model>__<task>`` subtree of the run output)."""
        return _default_compare_plot(model_dirs, out_dir)

    @property
    def compare_task_name(self) -> str:
        """Canonical task name used to group result directories when
        the cross-model comparison runner discovers them. Multiple
        ``TASKS`` registry entries that point at the same benchmark
        class (e.g. ``brooks_scramble`` for Yorzoi-window and
        ``brooks_scramble_shorkie`` for Shorkie-window) override this
        property to return a single shared name so the runner pairs
        their results when computing cross-model metrics. Defaults to
        ``self.info.name``."""
        return self.info.name


# ── Default compare plot ─────────────────────────────────────────────────


def _flat_scalar_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    """Pick numeric scalars (int / float, finite) from a summary dict.
    Skips nested structures (lists, dicts) — they'd need a custom plot."""
    out: dict[str, float] = {}
    for key, value in summary.items():
        if isinstance(value, bool):
            continue  # booleans aren't useful as bar heights
        if isinstance(value, (int, float)):
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
                continue
            out[key] = v
    return out


def _default_compare_plot(
    model_dirs: Mapping[str, Path], out_dir: Path,
) -> Path | None:
    """Grouped bar chart: one group per scalar metric, one bar per model.

    Metrics are the intersection of numeric scalars across *all* models'
    `summary.json` files — keeps the plot honest when models report
    different fields. NaNs / infs / booleans / nested structures are
    skipped."""
    import matplotlib.pyplot as plt
    import numpy as np

    per_model: dict[str, dict[str, float]] = {}
    for model_name, mdir in model_dirs.items():
        summary_path = Path(mdir) / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            continue
        per_model[model_name] = _flat_scalar_metrics(summary)
    if len(per_model) < 2:
        return None
    # Intersect metric keys across all models so the plot stays apples-to-apples.
    common = set.intersection(*(set(m.keys()) for m in per_model.values()))
    # Skip count-style keys that swamp the y-axis (heuristic: integer-valued
    # keys starting with ``n_`` are usually sample counts).
    metrics = sorted(
        k for k in common
        if not (k.startswith("n_") and all(
            float(m[k]).is_integer() for m in per_model.values()
        ))
    )
    if not metrics:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = sorted(per_model.keys())
    n_metrics = len(metrics)
    n_models = len(model_names)
    width = 0.8 / n_models
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * n_metrics), 4.5))
    for i, model in enumerate(model_names):
        ys = [per_model[model][m] for m in metrics]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, ys, width=width, label=model)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("metric value")
    ax.legend(fontsize=8)
    ax.set_title("model comparison — scalar metrics", fontsize=10)
    fig.tight_layout()
    out_path = out_dir / "plot.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
