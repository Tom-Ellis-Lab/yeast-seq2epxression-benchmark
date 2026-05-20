from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, Mapping, Sequence, TypeVar

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
        the headline metrics declared by ``headline_metric_labels``;
        returns the path of the resulting PNG (or ``None`` if there is
        nothing plottable). Override for richer per-protocol comparisons.

        ``model_dirs`` maps model name → result directory (the
        ``<model>__<task>`` subtree of the run output)."""
        return _default_compare_plot(
            model_dirs, out_dir,
            metric_labels=self.headline_metric_labels(),
            title=self.compare_plot_title(),
        )

    def headline_metric_labels(self) -> dict[str, str] | None:
        """Curated headline-metric → display-label map used by the
        default compare plot. Order matters (display order on the
        x-axis). Return ``None`` (default) to fall back to the legacy
        auto-discovery behaviour: every numeric scalar in
        ``summary.json`` except keys prefixed with ``n_`` or ending in
        ``_n_pos`` / ``_n`` / ``_n_total``.

        Benchmarks should override this so the comparison plot picks
        the right axes — auto-discovery is fragile (it can't tell
        ``extreme_high_n_pos`` from a metric)."""
        return None

    def compare_plot_title(self) -> str:
        """Title shown above the default compare plot. Defaults to the
        task name; override for a nicer display string."""
        return self.info.name

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
#
# A single shared palette used by the default compare plot AND by
# per-benchmark overrides (e.g. Brooks). Models are sorted by name
# before colour assignment so the same model gets the same colour in
# every panel of the mosaic.

MODEL_COLORS: tuple[str, ...] = (
    "#d62728",  # red
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # olive
    "#17becf",  # cyan
)


def model_color(model_name: str, ordered_names: Sequence[str]) -> str:
    """Pick a deterministic colour for *model_name* given the full
    sorted list of models in this comparison."""
    idx = list(ordered_names).index(model_name)
    return MODEL_COLORS[idx % len(MODEL_COLORS)]


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


def _is_count_key(key: str) -> bool:
    """Heuristic: does this summary key look like a sample count rather
    than a metric? Used by the auto-discovery fallback when a benchmark
    doesn't override `headline_metric_labels`."""
    if key.startswith("n_"):
        return True
    for suffix in ("_n", "_n_pos", "_n_neg", "_n_total"):
        if key.endswith(suffix):
            return True
    return False


def _default_compare_plot(
    model_dirs: Mapping[str, Path],
    out_dir: Path,
    *,
    metric_labels: Mapping[str, str] | None = None,
    title: str | None = None,
) -> Path | None:
    """Grouped bar chart: one group per metric, one bar per model.

    If ``metric_labels`` is provided, those are the metrics shown (in
    that order, with their human-readable labels on the x-axis). When
    ``None``, auto-discovers: intersection of numeric scalars across
    all models' `summary.json`, with `_is_count_key` filtered out.

    NaNs / infs / booleans / nested structures are skipped. Returns
    None when fewer than 2 models have summaries or there are no
    metrics to plot."""
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

    if metric_labels is not None:
        # Use the curated list — preserve insertion order, drop metrics
        # any model is missing (with a quiet skip rather than blowing up).
        metrics_order = [
            k for k in metric_labels
            if all(k in m for m in per_model.values())
        ]
        labels = {k: metric_labels[k] for k in metrics_order}
    else:
        # Auto-discovery fallback: numeric scalars common to every model,
        # minus count-style keys.
        common = set.intersection(*(set(m.keys()) for m in per_model.values()))
        metrics_order = sorted(k for k in common if not _is_count_key(k))
        labels = {k: k for k in metrics_order}
    if not metrics_order:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = sorted(per_model.keys())
    n_metrics = len(metrics_order)
    n_models = len(model_names)
    width = 0.8 / n_models
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * n_metrics), 4.5))
    for i, model in enumerate(model_names):
        ys = [per_model[model][m] for m in metrics_order]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(
            x + offset, ys, width=width, label=model,
            color=model_color(model, model_names),
        )
        # Value labels above each bar so the exact number is readable.
        for b, v in zip(bars, ys):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + (0.005 if v >= 0 else -0.025),
                f"{v:+.3f}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=7, color="black", alpha=0.8,
            )
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [labels[k] for k in metrics_order], rotation=20, ha="right", fontsize=9,
    )
    ax.legend(fontsize=9, loc="best")
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    out_path = out_dir / "plot.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path
