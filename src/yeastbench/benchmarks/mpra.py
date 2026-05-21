"""MPRA marginalized benchmark — Rafi / deBoer et al. random promoter expression.

Evaluates zero-shot prediction of scalar expression for 71,103 80-bp random
promoter sequences across eight DREAM test-set strata. Each insert is
scored by marginalizing its logSED over 22 native host-gene contexts (the
``MarginalizedSequenceExpressionPredictor`` protocol). The earlier
fixed-context plasmid-construct variant was retired on 2026-05-21 — the
marginalized / native-position evaluation is what Shorkie / Yorzoi were
trained for and is the canonical Rafi benchmark going forward.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from yeastbench.adapters.protocols import MarginalizedSequenceExpressionPredictor
from yeastbench.benchmarks._metrics import MPRAStratumResult
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo, model_color


# ── Stratum catalogue ─────────────────────────────────────────

# Non-pair strata: CSV has columns (tag, sequence, pos, exp).
# Pair strata:     CSV has columns (alt_tag, ref_tag, alt_sequence, ref_sequence,
#                                   alt_pos, ref_pos, alt_exp, ref_exp).
# In both cases `pos` / `alt_pos` / `ref_pos` are 0-based row indices into
# the 71,103-row master file.

STRATA_FILES: dict[str, str] = {
    "high_exp": "high_exp_seqs.csv",
    "low_exp": "low_exp_seqs.csv",
    "yeast_exp": "yeast_seqs.csv",
    "random_exp": "all_random_seqs.csv",
    "challenging": "challenging_seqs.csv",
    "SNVs": "all_SNVs_seqs.csv",
    "motif_perturbation": "motif_perturbation.csv",
    "motif_tiling": "motif_tiling_seqs.csv",
}

PAIR_STRATA = {"SNVs", "motif_perturbation", "motif_tiling"}


# ── Results ───────────────────────────────────────────────────
# `MPRAStratumResult` lives in `_metrics.py` (shared with Shalem).


@dataclass(frozen=True)
class MPRAPairStratumResult:
    name: str
    n_pairs: int
    pearson_r: float         # on raw predictions
    spearman_rho: float      # on raw predictions
    diff_pearson_r: float    # on (pred_diff, measured_diff) per pair
    diff_spearman_rho: float


@dataclass(frozen=True)
class MPRAResults:
    scores: np.ndarray                          # (N,) predicted expression
    labels: np.ndarray                          # (N,) measured expression
    overall: MPRAStratumResult
    per_stratum: list[MPRAStratumResult]
    per_pair_stratum: list[MPRAPairStratumResult]
    strata_indices: dict[str, np.ndarray]       # name → row indices
    pair_indices: dict[str, np.ndarray]         # name → (n_pairs, 2) alt/ref indices


# ── Benchmark ─────────────────────────────────────────────────


class MPRAMarginalizedBenchmark(
    Benchmark[MarginalizedSequenceExpressionPredictor, MPRAResults]
):
    """Rafi / deBoer MPRA, marginalized / native-position scoring.

    Adapters implement ``predict_marginalized_expressions`` and return a
    scalar per insert that's the mean logSED across 22 native host-gene
    contexts (configured in the adapter, not the benchmark).
    """

    adapter_protocol: ClassVar[type] = MarginalizedSequenceExpressionPredictor

    def __init__(
        self,
        data_dir: Path,
        fasta_path: Path,
        gtf_path: Path,
        info: BenchmarkInfo,
    ) -> None:
        self.data_dir = Path(data_dir)
        self._fasta_path = Path(fasta_path)
        self._gtf_path = Path(gtf_path)
        self.info = info

        # Load master file (no header, two columns: seq, el)
        master = pd.read_csv(
            self.data_dir / "filtered_test_data_with_MAUDE_expression.txt",
            sep="\t",
            header=None,
            names=["seq", "el"],
        )
        self.sequences: list[str] = master["seq"].tolist()
        self.labels: np.ndarray = master["el"].to_numpy(dtype=float)

        # Load stratum indices
        subset_dir = self.data_dir / "test_subset_ids"
        self.strata_indices: dict[str, np.ndarray] = {}
        self.pair_indices: dict[str, np.ndarray] = {}

        for stratum, fname in STRATA_FILES.items():
            csv_path = subset_dir / fname
            df = pd.read_csv(csv_path)
            if stratum in PAIR_STRATA:
                alt_idx = df["alt_pos"].to_numpy(dtype=int)
                ref_idx = df["ref_pos"].to_numpy(dtype=int)
                self.pair_indices[stratum] = np.column_stack([alt_idx, ref_idx])
                # Unique row indices that appear in this stratum
                self.strata_indices[stratum] = np.unique(
                    np.concatenate([alt_idx, ref_idx])
                )
            else:
                self.strata_indices[stratum] = df["pos"].to_numpy(dtype=int)

    @property
    def fasta_path(self) -> Path:
        return self._fasta_path

    @property
    def gtf_path(self) -> Path:
        return self._gtf_path

    def evaluate(
        self, adapter: MarginalizedSequenceExpressionPredictor
    ) -> MPRAResults:
        scores = np.asarray(
            adapter.predict_marginalized_expressions(self.sequences), dtype=float
        )
        assert len(scores) == len(self.labels)

        overall = _stratum_result("overall", scores, self.labels)

        per_stratum: list[MPRAStratumResult] = []
        for name, idx in self.strata_indices.items():
            per_stratum.append(
                _stratum_result(name, scores[idx], self.labels[idx])
            )

        per_pair_stratum: list[MPRAPairStratumResult] = []
        for name, pairs in self.pair_indices.items():
            per_pair_stratum.append(
                _pair_stratum_result(name, scores, self.labels, pairs)
            )

        return MPRAResults(
            scores=scores,
            labels=self.labels,
            overall=overall,
            per_stratum=per_stratum,
            per_pair_stratum=per_pair_stratum,
            strata_indices=self.strata_indices,
            pair_indices=self.pair_indices,
        )

    def plot(self, results: MPRAResults, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        title_model = out_dir.name.split("__")[0] if "__" in out_dir.name else ""

        _plot_scatter_grid(results, out_dir / "scatter_per_stratum.png", title_model)
        _plot_summary_bar(results, out_dir / "pearson_summary.png", title_model)

    def save_results(self, results: MPRAResults, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "scores.npy", results.scores)
        np.save(out_dir / "labels.npy", results.labels)

    def load_results(self, out_dir: Path) -> MPRAResults:
        out_dir = Path(out_dir)
        scores = np.load(out_dir / "scores.npy")
        labels = np.load(out_dir / "labels.npy")

        overall = _stratum_result("overall", scores, labels)

        per_stratum: list[MPRAStratumResult] = []
        for name, idx in self.strata_indices.items():
            per_stratum.append(
                _stratum_result(name, scores[idx], labels[idx])
            )

        per_pair_stratum: list[MPRAPairStratumResult] = []
        for name, pairs in self.pair_indices.items():
            per_pair_stratum.append(
                _pair_stratum_result(name, scores, labels, pairs)
            )

        return MPRAResults(
            scores=scores,
            labels=labels,
            overall=overall,
            per_stratum=per_stratum,
            per_pair_stratum=per_pair_stratum,
            strata_indices=self.strata_indices,
            pair_indices=self.pair_indices,
        )

    def summary_dict(self, results: MPRAResults) -> dict[str, Any]:
        return {
            "n_sequences": len(results.scores),
            "overall_pearson_r": results.overall.pearson_r,
            "overall_spearman_rho": results.overall.spearman_rho,
            "per_stratum": [
                {
                    "name": s.name,
                    "n": s.n,
                    "pearson_r": s.pearson_r,
                    "spearman_rho": s.spearman_rho,
                }
                for s in results.per_stratum
            ],
            "per_pair_stratum": [
                {
                    "name": s.name,
                    "n_pairs": s.n_pairs,
                    "pearson_r": s.pearson_r,
                    "spearman_rho": s.spearman_rho,
                    "diff_pearson_r": s.diff_pearson_r,
                    "diff_spearman_rho": s.diff_spearman_rho,
                }
                for s in results.per_pair_stratum
            ],
        }

    def headline(self, results: MPRAResults) -> str:
        return (
            f"overall Pearson r = {results.overall.pearson_r:.4f}  "
            f"Spearman ρ = {results.overall.spearman_rho:.4f}"
        )

    def headline_metric_labels(self) -> dict[str, str]:
        return {
            "overall_pearson_r": "Pearson r",
            "overall_spearman_rho": "Spearman ρ",
        }

    def compare_plot_title(self) -> str:
        return "DREAM MPRA — marginalized / native-position"

    def compare_plot(
        self,
        model_dirs: Mapping[str, Path],
        out_dir: Path,
    ) -> Path | None:
        """Per-stratum Pearson r + Spearman ρ across models. Reads
        ``per_stratum`` blocks from each model's ``summary.json`` (so
        no re-scoring) and renders a stacked 2-row SVG: top row Pearson,
        bottom row Spearman. Each stratum is a group on the x-axis with
        one bar per model; ``overall`` is the leftmost group.

        Returns the SVG path, or ``None`` when < 2 models have summary
        files."""
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        per_model: dict[str, dict[str, dict[str, float]]] = {}
        for name, mdir in model_dirs.items():
            sp = Path(mdir) / "summary.json"
            if not sp.exists():
                continue
            try:
                summary = json.loads(sp.read_text())
            except json.JSONDecodeError:
                continue
            entries: dict[str, dict[str, float]] = {
                "overall": {
                    "pearson_r": float(summary.get("overall_pearson_r", float("nan"))),
                    "spearman_rho": float(summary.get("overall_spearman_rho", float("nan"))),
                }
            }
            for s in summary.get("per_stratum", []):
                entries[str(s["name"])] = {
                    "pearson_r": float(s.get("pearson_r", float("nan"))),
                    "spearman_rho": float(s.get("spearman_rho", float("nan"))),
                }
            per_model[name] = entries
        if len(per_model) < 2:
            return None

        model_names = sorted(per_model.keys())
        # Stratum order: "overall" first, then the registry order from
        # `STRATA_FILES`. Drop any stratum not present in every model.
        ordered: list[str] = ["overall"] + [s for s in STRATA_FILES if all(
            s in per_model[m] for m in model_names
        )]
        ordered = [s for s in ordered if all(s in per_model[m] for m in model_names)]
        if len(ordered) < 1:
            return None

        x = np.arange(len(ordered))
        n_models = len(model_names)
        width = 0.8 / n_models
        colors = {m: model_color(m, model_names) for m in model_names}

        fig, (ax_p, ax_s) = plt.subplots(2, 1, figsize=(max(8.0, 1.1 * len(ordered)), 8))
        for metric_key, ax, ylabel in (
            ("pearson_r", ax_p, "Pearson r"),
            ("spearman_rho", ax_s, "Spearman ρ"),
        ):
            for i, m in enumerate(model_names):
                ys = [per_model[m][s][metric_key] for s in ordered]
                offset = (i - (n_models - 1) / 2) * width
                bars = ax.bar(
                    x + offset, ys, width=width, label=m, color=colors[m],
                )
                for b, v in zip(bars, ys):
                    if not np.isfinite(v):
                        continue
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        v + (0.005 if v >= 0 else -0.02),
                        f"{v:+.3f}",
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=6.5, alpha=0.8,
                    )
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(ordered, rotation=25, ha="right", fontsize=9)
            ax.set_ylabel(ylabel)
            ax.legend(loc="best", fontsize=9)

        fig.suptitle(self.compare_plot_title() + " — per-stratum correlations", fontsize=12)
        fig.tight_layout()
        out_path = out_dir / "plot.svg"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path


# ── Metric helpers ────────────────────────────────────────────


def _stratum_result(
    name: str, pred: np.ndarray, measured: np.ndarray
) -> MPRAStratumResult:
    mask = np.isfinite(pred) & np.isfinite(measured)
    p, m = pred[mask], measured[mask]
    if len(p) < 2:
        return MPRAStratumResult(name=name, n=int(mask.sum()), pearson_r=float("nan"), spearman_rho=float("nan"))
    return MPRAStratumResult(
        name=name,
        n=int(mask.sum()),
        pearson_r=float(pearsonr(p, m).statistic),
        spearman_rho=float(spearmanr(p, m).statistic),
    )


def _pair_stratum_result(
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    pairs: np.ndarray,
) -> MPRAPairStratumResult:
    """Compute raw and difference-based correlations for a pair-structured stratum.

    *pairs* is (n_pairs, 2) with columns [alt_idx, ref_idx].
    """
    alt_idx, ref_idx = pairs[:, 0], pairs[:, 1]

    # Raw per-sequence correlations (on the union of all sequences in this stratum)
    all_idx = np.unique(np.concatenate([alt_idx, ref_idx]))
    raw = _stratum_result(name, scores[all_idx], labels[all_idx])

    # Difference correlations
    pred_diff = scores[alt_idx] - scores[ref_idx]
    meas_diff = labels[alt_idx] - labels[ref_idx]
    mask = np.isfinite(pred_diff) & np.isfinite(meas_diff)
    pd_, md_ = pred_diff[mask], meas_diff[mask]
    if len(pd_) < 2:
        diff_r, diff_rho = float("nan"), float("nan")
    else:
        diff_r = float(pearsonr(pd_, md_).statistic)
        diff_rho = float(spearmanr(pd_, md_).statistic)

    return MPRAPairStratumResult(
        name=name,
        n_pairs=len(pairs),
        pearson_r=raw.pearson_r,
        spearman_rho=raw.spearman_rho,
        diff_pearson_r=diff_r,
        diff_spearman_rho=diff_rho,
    )


# ── Plotting ──────────────────────────────────────────────────


def _plot_scatter_grid(
    results: MPRAResults, out_path: Path, model_label: str
) -> None:
    import matplotlib.pyplot as plt

    all_strata = results.per_stratum
    n = len(all_strata) + 1  # +1 for overall
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    # Overall
    _scatter_one(
        axes[0], results.scores, results.labels,
        results.overall, "overall",
    )
    # Per stratum
    for i, sr in enumerate(all_strata, start=1):
        idx = results.strata_indices[sr.name]
        _scatter_one(axes[i], results.scores[idx], results.labels[idx], sr, sr.name)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    title = f"Rafi MPRA — pred vs measured"
    if model_label:
        title += f" — {model_label}"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _scatter_one(
    ax, pred: np.ndarray, measured: np.ndarray,
    sr: MPRAStratumResult, label: str,
) -> None:
    ax.scatter(measured, pred, s=2, alpha=0.3, rasterized=True)
    ax.set_xlabel("measured (el)")
    ax.set_ylabel("predicted")
    ax.set_title(
        f"{label} (n={sr.n})\n"
        f"r={sr.pearson_r:.3f}  ρ={sr.spearman_rho:.3f}",
        fontsize=9,
    )
    # Regression line
    mask = np.isfinite(pred) & np.isfinite(measured)
    if mask.sum() > 1:
        m, b = np.polyfit(measured[mask], pred[mask], 1)
        xlim = ax.get_xlim()
        xs = np.linspace(xlim[0], xlim[1], 50)
        ax.plot(xs, m * xs + b, color="red", linewidth=1, alpha=0.7)


def _plot_summary_bar(
    results: MPRAResults, out_path: Path, model_label: str
) -> None:
    import matplotlib.pyplot as plt

    names = ["overall"] + [s.name for s in results.per_stratum]
    pearson_vals = [results.overall.pearson_r] + [s.pearson_r for s in results.per_stratum]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 5))
    x = np.arange(len(names))
    ax.bar(x, pearson_vals, color="C0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r")
    ax.axhline(0, color="grey", linewidth=0.5)
    title = "Rafi MPRA — per-stratum Pearson r"
    if model_label:
        title += f" — {model_label}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
