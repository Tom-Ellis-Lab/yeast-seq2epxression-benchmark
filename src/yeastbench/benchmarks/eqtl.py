from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from yeastbench.adapters.protocols import Variant, VariantEffectScorer
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo


# Locked in the spec: see benchmarks/caudal_eqtl.md § Evaluation protocol.
DISTANCE_BINS: tuple[tuple[int, int], ...] = (
    (0, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 8000),
    (8000, 16000),
    (16000, 30000),
)
CLOSE_ONLY_THRESHOLD_BP = 2000


@dataclass(frozen=True)
class EQTLIterationResult:
    name: str
    scores: np.ndarray            # signed logSED_agg, shape (2N,)
    labels: np.ndarray            # 1/0 alternating pos/neg, shape (2N,)
    pairs: pd.DataFrame           # pair-level metadata (pair_id, distances, ...)
    auroc_signed: float
    auprc_signed: float
    auroc_abs: float              # |score| is the classification signal
    auprc_abs: float


@dataclass(frozen=True)
class EQTLResults:
    per_iter: list[EQTLIterationResult]

    def _agg(self, key: str) -> tuple[float, float]:
        xs = np.asarray([getattr(r, key) for r in self.per_iter], dtype=float)
        mean = float(xs.mean())
        sem = float(xs.std(ddof=1) / np.sqrt(xs.size)) if xs.size > 1 else 0.0
        return mean, sem

    @property
    def mean_auroc(self) -> float:
        return self._agg("auroc_abs")[0]

    @property
    def sem_auroc(self) -> float:
        return self._agg("auroc_abs")[1]

    @property
    def mean_auprc(self) -> float:
        return self._agg("auprc_abs")[0]

    @property
    def sem_auprc(self) -> float:
        return self._agg("auprc_abs")[1]


class EQTLClassificationBenchmark(Benchmark[VariantEffectScorer, EQTLResults]):
    adapter_protocol: ClassVar[type] = VariantEffectScorer

    def __init__(self, distribution_dir: Path, info: BenchmarkInfo) -> None:
        self.distribution_dir = Path(distribution_dir)
        self.info = info
        self.iteration_files = sorted(self.distribution_dir.glob("negset_*.tsv"))

    @property
    def fasta_path(self) -> Path:
        return self.distribution_dir / "reference" / "R64-1-1.fa"

    @property
    def gtf_path(self) -> Path:
        return self.distribution_dir / "reference" / "R64-1-1.115.gtf"

    def evaluate(self, adapter: VariantEffectScorer) -> EQTLResults:
        per_iter: list[EQTLIterationResult] = []
        for tsv in self.iteration_files:
            pairs = pd.read_csv(
                tsv, sep="\t", dtype={"pos_chrom": str, "neg_chrom": str}
            )
            variants: list[Variant] = []
            labels: list[int] = []
            meta_rows: list[dict] = []
            for row in pairs.itertuples():
                variants.append(
                    Variant(
                        row.pos_chrom, row.pos_pos, row.pos_ref, row.pos_alt, row.pos_gene
                    )
                )
                variants.append(
                    Variant(
                        row.neg_chrom, row.neg_pos, row.neg_ref, row.neg_alt, row.neg_gene
                    )
                )
                labels.extend([1, 0])
                meta_rows.append(
                    {
                        "pair_id": int(row.pair_id),
                        "pos_distance_to_tss": int(row.pos_distance_to_tss),
                        "neg_distance_to_tss": int(row.neg_distance_to_tss),
                    }
                )
            scores = np.asarray(adapter.score_variants(variants), dtype=float)
            labels_a = np.asarray(labels)
            per_iter.append(
                EQTLIterationResult(
                    name=tsv.stem,
                    scores=scores,
                    labels=labels_a,
                    pairs=pd.DataFrame(meta_rows),
                    auroc_signed=float(roc_auc_score(labels_a, scores)),
                    auprc_signed=float(average_precision_score(labels_a, scores)),
                    auroc_abs=float(roc_auc_score(labels_a, np.abs(scores))),
                    auprc_abs=float(average_precision_score(labels_a, np.abs(scores))),
                )
            )
        return EQTLResults(per_iter=per_iter)

    def plot(self, results: EQTLResults, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        title_model = out_dir.name.split("__")[0] if "__" in out_dir.name else ""
        _plot_roc_pr(
            results.per_iter,
            out_dir / "primary_roc_pr.png",
            subset=None,
            title=f"{self.info.name} — full set (|score|)"
            + (f" — {title_model}" if title_model else ""),
        )
        _plot_roc_pr(
            results.per_iter,
            out_dir / "close_only_roc_pr.png",
            subset=_close_only_mask,
            title=f"{self.info.name} — pos_distance_to_tss ≤ {CLOSE_ONLY_THRESHOLD_BP} bp (|score|)"
            + (f" — {title_model}" if title_model else ""),
        )
        _plot_distance_stratified(
            results.per_iter,
            out_dir / "distance_stratified.png",
            title=f"{self.info.name} — |score| AUROC / AUPRC by distance-to-TSS bin"
            + (f" — {title_model}" if title_model else ""),
        )

    def save_results(self, results: EQTLResults, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in results.per_iter:
            np.save(out_dir / f"{r.name}_scores.npy", r.scores)
            np.save(out_dir / f"{r.name}_labels.npy", r.labels)
            r.pairs.to_csv(out_dir / f"{r.name}_pairs.tsv", sep="\t", index=False)

    def load_results(self, out_dir: Path) -> EQTLResults:
        out_dir = Path(out_dir)
        per_iter: list[EQTLIterationResult] = []
        for sp in sorted(out_dir.glob("*_scores.npy")):
            name = sp.name.replace("_scores.npy", "")
            scores = np.load(sp)
            labels = np.load(out_dir / f"{name}_labels.npy")
            pairs = pd.read_csv(out_dir / f"{name}_pairs.tsv", sep="\t")
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
        return EQTLResults(per_iter=per_iter)

    def summary_dict(self, results: EQTLResults) -> dict[str, Any]:
        return {
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

    def headline(self, results: EQTLResults) -> str:
        return (
            f"|score| AUROC {results.mean_auroc:.4f} \u00b1 {results.sem_auroc:.4f}  "
            f"AUPRC {results.mean_auprc:.4f} \u00b1 {results.sem_auprc:.4f}"
        )


# ──────────────────────────────────────────────────────────────
# Plotting internals
# ──────────────────────────────────────────────────────────────


PairMaskFn = Callable[[pd.DataFrame], np.ndarray]


def _close_only_mask(pairs: pd.DataFrame) -> np.ndarray:
    return (pairs["pos_distance_to_tss"].to_numpy() <= CLOSE_ONLY_THRESHOLD_BP)


def _bin_mask(pairs: pd.DataFrame, lo: int, hi: int) -> np.ndarray:
    d = pairs["pos_distance_to_tss"].to_numpy()
    return (d > lo) & (d <= hi)


def _pair_mask_to_variant_mask(pair_mask: np.ndarray) -> np.ndarray:
    """Pair-level mask (length N) → variant-level mask (length 2N, pos/neg alternating)."""
    return np.repeat(pair_mask, 2)


def _iteration_subset(
    r: EQTLIterationResult, subset: PairMaskFn | None
) -> tuple[np.ndarray, np.ndarray]:
    if subset is None:
        return np.abs(r.scores), r.labels
    pair_mask = subset(r.pairs)
    var_mask = _pair_mask_to_variant_mask(pair_mask)
    return np.abs(r.scores[var_mask]), r.labels[var_mask]


def _interp_roc(
    scores: np.ndarray, labels: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    fpr, tpr, _ = roc_curve(labels, scores)
    return np.interp(grid, fpr, tpr)


def _interp_pr(
    scores: np.ndarray, labels: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    precision, recall, _ = precision_recall_curve(labels, scores)
    # sklearn returns recall descending from 1→0; sort ascending for np.interp.
    order = np.argsort(recall)
    return np.interp(grid, recall[order], precision[order])


def _mean_sem(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and SEM across rows (iterations)."""
    mean = arr.mean(axis=0)
    if arr.shape[0] < 2:
        return mean, np.zeros_like(mean)
    return mean, arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])


def _plot_roc_pr(
    per_iter: list[EQTLIterationResult],
    out_path: Path,
    subset: PairMaskFn | None,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    grid = np.linspace(0.0, 1.0, 101)
    roc_curves: list[np.ndarray] = []
    pr_curves: list[np.ndarray] = []
    aurocs: list[float] = []
    auprcs: list[float] = []
    ns: list[int] = []
    base_rates: list[float] = []

    for r in per_iter:
        sub_scores, sub_labels = _iteration_subset(r, subset)
        if sub_labels.size == 0 or len(set(sub_labels)) < 2:
            continue
        roc_curves.append(_interp_roc(sub_scores, sub_labels, grid))
        pr_curves.append(_interp_pr(sub_scores, sub_labels, grid))
        aurocs.append(float(roc_auc_score(sub_labels, sub_scores)))
        auprcs.append(float(average_precision_score(sub_labels, sub_scores)))
        ns.append(int(sub_labels.size // 2))
        base_rates.append(float(sub_labels.mean()))

    if not roc_curves:
        return  # nothing to plot (e.g., filter left the iteration empty)

    roc_arr = np.stack(roc_curves, axis=0)
    pr_arr = np.stack(pr_curves, axis=0)
    roc_mean, roc_sem = _mean_sem(roc_arr)
    pr_mean, pr_sem = _mean_sem(pr_arr)

    auroc_mean = float(np.mean(aurocs))
    auroc_sem = (
        float(np.std(aurocs, ddof=1) / np.sqrt(len(aurocs))) if len(aurocs) > 1 else 0.0
    )
    auprc_mean = float(np.mean(auprcs))
    auprc_sem = (
        float(np.std(auprcs, ddof=1) / np.sqrt(len(auprcs))) if len(auprcs) > 1 else 0.0
    )
    n = int(np.mean(ns))
    base_rate = float(np.mean(base_rates))

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 5))

    # --- ROC
    ax_roc.plot(grid, roc_mean, color="C0", label=f"AUROC = {auroc_mean:.3f} ± {auroc_sem:.3f}")
    ax_roc.fill_between(
        grid, roc_mean - roc_sem, roc_mean + roc_sem, color="C0", alpha=0.25,
        label="±1 SEM across iterations",
    )
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", label="random")
    ax_roc.plot([0, 0, 1], [0, 1, 1], linestyle=":", color="black", label="perfect")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_aspect("equal", adjustable="box")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.set_title("ROC")

    # --- PR
    ax_pr.plot(grid, pr_mean, color="C1", label=f"AUPRC = {auprc_mean:.3f} ± {auprc_sem:.3f}")
    ax_pr.fill_between(
        grid, pr_mean - pr_sem, pr_mean + pr_sem, color="C1", alpha=0.25,
        label="±1 SEM across iterations",
    )
    ax_pr.axhline(base_rate, linestyle="--", color="grey", label=f"random (base rate = {base_rate:.2f})")
    ax_pr.plot([0, 1, 1], [1, 1, base_rate], linestyle=":", color="black", label="perfect")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.set_aspect("equal", adjustable="box")
    ax_pr.legend(loc="lower left", fontsize=8)
    ax_pr.set_title("Precision–Recall")

    fig.suptitle(f"{title}   (N={n} pairs × {len(aurocs)} iterations)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_distance_stratified(
    per_iter: list[EQTLIterationResult],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    n_iter = len(per_iter)
    n_bins = len(DISTANCE_BINS)
    auroc_matrix = np.full((n_iter, n_bins), np.nan)
    auprc_matrix = np.full((n_iter, n_bins), np.nan)
    n_per_bin: list[int] = []

    for b_idx, (lo, hi) in enumerate(DISTANCE_BINS):
        bin_ns: list[int] = []
        for i_idx, r in enumerate(per_iter):
            pair_mask = _bin_mask(r.pairs, lo, hi)
            bin_ns.append(int(pair_mask.sum()))
            if pair_mask.sum() < 5:
                continue
            var_mask = _pair_mask_to_variant_mask(pair_mask)
            sub_scores = np.abs(r.scores[var_mask])
            sub_labels = r.labels[var_mask]
            if len(set(sub_labels)) < 2:
                continue
            auroc_matrix[i_idx, b_idx] = roc_auc_score(sub_labels, sub_scores)
            auprc_matrix[i_idx, b_idx] = average_precision_score(sub_labels, sub_scores)
        n_per_bin.append(int(round(float(np.mean(bin_ns)))))

    auroc_mean = np.nanmean(auroc_matrix, axis=0)
    auprc_mean = np.nanmean(auprc_matrix, axis=0)
    n_valid = (~np.isnan(auroc_matrix)).sum(axis=0)
    with np.errstate(invalid="ignore"):
        auroc_sem = np.nanstd(auroc_matrix, axis=0, ddof=1) / np.sqrt(
            np.maximum(n_valid, 1)
        )
        auprc_sem = np.nanstd(auprc_matrix, axis=0, ddof=1) / np.sqrt(
            np.maximum(n_valid, 1)
        )

    bin_labels = [
        f"{lo}–{hi}\n(n={n})" for (lo, hi), n in zip(DISTANCE_BINS, n_per_bin)
    ]
    x = np.arange(n_bins)

    fig, (ax_auroc, ax_auprc) = plt.subplots(1, 2, figsize=(13, 5))

    ax_auroc.bar(x, auroc_mean, yerr=auroc_sem, color="C0", capsize=3, alpha=0.85)
    ax_auroc.axhline(0.5, linestyle="--", color="grey", label="random")
    ax_auroc.set_xticks(x)
    ax_auroc.set_xticklabels(bin_labels, rotation=0, fontsize=8)
    ax_auroc.set_ylabel("|score| AUROC")
    ax_auroc.set_ylim(0.4, 1.0)
    ax_auroc.set_title("AUROC by distance-to-TSS bin")
    ax_auroc.legend(loc="upper right", fontsize=8)

    ax_auprc.bar(x, auprc_mean, yerr=auprc_sem, color="C1", capsize=3, alpha=0.85)
    ax_auprc.axhline(0.5, linestyle="--", color="grey", label="random")
    ax_auprc.set_xticks(x)
    ax_auprc.set_xticklabels(bin_labels, rotation=0, fontsize=8)
    ax_auprc.set_ylabel("|score| AUPRC")
    ax_auprc.set_ylim(0.4, 1.0)
    ax_auprc.set_title("AUPRC by distance-to-TSS bin")
    ax_auprc.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{title}   (error bars = ±1 SEM across {n_iter} iterations)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
