"""Wu et al. RFP-insertion position-effect benchmark.

Zero-shot prediction of OD600-normalised RFP plate-reader intensity for
one constant cassette integrated at 1044 single-ORF deletion loci.  Only
the genomic insertion site varies — this is a position-effect probe (see
``benchmarks/wu_rfpins.md``).

Primary metric: Pearson r + Spearman ρ between predicted and measured
relative fluorescence.  Secondary: agreement with the paper's five fixed
absolute-cutoff classes, scored scale-free by rank-matching predictions
to the ground-truth class sizes (quadratic-weighted Cohen's κ is the
primary class statistic — the bins are ordinal and very imbalanced).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from yeastbench.adapters._genome import parse_gene_annotations
from yeastbench.adapters._wu_scaffold import WuLocus, resolve_loci
from yeastbench.adapters.protocols import CassetteExpressionPredictor
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo

# Paper's five fixed absolute-cutoff classes (Wu et al. Fig. 1b).
# np.digitize(value, CLASS_EDGES) → 0..4.
CLASS_EDGES: tuple[float, ...] = (5.0, 6.0, 7.0, 8.0)
CLASS_NAMES: tuple[str, ...] = (
    "extreme_low", "low", "moderate", "high", "extreme_high",
)
CLASS_COLORS: tuple[str, ...] = ("red", "gold", "green", "violet", "blue")

LABEL_COL = "Relative_Fluorescence_Average"
ERROR_COL = "Relative_Fluorescence_Error"
ORF_COL = "ORF_name"


@dataclass(frozen=True)
class WuResults:
    scores: np.ndarray            # (N,) predicted; NaN for unresolved/uncontextualised loci
    labels: np.ndarray            # (N,) measured relative fluorescence
    gene_ids: list[str]           # (N,) ORF systematic names, row-aligned
    dropped_ids: list[str]        # ORFs not resolvable in the GTF
    n_scored: int
    pearson_r: float
    spearman_rho: float
    # 5-class (computed on the scored subset)
    accuracy: float
    macro_f1: float
    qwk: float                    # quadratic-weighted Cohen's κ
    confusion: np.ndarray         # (5, 5) rows = true class, cols = predicted


def _classify_absolute(values: np.ndarray) -> np.ndarray:
    """Paper's fixed-cutoff class index 0..4 for each measured value."""
    return np.digitize(values, CLASS_EDGES).astype(int)


def _rank_match_classes(
    pred: np.ndarray, true_cls: np.ndarray
) -> np.ndarray:
    """Assign predictions to classes 0..4 by rank so the predicted class
    sizes equal the ground-truth class sizes (scale-free — a zero-shot
    model's outputs are not on the assay's absolute scale)."""
    sizes = np.array([(true_cls == c).sum() for c in range(len(CLASS_NAMES))])
    order = np.argsort(pred, kind="stable")
    pred_cls = np.empty(len(pred), dtype=int)
    start = 0
    for c, n in enumerate(sizes):
        pred_cls[order[start : start + n]] = c
        start += n
    return pred_cls


def _metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(scores) & np.isfinite(labels)
    p, m = scores[mask], labels[mask]
    n = int(mask.sum())
    if n < 2:
        return dict(
            n_scored=n, pearson_r=float("nan"), spearman_rho=float("nan"),
            accuracy=float("nan"), macro_f1=float("nan"), qwk=float("nan"),
            confusion=np.zeros((5, 5), dtype=int),
        )
    true_cls = _classify_absolute(m)
    pred_cls = _rank_match_classes(p, true_cls)
    labels5 = list(range(len(CLASS_NAMES)))
    return dict(
        n_scored=n,
        pearson_r=float(pearsonr(p, m).statistic),
        spearman_rho=float(spearmanr(p, m).statistic),
        accuracy=float((pred_cls == true_cls).mean()),
        macro_f1=float(
            f1_score(
                true_cls, pred_cls, labels=labels5,
                average="macro", zero_division=0,
            )
        ),
        qwk=float(
            cohen_kappa_score(true_cls, pred_cls, labels=labels5, weights="quadratic")
        ),
        confusion=confusion_matrix(true_cls, pred_cls, labels=labels5),
    )


class RFPInsertionBenchmark(Benchmark[CassetteExpressionPredictor, WuResults]):
    adapter_protocol: ClassVar[type] = CassetteExpressionPredictor

    def __init__(
        self,
        cassette_seq: Path,
        labels_path: Path,
        fasta_path: Path,
        gtf_path: Path,
        info: BenchmarkInfo,
    ) -> None:
        self.cassette_seq = Path(cassette_seq)
        self.labels_path = Path(labels_path)
        self._fasta_path = Path(fasta_path)
        self._gtf_path = Path(gtf_path)
        self.info = info

        df = pd.read_csv(self.labels_path)
        for col in (ORF_COL, LABEL_COL):
            assert col in df.columns, f"{col} missing from {self.labels_path}"
        self.gene_ids: list[str] = df[ORF_COL].astype(str).tolist()
        self.labels: np.ndarray = df[LABEL_COL].to_numpy(dtype=float)
        self.errors: np.ndarray = (
            df[ERROR_COL].to_numpy(dtype=float)
            if ERROR_COL in df.columns
            else np.full(len(df), np.nan)
        )

        genes = parse_gene_annotations(self._gtf_path)
        self.loci, self.dropped = resolve_loci(self.gene_ids, genes)

    @property
    def fasta_path(self) -> Path:
        return self._fasta_path

    @property
    def gtf_path(self) -> Path:
        return self._gtf_path

    def evaluate(self, adapter: CassetteExpressionPredictor) -> WuResults:
        resolved_idx = [i for i, lc in enumerate(self.loci) if lc is not None]
        sub_loci: list[WuLocus] = [self.loci[i] for i in resolved_idx]  # type: ignore[misc]

        scores = np.full(len(self.labels), np.nan, dtype=np.float64)
        if sub_loci:
            sub_scores = np.asarray(
                adapter.predict_expressions(sub_loci), dtype=float
            )
            assert len(sub_scores) == len(sub_loci), (
                f"adapter returned {len(sub_scores)} scores for "
                f"{len(sub_loci)} loci"
            )
            scores[resolved_idx] = sub_scores

        m = _metrics(scores, self.labels)
        return WuResults(
            scores=scores,
            labels=self.labels,
            gene_ids=self.gene_ids,
            dropped_ids=self.dropped,
            **m,
        )

    def plot(self, results: WuResults, out_dir: Path) -> None:
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        title_model = out_dir.name.split("__")[0] if "__" in out_dir.name else ""

        mask = np.isfinite(results.scores) & np.isfinite(results.labels)
        pred = results.scores[mask]
        meas = results.labels[mask]

        # (1) scatter pred vs measured
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(meas, pred, s=8, alpha=0.4, rasterized=True)
        if mask.sum() > 1:
            a, b = np.polyfit(meas, pred, 1)
            xs = np.linspace(meas.min(), meas.max(), 50)
            ax.plot(xs, a * xs + b, color="red", linewidth=1, alpha=0.8)
        ax.set_xlabel("measured relative fluorescence (RFP / OD600)")
        ax.set_ylabel("predicted (mCherry-CDS track readout)")
        title = "Wu RFP-insertion position effect"
        if title_model:
            title += f" — {title_model}"
        ax.set_title(
            f"{title}\nn = {results.n_scored}  "
            f"r = {results.pearson_r:.3f}  ρ = {results.spearman_rho:.3f}"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "scatter.png", dpi=150)
        plt.close(fig)

        # (2) Fig. 1b-style measured histogram, coloured by the 5 classes
        fig, ax = plt.subplots(figsize=(7, 4))
        edges = [meas.min()] + list(CLASS_EDGES) + [max(meas.max(), 13.0)]
        for c, color in enumerate(CLASS_COLORS):
            lo, hi = edges[c], edges[c + 1]
            sel = (meas >= lo) & (meas < hi if c < 4 else meas >= lo)
            ax.hist(
                meas[sel], bins=np.arange(np.floor(meas.min()), 14.0, 0.5),
                color=color, alpha=0.8, label=CLASS_NAMES[c],
            )
        ax.set_xlabel("relative fluorescence intensity")
        ax.set_ylabel("number of loci")
        ax.set_title("Measured intensity by Wu class" +
                     (f" — {title_model}" if title_model else ""))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "measured_classes.png", dpi=150)
        plt.close(fig)

    def save_results(self, results: WuResults, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "scores.npy", results.scores)
        np.save(out_dir / "labels.npy", results.labels)
        (out_dir / "loci.json").write_text(
            json.dumps(
                {"gene_ids": results.gene_ids, "dropped_ids": results.dropped_ids},
                indent=2,
            )
        )

    def load_results(self, out_dir: Path) -> WuResults:
        out_dir = Path(out_dir)
        scores = np.load(out_dir / "scores.npy")
        labels = np.load(out_dir / "labels.npy")
        meta = json.loads((out_dir / "loci.json").read_text())
        m = _metrics(scores, labels)
        return WuResults(
            scores=scores,
            labels=labels,
            gene_ids=meta["gene_ids"],
            dropped_ids=meta["dropped_ids"],
            **m,
        )

    def summary_dict(self, results: WuResults) -> dict[str, Any]:
        return {
            "n_rows_total": int(len(results.scores)),
            "n_rows_scored": results.n_scored,
            "n_dropped_unresolved": len(results.dropped_ids),
            "pearson_r": results.pearson_r,
            "spearman_rho": results.spearman_rho,
            "class_accuracy": results.accuracy,
            "class_macro_f1": results.macro_f1,
            "class_qwk": results.qwk,
            "confusion": results.confusion.tolist(),
        }

    def headline(self, results: WuResults) -> str:
        return (
            f"Pearson r = {results.pearson_r:.4f}  "
            f"Spearman ρ = {results.spearman_rho:.4f}  "
            f"κ_w = {results.qwk:.3f}  (n = {results.n_scored})"
        )
