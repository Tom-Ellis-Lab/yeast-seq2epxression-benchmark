"""Shalem MPRA terminator benchmark (marginalized / native-position).

Evaluates zero-shot prediction of 3′-end MPRA expression for ~14 k
designed oligos by marginalizing predicted logSED across 22 native host
genes.  v1 reports overall Pearson and Spearman only — per-stratum
breakdown by the ``Description`` column's ``SetName`` is deferred to v2.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from yeastbench.adapters.protocols import TerminatorMarginalizedExpressionPredictor
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo
from yeastbench.benchmarks.mpra import MPRAStratumResult


RAW_TABLE_NAME = "segal_2015.tsv"
OLIGO_LEN = 150


@dataclass(frozen=True)
class ShalemResults:
    scores: np.ndarray    # (N=14956,) predicted expression — NaN for adapter-returned NaN (not sampled etc.)
    labels: np.ndarray    # (N=14956,) measured Expression with NaN for the 784 non-labelled rows
    overall: MPRAStratumResult


class ShalemMPRAMarginalizedBenchmark(
    Benchmark[TerminatorMarginalizedExpressionPredictor, ShalemResults]
):
    adapter_protocol: ClassVar[type] = TerminatorMarginalizedExpressionPredictor

    def __init__(
        self,
        data_path: Path,
        fasta_path: Path,
        gtf_path: Path,
        info: BenchmarkInfo,
    ) -> None:
        self.data_path = Path(data_path)
        self._fasta_path = Path(fasta_path)
        self._gtf_path = Path(gtf_path)
        self.info = info

        df = pd.read_csv(self.data_path, sep="\t")
        assert "Oligo Sequence" in df.columns, df.columns.tolist()
        assert "Expression" in df.columns
        assert (df["Oligo Sequence"].str.len() == OLIGO_LEN).all(), (
            "some oligos are not 150 bp"
        )
        self.sequences: list[str] = df["Oligo Sequence"].tolist()
        self.labels: np.ndarray = df["Expression"].to_numpy(dtype=float)

    @property
    def fasta_path(self) -> Path:
        return self._fasta_path

    @property
    def gtf_path(self) -> Path:
        return self._gtf_path

    def evaluate(
        self, adapter: TerminatorMarginalizedExpressionPredictor
    ) -> ShalemResults:
        scores = np.asarray(
            adapter.predict_terminator_marginalized(self.sequences), dtype=float
        )
        assert len(scores) == len(self.labels)

        overall = _scored_pearson("overall", scores, self.labels)
        return ShalemResults(scores=scores, labels=self.labels, overall=overall)

    def plot(self, results: ShalemResults, out_dir: Path) -> None:
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        title_model = out_dir.name.split("__")[0] if "__" in out_dir.name else ""

        mask = np.isfinite(results.scores) & np.isfinite(results.labels)
        pred = results.scores[mask]
        meas = results.labels[mask]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(meas, pred, s=3, alpha=0.35, rasterized=True)
        if mask.sum() > 1:
            m, b = np.polyfit(meas, pred, 1)
            xs = np.linspace(meas.min(), meas.max(), 50)
            ax.plot(xs, m * xs + b, color="red", linewidth=1, alpha=0.8)
        ax.set_xlabel("measured Expression")
        ax.set_ylabel("predicted (mean logSED across host genes)")
        title = "Shalem MPRA terminator (marginalized)"
        if title_model:
            title += f" — {title_model}"
        ax.set_title(
            f"{title}\n"
            f"n = {results.overall.n}  "
            f"r = {results.overall.pearson_r:.3f}  "
            f"ρ = {results.overall.spearman_rho:.3f}"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "scatter.png", dpi=150)
        plt.close(fig)

    def save_results(self, results: ShalemResults, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "scores.npy", results.scores)
        np.save(out_dir / "labels.npy", results.labels)

    def load_results(self, out_dir: Path) -> ShalemResults:
        out_dir = Path(out_dir)
        scores = np.load(out_dir / "scores.npy")
        labels = np.load(out_dir / "labels.npy")
        overall = _scored_pearson("overall", scores, labels)
        return ShalemResults(scores=scores, labels=labels, overall=overall)

    def summary_dict(self, results: ShalemResults) -> dict[str, Any]:
        return {
            "n_rows_total": int(len(results.scores)),
            "n_rows_scored": int(results.overall.n),
            "overall_pearson_r": results.overall.pearson_r,
            "overall_spearman_rho": results.overall.spearman_rho,
        }

    def headline(self, results: ShalemResults) -> str:
        return (
            f"overall Pearson r = {results.overall.pearson_r:.4f}  "
            f"Spearman \u03c1 = {results.overall.spearman_rho:.4f}  "
            f"(n = {results.overall.n})"
        )


def _scored_pearson(
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
