"""MYTK integration-site × promoter position-effect benchmark.

Zero-shot prediction of mScarlet reporter expression (mean fluorescence,
fold-over-background) for a ``[promoter]-mScarlet-[terminator]`` cassette
integrated at a genomic point (native flanks intact) across 11 sites
(``ura3`` control + ``Int.1`` … ``Int.10``) × 3 promoters
(``pTDH3`` / ``pRPL18B`` / ``pREV1``) = 33 constructs.

Source: a Multiplex MoClo (MYTK) toolkit for S. cerevisiae,
ACS Synth. Biol. 2024 (doi:10.1021/acssynbio.3c00423). See
``docs/benchmarks/mytk_ints_promoter.md``.

Two metrics:

- **Primary — per-promoter Spearman ρ across the 11 sites** (3 values +
  their mean). This is the position-effect signal: holding the promoter
  fixed, does the model rank integration sites correctly? n=11 each.
- **Secondary — pooled Spearman ρ across all 33.** Dominated by the
  >100× cross-promoter strength difference (pTDH3 ≫ pRPL18B ≫ pREV1), so
  any non-broken model scores high; reported for context, not as the
  headline. (Per-*site* ρ across the 3 promoters is intentionally not
  reported: n=3 is statistically degenerate and measures only the
  trivial promoter-strength ranking — see the spec.)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from yeastbench.adapters.protocols import (
    PromoterIntegrationConstruct,
    PromoterIntegrationExpressionPredictor,
)
from yeastbench.benchmarks._metrics import MPRAStratumResult
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo

LOCUS_ID_COL = "locus_id"
PROMOTER_COL = "promoter"
CHROM_COL = "chrom"
COORD_COL = "integration_coord"
LABEL_COL = "mean"
POOLED = "pooled"


@dataclass(frozen=True)
class MytkResults:
    scores: np.ndarray            # (N=33,) predicted; NaN for unscoreable
    labels: np.ndarray            # (N=33,) measured mean fluorescence
    promoters: np.ndarray         # (N=33,) promoter id per row
    locus_ids: list[str]          # (N=33,)
    per_promoter: dict[str, MPRAStratumResult]  # promoter id → result (n=11)
    pooled: MPRAStratumResult                    # across all rows (n=33)

    @property
    def mean_per_promoter_spearman(self) -> float:
        rhos = [r.spearman_rho for r in self.per_promoter.values()]
        rhos = [r for r in rhos if np.isfinite(r)]
        return float(np.mean(rhos)) if rhos else float("nan")


def _scored(name: str, pred: np.ndarray, measured: np.ndarray) -> MPRAStratumResult:
    mask = np.isfinite(pred) & np.isfinite(measured)
    p, m = pred[mask], measured[mask]
    if len(p) < 2 or np.unique(p).size < 2 or np.unique(m).size < 2:
        return MPRAStratumResult(name=name, n=int(mask.sum()),
                                 pearson_r=float("nan"), spearman_rho=float("nan"))
    return MPRAStratumResult(
        name=name,
        n=int(mask.sum()),
        pearson_r=float(pearsonr(p, m).statistic),
        spearman_rho=float(spearmanr(p, m).statistic),
    )


def _all_metrics(
    scores: np.ndarray, labels: np.ndarray, promoters: np.ndarray,
) -> tuple[dict[str, MPRAStratumResult], MPRAStratumResult]:
    per_promoter = {
        prom: _scored(prom, scores[promoters == prom], labels[promoters == prom])
        for prom in sorted(set(promoters.tolist()))
    }
    pooled = _scored(POOLED, scores, labels)
    return per_promoter, pooled


class MytkBenchmark(
    Benchmark[PromoterIntegrationExpressionPredictor, MytkResults]
):
    adapter_protocol: ClassVar[type] = PromoterIntegrationExpressionPredictor

    def __init__(
        self,
        data_path: Path,
        fasta_path: Path,
        info: BenchmarkInfo,
    ) -> None:
        self.data_path = Path(data_path)
        self._fasta_path = Path(fasta_path)
        self.info = info

        df = pd.read_csv(self.data_path, sep="\t")
        for col in (LOCUS_ID_COL, PROMOTER_COL, CHROM_COL, COORD_COL, LABEL_COL):
            assert col in df.columns, f"{col} missing from {self.data_path}"

        self.locus_ids: list[str] = df[LOCUS_ID_COL].astype(str).tolist()
        self.promoters: np.ndarray = df[PROMOTER_COL].astype(str).to_numpy()
        self.labels: np.ndarray = df[LABEL_COL].to_numpy(dtype=float)
        self.constructs: list[PromoterIntegrationConstruct] = [
            PromoterIntegrationConstruct(
                locus_id=str(r[LOCUS_ID_COL]),
                chrom=str(r[CHROM_COL]),
                integration_coord=int(r[COORD_COL]),
                promoter=str(r[PROMOTER_COL]),
            )
            for _, r in df.iterrows()
        ]

    @property
    def fasta_path(self) -> Path:
        return self._fasta_path

    def evaluate(
        self, adapter: PromoterIntegrationExpressionPredictor
    ) -> MytkResults:
        scores = np.asarray(
            adapter.predict_integrated_expressions(self.constructs), dtype=float
        )
        assert len(scores) == len(self.labels), (
            f"adapter returned {len(scores)} scores for {len(self.labels)} constructs"
        )
        per_promoter, pooled = _all_metrics(scores, self.labels, self.promoters)
        return MytkResults(
            scores=scores,
            labels=self.labels,
            promoters=self.promoters,
            locus_ids=self.locus_ids,
            per_promoter=per_promoter,
            pooled=pooled,
        )

    def plot(self, results: MytkResults, out_dir: Path) -> None:
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        title_model = out_dir.name.split("__")[0] if "__" in out_dir.name else ""

        promoters = sorted(results.per_promoter)
        panels = promoters + [POOLED]
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
        if len(panels) == 1:
            axes = [axes]

        for ax, name in zip(axes, panels):
            if name == POOLED:
                sel = np.ones(len(results.scores), dtype=bool)
                res = results.pooled
            else:
                sel = (results.promoters == name)
                res = results.per_promoter[name]
            mask = sel & np.isfinite(results.scores) & np.isfinite(results.labels)
            m, p = results.labels[mask], results.scores[mask]
            ax.scatter(m, p, s=20, alpha=0.7, rasterized=True)
            if len(m) > 1:
                a, b = np.polyfit(m, p, 1)
                xs = np.linspace(m.min(), m.max(), 50)
                ax.plot(xs, a * xs + b, color="red", linewidth=1, alpha=0.8)
            ax.set_xlabel("measured mScarlet (fold over background)")
            ax.set_ylabel("predicted expression")
            ax.set_title(f"{name}  n={res.n}  r={res.pearson_r:.3f}  ρ={res.spearman_rho:.3f}")

        title = "MYTK integration-site × promoter"
        if title_model:
            title += f" — {title_model}"
        title += (
            f"\nprimary: mean per-promoter ρ = "
            f"{results.mean_per_promoter_spearman:.3f}   "
            f"(secondary: pooled ρ = {results.pooled.spearman_rho:.3f})"
        )
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(out_dir / "scatter.png", dpi=150)
        plt.close(fig)

    def save_results(self, results: MytkResults, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "scores.npy", results.scores)
        np.save(out_dir / "labels.npy", results.labels)
        (out_dir / "meta.json").write_text(json.dumps({
            "locus_ids": results.locus_ids,
            "promoters": results.promoters.tolist(),
        }, indent=2))

    def load_results(self, out_dir: Path) -> MytkResults:
        out_dir = Path(out_dir)
        scores = np.load(out_dir / "scores.npy")
        labels = np.load(out_dir / "labels.npy")
        meta = json.loads((out_dir / "meta.json").read_text())
        promoters = np.asarray(meta["promoters"])
        per_promoter, pooled = _all_metrics(scores, labels, promoters)
        return MytkResults(
            scores=scores,
            labels=labels,
            promoters=promoters,
            locus_ids=meta["locus_ids"],
            per_promoter=per_promoter,
            pooled=pooled,
        )

    def summary_dict(self, results: MytkResults) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_rows_total": int(len(results.scores)),
            "mean_per_promoter_spearman_rho": results.mean_per_promoter_spearman,
            "pooled_spearman_rho": results.pooled.spearman_rho,
            "pooled_pearson_r": results.pooled.pearson_r,
            "pooled_n": results.pooled.n,
        }
        for prom, res in results.per_promoter.items():
            out[f"{prom}_spearman_rho"] = res.spearman_rho
            out[f"{prom}_pearson_r"] = res.pearson_r
            out[f"{prom}_n"] = res.n
        return out

    def headline(self, results: MytkResults) -> str:
        per = "  ".join(
            f"{prom} ρ={res.spearman_rho:.3f}"
            for prom, res in sorted(results.per_promoter.items())
        )
        return (
            f"Primary: mean per-promoter ρ = "
            f"{results.mean_per_promoter_spearman:.4f}  [{per}]  |  "
            f"Secondary: pooled ρ = {results.pooled.spearman_rho:.4f} "
            f"(n = {results.pooled.n})"
        )

    def headline_metric_labels(self) -> dict[str, str]:
        return {
            "mean_per_promoter_spearman_rho": "mean per-promoter ρ",
            "pooled_spearman_rho": "pooled ρ",
        }

    def compare_plot_title(self) -> str:
        return "MYTK integration-site × promoter position effects"
