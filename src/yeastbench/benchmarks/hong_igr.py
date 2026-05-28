"""Hong et al. IGR-insertion position-effect benchmark.

Zero-shot prediction of mCherry mean fluorescence (normalized to a
reference site ``IntTrain92``) of one constant
``TDH3p-mCherry-ADH1t`` cassette integrated at 150 intergenic regions
in *S. cerevisiae*. Only the integration site varies — this is a
position-effect probe, the IGR-flavor sibling of the Wu RFP-insertion
benchmark (see ``benchmarks/hong_igr.md``).

Two co-primary metrics, both reported as Spearman ρ on the IntProp
held-out tier:

- **Primary**: RNA-seq × mCherry-CDS readout (RNA-seq T0 for Shorkie,
  all-+ tracks for Yorzoi). Fixed per model; apples-to-apples across
  models; tests the biologically principled causal chain (mRNA →
  fluorescence).
- **IntTrain-fitted IntProp ρ**: same evaluation tier (IntProp) as
  Primary, but the (track group × readout region) is the one that
  maximises signed Spearman ρ on the IntTrain set. A soft form of
  supervised feature engineering — diagnoses the model's *upper
  bound* given its track inventory and any biologically-meaningful
  signal at the immediate native flank. Adapters opt in by
  implementing ``predict_diagnostic_readouts(loci) → dict``;
  adapters without it get only Primary reported.

See ``benchmarks/hong_igr.md`` for the full design + caveats about
the 0.847 cross-promoter ceiling vs noise-limited within-promoter
ceiling.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from yeastbench.adapters._hong_scaffold import HongLocus
from yeastbench.adapters.protocols import IGRInsertionExpressionPredictor
from yeastbench.benchmarks.base import (
    Benchmark,
    BenchmarkInfo,
)

LABEL_COL = "fluorescence_norm_intrain92"
SET_COL = "set"
LOCUS_ID_COL = "locus_id"
CHROM_COL = "chrom"
COORD_COL = "integration_coord"

TIERS: tuple[str, ...] = ("IntTrain", "IntProp", "pooled")
TOP_K: tuple[int, ...] = (3, 5, 7, 10)

# Hong et al. report SPCC = 0.847 between RT-qPCR mCherry mRNA and
# mCherry fluorescence across 30 promoter × IGR combinations (Fig. 2I).
# That number is inflated by cross-promoter rank ordering (5–6 of 6
# promoters in the matrix); the within-promoter ceiling for our
# fixed-TDH3p task is much lower. Surfaced for transparency.
MRNA_FLUO_CEILING_SPCC_PUBLISHED = 0.847


@dataclass(frozen=True)
class IntTrainFitted:
    """IntTrain-fitted IntProp ρ output: which (track × region) was
    picked on IntTrain, the corresponding selection ρ, and the
    evaluation ρ on IntProp + pooled."""
    readout_name: str            # e.g. "H3 (nucleosome density) × flank both 1kb"
    scores: np.ndarray           # (N,) per-locus signed scores, aligned to row order
    intrain_rho: float           # signed ρ on IntTrain (the selection score)
    intprop_rho: float           # signed ρ on IntProp (the eval score)
    pooled_rho: float            # signed ρ on the full 150-locus set
    intrain_pearson: float
    intprop_pearson: float
    n_candidates: int            # total number of candidate readouts considered


@dataclass(frozen=True)
class HongResults:
    # Primary
    scores: np.ndarray            # (N,) primary scores; NaN for unscoreable
    labels: np.ndarray            # (N,) measured fluorescence
    sets: np.ndarray              # (N,) tier label per row
    locus_ids: list[str]          # (N,)
    metrics: dict[str, dict[str, float]]   # tier → {n, pearson_r, spearman_rho, top_k_*}
    # Optional IntTrain-fitted IntProp ρ
    inttrain_fitted: IntTrainFitted | None = None


def _topk_enrichment(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    if len(scores) < k or len(scores) == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    top = labels[order[:k]]
    overall = labels.mean()
    if overall == 0 or not np.isfinite(overall):
        return float("nan")
    return float(top.mean() / overall)


def _tier_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(scores) & np.isfinite(labels)
    p, m = scores[mask], labels[mask]
    n = int(mask.sum())
    out: dict[str, float] = {"n": n}
    if n < 2:
        out["pearson_r"] = float("nan")
        out["spearman_rho"] = float("nan")
        for k in TOP_K:
            out[f"top_{k}_enrichment"] = float("nan")
        return out
    out["pearson_r"] = float(pearsonr(p, m).statistic)
    out["spearman_rho"] = float(spearmanr(p, m).statistic)
    for k in TOP_K:
        out[f"top_{k}_enrichment"] = _topk_enrichment(p, m, k)
    return out


def _all_metrics(
    scores: np.ndarray, labels: np.ndarray, sets: np.ndarray,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for tier in TIERS:
        if tier == "pooled":
            sel = np.ones(len(scores), dtype=bool)
        else:
            sel = (sets == tier)
        metrics[tier] = _tier_metrics(scores[sel], labels[sel])
    return metrics


def _safe_rho(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    """Signed Spearman ρ on the masked subset, NaN if not computable."""
    sel = mask & np.isfinite(scores) & np.isfinite(labels)
    if sel.sum() < 2:
        return float("nan")
    if np.unique(scores[sel]).size < 2:
        return float("nan")
    return float(spearmanr(scores[sel], labels[sel]).statistic)


def _safe_pearson(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    sel = mask & np.isfinite(scores) & np.isfinite(labels)
    if sel.sum() < 2:
        return float("nan")
    if np.unique(scores[sel]).size < 2:
        return float("nan")
    return float(pearsonr(scores[sel], labels[sel]).statistic)


def _select_inttrain_fitted(
    readouts: dict[str, np.ndarray],
    labels: np.ndarray,
    sets: np.ndarray,
) -> IntTrainFitted | None:
    """IntTrain-based selection over candidate readouts.

    Iterates over every key in ``readouts`` except ``'primary'`` and
    keys whose values are byte-identical to the primary. Picks the
    one maximizing signed ρ on IntTrain. Evaluates the picked combo on
    IntProp + pooled. Returns ``None`` if no candidate beats the
    Primary readout (signed) — IntTrain-fitted IntProp ρ is then "no
    improvement" and the benchmark reports Primary only.
    """
    if not readouts or "primary" not in readouts:
        return None

    primary = readouts["primary"]
    intrain_mask = (sets == "IntTrain")
    intprop_mask = (sets == "IntProp")
    pooled_mask = np.ones(len(labels), dtype=bool)

    primary_intrain_rho = _safe_rho(primary, labels, intrain_mask)

    candidate_keys = [k for k in readouts if k != "primary"]
    best_name: str | None = None
    best_intrain_rho = -np.inf
    for k in candidate_keys:
        scores = readouts[k]
        rho = _safe_rho(scores, labels, intrain_mask)
        if not np.isfinite(rho):
            continue
        if rho > best_intrain_rho:
            best_intrain_rho = rho
            best_name = k

    if best_name is None:
        return None
    # If the IntTrain pick doesn't beat the Primary readout's own
    # IntTrain ρ, the IntTrain-fitted selection isn't carrying
    # weight; report it as "no improvement" by reusing the Primary
    # readout's name.
    if np.isfinite(primary_intrain_rho) and best_intrain_rho <= primary_intrain_rho:
        # Find which named candidate equals the primary (if any)
        primary_named = next(
            (k for k in candidate_keys
             if np.allclose(readouts[k], primary, equal_nan=True)),
            best_name,
        )
        best_name = primary_named
        best_intrain_rho = primary_intrain_rho

    picked_scores = readouts[best_name]
    return IntTrainFitted(
        readout_name=best_name,
        scores=picked_scores,
        intrain_rho=best_intrain_rho,
        intprop_rho=_safe_rho(picked_scores, labels, intprop_mask),
        pooled_rho=_safe_rho(picked_scores, labels, pooled_mask),
        intrain_pearson=_safe_pearson(picked_scores, labels, intrain_mask),
        intprop_pearson=_safe_pearson(picked_scores, labels, intprop_mask),
        n_candidates=len(candidate_keys),
    )


class HongIGRInsertionBenchmark(
    Benchmark[IGRInsertionExpressionPredictor, HongResults]
):
    adapter_protocol: ClassVar[type] = IGRInsertionExpressionPredictor

    def __init__(
        self,
        labels_path: Path,
        cassette_seq: Path,
        fasta_path: Path,
        info: BenchmarkInfo,
    ) -> None:
        self.labels_path = Path(labels_path)
        self.cassette_seq = Path(cassette_seq)
        self._fasta_path = Path(fasta_path)
        self.info = info

        df = pd.read_csv(self.labels_path, sep="\t")
        for col in (LOCUS_ID_COL, SET_COL, CHROM_COL, COORD_COL, LABEL_COL):
            assert col in df.columns, f"{col} missing from {self.labels_path}"
        self.locus_ids: list[str] = df[LOCUS_ID_COL].astype(str).tolist()
        self.labels: np.ndarray = df[LABEL_COL].to_numpy(dtype=float)
        self.sets: np.ndarray = df[SET_COL].to_numpy()
        self.loci: list[HongLocus] = [
            HongLocus(
                locus_id=str(r[LOCUS_ID_COL]),
                set=str(r[SET_COL]),
                chrom=str(r[CHROM_COL]),
                integration_coord=int(r[COORD_COL]),
            )
            for _, r in df.iterrows()
        ]

    @property
    def fasta_path(self) -> Path:
        return self._fasta_path

    def evaluate(self, adapter: IGRInsertionExpressionPredictor) -> HongResults:
        # IntTrain-fitted path: adapters opting in implement
        # predict_diagnostic_readouts → dict including 'primary' key.
        if hasattr(adapter, "predict_diagnostic_readouts"):
            readouts = adapter.predict_diagnostic_readouts(self.loci)
            assert "primary" in readouts, (
                f"adapter {type(adapter).__name__} returned "
                f"predict_diagnostic_readouts without 'primary' key"
            )
            primary_scores = np.asarray(readouts["primary"], dtype=float)
            fit = _select_inttrain_fitted(readouts, self.labels, self.sets)
        else:
            primary_scores = np.asarray(
                adapter.predict_expressions(self.loci), dtype=float
            )
            fit = None

        assert len(primary_scores) == len(self.loci), (
            f"adapter returned {len(primary_scores)} scores for "
            f"{len(self.loci)} loci"
        )
        metrics = _all_metrics(primary_scores, self.labels, self.sets)
        return HongResults(
            scores=primary_scores,
            labels=self.labels,
            sets=self.sets,
            locus_ids=self.locus_ids,
            metrics=metrics,
            inttrain_fitted=fit,
        )

    def plot(self, results: HongResults, out_dir: Path) -> None:
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        title_model = out_dir.name.split("__")[0] if "__" in out_dir.name else ""

        mask = np.isfinite(results.scores) & np.isfinite(results.labels)
        pred = results.scores[mask]
        meas = results.labels[mask]
        sets = results.sets[mask]

        # (1) Per-tier Primary scatter
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, tier in zip(axes, ("IntTrain", "IntProp", "pooled")):
            if tier == "pooled":
                m, p, s = meas, pred, sets
            else:
                sel = (sets == tier)
                m, p, s = meas[sel], pred[sel], sets[sel]
            if tier == "pooled":
                for tier_name, color in (("IntTrain", "C0"), ("IntProp", "C1")):
                    sel_t = (s == tier_name)
                    ax.scatter(
                        m[sel_t], p[sel_t], s=12, alpha=0.6, label=tier_name,
                        color=color, rasterized=True,
                    )
                ax.legend(loc="best", fontsize=8)
            else:
                ax.scatter(m, p, s=12, alpha=0.6, rasterized=True)
            if len(m) > 1:
                a, b = np.polyfit(m, p, 1)
                xs = np.linspace(m.min(), m.max(), 50)
                ax.plot(xs, a * xs + b, color="red", linewidth=1, alpha=0.8)
            m_ = results.metrics[tier]
            ax.set_xlabel("measured (fluorescence / IntTrain92)")
            ax.set_ylabel("primary score (mCherry-CDS RNA-seq sum)")
            ax.set_title(
                f"{tier}  n={m_['n']}  "
                f"r={m_['pearson_r']:.3f}  ρ={m_['spearman_rho']:.3f}"
            )
        title = "Hong IGR-insertion — Primary (RNA-seq × mCherry CDS)"
        if title_model:
            title += f" — {title_model}"
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_dir / "scatter_primary.png", dpi=150)
        plt.close(fig)

        # (2) Top-k enrichment curve per tier (Primary)
        ks = np.arange(1, 31)
        fig, ax = plt.subplots(figsize=(7, 5))
        for tier, color in (
            ("IntTrain", "C0"), ("IntProp", "C1"), ("pooled", "k"),
        ):
            if tier == "pooled":
                sel = np.ones(len(meas), dtype=bool)
            else:
                sel = (sets == tier)
            m_t, p_t = meas[sel], pred[sel]
            if len(m_t) == 0:
                continue
            ys = np.array([_topk_enrichment(p_t, m_t, k) for k in ks])
            ax.plot(ks, ys, marker="o", markersize=3, color=color, label=tier)
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.7, label="random (1.0)")
        ax.set_xlabel("top-k candidates")
        ax.set_ylabel("mean(top-k fluorescence) / overall mean")
        title2 = "Hong top-k enrichment (Primary)"
        if title_model:
            title2 += f" — {title_model}"
        ax.set_title(title2)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "top_k_enrichment.png", dpi=150)
        plt.close(fig)

        # (3) IntTrain-fitted IntProp ρ per-tier scatter, if available
        if results.inttrain_fitted is not None:
            fit = results.inttrain_fitted
            mask_d = np.isfinite(fit.scores) & np.isfinite(results.labels)
            d_pred = fit.scores[mask_d]
            d_meas = results.labels[mask_d]
            d_sets = results.sets[mask_d]
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            for ax, tier in zip(axes, ("IntTrain", "IntProp", "pooled")):
                if tier == "pooled":
                    m, p = d_meas, d_pred
                else:
                    sel = (d_sets == tier)
                    m, p = d_meas[sel], d_pred[sel]
                if len(m) < 2:
                    ax.set_title(f"{tier}: too few points")
                    continue
                ax.scatter(m, p, s=12, alpha=0.6, rasterized=True, color="C2")
                a, b = np.polyfit(m, p, 1)
                xs = np.linspace(m.min(), m.max(), 50)
                ax.plot(xs, a * xs + b, color="red", linewidth=1, alpha=0.8)
                r = pearsonr(m, p).statistic
                rho = spearmanr(m, p).statistic
                ax.set_xlabel("measured (fluorescence / IntTrain92)")
                ax.set_ylabel("IntTrain-fitted signed score")
                ax.set_title(f"{tier}  n={len(m)}  r={r:.3f}  ρ={rho:.3f}")
            t = f"Hong IntTrain-fitted — {fit.readout_name}"
            if title_model:
                t += f" ({title_model})"
            fig.suptitle(t, fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(out_dir / "scatter_inttrain_fitted.png", dpi=150)
            plt.close(fig)

    def save_results(self, results: HongResults, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "scores.npy", results.scores)
        np.save(out_dir / "labels.npy", results.labels)
        meta: dict[str, Any] = {
            "locus_ids": results.locus_ids,
            "sets": results.sets.tolist(),
        }
        if results.inttrain_fitted is not None:
            np.save(out_dir / "inttrain_fitted_scores.npy", results.inttrain_fitted.scores)
            meta["inttrain_fitted"] = {
                "readout_name": results.inttrain_fitted.readout_name,
                "intrain_rho": results.inttrain_fitted.intrain_rho,
                "intprop_rho": results.inttrain_fitted.intprop_rho,
                "pooled_rho": results.inttrain_fitted.pooled_rho,
                "intrain_pearson": results.inttrain_fitted.intrain_pearson,
                "intprop_pearson": results.inttrain_fitted.intprop_pearson,
                "n_candidates": results.inttrain_fitted.n_candidates,
            }
        (out_dir / "loci.json").write_text(json.dumps(meta, indent=2))

    def load_results(self, out_dir: Path) -> HongResults:
        out_dir = Path(out_dir)
        scores = np.load(out_dir / "scores.npy")
        labels = np.load(out_dir / "labels.npy")
        meta = json.loads((out_dir / "loci.json").read_text())
        sets = np.asarray(meta["sets"])
        metrics = _all_metrics(scores, labels, sets)
        fit = None
        if "inttrain_fitted" in meta and (out_dir / "inttrain_fitted_scores.npy").exists():
            d = meta["inttrain_fitted"]
            fit = IntTrainFitted(
                readout_name=d["readout_name"],
                scores=np.load(out_dir / "inttrain_fitted_scores.npy"),
                intrain_rho=float(d["intrain_rho"]),
                intprop_rho=float(d["intprop_rho"]),
                pooled_rho=float(d["pooled_rho"]),
                intrain_pearson=float(d.get("intrain_pearson", float("nan"))),
                intprop_pearson=float(d.get("intprop_pearson", float("nan"))),
                n_candidates=int(d.get("n_candidates", 0)),
            )
        return HongResults(
            scores=scores,
            labels=labels,
            sets=sets,
            locus_ids=meta["locus_ids"],
            metrics=metrics,
            inttrain_fitted=fit,
        )

    def summary_dict(self, results: HongResults) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_rows_total": int(len(results.scores)),
            "mrna_fluo_ceiling_spcc_published": MRNA_FLUO_CEILING_SPCC_PUBLISHED,
        }
        for tier, m in results.metrics.items():
            for key, val in m.items():
                out[f"{tier}_{key}"] = val
        if results.inttrain_fitted is not None:
            d = results.inttrain_fitted
            out["inttrain_fitted_readout_name"] = d.readout_name
            out["inttrain_fitted_intrain_spearman_rho"] = d.intrain_rho
            out["inttrain_fitted_intprop_spearman_rho"] = d.intprop_rho
            out["inttrain_fitted_pooled_spearman_rho"] = d.pooled_rho
            out["inttrain_fitted_intrain_pearson_r"] = d.intrain_pearson
            out["inttrain_fitted_intprop_pearson_r"] = d.intprop_pearson
            out["inttrain_fitted_n_candidates"] = d.n_candidates
        return out

    def headline(self, results: HongResults) -> str:
        ip = results.metrics["IntProp"]
        it = results.metrics["IntTrain"]
        line = (
            f"Primary: IntProp ρ = {ip['spearman_rho']:.4f} "
            f"(IntTrain ρ = {it['spearman_rho']:.4f}, n = {ip['n']})"
        )
        if results.inttrain_fitted is not None:
            d = results.inttrain_fitted
            line += (
                f" | IntTrain-fitted [{d.readout_name}]: "
                f"IntProp ρ = {d.intprop_rho:.4f} "
                f"(IntTrain ρ = {d.intrain_rho:.4f}, "
                f"selected from {d.n_candidates} candidates)"
            )
        else:
            line += " | IntTrain-fitted: not available (adapter doesn't expose readouts)"
        return line

    def headline_metric_labels(self) -> dict[str, str]:
        return {
            "IntProp_spearman_rho":        "Primary IntProp ρ",
            "inttrain_fitted_intprop_spearman_rho": "IntTrain-fitted IntProp ρ",
            "IntTrain_spearman_rho":       "Primary IntTrain ρ",
            "pooled_spearman_rho":         "Primary pooled ρ",
            "IntProp_pearson_r":           "Primary IntProp r",
            "IntProp_top_7_enrichment":    "Primary IntProp top-7",
        }

    def compare_plot_title(self) -> str:
        return "Hong et al. IGR-insertion position effects"
