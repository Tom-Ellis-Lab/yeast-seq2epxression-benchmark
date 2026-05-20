"""Brooks et al. SCRaMBLE structural-rearrangement expression benchmark.

Two tiers (see ``benchmarks/brooks_scramble.md``):

  Tier 1 — scalar LFC.  log2((sum pred over CDS bp + 1) /
    (sum native pred over CDS bp + 1)) vs the distribution's
    ``true_lfc`` (computed from library-size-normalised raw CDS pileups).
    Metrics: direction balanced accuracy → Spearman ρ → Pearson r,
    reported against the JS94×3 reproducibility ceiling derivable from
    ``norm_cov_js94_runs`` in the same file.

  Tier 2 — coverage shape.  Per-base predicted vs per-base true
    Nanopore pileup over the central ``seq_len - 2 * crop`` region;
    metrics: Pearson + Jensen–Shannon divergence per sample, mean across.

**Units.** The benchmark expects adapter predictions in **raw per-base
predicted-count units** (i.e. with any model-specific training transform
inverted inside the adapter); the distribution's ``true_cov_*`` columns
are raw per-base Nanopore pileups. Library size cancels in the LFC
ratio and is normalised away by the sum-to-1 step before the shape
metrics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score

from yeastbench.adapters.protocols import CoverageTrackPredictor
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo

WINDOW_LEN = 4992          # gene-centred window length (set by the builder)
PSEUDOCOUNT = 1.0


@dataclass(frozen=True)
class BrooksResults:
    sample_ids: list[str]
    true_lfc: np.ndarray             # (N,)
    pred_lfc: np.ndarray             # (N,)
    low_support: np.ndarray          # (N,) bool
    # Tier-1 scalars (on the *well-supported* subset)
    n_total: int
    n_scored: int
    dir_balanced_acc: float
    pearson_r: float
    spearman_rho: float
    # Reproducibility ceiling from JS94 deep-run pairs (LFC of two
    # control runs vs their mean; symmetric per sample)
    ceiling_dir_acc: float
    ceiling_pearson_r: float
    # Tier-2 (mean over scored samples; alt construct, full window)
    tier2_pearson_mean: float
    tier2_js_mean: float


# ── shape metric helpers ─────────────────────────────────────


def _crop_to_output(per_base: np.ndarray, crop: int, out_len: int) -> np.ndarray:
    """Slice a length-`window_len` per-base vector to the
    `[crop, crop + out_len)` region the adapter actually predicts."""
    return per_base[crop : crop + out_len].astype(np.float64)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JS divergence in bits; symmetric, bounded [0, 1], no smoothing
    needed (mixture absorbs zeros). p and q must sum to 1."""
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * (_kl(p, m) + _kl(q, m))


# ── benchmark class ──────────────────────────────────────────


class BrooksScrambleBenchmark(Benchmark[CoverageTrackPredictor, BrooksResults]):
    adapter_protocol: ClassVar[type] = CoverageTrackPredictor

    def __init__(self, data_path: Path, info: BenchmarkInfo) -> None:
        self.data_path = Path(data_path)
        self.info = info
        df = pd.read_csv(self.data_path, sep="\t")
        for col in ("alt_seq", "native_seq", "cds_start_in_window",
                    "cds_end_in_window", "true_lfc", "norm_cov_js94_runs",
                    "true_cov_alt", "true_cov_native", "low_support",
                    "strand", "sample_id"):
            assert col in df.columns, f"{col} missing from {self.data_path}"
        assert (df.alt_seq.str.len() == WINDOW_LEN).all()
        assert (df.native_seq.str.len() == WINDOW_LEN).all()
        self.df = df.reset_index(drop=True)

    def _parse_cov(self, s: str) -> np.ndarray:
        return np.fromstring(s, sep=",", dtype=np.int32)

    def _parse_runs(self, s: str) -> np.ndarray:
        return np.fromstring(s, sep=",", dtype=np.float64)

    def evaluate(self, adapter: CoverageTrackPredictor) -> BrooksResults:
        n = len(self.df)
        pred_lfc = np.full(n, np.nan)
        tier2_pearson = np.full(n, np.nan)
        tier2_js = np.full(n, np.nan)

        crop = adapter.crop_bp_each_side
        out_len = adapter.seq_len - 2 * crop  # per-base prediction length

        for i, row in self.df.iterrows():
            pred_alt = np.asarray(
                adapter.predict_coverage(row.alt_seq, row.strand), dtype=float
            )
            pred_nat = np.asarray(
                adapter.predict_coverage(row.native_seq, row.strand), dtype=float
            )
            assert pred_alt.shape == pred_nat.shape == (out_len,), (
                f"adapter must return per-base length {out_len}; got "
                f"{pred_alt.shape}"
            )
            # CDS interval mapped to the predicted (cropped) region
            cs = max(0, int(row.cds_start_in_window) - crop)
            ce = min(out_len, int(row.cds_end_in_window) - crop)
            if ce <= cs:
                continue
            # Tier 1: per-base CDS sum → LFC. Both sides are in the same
            # untransformed predicted-count units; library size cancels.
            alt_cds = pred_alt[cs:ce].sum()
            nat_cds = pred_nat[cs:ce].sum()
            pred_lfc[i] = float(np.log2(
                (alt_cds + PSEUDOCOUNT) / (nat_cds + PSEUDOCOUNT)
            ))
            # Tier 2: per-base coverage shape over the predicted region
            true_alt = _crop_to_output(
                self._parse_cov(row.true_cov_alt), crop, out_len
            )
            if true_alt.sum() > 0 and pred_alt.sum() > 0:
                tier2_pearson[i] = float(
                    pearsonr(true_alt, pred_alt).statistic
                )
                p = true_alt / true_alt.sum()
                q = pred_alt / pred_alt.sum()
                tier2_js[i] = _js_divergence(p, q)

        true_lfc = self.df.true_lfc.to_numpy(dtype=float)
        low = self.df.low_support.to_numpy(dtype=bool)
        mask = np.isfinite(pred_lfc) & np.isfinite(true_lfc) & (~low)
        n_scored = int(mask.sum())
        if n_scored < 2:
            dir_acc = pr = sr = float("nan")
        else:
            dir_acc = float(balanced_accuracy_score(
                np.sign(true_lfc[mask]).astype(int),
                np.sign(pred_lfc[mask]).astype(int),
            ))
            pr = float(pearsonr(pred_lfc[mask], true_lfc[mask]).statistic)
            sr = float(spearmanr(pred_lfc[mask], true_lfc[mask]).statistic)

        # Reproducibility ceiling: per sample, compute LFC of each JS94
        # run vs the run-mean (a control-vs-control "expected" zero LFC);
        # use direction acc + Pearson r of (ctrl_LFC vs true_LFC) — both
        # should be ~ at-chance, so this is the noise floor.
        ceiling_lfc = np.full(n, np.nan)
        for i, row in self.df.iterrows():
            runs = self._parse_runs(row.norm_cov_js94_runs)
            mean_run = runs.mean()
            # pick the most-deviant single run as the worst-case "ctrl LFC"
            dev = np.log2((runs + PSEUDOCOUNT) / (mean_run + PSEUDOCOUNT))
            ceiling_lfc[i] = float(dev[np.argmax(np.abs(dev))])
        cm = mask & np.isfinite(ceiling_lfc)
        if cm.sum() >= 2:
            ceiling_dir = float(balanced_accuracy_score(
                np.sign(true_lfc[cm]).astype(int),
                np.sign(ceiling_lfc[cm]).astype(int),
            ))
            ceiling_pr = float(pearsonr(ceiling_lfc[cm], true_lfc[cm]).statistic)
        else:
            ceiling_dir = ceiling_pr = float("nan")

        t2p = float(np.nanmean(tier2_pearson[mask])) if mask.any() else float("nan")
        t2j = float(np.nanmean(tier2_js[mask])) if mask.any() else float("nan")

        return BrooksResults(
            sample_ids=self.df.sample_id.tolist(),
            true_lfc=true_lfc, pred_lfc=pred_lfc, low_support=low,
            n_total=n, n_scored=n_scored,
            dir_balanced_acc=dir_acc, pearson_r=pr, spearman_rho=sr,
            ceiling_dir_acc=ceiling_dir, ceiling_pearson_r=ceiling_pr,
            tier2_pearson_mean=t2p, tier2_js_mean=t2j,
        )

    def plot(self, results: BrooksResults, out_dir: Path) -> None:
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        title_model = out_dir.name.split("__")[0] if "__" in out_dir.name else ""

        m = (np.isfinite(results.pred_lfc) & np.isfinite(results.true_lfc)
             & ~results.low_support)
        p, t = results.pred_lfc[m], results.true_lfc[m]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
        ax.scatter(t, p, s=24, alpha=0.7)
        for sid, ti, pi in zip(np.array(results.sample_ids)[m], t, p):
            ax.annotate(sid.split(":")[1], (ti, pi), fontsize=6, alpha=0.6)
        lim = max(np.nanmax(np.abs(t)), np.nanmax(np.abs(p)), 1) + 0.5
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("true log2 LFC (alt / native)")
        ax.set_ylabel("predicted log2 LFC (alt / native)")
        ax.set_title(
            f"Brooks SCRaMBLE — Tier 1"
            + (f" — {title_model}" if title_model else "")
            + f"\nn={results.n_scored}  dir-acc={results.dir_balanced_acc:.3f}  "
            f"r={results.pearson_r:.3f}  ρ={results.spearman_rho:.3f}\n"
            f"ceiling: dir-acc={results.ceiling_dir_acc:.3f}  "
            f"r={results.ceiling_pearson_r:.3f}"
        )
        fig.tight_layout(); fig.savefig(out_dir / "tier1_scatter.png", dpi=150)
        plt.close(fig)

    def save_results(self, results: BrooksResults, out_dir: Path) -> None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "pred_lfc.npy", results.pred_lfc)
        np.save(out_dir / "true_lfc.npy", results.true_lfc)
        (out_dir / "samples.json").write_text(json.dumps({
            "sample_ids": results.sample_ids,
            "low_support": results.low_support.tolist(),
        }, indent=2))

    def load_results(self, out_dir: Path) -> BrooksResults:
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "samples.json").read_text())
        pred = np.load(out_dir / "pred_lfc.npy")
        true = np.load(out_dir / "true_lfc.npy")
        low = np.asarray(meta["low_support"], dtype=bool)
        m = np.isfinite(pred) & np.isfinite(true) & ~low
        if m.sum() >= 2:
            dir_acc = float(balanced_accuracy_score(
                np.sign(true[m]).astype(int), np.sign(pred[m]).astype(int)))
            pr = float(pearsonr(pred[m], true[m]).statistic)
            sr = float(spearmanr(pred[m], true[m]).statistic)
        else:
            dir_acc = pr = sr = float("nan")
        return BrooksResults(
            sample_ids=meta["sample_ids"], true_lfc=true, pred_lfc=pred,
            low_support=low, n_total=len(true), n_scored=int(m.sum()),
            dir_balanced_acc=dir_acc, pearson_r=pr, spearman_rho=sr,
            ceiling_dir_acc=float("nan"), ceiling_pearson_r=float("nan"),
            tier2_pearson_mean=float("nan"), tier2_js_mean=float("nan"),
        )

    def summary_dict(self, results: BrooksResults) -> dict[str, Any]:
        return {
            "n_total": results.n_total,
            "n_scored": results.n_scored,
            "n_low_support": int(results.low_support.sum()),
            "tier1_dir_balanced_acc": results.dir_balanced_acc,
            "tier1_pearson_r": results.pearson_r,
            "tier1_spearman_rho": results.spearman_rho,
            "ceiling_dir_balanced_acc": results.ceiling_dir_acc,
            "ceiling_pearson_r": results.ceiling_pearson_r,
            "tier2_pearson_mean": results.tier2_pearson_mean,
            "tier2_js_mean": results.tier2_js_mean,
        }

    def headline(self, results: BrooksResults) -> str:
        return (
            f"Tier-1: dir-acc {results.dir_balanced_acc:.3f}  "
            f"r {results.pearson_r:.3f}  ρ {results.spearman_rho:.3f}  | "
            f"Tier-2: r̄ {results.tier2_pearson_mean:.3f}  "
            f"JS̄ {results.tier2_js_mean:.3f}  | "
            f"ceiling: dir-acc {results.ceiling_dir_acc:.3f}, "
            f"r {results.ceiling_pearson_r:.3f}  (n = {results.n_scored})"
        )
