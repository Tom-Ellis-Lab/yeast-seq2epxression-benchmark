"""Brooks et al. SCRaMBLE structural-rearrangement expression benchmark.

Two tiers (see ``benchmarks/brooks_scramble.md``):

  Tier 1 — scalar LFC.  **Per-replicate** true LFCs: for each sample,
    compute ``log2((norm_cov_strain + 1) / (norm_cov_js94_k + 1))`` for
    each JS94 deep run ``k`` whose raw CDS read count for the gene
    meets ``MIN_READS_PER_RUN`` (default 10). Yields 0–3 supporting
    LFCs per sample. Predicted LFC is a single scalar (from per-base
    predicted-count units, alt CDS sum vs native CDS sum).
    Headline metrics over ``n_reps ≥ 1`` AND not ``low_support``:
      * Pearson r and Spearman ρ of ``pred_lfc`` vs mean true LFC
      * Direction balanced accuracy on the sign of the mean true LFC.
    Calibration metrics over the ``n_reps ≥ 2`` subset (range defined):
      * Within-range hit rate — fraction of samples where ``pred_lfc``
        lies in ``[min(true_lfcs), max(true_lfcs)]``.
      * Mean standardised residual ``|z|`` where
        ``z = (pred - mean) / max(range, eps)``.

  Tier 2 — coverage shape.  Per-base predicted vs per-base true
    Nanopore pileup over the central ``seq_len - 2 * crop`` region;
    metrics: Pearson + Jensen–Shannon divergence per sample, mean across
    the ``n_reps ≥ 1`` AND not ``low_support`` cohort.

**Units.** The benchmark expects adapter predictions in **raw per-base
predicted-count units** (i.e. with any model-specific training transform
inverted inside the adapter); the distribution's ``true_cov_*`` columns
are raw per-base Nanopore pileups. Library size cancels in the LFC
ratio and is normalised away by the sum-to-1 step before the shape
metrics.
"""
from __future__ import annotations

import json
import warnings
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
MIN_READS_PER_RUN = 10     # per-JS94-run raw read floor for that run to
                           # contribute a per-replicate true_lfc for the sample
RANGE_EPS = 1e-6           # avoid division by zero in |z| when min == max


@dataclass(frozen=True)
class BrooksResults:
    sample_ids: list[str]
    pred_lfc: np.ndarray             # (N,)
    # Per-replicate true LFCs derived from the 3 JS94 deep runs. For each
    # sample, only runs with raw_reads >= MIN_READS_PER_RUN contribute,
    # so each row holds 0–3 finite values padded with NaN to width 3.
    true_lfc_runs: np.ndarray        # (N, 3) float64
    n_reps_supported: np.ndarray     # (N,) int — finite-count per row
    low_support: np.ndarray          # (N,) bool — strain-side only
    # Cohort counts
    n_total: int
    n_scored: int                    # n_reps >= 1 AND not low_support
    n_calibration: int               # n_reps >= 2 AND not low_support
    n_weak_baseline: int             # n_reps == 0 (per-gene, all JS94 thin)
    n_low_support: int               # low_support == True
    # Headline (rank/correlation) — over n_scored
    dir_balanced_acc: float
    pearson_r: float
    spearman_rho: float
    # Calibration — over n_calibration
    within_range_rate: float
    mean_abs_z: float
    # Tier-2 (mean over n_scored; alt construct, full predicted region)
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
                    "cds_end_in_window", "norm_cov_strain",
                    "norm_cov_js94_runs", "js94_reads_runs",
                    "true_cov_alt", "true_cov_native", "low_support",
                    "strand", "sample_id"):
            assert col in df.columns, f"{col} missing from {self.data_path}"
        assert (df.alt_seq.str.len() == WINDOW_LEN).all()
        assert (df.native_seq.str.len() == WINDOW_LEN).all()
        self.df = df.reset_index(drop=True)

    def _parse_cov(self, s: str) -> np.ndarray:
        return np.fromstring(s, sep=",", dtype=np.int32)

    def _parse_norm_runs(self, s: str) -> np.ndarray:
        return np.fromstring(s, sep=",", dtype=np.float64)

    def _parse_raw_runs(self, s: str) -> np.ndarray:
        return np.fromstring(s, sep=",", dtype=np.int64)

    def evaluate(self, adapter: CoverageTrackPredictor) -> BrooksResults:
        n = len(self.df)
        pred_lfc = np.full(n, np.nan)
        tier2_pearson = np.full(n, np.nan)
        tier2_js = np.full(n, np.nan)
        # Per-replicate true LFCs. NaN where the run is below MIN_READS_PER_RUN
        # for this gene; samples can have 0–3 finite entries per row.
        true_lfc_runs = np.full((n, 3), np.nan, dtype=np.float64)

        crop = adapter.crop_bp_each_side
        out_len = adapter.seq_len - 2 * crop  # per-base prediction length

        for i, row in self.df.iterrows():
            # Per-replicate true LFCs from the JS94 deep runs.
            s_norm = float(row.norm_cov_strain)
            j_norms = self._parse_norm_runs(row.norm_cov_js94_runs)
            j_raws = self._parse_raw_runs(row.js94_reads_runs)
            for k in range(min(len(j_norms), len(j_raws), 3)):
                if j_raws[k] < MIN_READS_PER_RUN:
                    continue
                true_lfc_runs[i, k] = float(np.log2(
                    (s_norm + PSEUDOCOUNT) / (j_norms[k] + PSEUDOCOUNT)
                ))

            # Predicted LFC + Tier-2 shape from the adapter.
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
            cs = max(0, int(row.cds_start_in_window) - crop)
            ce = min(out_len, int(row.cds_end_in_window) - crop)
            if ce <= cs:
                continue
            alt_cds = pred_alt[cs:ce].sum()
            nat_cds = pred_nat[cs:ce].sum()
            pred_lfc[i] = float(np.log2(
                (alt_cds + PSEUDOCOUNT) / (nat_cds + PSEUDOCOUNT)
            ))
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

        # Per-sample summary of the replicate LFCs.
        finite = np.isfinite(true_lfc_runs)
        n_reps_supported = finite.sum(axis=1)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_true = np.where(
                n_reps_supported > 0,
                np.nanmean(np.where(finite, true_lfc_runs, np.nan), axis=1),
                np.nan,
            )

        low = self.df.low_support.to_numpy(dtype=bool)
        scored_mask = (
            (~low) & (n_reps_supported >= 1)
            & np.isfinite(pred_lfc) & np.isfinite(mean_true)
        )
        calib_mask = scored_mask & (n_reps_supported >= 2)

        n_scored = int(scored_mask.sum())
        n_calibration = int(calib_mask.sum())
        n_weak_baseline = int((n_reps_supported == 0).sum())
        n_low = int(low.sum())

        if n_scored < 2:
            dir_acc = pr = sr = float("nan")
        else:
            dir_acc = float(balanced_accuracy_score(
                np.sign(mean_true[scored_mask]).astype(int),
                np.sign(pred_lfc[scored_mask]).astype(int),
            ))
            pr = float(pearsonr(pred_lfc[scored_mask],
                                 mean_true[scored_mask]).statistic)
            sr = float(spearmanr(pred_lfc[scored_mask],
                                  mean_true[scored_mask]).statistic)

        # Calibration on the n_reps >= 2 subset: how often does the
        # prediction land inside the JS94 replicate envelope, and how
        # many "envelope widths" off is it on average?
        if n_calibration < 1:
            within_range_rate = mean_abs_z = float("nan")
        else:
            idx = np.where(calib_mask)[0]
            hits = 0
            zs = []
            for ii in idx:
                row_runs = true_lfc_runs[ii][finite[ii]]
                lo, hi = float(row_runs.min()), float(row_runs.max())
                if lo <= pred_lfc[ii] <= hi:
                    hits += 1
                rng = max(hi - lo, RANGE_EPS)
                zs.append(abs(pred_lfc[ii] - mean_true[ii]) / rng)
            within_range_rate = hits / n_calibration
            mean_abs_z = float(np.mean(zs))

        t2p = (float(np.nanmean(tier2_pearson[scored_mask]))
               if n_scored else float("nan"))
        t2j = (float(np.nanmean(tier2_js[scored_mask]))
               if n_scored else float("nan"))

        return BrooksResults(
            sample_ids=self.df.sample_id.tolist(),
            pred_lfc=pred_lfc,
            true_lfc_runs=true_lfc_runs,
            n_reps_supported=n_reps_supported.astype(np.int64),
            low_support=low,
            n_total=n, n_scored=n_scored, n_calibration=n_calibration,
            n_weak_baseline=n_weak_baseline, n_low_support=n_low,
            dir_balanced_acc=dir_acc, pearson_r=pr, spearman_rho=sr,
            within_range_rate=within_range_rate, mean_abs_z=mean_abs_z,
            tier2_pearson_mean=t2p, tier2_js_mean=t2j,
        )

    def plot(self, results: BrooksResults, out_dir: Path) -> None:
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        title_model = out_dir.name.split("__")[0] if "__" in out_dir.name else ""

        finite = np.isfinite(results.true_lfc_runs)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_true = np.where(
                results.n_reps_supported > 0,
                np.nanmean(
                    np.where(finite, results.true_lfc_runs, np.nan), axis=1
                ),
                np.nan,
            )
        m = (~results.low_support) & (results.n_reps_supported >= 1) \
            & np.isfinite(results.pred_lfc) & np.isfinite(mean_true)

        p_arr, t_arr = results.pred_lfc[m], mean_true[m]
        # Per-sample replicate envelope (min..max of supported runs)
        envelope_lo = np.full(m.sum(), np.nan)
        envelope_hi = np.full(m.sum(), np.nan)
        for j, ii in enumerate(np.where(m)[0]):
            runs = results.true_lfc_runs[ii][finite[ii]]
            envelope_lo[j], envelope_hi[j] = runs.min(), runs.max()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
        # Horizontal error bars = JS94 replicate envelope (only where n_reps>=2)
        err_lo = t_arr - envelope_lo
        err_hi = envelope_hi - t_arr
        ax.errorbar(
            t_arr, p_arr, xerr=[err_lo, err_hi], fmt="o", ms=4,
            ecolor="lightgrey", elinewidth=1, alpha=0.7,
        )
        for sid, ti, pi in zip(np.array(results.sample_ids)[m], t_arr, p_arr):
            ax.annotate(sid.split(":")[1], (ti, pi), fontsize=6, alpha=0.6)
        lim = max(np.nanmax(np.abs(t_arr)), np.nanmax(np.abs(p_arr)), 1) + 0.5
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("true log2 LFC (mean over supporting JS94 runs)")
        ax.set_ylabel("predicted log2 LFC (alt / native)")
        ax.set_title(
            f"Brooks SCRaMBLE — Tier 1"
            + (f" — {title_model}" if title_model else "")
            + f"\nn_scored={results.n_scored}  "
            f"dir-acc={results.dir_balanced_acc:.3f}  "
            f"r={results.pearson_r:.3f}  ρ={results.spearman_rho:.3f}\n"
            f"calibration (n={results.n_calibration}): "
            f"within-range={results.within_range_rate:.3f}  "
            f"|z|={results.mean_abs_z:.3f}"
        )
        fig.tight_layout(); fig.savefig(out_dir / "tier1_scatter.png", dpi=150)
        plt.close(fig)

    def save_results(self, results: BrooksResults, out_dir: Path) -> None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "pred_lfc.npy", results.pred_lfc)
        np.save(out_dir / "true_lfc_runs.npy", results.true_lfc_runs)
        np.save(out_dir / "n_reps_supported.npy", results.n_reps_supported)
        (out_dir / "samples.json").write_text(json.dumps({
            "sample_ids": results.sample_ids,
            "low_support": results.low_support.tolist(),
        }, indent=2))

    def load_results(self, out_dir: Path) -> BrooksResults:
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "samples.json").read_text())
        pred = np.load(out_dir / "pred_lfc.npy")
        true_lfc_runs = np.load(out_dir / "true_lfc_runs.npy")
        n_reps = np.load(out_dir / "n_reps_supported.npy")
        low = np.asarray(meta["low_support"], dtype=bool)

        finite = np.isfinite(true_lfc_runs)
        mean_true = np.where(
            n_reps > 0,
            np.nanmean(np.where(finite, true_lfc_runs, np.nan), axis=1),
            np.nan,
        )
        scored = (~low) & (n_reps >= 1) & np.isfinite(pred) & np.isfinite(mean_true)
        calib = scored & (n_reps >= 2)
        if scored.sum() >= 2:
            dir_acc = float(balanced_accuracy_score(
                np.sign(mean_true[scored]).astype(int),
                np.sign(pred[scored]).astype(int)))
            pr = float(pearsonr(pred[scored], mean_true[scored]).statistic)
            sr = float(spearmanr(pred[scored], mean_true[scored]).statistic)
        else:
            dir_acc = pr = sr = float("nan")
        if calib.sum() >= 1:
            hits = 0
            zs = []
            for ii in np.where(calib)[0]:
                runs = true_lfc_runs[ii][finite[ii]]
                lo, hi = float(runs.min()), float(runs.max())
                if lo <= pred[ii] <= hi:
                    hits += 1
                zs.append(abs(pred[ii] - mean_true[ii]) / max(hi - lo, RANGE_EPS))
            within = hits / calib.sum()
            mz = float(np.mean(zs))
        else:
            within = mz = float("nan")
        return BrooksResults(
            sample_ids=meta["sample_ids"], pred_lfc=pred,
            true_lfc_runs=true_lfc_runs, n_reps_supported=n_reps.astype(np.int64),
            low_support=low, n_total=len(pred), n_scored=int(scored.sum()),
            n_calibration=int(calib.sum()),
            n_weak_baseline=int((n_reps == 0).sum()),
            n_low_support=int(low.sum()),
            dir_balanced_acc=dir_acc, pearson_r=pr, spearman_rho=sr,
            within_range_rate=within, mean_abs_z=mz,
            tier2_pearson_mean=float("nan"), tier2_js_mean=float("nan"),
        )

    def summary_dict(self, results: BrooksResults) -> dict[str, Any]:
        return {
            "n_total": results.n_total,
            "n_scored": results.n_scored,
            "n_calibration": results.n_calibration,
            "n_weak_baseline": results.n_weak_baseline,
            "n_low_support": results.n_low_support,
            "tier1_dir_balanced_acc": results.dir_balanced_acc,
            "tier1_pearson_r": results.pearson_r,
            "tier1_spearman_rho": results.spearman_rho,
            "tier1_within_range_rate": results.within_range_rate,
            "tier1_mean_abs_z": results.mean_abs_z,
            "tier2_pearson_mean": results.tier2_pearson_mean,
            "tier2_js_mean": results.tier2_js_mean,
        }

    def headline(self, results: BrooksResults) -> str:
        return (
            f"Tier-1 (n_scored={results.n_scored}): "
            f"dir-acc {results.dir_balanced_acc:.3f}  "
            f"r {results.pearson_r:.3f}  ρ {results.spearman_rho:.3f}  | "
            f"calibration (n={results.n_calibration}): "
            f"within-range {results.within_range_rate:.3f}  "
            f"|z| {results.mean_abs_z:.3f}  | "
            f"Tier-2: r̄ {results.tier2_pearson_mean:.3f}  "
            f"JS̄ {results.tier2_js_mean:.3f}"
        )
