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
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score

from yeastbench.adapters.protocols import CoverageTrackPredictor
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo

WINDOW_LEN = 4992          # Legacy Yorzoi window — kept as an importable
                           # default for tests and scripts. The benchmark
                           # itself reads the window length from the loaded
                           # TSV's `window_len` column so a single benchmark
                           # class supports both the Yorzoi (4992) and the
                           # Shorkie (16384) distributions.
PSEUDOCOUNT = 1.0
MIN_READS_PER_RUN = 10     # per-JS94-run raw read floor for that run to
                           # contribute a per-replicate true_lfc for the sample
RANGE_EPS = 1e-6           # avoid division by zero in |z| when min == max


JS94_REPLICATE_STRAIN_KEYS: tuple[str, ...] = ("JS94_r0", "JS94_r1", "JS94_r2")


@dataclass(frozen=True)
class BrooksResults:
    sample_ids: list[str]
    # Per-replicate predicted LFCs. For each sample and each JS94 deep WT
    # run k, pred_lfc_runs[i, k] = log2((alt_cds + 1) / (nat_cds_k + 1)),
    # where nat_cds_k is the model's prediction using only JS94's k-th
    # track. NaN if the truth side's j_raws[i, k] is below
    # MIN_READS_PER_RUN (no useful comparison axis).
    pred_lfc_runs: np.ndarray        # (N, 3) float64
    # Per-replicate true LFCs (same shape, same NaN structure).
    true_lfc_runs: np.ndarray        # (N, 3) float64
    n_reps_supported: np.ndarray     # (N,) int — finite-count per row
    low_support: np.ndarray          # (N,) bool — strain-side only
    # Cohort counts
    n_total: int
    n_scored: int                    # n_reps >= 1 AND not low_support
    n_calibration: int               # n_reps >= 2 AND not low_support
    n_weak_baseline: int             # n_reps == 0 (per-gene, all JS94 thin)
    n_low_support: int               # low_support == True
    # Per-replicate headline + ceiling. Each k uses only samples where
    # both true_lfc_runs[:, k] and pred_lfc_runs[:, k] are finite and
    # the sample is not low_support; ceiling_k uses the mean of the
    # *other* JS94 replicates as a "test-retest predictor".
    pearson_r_per_rep: np.ndarray       # (3,) float64
    spearman_rho_per_rep: np.ndarray    # (3,) float64
    dir_balanced_acc_per_rep: np.ndarray  # (3,) float64
    ceiling_r_per_rep: np.ndarray       # (3,) float64
    ceiling_dir_acc_per_rep: np.ndarray  # (3,) float64
    # Headline = mean across replicates (NaN-aware).
    pearson_r: float
    spearman_rho: float
    dir_balanced_acc: float
    ceiling_pearson_r: float
    ceiling_dir_balanced_acc: float
    # Calibration on the sample-level mean LFCs (n_reps >= 2 cohort)
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
                    "strand", "sample_id", "window_len"):
            assert col in df.columns, f"{col} missing from {self.data_path}"
        # Window length is set by the builder per distribution file. A
        # single TSV must use one consistent window length.
        window_len = int(df.window_len.iloc[0])
        assert (df.window_len == window_len).all(), (
            f"all rows of {self.data_path} must share window_len"
        )
        assert (df.alt_seq.str.len() == window_len).all()
        assert (df.native_seq.str.len() == window_len).all()
        self.window_len = window_len
        self.df = df.reset_index(drop=True)

    def _parse_cov(self, s: str) -> np.ndarray:
        return np.fromstring(s, sep=",", dtype=np.int32)

    def _parse_norm_runs(self, s: str) -> np.ndarray:
        return np.fromstring(s, sep=",", dtype=np.float64)

    def _parse_raw_runs(self, s: str) -> np.ndarray:
        return np.fromstring(s, sep=",", dtype=np.int64)

    def _run_batched(
        self,
        adapter: CoverageTrackPredictor,
        seqs: list[str],
        strands: list[str],
        strains: list[str | None],
        *,
        desc: str,
    ) -> np.ndarray:
        """Chunk a list of inputs into adapter-sized batches and call
        ``predict_coverage_batch`` on each chunk. Returns shape
        ``(len(seqs), out_len)``."""
        from tqdm import tqdm

        if not seqs:
            return np.empty((0, 0), dtype=np.float64)
        bs = max(1, int(getattr(adapter, "batch_size", 1)))
        out_chunks: list[np.ndarray] = []
        for start in tqdm(range(0, len(seqs), bs), desc=desc, ncols=80):
            end = min(start + bs, len(seqs))
            arr = adapter.predict_coverage_batch(
                seqs=seqs[start:end],
                strands=strands[start:end],
                strains=strains[start:end],
            )
            out_chunks.append(np.asarray(arr, dtype=np.float64))
        return np.concatenate(out_chunks, axis=0)

    def evaluate(self, adapter: CoverageTrackPredictor) -> BrooksResults:
        n = len(self.df)
        n_reps = len(JS94_REPLICATE_STRAIN_KEYS)
        # Per-replicate prediction + truth LFCs, same (N, 3) shape.
        pred_lfc_runs = np.full((n, n_reps), np.nan, dtype=np.float64)
        true_lfc_runs = np.full((n, n_reps), np.nan, dtype=np.float64)
        tier2_pearson = np.full(n, np.nan)
        tier2_js = np.full(n, np.nan)

        crop = adapter.crop_bp_each_side
        out_len = adapter.seq_len - 2 * crop  # per-base prediction length
        assert adapter.seq_len == self.window_len, (
            f"adapter seq_len {adapter.seq_len} != distribution window_len "
            f"{self.window_len}; pick a TSV that matches the model "
            "(`brooks_scramble_v1.tsv` for Yorzoi @ 4992, "
            "`brooks_scramble_v1_w16384.tsv` for Shorkie @ 16384)."
        )
        varies_by_strain = bool(getattr(adapter, "varies_by_strain", True))

        # ── Phase 1: per-replicate true LFCs (no GPU work) ──
        j_raws_all = np.zeros((n, n_reps), dtype=np.int64)
        for i, row in self.df.iterrows():
            s_norm = float(row.norm_cov_strain)
            j_norms = self._parse_norm_runs(row.norm_cov_js94_runs)
            j_raws = self._parse_raw_runs(row.js94_reads_runs)
            for k in range(min(len(j_norms), len(j_raws), n_reps)):
                j_raws_all[i, k] = int(j_raws[k])
                if j_raws[k] < MIN_READS_PER_RUN:
                    continue
                true_lfc_runs[i, k] = float(np.log2(
                    (s_norm + PSEUDOCOUNT) / (j_norms[k] + PSEUDOCOUNT)
                ))

        # ── Phase 2: batched alt predictions across all samples ──
        all_alt_seqs = self.df.alt_seq.tolist()
        all_strands = self.df.strand.tolist()
        all_strains = self.df.strain.tolist()
        pred_alt_all = self._run_batched(
            adapter, all_alt_seqs, all_strands, all_strains,
            desc=f"alt   (n={n})",
        )
        assert pred_alt_all.shape == (n, out_len), (
            f"adapter returned {pred_alt_all.shape}, expected "
            f"({n}, {out_len})"
        )

        # ── Phase 3: batched native predictions ──
        # `pred_nat_runs[i, k]` is the model's prediction for sample i
        # against JS94 replicate k. NaN where unused (truth NaN).
        pred_nat_runs = np.full((n, n_reps, out_len), np.nan,
                                 dtype=np.float64)
        all_native_seqs = self.df.native_seq.tolist()
        if varies_by_strain:
            # One batched call per JS94 replicate; restrict to samples
            # that need this replicate (truth is finite for it).
            for k, alias in enumerate(JS94_REPLICATE_STRAIN_KEYS):
                mask = np.isfinite(true_lfc_runs[:, k])
                idx = np.where(mask)[0]
                if idx.size == 0:
                    continue
                sub_pred = self._run_batched(
                    adapter,
                    seqs=[all_native_seqs[i] for i in idx],
                    strands=[all_strands[i] for i in idx],
                    strains=[alias] * idx.size,
                    desc=f"nat {alias} (n={idx.size})",
                )
                pred_nat_runs[idx, k] = sub_pred
        else:
            # One forward across all samples; broadcast into supported reps.
            pred_nat_one = self._run_batched(
                adapter,
                seqs=all_native_seqs,
                strands=all_strands,
                strains=["JS94"] * n,
                desc=f"nat   (n={n})",
            )
            for k in range(n_reps):
                mask = np.isfinite(true_lfc_runs[:, k])
                pred_nat_runs[mask, k] = pred_nat_one[mask]

        # ── Phase 4: per-sample LFCs + Tier-2 shape (CPU only) ──
        for i, row in self.df.iterrows():
            pred_alt = pred_alt_all[i]
            cs = max(0, int(row.cds_start_in_window) - crop)
            ce = min(out_len, int(row.cds_end_in_window) - crop)
            if ce <= cs:
                continue
            alt_cds = pred_alt[cs:ce].sum()

            for k in range(n_reps):
                if not np.isfinite(true_lfc_runs[i, k]):
                    continue
                nat_cds_k = pred_nat_runs[i, k, cs:ce].sum()
                pred_lfc_runs[i, k] = float(np.log2(
                    (alt_cds + PSEUDOCOUNT) / (nat_cds_k + PSEUDOCOUNT)
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

        # Per-sample replicate counts (truth side; pred side mirrors it
        # by construction since we only ran pred when truth was finite).
        finite_true = np.isfinite(true_lfc_runs)
        n_reps_supported = finite_true.sum(axis=1).astype(np.int64)

        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_true = np.where(
                n_reps_supported > 0,
                np.nanmean(true_lfc_runs, axis=1),
                np.nan,
            )
            mean_pred = np.where(
                n_reps_supported > 0,
                np.nanmean(pred_lfc_runs, axis=1),
                np.nan,
            )

        low = self.df.low_support.to_numpy(dtype=bool)
        scored_mask = (
            (~low) & (n_reps_supported >= 1)
            & np.isfinite(mean_pred) & np.isfinite(mean_true)
        )
        calib_mask = scored_mask & (n_reps_supported >= 2)
        n_scored = int(scored_mask.sum())
        n_calibration = int(calib_mask.sum())
        n_weak_baseline = int((n_reps_supported == 0).sum())
        n_low = int(low.sum())

        # ── per-replicate headline metrics ───────────────────────
        # For each JS94 replicate k, compute r/ρ/dir-acc over samples
        # where both pred and truth are finite for that replicate and
        # the sample is not low_support. Headline = mean across k.
        pr_per = np.full(n_reps, np.nan)
        sr_per = np.full(n_reps, np.nan)
        da_per = np.full(n_reps, np.nan)
        for k in range(n_reps):
            mk = (
                (~low) & finite_true[:, k] & np.isfinite(pred_lfc_runs[:, k])
            )
            if mk.sum() < 2:
                continue
            t_k = true_lfc_runs[mk, k]
            p_k = pred_lfc_runs[mk, k]
            pr_per[k] = float(pearsonr(p_k, t_k).statistic)
            sr_per[k] = float(spearmanr(p_k, t_k).statistic)
            da_per[k] = float(balanced_accuracy_score(
                np.sign(t_k).astype(int), np.sign(p_k).astype(int),
            ))

        # ── LOO reproducibility ceiling ──────────────────────────
        # For each k, compare true_lfc_runs[:, k] against the mean of
        # the other replicates' true LFCs. The correlation between the
        # two is an upper bound on what any predictor can achieve on
        # the per-replicate k label (denominator-side noise only;
        # strain numerator is single-rep, so this is conservative).
        ceil_pr_per = np.full(n_reps, np.nan)
        ceil_da_per = np.full(n_reps, np.nan)
        for k in range(n_reps):
            others = [j for j in range(n_reps) if j != k]
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                loo_true = np.nanmean(true_lfc_runs[:, others], axis=1)
            mk = (~low) & finite_true[:, k] & np.isfinite(loo_true)
            if mk.sum() < 2:
                continue
            t_k = true_lfc_runs[mk, k]
            l_k = loo_true[mk]
            ceil_pr_per[k] = float(pearsonr(l_k, t_k).statistic)
            ceil_da_per[k] = float(balanced_accuracy_score(
                np.sign(t_k).astype(int), np.sign(l_k).astype(int),
            ))

        pr = float(np.nanmean(pr_per)) if np.any(np.isfinite(pr_per)) else float("nan")
        sr = float(np.nanmean(sr_per)) if np.any(np.isfinite(sr_per)) else float("nan")
        da = float(np.nanmean(da_per)) if np.any(np.isfinite(da_per)) else float("nan")
        ceiling_pr = (float(np.nanmean(ceil_pr_per))
                      if np.any(np.isfinite(ceil_pr_per)) else float("nan"))
        ceiling_da = (float(np.nanmean(ceil_da_per))
                      if np.any(np.isfinite(ceil_da_per)) else float("nan"))

        # ── calibration on the sample-mean LFCs (n_reps >= 2 cohort) ──
        if n_calibration < 1:
            within_range_rate = mean_abs_z = float("nan")
        else:
            idx = np.where(calib_mask)[0]
            hits = 0
            zs = []
            for ii in idx:
                row_runs = true_lfc_runs[ii][finite_true[ii]]
                lo_v, hi_v = float(row_runs.min()), float(row_runs.max())
                if lo_v <= mean_pred[ii] <= hi_v:
                    hits += 1
                rng = max(hi_v - lo_v, RANGE_EPS)
                zs.append(abs(mean_pred[ii] - mean_true[ii]) / rng)
            within_range_rate = hits / n_calibration
            mean_abs_z = float(np.mean(zs))

        t2p = (float(np.nanmean(tier2_pearson[scored_mask]))
               if n_scored else float("nan"))
        t2j = (float(np.nanmean(tier2_js[scored_mask]))
               if n_scored else float("nan"))

        return BrooksResults(
            sample_ids=self.df.sample_id.tolist(),
            pred_lfc_runs=pred_lfc_runs,
            true_lfc_runs=true_lfc_runs,
            n_reps_supported=n_reps_supported,
            low_support=low,
            n_total=n, n_scored=n_scored, n_calibration=n_calibration,
            n_weak_baseline=n_weak_baseline, n_low_support=n_low,
            pearson_r_per_rep=pr_per, spearman_rho_per_rep=sr_per,
            dir_balanced_acc_per_rep=da_per,
            ceiling_r_per_rep=ceil_pr_per,
            ceiling_dir_acc_per_rep=ceil_da_per,
            pearson_r=pr, spearman_rho=sr, dir_balanced_acc=da,
            ceiling_pearson_r=ceiling_pr, ceiling_dir_balanced_acc=ceiling_da,
            within_range_rate=within_range_rate, mean_abs_z=mean_abs_z,
            tier2_pearson_mean=t2p, tier2_js_mean=t2j,
        )

    def plot(self, results: BrooksResults, out_dir: Path) -> None:
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        title_model = out_dir.name.split("__")[0] if "__" in out_dir.name else ""

        finite_true = np.isfinite(results.true_lfc_runs)
        finite_pred = np.isfinite(results.pred_lfc_runs)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_true = np.where(
                results.n_reps_supported > 0,
                np.nanmean(results.true_lfc_runs, axis=1),
                np.nan,
            )
            mean_pred = np.where(
                results.n_reps_supported > 0,
                np.nanmean(results.pred_lfc_runs, axis=1),
                np.nan,
            )
        m = (~results.low_support) & (results.n_reps_supported >= 1) \
            & np.isfinite(mean_pred) & np.isfinite(mean_true)

        # ── Tier-1 scatter — mean pred vs mean true, with replicate
        # envelopes shown as crosshair error bars on both axes ──
        p_arr, t_arr = mean_pred[m], mean_true[m]
        true_lo = np.array([
            results.true_lfc_runs[ii][finite_true[ii]].min()
            for ii in np.where(m)[0]
        ])
        true_hi = np.array([
            results.true_lfc_runs[ii][finite_true[ii]].max()
            for ii in np.where(m)[0]
        ])
        pred_lo = np.array([
            results.pred_lfc_runs[ii][finite_pred[ii]].min()
            if finite_pred[ii].any() else mean_pred[ii]
            for ii in np.where(m)[0]
        ])
        pred_hi = np.array([
            results.pred_lfc_runs[ii][finite_pred[ii]].max()
            if finite_pred[ii].any() else mean_pred[ii]
            for ii in np.where(m)[0]
        ])

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
        # Clamp to >=0; tiny float-precision noise around mean ≈ min ≈ max
        # for broadcast (varies_by_strain=False) predictions has slipped
        # below zero in practice and matplotlib's errorbar rejects it.
        x_lo = np.maximum(t_arr - true_lo, 0.0)
        x_hi = np.maximum(true_hi - t_arr, 0.0)
        y_lo = np.maximum(p_arr - pred_lo, 0.0)
        y_hi = np.maximum(pred_hi - p_arr, 0.0)
        ax.errorbar(
            t_arr, p_arr, xerr=[x_lo, x_hi], yerr=[y_lo, y_hi],
            fmt="o", ms=4, ecolor="lightgrey", elinewidth=1, alpha=0.7,
        )
        lim = max(np.nanmax(np.abs(t_arr)), np.nanmax(np.abs(p_arr)), 1) + 0.5
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("true log2 LFC (mean over supporting JS94 runs)")
        ax.set_ylabel("predicted log2 LFC (mean over supporting JS94 runs)")
        ax.set_title(
            f"Brooks SCRaMBLE — Tier 1"
            + (f" — {title_model}" if title_model else "")
            + f"\nn_scored={results.n_scored}  "
            f"dir-acc={results.dir_balanced_acc:.3f} "
            f"(ceiling {results.ceiling_dir_balanced_acc:.3f})  "
            f"r={results.pearson_r:.3f} "
            f"(ceiling {results.ceiling_pearson_r:.3f})  "
            f"ρ={results.spearman_rho:.3f}\n"
            f"calibration (n={results.n_calibration}): "
            f"within-range={results.within_range_rate:.3f}  "
            f"|z|={results.mean_abs_z:.3f}"
        )
        fig.tight_layout(); fig.savefig(out_dir / "tier1_scatter.png", dpi=150)
        plt.close(fig)

        # ── Per-sample interval plot — every scored sample side by
        # side, true (blue) and pred (orange) ranges with mean dot.
        # Wide canvas; sort by mean true LFC for visual order. ──
        idx_sorted = np.array(sorted(
            np.where(m)[0], key=lambda i: mean_true[i],
        ))
        K = len(idx_sorted)
        if K > 0:
            fig_w = max(8.0, 0.08 * K)        # ~0.08" per sample
            fig, ax = plt.subplots(figsize=(fig_w, 6))
            ax.axhline(0, color="grey", lw=0.3)
            x = np.arange(K, dtype=float)
            off = 0.18
            # True (blue)
            true_means = mean_true[idx_sorted]
            t_lo = np.array([
                results.true_lfc_runs[ii][finite_true[ii]].min()
                for ii in idx_sorted
            ])
            t_hi = np.array([
                results.true_lfc_runs[ii][finite_true[ii]].max()
                for ii in idx_sorted
            ])
            ax.vlines(x - off, t_lo, t_hi, colors="#1f77b4",
                      lw=1.0, alpha=0.7)
            ax.scatter(x - off, true_means, s=8, c="#1f77b4",
                       label="true")
            # Pred (orange) — handle samples with only 1 finite pred (no range)
            pred_means = mean_pred[idx_sorted]
            p_lo = np.array([
                results.pred_lfc_runs[ii][finite_pred[ii]].min()
                if finite_pred[ii].any() else mean_pred[ii]
                for ii in idx_sorted
            ])
            p_hi = np.array([
                results.pred_lfc_runs[ii][finite_pred[ii]].max()
                if finite_pred[ii].any() else mean_pred[ii]
                for ii in idx_sorted
            ])
            ax.vlines(x + off, p_lo, p_hi, colors="#ff7f0e",
                      lw=1.0, alpha=0.7)
            ax.scatter(x + off, pred_means, s=8, c="#ff7f0e",
                       label="pred")
            ax.set_xlim(-1, K)
            ax.set_xlabel(f"sample (sorted by mean true LFC, n={K})")
            ax.set_ylabel("log2 LFC (alt / native)")
            ax.set_title(
                "Brooks SCRaMBLE — per-sample LFC ranges"
                + (f" — {title_model}" if title_model else "")
                + f"   |   r={results.pearson_r:.3f} "
                f"(ceiling {results.ceiling_pearson_r:.3f})"
            )
            ax.legend(loc="upper left", fontsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / "tier1_per_sample.png", dpi=100)
            plt.close(fig)

    def save_results(self, results: BrooksResults, out_dir: Path) -> None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "pred_lfc_runs.npy", results.pred_lfc_runs)
        np.save(out_dir / "true_lfc_runs.npy", results.true_lfc_runs)
        np.save(out_dir / "n_reps_supported.npy", results.n_reps_supported)
        (out_dir / "samples.json").write_text(json.dumps({
            "sample_ids": results.sample_ids,
            "low_support": results.low_support.tolist(),
        }, indent=2))

    def load_results(self, out_dir: Path) -> BrooksResults:
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "samples.json").read_text())
        pred_lfc_runs = np.load(out_dir / "pred_lfc_runs.npy")
        true_lfc_runs = np.load(out_dir / "true_lfc_runs.npy")
        n_reps_supported = np.load(out_dir / "n_reps_supported.npy").astype(np.int64)
        low = np.asarray(meta["low_support"], dtype=bool)
        n = len(low)
        n_reps_k = pred_lfc_runs.shape[1]

        finite_true = np.isfinite(true_lfc_runs)
        finite_pred = np.isfinite(pred_lfc_runs)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_true = np.where(
                n_reps_supported > 0, np.nanmean(true_lfc_runs, axis=1), np.nan
            )
            mean_pred = np.where(
                n_reps_supported > 0, np.nanmean(pred_lfc_runs, axis=1), np.nan
            )
        scored = (~low) & (n_reps_supported >= 1) \
            & np.isfinite(mean_pred) & np.isfinite(mean_true)
        calib = scored & (n_reps_supported >= 2)

        pr_per = np.full(n_reps_k, np.nan)
        sr_per = np.full(n_reps_k, np.nan)
        da_per = np.full(n_reps_k, np.nan)
        ceil_pr_per = np.full(n_reps_k, np.nan)
        ceil_da_per = np.full(n_reps_k, np.nan)
        for k in range(n_reps_k):
            mk = (~low) & finite_true[:, k] & finite_pred[:, k]
            if mk.sum() >= 2:
                t_k = true_lfc_runs[mk, k]; p_k = pred_lfc_runs[mk, k]
                pr_per[k] = float(pearsonr(p_k, t_k).statistic)
                sr_per[k] = float(spearmanr(p_k, t_k).statistic)
                da_per[k] = float(balanced_accuracy_score(
                    np.sign(t_k).astype(int), np.sign(p_k).astype(int)))
            others = [j for j in range(n_reps_k) if j != k]
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                loo = np.nanmean(true_lfc_runs[:, others], axis=1)
            ck = (~low) & finite_true[:, k] & np.isfinite(loo)
            if ck.sum() >= 2:
                t_k = true_lfc_runs[ck, k]; l_k = loo[ck]
                ceil_pr_per[k] = float(pearsonr(l_k, t_k).statistic)
                ceil_da_per[k] = float(balanced_accuracy_score(
                    np.sign(t_k).astype(int), np.sign(l_k).astype(int)))

        if calib.sum() >= 1:
            hits = 0; zs = []
            for ii in np.where(calib)[0]:
                runs = true_lfc_runs[ii][finite_true[ii]]
                lo_v, hi_v = float(runs.min()), float(runs.max())
                if lo_v <= mean_pred[ii] <= hi_v:
                    hits += 1
                zs.append(abs(mean_pred[ii] - mean_true[ii])
                          / max(hi_v - lo_v, RANGE_EPS))
            within = hits / calib.sum()
            mz = float(np.mean(zs))
        else:
            within = mz = float("nan")

        return BrooksResults(
            sample_ids=meta["sample_ids"],
            pred_lfc_runs=pred_lfc_runs, true_lfc_runs=true_lfc_runs,
            n_reps_supported=n_reps_supported, low_support=low,
            n_total=n, n_scored=int(scored.sum()),
            n_calibration=int(calib.sum()),
            n_weak_baseline=int((n_reps_supported == 0).sum()),
            n_low_support=int(low.sum()),
            pearson_r_per_rep=pr_per, spearman_rho_per_rep=sr_per,
            dir_balanced_acc_per_rep=da_per,
            ceiling_r_per_rep=ceil_pr_per,
            ceiling_dir_acc_per_rep=ceil_da_per,
            pearson_r=float(np.nanmean(pr_per)) if np.any(np.isfinite(pr_per)) else float("nan"),
            spearman_rho=float(np.nanmean(sr_per)) if np.any(np.isfinite(sr_per)) else float("nan"),
            dir_balanced_acc=float(np.nanmean(da_per)) if np.any(np.isfinite(da_per)) else float("nan"),
            ceiling_pearson_r=float(np.nanmean(ceil_pr_per)) if np.any(np.isfinite(ceil_pr_per)) else float("nan"),
            ceiling_dir_balanced_acc=float(np.nanmean(ceil_da_per)) if np.any(np.isfinite(ceil_da_per)) else float("nan"),
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
            "tier1_ceiling_dir_balanced_acc": results.ceiling_dir_balanced_acc,
            "tier1_ceiling_pearson_r": results.ceiling_pearson_r,
            "tier1_pearson_r_per_rep": results.pearson_r_per_rep.tolist(),
            "tier1_spearman_rho_per_rep": results.spearman_rho_per_rep.tolist(),
            "tier1_dir_balanced_acc_per_rep":
                results.dir_balanced_acc_per_rep.tolist(),
            "tier1_ceiling_r_per_rep": results.ceiling_r_per_rep.tolist(),
            "tier1_ceiling_dir_acc_per_rep":
                results.ceiling_dir_acc_per_rep.tolist(),
            "tier1_within_range_rate": results.within_range_rate,
            "tier1_mean_abs_z": results.mean_abs_z,
            "tier2_pearson_mean": results.tier2_pearson_mean,
            "tier2_js_mean": results.tier2_js_mean,
        }

    def headline(self, results: BrooksResults) -> str:
        return (
            f"Tier-1 (n_scored={results.n_scored}): "
            f"dir-acc {results.dir_balanced_acc:.3f} "
            f"(ceiling {results.ceiling_dir_balanced_acc:.3f})  "
            f"r {results.pearson_r:.3f} "
            f"(ceiling {results.ceiling_pearson_r:.3f})  "
            f"ρ {results.spearman_rho:.3f}  | "
            f"calibration (n={results.n_calibration}): "
            f"within-range {results.within_range_rate:.3f}  "
            f"|z| {results.mean_abs_z:.3f}  | "
            f"Tier-2: r̄ {results.tier2_pearson_mean:.3f}  "
            f"JS̄ {results.tier2_js_mean:.3f}"
        )

    # ── Cross-model comparison override ──────────────────────────────────
    #
    # Different receptive fields → different sample sets per model
    # (Yorzoi @ 4992 bp dedups byte-identical copies that Shorkie @
    # 16,384 bp keeps separate). Headline numbers are computed on the
    # **intersection of sample_ids** so the comparison is apples-to-
    # apples; per-model full-set metrics are recorded as secondary.
    # Generalises to N models, not just two.

    @property
    def compare_task_name(self) -> str:
        """Brooks ships two registry entries (`brooks_scramble` for
        Yorzoi @ 4992, `brooks_scramble_shorkie` for Shorkie @ 16384)
        because the two models need differently-sized distributions.
        Cross-model comparisons should treat them as the same task —
        the canonical name is `brooks_scramble`."""
        return "brooks_scramble"

    def compare_plot(
        self,
        model_dirs: Mapping[str, Path],
        out_dir: Path,
    ) -> Path | None:
        """Shared-cohort comparison across N models. Writes:

          - ``shared_tier1.png``: bar chart of Pearson r / Spearman ρ /
            dir-acc per model on the shared cohort, with the LOO
            reproducibility ceiling marked as a grey dashed line.
          - ``shared_per_sample.png``: per-sample interval plot — every
            shared-cohort sample gets one blue range (truth) plus one
            range per model, sorted left-to-right by mean true LFC.
          - ``summary.json``: shared-cohort + secondary full-set numbers
            for every model.

        Returns the Tier-1 plot path so the runner can include it in
        the cross-task mosaic."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        loaded = {
            name: _load_brooks_run_dir(Path(d)) for name, d in model_dirs.items()
        }
        # Drop any that don't have the per-replicate arrays (older / partial
        # runs). Need at least 2 to compare.
        loaded = {n: r for n, r in loaded.items() if r is not None}
        if len(loaded) < 2:
            return None

        shared = sorted(
            set.intersection(*(set(r["sample_ids"]) for r in loaded.values()))
        )
        if not shared:
            return None
        indexers = {
            name: np.array([{sid: i for i, sid in enumerate(r["sample_ids"])}[s]
                              for s in shared])
            for name, r in loaded.items()
        }

        shared_cohort: dict[str, Any] = {
            "sample_ids": shared,
            "n": len(shared),
        }
        for name, r in loaded.items():
            shared_cohort[name] = _brooks_metrics(r, indexers[name])

        secondary = {
            name: _brooks_metrics(r, np.arange(len(r["sample_ids"])))
            for name, r in loaded.items()
        }

        out_summary = {
            "shared_cohort": shared_cohort,
            "secondary_full_set": secondary,
            "note": (
                "Headline metrics are computed on the intersection of all "
                "models' sample sets. Full-set metrics for each model are "
                "kept as secondary so the gap is documented."
            ),
        }
        (out_dir / "summary.json").write_text(json.dumps(out_summary, indent=2))

        plot_path = out_dir / "shared_tier1.png"
        _plot_brooks_shared_metrics(loaded, indexers, shared_cohort, plot_path)
        _plot_brooks_shared_per_sample(
            loaded, indexers, out_dir / "shared_per_sample.png"
        )
        return plot_path


# ── Brooks-specific compare helpers (lifted from
#    scripts/brooks/compare_models.py) ──────────────────────────────────


def _load_brooks_run_dir(model_dir: Path) -> dict | None:
    """Load one model's per-replicate prediction arrays + sample IDs.
    Returns None if the expected files aren't present (older / partial
    runs from before the per-replicate framework)."""
    samples_path = model_dir / "samples.json"
    pred_path = model_dir / "pred_lfc_runs.npy"
    true_path = model_dir / "true_lfc_runs.npy"
    n_reps_path = model_dir / "n_reps_supported.npy"
    if not all(p.exists() for p in (samples_path, pred_path, true_path,
                                       n_reps_path)):
        return None
    meta = json.loads(samples_path.read_text())
    return {
        "sample_ids": meta["sample_ids"],
        "low_support": np.asarray(meta["low_support"], dtype=bool),
        "pred_lfc_runs": np.load(pred_path),
        "true_lfc_runs": np.load(true_path),
        "n_reps_supported": np.load(n_reps_path),
    }


def _brooks_metrics(d: dict, idx: np.ndarray) -> dict:
    """Per-replicate r / ρ / dir-acc + LOO ceiling + calibration metrics
    on the row subset ``idx``. Mirrors the in-class compute path so the
    cross-model comparison uses identical definitions."""
    pred = d["pred_lfc_runs"][idx]
    true = d["true_lfc_runs"][idx]
    n_reps = d["n_reps_supported"][idx]
    low = d["low_support"][idx]
    nk = pred.shape[1]

    pr_per = np.full(nk, np.nan)
    sr_per = np.full(nk, np.nan)
    da_per = np.full(nk, np.nan)
    ceil_pr_per = np.full(nk, np.nan)
    ceil_da_per = np.full(nk, np.nan)

    finite_t = np.isfinite(true)
    finite_p = np.isfinite(pred)
    for k in range(nk):
        mk = (~low) & finite_t[:, k] & finite_p[:, k]
        if mk.sum() >= 2:
            t_k, p_k = true[mk, k], pred[mk, k]
            pr_per[k] = float(pearsonr(p_k, t_k).statistic)
            sr_per[k] = float(spearmanr(p_k, t_k).statistic)
            da_per[k] = float(balanced_accuracy_score(
                np.sign(t_k).astype(int), np.sign(p_k).astype(int)))
        others = [j for j in range(nk) if j != k]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            loo = np.nanmean(true[:, others], axis=1)
        ck = (~low) & finite_t[:, k] & np.isfinite(loo)
        if ck.sum() >= 2:
            t_k, l_k = true[ck, k], loo[ck]
            ceil_pr_per[k] = float(pearsonr(l_k, t_k).statistic)
            ceil_da_per[k] = float(balanced_accuracy_score(
                np.sign(t_k).astype(int), np.sign(l_k).astype(int)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_true = np.where(n_reps > 0, np.nanmean(true, axis=1), np.nan)
        mean_pred = np.where(n_reps > 0, np.nanmean(pred, axis=1), np.nan)

    calib_mask = (
        (~low) & (n_reps >= 2)
        & np.isfinite(mean_true) & np.isfinite(mean_pred)
    )
    if calib_mask.sum() >= 1:
        hits = 0
        zs = []
        for ii in np.where(calib_mask)[0]:
            runs = true[ii][finite_t[ii]]
            lo, hi = float(runs.min()), float(runs.max())
            if lo <= mean_pred[ii] <= hi:
                hits += 1
            zs.append(abs(mean_pred[ii] - mean_true[ii])
                       / max(hi - lo, RANGE_EPS))
        within = hits / calib_mask.sum()
        mz = float(np.mean(zs))
    else:
        within = mz = float("nan")

    scored = (~low) & (n_reps >= 1)
    return {
        "n_total": int(len(idx)),
        "n_low_support": int(low.sum()),
        "n_scored": int(scored.sum()),
        "n_calibration": int(calib_mask.sum()),
        "n_weak_baseline": int((n_reps == 0).sum()),
        "pearson_r": (float(np.nanmean(pr_per))
                       if np.any(np.isfinite(pr_per)) else float("nan")),
        "spearman_rho": (float(np.nanmean(sr_per))
                          if np.any(np.isfinite(sr_per)) else float("nan")),
        "dir_balanced_acc": (float(np.nanmean(da_per))
                              if np.any(np.isfinite(da_per)) else float("nan")),
        "ceiling_pearson_r": (float(np.nanmean(ceil_pr_per))
                               if np.any(np.isfinite(ceil_pr_per)) else float("nan")),
        "ceiling_dir_balanced_acc": (float(np.nanmean(ceil_da_per))
                                      if np.any(np.isfinite(ceil_da_per))
                                      else float("nan")),
        "pearson_r_per_rep": pr_per.tolist(),
        "spearman_rho_per_rep": sr_per.tolist(),
        "dir_balanced_acc_per_rep": da_per.tolist(),
        "ceiling_pearson_r_per_rep": ceil_pr_per.tolist(),
        "ceiling_dir_balanced_acc_per_rep": ceil_da_per.tolist(),
        "within_range_rate": within,
        "mean_abs_z": mz,
    }


def _plot_brooks_shared_metrics(
    loaded: Mapping[str, dict],
    indexers: Mapping[str, np.ndarray],
    shared_cohort: dict,
    out_path: Path,
) -> None:
    """Bar chart of Pearson r / Spearman ρ / dir-acc per model on the
    shared cohort. Ceiling marked as a grey dashed line where defined."""
    import matplotlib.pyplot as plt

    rows = [
        ("Pearson r",   "pearson_r",         "ceiling_pearson_r"),
        ("Spearman ρ",  "spearman_rho",      None),
        ("dir-acc",     "dir_balanced_acc",  "ceiling_dir_balanced_acc"),
    ]
    model_names = sorted(loaded.keys())
    fig, axes = plt.subplots(len(rows), 1, figsize=(8, 1.7 * len(rows)),
                              squeeze=False)
    for i, (metric_name, key, ceil_key) in enumerate(rows):
        ax = axes[i, 0]
        ys = [shared_cohort[m][key] for m in model_names]
        bars = ax.barh(model_names, ys, alpha=0.85)
        ax.axvline(0, color="grey", lw=0.5)
        if ceil_key:
            # The ceiling depends only on the truth labels, which are
            # identical for all models on the shared cohort — read it
            # from the first model.
            ceil = shared_cohort[model_names[0]][ceil_key]
            ax.axvline(ceil, color="grey", lw=1.0, ls="--",
                        label=f"LOO ceiling {ceil:+.3f}")
            ax.legend(loc="lower right", fontsize=8)
        for b, v in zip(bars, ys):
            ax.text(v + 0.005, b.get_y() + b.get_height() / 2,
                     f"{v:+.3f}", va="center", fontsize=9)
        all_vals = ys + ([shared_cohort[model_names[0]].get(ceil_key, 0)]
                          if ceil_key else [])
        ax.set_xlim(min(-0.05, min(all_vals) - 0.05),
                    max(1.0, *all_vals) * 1.05 + 0.05)
        ax.set_title(
            f"{metric_name}  (shared cohort, n_scored="
            f"{shared_cohort[model_names[0]]['n_scored']})",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# Palette for the per-sample plot: truth is blue, models cycle through
# matplotlib's default cycle after that.
_TRUTH_COLOR = "#1f77b4"
_MODEL_COLORS = ("#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")


def _plot_brooks_shared_per_sample(
    loaded: Mapping[str, dict],
    indexers: Mapping[str, np.ndarray],
    out_path: Path,
) -> None:
    """Per-sample interval plot on the shared cohort. Each scored sample
    becomes one column: truth (blue) + one range per model (red, green,
    purple, …). Models with `varies_by_strain=False` collapse to a dot.
    Sorted left-to-right by mean true LFC."""
    import matplotlib.pyplot as plt

    model_names = sorted(loaded.keys())
    # Use the first model's truth as the canonical truth (identical
    # across models on the shared cohort by construction).
    ref = loaded[model_names[0]]
    ref_idx = indexers[model_names[0]]
    true = ref["true_lfc_runs"][ref_idx]
    n_reps_ref = ref["n_reps_supported"][ref_idx]
    finite_t = np.isfinite(true)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_true = np.where(n_reps_ref > 0, np.nanmean(true, axis=1), np.nan)

    # Which samples are scored on every model?
    scored_mask = (~ref["low_support"][ref_idx]) & (n_reps_ref >= 1) & np.isfinite(mean_true)
    for m in model_names:
        idx = indexers[m]
        scored_mask = scored_mask & (~loaded[m]["low_support"][idx])
        pred = loaded[m]["pred_lfc_runs"][idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            finite_p = np.isfinite(pred)
            mean_pred = np.where(
                finite_p.any(axis=1), np.nanmean(pred, axis=1), np.nan
            )
        scored_mask = scored_mask & np.isfinite(mean_pred)

    K = int(scored_mask.sum())
    if K == 0:
        return
    idx_sorted = np.array(sorted(np.where(scored_mask)[0],
                                  key=lambda i: mean_true[i]))
    x = np.arange(K, dtype=float)

    # Layout: 1 (truth) + N model entries; one column per sample with
    # equal-spaced x-offsets.
    n_series = 1 + len(model_names)
    total_width = 0.85
    spacing = total_width / n_series
    offsets = np.linspace(
        -total_width / 2 + spacing / 2,
        total_width / 2 - spacing / 2,
        n_series,
    )

    fig_w = max(8.0, 0.08 * K)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    ax.axhline(0, color="grey", lw=0.3)

    def _ranges(arr: np.ndarray, mask: np.ndarray) -> tuple[
        np.ndarray, np.ndarray, np.ndarray
    ]:
        lo = np.array([
            arr[i][mask[i]].min() if mask[i].any() else np.nan
            for i in idx_sorted
        ])
        hi = np.array([
            arr[i][mask[i]].max() if mask[i].any() else np.nan
            for i in idx_sorted
        ])
        mn = np.array([
            arr[i][mask[i]].mean() if mask[i].any() else np.nan
            for i in idx_sorted
        ])
        return lo, hi, mn

    t_lo, t_hi, t_mn = _ranges(true, finite_t)
    ax.vlines(x + offsets[0], t_lo, t_hi, colors=_TRUTH_COLOR, lw=1.0, alpha=0.7)
    ax.scatter(x + offsets[0], t_mn, s=8, c=_TRUTH_COLOR, label="true")

    for j, m in enumerate(model_names):
        idx = indexers[m]
        pred = loaded[m]["pred_lfc_runs"][idx]
        finite_p = np.isfinite(pred)
        lo, hi, mn = _ranges(pred, finite_p)
        color = _MODEL_COLORS[j % len(_MODEL_COLORS)]
        ax.vlines(x + offsets[1 + j], lo, hi, colors=color, lw=1.0, alpha=0.7)
        ax.scatter(x + offsets[1 + j], mn, s=8, c=color, label=f"{m} pred")

    ax.set_xlim(-1, K)
    ax.set_xlabel(f"sample (sorted by mean true LFC, n={K})")
    ax.set_ylabel("log2 LFC (alt / native)")
    ax.set_title("Brooks SCRaMBLE — per-sample LFC ranges (shared cohort)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
