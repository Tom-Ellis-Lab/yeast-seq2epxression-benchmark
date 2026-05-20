"""Tests for the Brooks SCRaMBLE benchmark.

The benchmark depends on a single self-contained TSV; the tests build a
tiny one and verify Tier-1 LFC + Tier-2 shape end-to-end against a
mock CoverageTrackPredictor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yeastbench.adapters.protocols import CoverageTrackPredictor
from yeastbench.benchmarks.base import BenchmarkInfo
from yeastbench.benchmarks.brooks import (
    BrooksScrambleBenchmark,
    MIN_READS_PER_RUN,
    WINDOW_LEN,
    _js_divergence,
)
from yeastbench.registry import TASKS

INFO = BenchmarkInfo(name="test_brooks", version="test", description="t",
                     distribution_uri="")

# Mock model geometry (matches Yorzoi numerically; expressed locally so
# the test isn't coupled to the real adapter)
CROP = 996
OUT_LEN = WINDOW_LEN - 2 * CROP   # 3000


def _make_seq(rng: np.random.Generator) -> str:
    return "".join(rng.choice(list("ACGT"), size=WINDOW_LEN))


def _make_cov(rng: np.random.Generator, mean: float = 0.5) -> str:
    return ",".join(map(str, rng.poisson(mean, WINDOW_LEN).astype(int).tolist()))


@pytest.fixture
def brooks_tsv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        # half of samples in each tail of true_lfc, so dir-acc is informative.
        # norm_cov_strain set so that LFC = log2((s+1)/(j+1)) ≈ ±1.5 with
        # a normalised JS94 mean of 10:
        #   2^1.5 * 11 - 1 = 30.10  → s_norm ≈ 30 for +1.5
        #   2^-1.5 * 11 - 1 = 2.89 → s_norm ≈ 2.9 for -1.5
        # We give each gene 3 JS94 raw runs all ≥ MIN_READS_PER_RUN so all
        # samples are in the calibration cohort (n_reps == 3).
        is_up = (i % 2 == 0)
        s_norm = 30.0 if is_up else 2.9
        true_lfc = 1.5 if is_up else -1.5
        norm_runs = ",".join(f"{v:.3f}" for v in rng.uniform(9.5, 10.5, 3))
        raw_runs = ",".join(str(v) for v in rng.integers(20, 60, 3))
        rows.append({
            "sample_id": f"JS6{i:02d}:YIR0{i:02d}W:0",
            "gene_id": f"YIR0{i:02d}W",
            "strain": f"JS60{i}",
            "copy_idx": 0, "n_copies": 1, "strand": "+",
            "rearr_class": "context_change",
            "syn_contig": "JS60_1", "cds_start": 100, "cds_end": 1000,
            "window_len": WINDOW_LEN,
            "cds_start_in_window": 2000, "cds_end_in_window": 2700,
            "alt_seq": _make_seq(rng), "native_seq": _make_seq(rng),
            "true_cov_alt": _make_cov(rng, mean=0.5),
            "true_cov_native": _make_cov(rng, mean=0.5),
            "strain_reads": 250, "js94_reads_runs": raw_runs,
            "size_factor_strain": 4.0,
            "norm_cov_strain": s_norm,
            "norm_cov_js94_mean": 10.0, "norm_cov_js94_runs": norm_runs,
            "true_lfc": true_lfc, "low_support": False,
        })
    p = tmp_path / "mini.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return p


class _MockAdapter:
    """Perfect per-base predictor: returns coverage that yields
    ``pred_lfc = true_lfc``. Alt CDS-region intensity is scaled by
    ``2^true_lfc`` relative to native; everything else is constant noise.
    Returns a length-``OUT_LEN`` (= 3000) per-base vector — already
    untransformed/unbinned, per the protocol contract."""
    seq_len = WINDOW_LEN
    crop_bp_each_side = CROP

    def __init__(self, df: pd.DataFrame, seed: int = 0):
        self._df = df.set_index("alt_seq")
        self._rng = np.random.default_rng(seed)

    def predict_coverage(self, construct_seq: str, strand: str,
                         strain: str | None = None) -> np.ndarray:
        if construct_seq in self._df.index:
            tl = float(self._df.loc[construct_seq, "true_lfc"])
            scale = float(2.0 ** tl)         # alt = scale * native
        else:
            scale = 1.0
        base = self._rng.normal(1.0, 0.05, OUT_LEN).clip(0.1)
        return base * scale


assert isinstance(_MockAdapter(pd.DataFrame({"alt_seq": [], "native_seq": [],
                                             "true_lfc": []})),
                  CoverageTrackPredictor)


def _make_raw_runs(rng: np.random.Generator, *, all_supported: bool = True,
                   below: int = 0) -> str:
    """Three per-JS94-run raw counts. If `below` > 0, that many runs are
    below MIN_READS_PER_RUN (forced to 0) so n_reps_supported drops."""
    vals = list(rng.integers(20, 60, 3))
    for k in range(below):
        vals[k] = 0
    return ",".join(str(v) for v in vals)


# ── helpers ───────────────────────────────────────────────────


class TestHelpers:
    def test_js_divergence_symmetric_and_zero(self):
        p = np.array([0.2, 0.3, 0.5])
        q = np.array([0.5, 0.3, 0.2])
        assert _js_divergence(p, p) < 1e-9
        assert abs(_js_divergence(p, q) - _js_divergence(q, p)) < 1e-12
        # JS bounded by log2(2) = 1
        assert 0 <= _js_divergence(p, q) <= 1.0


# ── benchmark ─────────────────────────────────────────────────


class TestBrooksBenchmark:
    def test_init_loads(self, brooks_tsv):
        b = BrooksScrambleBenchmark(brooks_tsv, INFO)
        assert len(b.df) == 6
        assert (b.df.alt_seq.str.len() == WINDOW_LEN).all()
        assert (b.df.native_seq.str.len() == WINDOW_LEN).all()

    def test_evaluate_perfect_predictor(self, brooks_tsv):
        b = BrooksScrambleBenchmark(brooks_tsv, INFO)
        res = b.evaluate(_MockAdapter(b.df))
        # 6 samples, none flagged low_support, all with n_reps == 3
        assert res.n_total == 6 and res.n_scored == 6
        assert res.n_calibration == 6 and res.n_weak_baseline == 0
        # Mock makes pred_lfc track true_lfc up to a small noise floor
        assert res.dir_balanced_acc > 0.99
        assert res.pearson_r > 0.95
        # Each row has 3 finite per-replicate LFCs
        assert (res.n_reps_supported == 3).all()

    def test_save_load_roundtrip(self, brooks_tsv, tmp_path):
        b = BrooksScrambleBenchmark(brooks_tsv, INFO)
        res = b.evaluate(_MockAdapter(b.df))
        out = tmp_path / "out"
        b.save_results(res, out)
        loaded = b.load_results(out)
        np.testing.assert_array_almost_equal(loaded.pred_lfc_runs,
                                              res.pred_lfc_runs)
        np.testing.assert_array_almost_equal(loaded.true_lfc_runs,
                                              res.true_lfc_runs)
        np.testing.assert_array_equal(loaded.n_reps_supported,
                                       res.n_reps_supported)
        assert loaded.sample_ids == res.sample_ids
        assert loaded.n_scored == res.n_scored
        assert loaded.n_calibration == res.n_calibration
        # Per-replicate r and ceiling round-trip too
        np.testing.assert_array_almost_equal(loaded.pearson_r_per_rep,
                                              res.pearson_r_per_rep)
        np.testing.assert_array_almost_equal(loaded.ceiling_r_per_rep,
                                              res.ceiling_r_per_rep)

    def test_plot_and_summary_and_headline(self, brooks_tsv, tmp_path):
        b = BrooksScrambleBenchmark(brooks_tsv, INFO)
        res = b.evaluate(_MockAdapter(b.df))
        b.plot(res, tmp_path / "p")
        assert (tmp_path / "p" / "tier1_scatter.png").exists()
        # Also writes the per-sample interval plot
        assert (tmp_path / "p" / "tier1_per_sample.png").exists()
        s = b.summary_dict(res)
        for k in ("n_total", "n_scored", "n_calibration", "n_weak_baseline",
                  "tier1_dir_balanced_acc", "tier1_pearson_r",
                  "tier1_ceiling_pearson_r",
                  "tier1_ceiling_dir_balanced_acc",
                  "tier1_pearson_r_per_rep", "tier1_ceiling_r_per_rep",
                  "tier1_within_range_rate", "tier1_mean_abs_z",
                  "tier2_pearson_mean", "tier2_js_mean"):
            assert k in s
        # Per-rep arrays serialise as 3-element lists
        assert len(s["tier1_pearson_r_per_rep"]) == 3
        assert len(s["tier1_ceiling_r_per_rep"]) == 3
        h = b.headline(res)
        assert "Tier-1" in h and "ceiling" in h and "Tier-2" in h

    def test_low_support_dropped(self, brooks_tsv):
        # flip one row to low_support and confirm it's excluded
        df = pd.read_csv(brooks_tsv, sep="\t")
        df.loc[0, "low_support"] = True
        df.to_csv(brooks_tsv, sep="\t", index=False)
        b = BrooksScrambleBenchmark(brooks_tsv, INFO)
        res = b.evaluate(_MockAdapter(b.df))
        assert res.n_total == 6
        assert res.n_scored == 5
        assert res.n_calibration == 5

    def test_weak_baseline_excluded_from_scored(self, brooks_tsv):
        # zero out all JS94 raw runs on row 0 → n_reps == 0 → weak baseline
        df = pd.read_csv(brooks_tsv, sep="\t")
        df.loc[0, "js94_reads_runs"] = "0,0,0"
        df.to_csv(brooks_tsv, sep="\t", index=False)
        b = BrooksScrambleBenchmark(brooks_tsv, INFO)
        res = b.evaluate(_MockAdapter(b.df))
        assert res.n_scored == 5
        assert res.n_calibration == 5
        assert res.n_weak_baseline == 1

    def test_ceiling_is_high_when_replicates_agree(self, brooks_tsv):
        # The fixture's JS94 norm_runs are drawn from uniform(9.5, 10.5) so
        # the three replicates produce nearly-identical true LFCs per
        # sample → the LOO ceiling should be ~1.0 (test-retest of an
        # almost-noiseless label is near-perfect).
        b = BrooksScrambleBenchmark(brooks_tsv, INFO)
        res = b.evaluate(_MockAdapter(b.df))
        assert res.ceiling_pearson_r > 0.99
        assert res.ceiling_dir_balanced_acc > 0.99
        # All three per-replicate ceilings are computed
        assert np.all(np.isfinite(res.ceiling_r_per_rep))

    def test_partial_replicate_support_keeps_sample_but_lowers_calibration(
            self, brooks_tsv):
        # one JS94 run below threshold on row 0 → n_reps == 2 (still calib);
        # two below on row 1 → n_reps == 1 (scored but NOT calibration)
        df = pd.read_csv(brooks_tsv, sep="\t")
        df.loc[0, "js94_reads_runs"] = "0,30,40"          # n_reps == 2
        df.loc[1, "js94_reads_runs"] = "0,0,40"           # n_reps == 1
        df.to_csv(brooks_tsv, sep="\t", index=False)
        b = BrooksScrambleBenchmark(brooks_tsv, INFO)
        res = b.evaluate(_MockAdapter(b.df))
        assert res.n_scored == 6                          # both rows scored
        assert res.n_calibration == 5                     # row 1 falls out
        assert res.n_reps_supported[0] == 2
        assert res.n_reps_supported[1] == 1


# ── registry ──────────────────────────────────────────────────


class TestBrooksRegistry:
    def test_task_registered(self):
        assert "brooks_scramble" in TASKS

    def test_factory_builds_benchmark(self, brooks_tsv):
        task = TASKS["brooks_scramble"](data_path=brooks_tsv)
        assert isinstance(task, BrooksScrambleBenchmark)
        assert task.adapter_protocol is CoverageTrackPredictor
