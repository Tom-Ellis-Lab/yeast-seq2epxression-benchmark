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
    WINDOW_LEN,
    _bin_per_base,
    _js_divergence,
)
from yeastbench.registry import TASKS

INFO = BenchmarkInfo(name="test_brooks", version="test", description="t",
                     distribution_uri="")

# Model geometry the tests use (matches Yorzoi but expressed locally so
# the test isn't coupled to the real adapter)
BIN = 10
CROP = 996
OB = 300


def _make_seq(rng: np.random.Generator) -> str:
    return "".join(rng.choice(list("ACGT"), size=WINDOW_LEN))


def _make_cov(rng: np.random.Generator, mean: float = 0.5) -> str:
    return ",".join(map(str, rng.poisson(mean, WINDOW_LEN).astype(int).tolist()))


@pytest.fixture
def brooks_tsv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        # half of samples in each tail of true_lfc, so dir-acc is informative
        true_lfc = 1.5 if i % 2 == 0 else -1.5
        # JS94 deep-run replicate coverages (3 values)
        runs = ",".join(f"{v:.3f}" for v in rng.uniform(8, 12, 3))
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
            "strain_reads": 250, "js94_reads_mean": 30.0,
            "size_factor_strain": 4.0,
            "norm_cov_strain": 62.5, "norm_cov_js94_mean": 10.0,
            "norm_cov_js94_runs": runs,
            "true_lfc": true_lfc, "low_support": False,
        })
    p = tmp_path / "mini.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return p


class _MockAdapter:
    """Perfect predictor: returns coverage that gives `pred_lfc = true_lfc`.

    For + strand samples we make alt = 2^|true_lfc| × native over the CDS
    bins, with the sign baked in (alt > native → positive pred_lfc)."""
    bin_width = BIN
    crop_bp_each_side = CROP
    output_bins = OB

    def __init__(self, df: pd.DataFrame, seed: int = 0):
        self._df = df.set_index("alt_seq")
        self._native_lookup = df.set_index("native_seq")
        self._rng = np.random.default_rng(seed)

    def predict_coverage(self, construct_seq: str, strand: str) -> np.ndarray:
        # Determine whether this is an alt or native call and the
        # corresponding true_lfc for the sample
        if construct_seq in self._df.index:
            tl = float(self._df.loc[construct_seq, "true_lfc"])
            scale = float(2.0 ** tl)         # alt = scale * native
        else:
            scale = 1.0                       # native baseline
        # smooth profile peaking in the CDS region; same shape both ways
        base = self._rng.normal(1.0, 0.05, OB).clip(0.1)
        return base * scale


assert isinstance(_MockAdapter(pd.DataFrame({"alt_seq": [], "native_seq": [],
                                             "true_lfc": []})),
                  CoverageTrackPredictor)


# ── helpers ───────────────────────────────────────────────────


class TestHelpers:
    def test_bin_per_base_shape_and_sum(self):
        v = np.ones(WINDOW_LEN, dtype=np.int32)
        b = _bin_per_base(v, BIN, CROP, OB)
        assert b.shape == (OB,)
        assert b.sum() == BIN * OB        # 3000 base positions covered

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
        # 6 samples, none flagged low_support
        assert res.n_total == 6 and res.n_scored == 6
        # Mock makes pred_lfc track true_lfc up to a small noise floor
        assert res.dir_balanced_acc > 0.99
        assert res.pearson_r > 0.95

    def test_save_load_roundtrip(self, brooks_tsv, tmp_path):
        b = BrooksScrambleBenchmark(brooks_tsv, INFO)
        res = b.evaluate(_MockAdapter(b.df))
        out = tmp_path / "out"
        b.save_results(res, out)
        loaded = b.load_results(out)
        np.testing.assert_array_almost_equal(loaded.pred_lfc, res.pred_lfc)
        np.testing.assert_array_almost_equal(loaded.true_lfc, res.true_lfc)
        assert loaded.sample_ids == res.sample_ids

    def test_plot_and_summary_and_headline(self, brooks_tsv, tmp_path):
        b = BrooksScrambleBenchmark(brooks_tsv, INFO)
        res = b.evaluate(_MockAdapter(b.df))
        b.plot(res, tmp_path / "p")
        assert (tmp_path / "p" / "tier1_scatter.png").exists()
        s = b.summary_dict(res)
        for k in ("n_total", "n_scored", "tier1_dir_balanced_acc",
                  "tier1_pearson_r", "tier2_pearson_mean", "tier2_js_mean"):
            assert k in s
        h = b.headline(res)
        assert "Tier-1" in h and "Tier-2" in h

    def test_low_support_dropped(self, brooks_tsv):
        # flip one row to low_support and confirm it's excluded
        df = pd.read_csv(brooks_tsv, sep="\t")
        df.loc[0, "low_support"] = True
        df.to_csv(brooks_tsv, sep="\t", index=False)
        b = BrooksScrambleBenchmark(brooks_tsv, INFO)
        res = b.evaluate(_MockAdapter(b.df))
        assert res.n_total == 6
        assert res.n_scored == 5


# ── registry ──────────────────────────────────────────────────


class TestBrooksRegistry:
    def test_task_registered(self):
        assert "brooks_scramble" in TASKS

    def test_factory_builds_benchmark(self, brooks_tsv):
        task = TASKS["brooks_scramble"](data_path=brooks_tsv)
        assert isinstance(task, BrooksScrambleBenchmark)
        assert task.adapter_protocol is CoverageTrackPredictor
