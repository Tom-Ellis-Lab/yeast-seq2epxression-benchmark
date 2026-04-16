"""Tests for MPRA regression benchmark."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from yeastbench.adapters.protocols import SequenceExpressionPredictor
from yeastbench.benchmarks.base import BenchmarkInfo
from yeastbench.benchmarks.mpra import (
    PAIR_STRATA,
    STRATA_FILES,
    MPRAPairStratumResult,
    MPRARegressionBenchmark,
    MPRAResults,
    MPRAStratumResult,
    _pair_stratum_result,
    _stratum_result,
)


# ── Mock adapter ──────────────────────────────────────────────


class LinearPredictor:
    """Predicts expression as a noisy linear function of the true label.

    Given the true labels at construction time, returns
    ``labels * scale + noise`` from predict_expressions.
    """

    def __init__(self, labels: np.ndarray, scale: float = 1.0, noise_std: float = 0.05, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.predictions = labels * scale + rng.normal(0, noise_std, size=len(labels))

    def predict_expressions(self, seqs: Sequence[str]) -> np.ndarray:
        return self.predictions


class ConstantPredictor:
    """Predicts a constant for every sequence."""

    def __init__(self, value: float = 0.5):
        self.value = value

    def predict_expressions(self, seqs: Sequence[str]) -> np.ndarray:
        return np.full(len(seqs), self.value)


assert isinstance(LinearPredictor(np.array([0.0])), SequenceExpressionPredictor)
assert isinstance(ConstantPredictor(), SequenceExpressionPredictor)


# ── Synthetic distribution fixture ────────────────────────────

N_SEQS = 200
POLYT_OVERHANG = "TGCATTTTTTTCACATC"
POLYA_OVERHANG = "GGTTACGGCTGTT"
INFO = BenchmarkInfo(name="test_mpra", version="test", description="test", distribution_uri="")


def _random_seq(rng: np.random.Generator) -> str:
    bases = list("ACGT")
    n80 = "".join(rng.choice(bases, 80))
    return POLYT_OVERHANG + n80 + POLYA_OVERHANG


@pytest.fixture
def mpra_distribution(tmp_path: Path) -> Path:
    """Build a minimal synthetic MPRA distribution."""
    rng = np.random.default_rng(42)
    data_dir = tmp_path / "deboer"
    data_dir.mkdir()
    subset_dir = data_dir / "test_subset_ids"
    subset_dir.mkdir()

    # Master file
    seqs = [_random_seq(rng) for _ in range(N_SEQS)]
    labels = rng.uniform(0, 1, size=N_SEQS)
    master = pd.DataFrame({"seq": seqs, "el": labels})
    master.to_csv(
        data_dir / "filtered_test_data_with_MAUDE_expression.txt",
        sep="\t", header=False, index=False,
    )

    # Non-pair strata
    all_idx = np.arange(N_SEQS)
    rng.shuffle(all_idx)
    non_pair_strata = {
        "high_exp": all_idx[:10],
        "low_exp": all_idx[10:20],
        "yeast_exp": all_idx[20:30],
        "random_exp": all_idx[30:60],
        "challenging": all_idx[60:80],
    }
    for name, idx in non_pair_strata.items():
        df = pd.DataFrame({
            "tag": [f"tag_{i}" for i in idx],
            "sequence": [seqs[i] for i in idx],
            "pos": idx,
            "exp": [None] * len(idx),
        })
        df.to_csv(subset_dir / STRATA_FILES[name], index=True)

    # Pair strata
    pair_strata = {
        "SNVs": (all_idx[80:120].reshape(-1, 2)),
        "motif_perturbation": (all_idx[120:140].reshape(-1, 2)),
        "motif_tiling": (all_idx[140:160].reshape(-1, 2)),
    }
    for name, pairs in pair_strata.items():
        df = pd.DataFrame({
            "alt_tag": [f"alt_{i}" for i in range(len(pairs))],
            "ref_tag": [f"ref_{i}" for i in range(len(pairs))],
            "alt_sequence": [seqs[p[0]] for p in pairs],
            "ref_sequence": [seqs[p[1]] for p in pairs],
            "alt_pos": pairs[:, 0],
            "ref_pos": pairs[:, 1],
            "alt_exp": [None] * len(pairs),
            "ref_exp": [None] * len(pairs),
        })
        df.to_csv(subset_dir / STRATA_FILES[name], index=True)

    return data_dir


# ── Metric helper tests ──────────────────────────────────────


class TestStratumResult:
    def test_perfect_prediction(self):
        pred = np.array([0.1, 0.5, 0.9, 0.3])
        sr = _stratum_result("test", pred, pred)
        assert sr.pearson_r == pytest.approx(1.0)
        assert sr.spearman_rho == pytest.approx(1.0)
        assert sr.n == 4

    def test_constant_prediction_nan(self):
        pred = np.array([0.5, 0.5, 0.5])
        meas = np.array([0.1, 0.5, 0.9])
        sr = _stratum_result("test", pred, meas)
        # Pearson of constant vs variable is nan or 0
        assert sr.n == 3

    def test_single_element(self):
        sr = _stratum_result("test", np.array([1.0]), np.array([1.0]))
        assert sr.n == 1
        assert np.isnan(sr.pearson_r)

    def test_nan_handling(self):
        pred = np.array([0.1, float("nan"), 0.9])
        meas = np.array([0.2, 0.5, 0.8])
        sr = _stratum_result("test", pred, meas)
        assert sr.n == 2


class TestPairStratumResult:
    def test_perfect_differences(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        labels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        pairs = np.array([[0, 1], [2, 3], [4, 5]])
        result = _pair_stratum_result("test", scores, labels, pairs)
        assert result.diff_pearson_r == pytest.approx(1.0)
        assert result.n_pairs == 3


# ── Benchmark class tests ────────────────────────────────────


class TestMPRARegressionBenchmark:
    def test_init_loads_data(self, mpra_distribution):
        bench = MPRARegressionBenchmark(mpra_distribution, INFO)
        assert len(bench.sequences) == N_SEQS
        assert len(bench.labels) == N_SEQS
        assert len(bench.strata_indices) == len(STRATA_FILES)
        assert len(bench.pair_indices) == len(PAIR_STRATA)

    def test_sequences_are_110bp(self, mpra_distribution):
        bench = MPRARegressionBenchmark(mpra_distribution, INFO)
        for seq in bench.sequences:
            assert len(seq) == 110

    def test_evaluate_structure(self, mpra_distribution):
        bench = MPRARegressionBenchmark(mpra_distribution, INFO)
        adapter = LinearPredictor(bench.labels)
        results = bench.evaluate(adapter)

        assert isinstance(results, MPRAResults)
        assert isinstance(results.overall, MPRAStratumResult)
        assert results.overall.name == "overall"
        assert results.overall.n == N_SEQS
        assert len(results.per_stratum) == len(STRATA_FILES)
        assert len(results.per_pair_stratum) == len(PAIR_STRATA)

    def test_linear_predictor_high_correlation(self, mpra_distribution):
        bench = MPRARegressionBenchmark(mpra_distribution, INFO)
        adapter = LinearPredictor(bench.labels, noise_std=0.01)
        results = bench.evaluate(adapter)
        assert results.overall.pearson_r > 0.95

    def test_constant_predictor_near_zero(self, mpra_distribution):
        bench = MPRARegressionBenchmark(mpra_distribution, INFO)
        adapter = ConstantPredictor(0.5)
        results = bench.evaluate(adapter)
        # Constant predictions → Pearson is nan (zero variance)
        # Just verify it doesn't crash and returns a result
        assert isinstance(results.overall, MPRAStratumResult)

    def test_evaluate_pair_strata(self, mpra_distribution):
        bench = MPRARegressionBenchmark(mpra_distribution, INFO)
        adapter = LinearPredictor(bench.labels, noise_std=0.01)
        results = bench.evaluate(adapter)
        for ps in results.per_pair_stratum:
            assert isinstance(ps, MPRAPairStratumResult)
            assert ps.n_pairs > 0
            assert ps.name in PAIR_STRATA

    def test_plot_runs_without_error(self, mpra_distribution, tmp_path):
        bench = MPRARegressionBenchmark(mpra_distribution, INFO)
        adapter = LinearPredictor(bench.labels)
        results = bench.evaluate(adapter)
        plot_dir = tmp_path / "plots"
        bench.plot(results, plot_dir)
        assert (plot_dir / "scatter_per_stratum.png").exists()
        assert (plot_dir / "pearson_summary.png").exists()

    def test_save_load_roundtrip(self, mpra_distribution, tmp_path):
        bench = MPRARegressionBenchmark(mpra_distribution, INFO)
        adapter = LinearPredictor(bench.labels)
        results = bench.evaluate(adapter)

        out = tmp_path / "save_test"
        bench.save_results(results, out)
        loaded = bench.load_results(out)

        np.testing.assert_array_almost_equal(loaded.scores, results.scores)
        np.testing.assert_array_almost_equal(loaded.labels, results.labels)
        assert loaded.overall.pearson_r == pytest.approx(results.overall.pearson_r)
        assert len(loaded.per_stratum) == len(results.per_stratum)

    def test_summary_dict_schema(self, mpra_distribution):
        bench = MPRARegressionBenchmark(mpra_distribution, INFO)
        adapter = LinearPredictor(bench.labels)
        results = bench.evaluate(adapter)
        summary = bench.summary_dict(results)

        assert "n_sequences" in summary
        assert "overall_pearson_r" in summary
        assert "overall_spearman_rho" in summary
        assert "per_stratum" in summary
        assert "per_pair_stratum" in summary
        assert len(summary["per_stratum"]) == len(STRATA_FILES)
        assert len(summary["per_pair_stratum"]) == len(PAIR_STRATA)

    def test_headline(self, mpra_distribution):
        bench = MPRARegressionBenchmark(mpra_distribution, INFO)
        adapter = LinearPredictor(bench.labels)
        results = bench.evaluate(adapter)
        h = bench.headline(results)
        assert "Pearson" in h
        assert "Spearman" in h


# ── Registry integration ──────────────────────────────────────


class TestMPRARegistry:
    def test_rafi_mpra_registered(self):
        from yeastbench.registry import TASKS
        assert "rafi_mpra_promoter" in TASKS

    def test_rafi_factory_produces_benchmark(self, mpra_distribution):
        from yeastbench.registry import TASKS
        task = TASKS["rafi_mpra_promoter"](data_dir=mpra_distribution)
        assert isinstance(task, MPRARegressionBenchmark)
        assert task.adapter_protocol is SequenceExpressionPredictor


# ── Real data regression (skipped if not present) ─────────────

REAL_DATA = Path("data/raw/deboer")


@pytest.mark.skipif(
    not (REAL_DATA / "filtered_test_data_with_MAUDE_expression.txt").exists(),
    reason="Real deBoer data not present",
)
class TestRealDataLoading:
    def test_loads_71103_sequences(self):
        bench = MPRARegressionBenchmark(REAL_DATA, INFO)
        assert len(bench.sequences) == 71103
        assert len(bench.labels) == 71103

    def test_all_sequences_110bp(self):
        bench = MPRARegressionBenchmark(REAL_DATA, INFO)
        lengths = {len(s) for s in bench.sequences}
        assert lengths == {110}

    def test_labels_finite(self):
        bench = MPRARegressionBenchmark(REAL_DATA, INFO)
        assert np.all(np.isfinite(bench.labels))

    def test_stratum_counts(self):
        bench = MPRARegressionBenchmark(REAL_DATA, INFO)
        # From the spec / wc -l
        assert len(bench.strata_indices["high_exp"]) == 968
        assert len(bench.strata_indices["low_exp"]) == 997
        assert bench.pair_indices["SNVs"].shape[0] == 46236
        assert bench.pair_indices["motif_perturbation"].shape[0] == 3527
        assert bench.pair_indices["motif_tiling"].shape[0] == 2653

    def test_sequences_start_with_polyT(self):
        bench = MPRARegressionBenchmark(REAL_DATA, INFO)
        for seq in bench.sequences[:100]:
            assert seq.startswith("TGCATTTTTTTCACATC")

    def test_sequences_end_with_polyA(self):
        bench = MPRARegressionBenchmark(REAL_DATA, INFO)
        for seq in bench.sequences[:100]:
            assert seq.endswith("GGTTACGGCTGTT")
