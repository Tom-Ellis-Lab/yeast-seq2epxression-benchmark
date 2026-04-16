"""Tests for eQTL benchmark logic: results aggregation, evaluate, plotting helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yeastbench.benchmarks.base import BenchmarkInfo
from yeastbench.benchmarks.eqtl import (
    CLOSE_ONLY_THRESHOLD_BP,
    DISTANCE_BINS,
    EQTLClassificationBenchmark,
    EQTLIterationResult,
    EQTLResults,
    _bin_mask,
    _close_only_mask,
    _iteration_subset,
    _pair_mask_to_variant_mask,
)

from conftest import DeterministicScorer, PerfectScorer


# ── Helpers ───────────────────────────────────────────────────

INFO = BenchmarkInfo(
    name="test_eqtl",
    version="test",
    description="test",
    distribution_uri="file:///dev/null",
)


def _make_iter_result(
    n_pairs: int, auroc_abs: float, auprc_abs: float, seed: int = 0
) -> EQTLIterationResult:
    """Build a synthetic EQTLIterationResult with controlled metrics."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(0, 1, size=2 * n_pairs)
    labels = np.tile([1, 0], n_pairs)
    pairs = pd.DataFrame({
        "pair_id": range(n_pairs),
        "pos_distance_to_tss": rng.integers(50, 25000, size=n_pairs),
        "neg_distance_to_tss": rng.integers(50, 25000, size=n_pairs),
    })
    return EQTLIterationResult(
        name=f"negset_{seed}",
        scores=scores,
        labels=labels,
        pairs=pairs,
        auroc_signed=0.5,
        auprc_signed=0.5,
        auroc_abs=auroc_abs,
        auprc_abs=auprc_abs,
    )


# ── EQTLResults aggregation ──────────────────────────────────


class TestEQTLResultsAggregation:
    def test_single_iteration_sem_is_zero(self):
        results = EQTLResults(per_iter=[_make_iter_result(10, 0.75, 0.60)])
        assert results.mean_auroc == pytest.approx(0.75)
        assert results.sem_auroc == 0.0
        assert results.mean_auprc == pytest.approx(0.60)
        assert results.sem_auprc == 0.0

    def test_multiple_iterations_mean(self):
        iters = [
            _make_iter_result(10, 0.70, 0.55, seed=1),
            _make_iter_result(10, 0.80, 0.65, seed=2),
            _make_iter_result(10, 0.90, 0.75, seed=3),
        ]
        results = EQTLResults(per_iter=iters)
        assert results.mean_auroc == pytest.approx(0.80)
        assert results.mean_auprc == pytest.approx(0.65)

    def test_multiple_iterations_sem(self):
        iters = [
            _make_iter_result(10, 0.70, 0.55, seed=1),
            _make_iter_result(10, 0.80, 0.65, seed=2),
        ]
        results = EQTLResults(per_iter=iters)
        expected_sem = np.std([0.70, 0.80], ddof=1) / np.sqrt(2)
        assert results.sem_auroc == pytest.approx(expected_sem)

    def test_identical_iterations_zero_sem(self):
        iters = [_make_iter_result(10, 0.75, 0.60, seed=i) for i in range(4)]
        results = EQTLResults(per_iter=iters)
        assert results.sem_auroc == pytest.approx(0.0)


# ── Plotting helpers ──────────────────────────────────────────


class TestPlottingHelpers:
    @pytest.fixture
    def pairs_df(self):
        return pd.DataFrame({
            "pair_id": range(10),
            "pos_distance_to_tss": [100, 400, 800, 1500, 2500, 5000, 10000, 20000, 28000, 50],
            "neg_distance_to_tss": [200, 300, 700, 1200, 3000, 6000, 12000, 22000, 29000, 100],
        })

    def test_close_only_mask(self, pairs_df):
        mask = _close_only_mask(pairs_df)
        assert mask.dtype == bool
        # Distances <= 2000: 100, 400, 800, 1500, 50 → indices 0,1,2,3,9
        assert mask.sum() == 5
        assert mask[0] and mask[1] and mask[2] and mask[3] and mask[9]
        assert not mask[4]  # 2500 > 2000

    def test_bin_mask(self, pairs_df):
        # (0, 500] bin should match distances 100, 400, 50
        mask = _bin_mask(pairs_df, 0, 500)
        assert mask.sum() == 3

        # (500, 1000] should match 800
        mask = _bin_mask(pairs_df, 500, 1000)
        assert mask.sum() == 1
        assert mask[2]

    def test_pair_mask_to_variant_mask(self):
        pair_mask = np.array([True, False, True, False])
        var_mask = _pair_mask_to_variant_mask(pair_mask)
        assert len(var_mask) == 8
        np.testing.assert_array_equal(
            var_mask, [True, True, False, False, True, True, False, False]
        )

    def test_iteration_subset_no_filter(self):
        r = _make_iter_result(20, 0.75, 0.60, seed=42)
        scores, labels = _iteration_subset(r, subset=None)
        np.testing.assert_array_equal(scores, np.abs(r.scores))
        np.testing.assert_array_equal(labels, r.labels)

    def test_iteration_subset_with_filter(self):
        r = _make_iter_result(20, 0.75, 0.60, seed=42)
        # Apply close_only_mask
        scores, labels = _iteration_subset(r, subset=_close_only_mask)
        pair_mask = _close_only_mask(r.pairs)
        n_kept = pair_mask.sum()
        assert len(scores) == 2 * n_kept
        assert len(labels) == 2 * n_kept


# ── EQTLClassificationBenchmark ──────────────────────────────


class TestEQTLClassificationBenchmark:
    def test_init_loads_iteration_files(self, synthetic_distribution):
        bench = EQTLClassificationBenchmark(synthetic_distribution, INFO)
        assert len(bench.iteration_files) == 3
        for f in bench.iteration_files:
            assert f.name.startswith("negset_")
            assert f.suffix == ".tsv"

    def test_fasta_gtf_paths(self, synthetic_distribution):
        bench = EQTLClassificationBenchmark(synthetic_distribution, INFO)
        assert bench.fasta_path == synthetic_distribution / "reference" / "R64-1-1.fa"
        assert bench.gtf_path == synthetic_distribution / "reference" / "R64-1-1.115.gtf"

    def test_evaluate_returns_correct_structure(self, synthetic_distribution):
        bench = EQTLClassificationBenchmark(synthetic_distribution, INFO)
        scorer = DeterministicScorer()
        results = bench.evaluate(scorer)

        assert isinstance(results, EQTLResults)
        assert len(results.per_iter) == 3

        for r in results.per_iter:
            assert isinstance(r, EQTLIterationResult)
            # 50 pairs → 100 variants
            assert len(r.scores) == 100
            assert len(r.labels) == 100
            assert len(r.pairs) == 50
            # Labels alternate 1, 0
            np.testing.assert_array_equal(r.labels, np.tile([1, 0], 50))

    def test_evaluate_scores_are_finite(self, synthetic_distribution):
        bench = EQTLClassificationBenchmark(synthetic_distribution, INFO)
        results = bench.evaluate(DeterministicScorer())
        for r in results.per_iter:
            assert np.all(np.isfinite(r.scores))

    def test_evaluate_metrics_in_range(self, synthetic_distribution):
        bench = EQTLClassificationBenchmark(synthetic_distribution, INFO)
        results = bench.evaluate(DeterministicScorer())
        for r in results.per_iter:
            assert 0.0 <= r.auroc_abs <= 1.0
            assert 0.0 <= r.auprc_abs <= 1.0
            assert 0.0 <= r.auroc_signed <= 1.0
            assert 0.0 <= r.auprc_signed <= 1.0

    def test_perfect_scorer_auroc_one(self, synthetic_distribution):
        bench = EQTLClassificationBenchmark(synthetic_distribution, INFO)
        results = bench.evaluate(PerfectScorer())
        for r in results.per_iter:
            assert r.auroc_abs == pytest.approx(1.0)

    def test_evaluate_pairs_metadata(self, synthetic_distribution):
        bench = EQTLClassificationBenchmark(synthetic_distribution, INFO)
        results = bench.evaluate(DeterministicScorer())
        for r in results.per_iter:
            assert "pair_id" in r.pairs.columns
            assert "pos_distance_to_tss" in r.pairs.columns
            assert "neg_distance_to_tss" in r.pairs.columns
            assert len(r.pairs) == 50

    def test_plot_runs_without_error(self, synthetic_distribution, tmp_path):
        bench = EQTLClassificationBenchmark(synthetic_distribution, INFO)
        results = bench.evaluate(DeterministicScorer())
        plot_dir = tmp_path / "plots"
        bench.plot(results, plot_dir)

        assert (plot_dir / "primary_roc_pr.png").exists()
        assert (plot_dir / "close_only_roc_pr.png").exists()
        assert (plot_dir / "distance_stratified.png").exists()


# ── save / load / summary round-trip ─────────────────────────


class TestEQTLPersistence:
    def test_save_load_roundtrip(self, synthetic_distribution, tmp_path):
        bench = EQTLClassificationBenchmark(synthetic_distribution, INFO)
        results = bench.evaluate(DeterministicScorer())

        out = tmp_path / "save_test"
        bench.save_results(results, out)
        loaded = bench.load_results(out)

        assert len(loaded.per_iter) == len(results.per_iter)
        for orig, back in zip(results.per_iter, loaded.per_iter):
            assert orig.name == back.name
            np.testing.assert_array_almost_equal(orig.scores, back.scores)
            np.testing.assert_array_equal(orig.labels, back.labels)
            assert len(orig.pairs) == len(back.pairs)
            # Recomputed metrics should match
            assert back.auroc_abs == pytest.approx(orig.auroc_abs)
            assert back.auprc_abs == pytest.approx(orig.auprc_abs)

    def test_save_creates_expected_files(self, synthetic_distribution, tmp_path):
        bench = EQTLClassificationBenchmark(synthetic_distribution, INFO)
        results = bench.evaluate(DeterministicScorer())
        out = tmp_path / "files_test"
        bench.save_results(results, out)

        for r in results.per_iter:
            assert (out / f"{r.name}_scores.npy").exists()
            assert (out / f"{r.name}_labels.npy").exists()
            assert (out / f"{r.name}_pairs.tsv").exists()

    def test_summary_dict_schema(self, synthetic_distribution):
        bench = EQTLClassificationBenchmark(synthetic_distribution, INFO)
        results = bench.evaluate(DeterministicScorer())
        summary = bench.summary_dict(results)

        assert "per_iteration" in summary
        assert len(summary["per_iteration"]) == 3
        assert "auroc_abs_mean" in summary
        assert "auroc_abs_sem" in summary
        assert "auprc_abs_mean" in summary
        assert "auprc_abs_sem" in summary
        for it in summary["per_iteration"]:
            assert "name" in it
            assert "n_pairs" in it
            assert "auroc_abs" in it

    def test_summary_dict_matches_results(self, synthetic_distribution):
        bench = EQTLClassificationBenchmark(synthetic_distribution, INFO)
        results = bench.evaluate(DeterministicScorer())
        summary = bench.summary_dict(results)

        assert summary["auroc_abs_mean"] == pytest.approx(results.mean_auroc)
        assert summary["auroc_abs_sem"] == pytest.approx(results.sem_auroc)

    def test_headline(self, synthetic_distribution):
        bench = EQTLClassificationBenchmark(synthetic_distribution, INFO)
        results = bench.evaluate(DeterministicScorer())
        h = bench.headline(results)
        assert "AUROC" in h
        assert "AUPRC" in h
        assert "±" in h


# ── Distance bins sanity ──────────────────────────────────────


class TestDistanceBins:
    def test_bins_are_contiguous(self):
        for i in range(len(DISTANCE_BINS) - 1):
            assert DISTANCE_BINS[i][1] == DISTANCE_BINS[i + 1][0]

    def test_bins_start_at_zero(self):
        assert DISTANCE_BINS[0][0] == 0

    def test_close_only_threshold_falls_in_a_bin(self):
        found = False
        for lo, hi in DISTANCE_BINS:
            if lo < CLOSE_ONLY_THRESHOLD_BP <= hi:
                found = True
                break
        assert found
