"""Tests for the MYTK integration-site × promoter benchmark.

Uses a mock adapter + a synthetic 33-row distribution, so nothing here
needs downloaded data, model weights, or a GPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from yeastbench.adapters.protocols import (
    PromoterIntegrationConstruct,
    PromoterIntegrationExpressionPredictor,
)
from yeastbench.benchmarks.base import BenchmarkInfo
from yeastbench.benchmarks.mytk import MytkBenchmark
from yeastbench.registry import TASKS

INFO = BenchmarkInfo(name="test_mytk", version="test", description="t", distribution_uri="")

PROMOTERS = ("pTDH3", "pRPL18B", "pREV1")
N_LOCI = 11


# ── Synthetic distribution ────────────────────────────────────


@pytest.fixture
def mytk_data(tmp_path: Path) -> Path:
    """A synthetic 11-loci × 3-promoter table mirroring the real one.

    Per-promoter base levels differ ~100× (like pTDH3/pRPL18B/pREV1) and
    each locus adds a small position effect, so a faithful adapter can
    score positively both within and across promoters.
    """
    base = {"pTDH3": 1300.0, "pRPL18B": 185.0, "pREV1": 8.0}
    rng = np.random.default_rng(0)
    rows = []
    for i in range(N_LOCI):
        locus = "ura3" if i == 0 else f"Int.{i}"
        position_effect = rng.uniform(-0.15, 0.15)  # ±15% per-locus shift
        for prom in PROMOTERS:
            mean = base[prom] * (1.0 + position_effect)
            rows.append({
                "locus_id": locus,
                "promoter": prom,
                "chrom": "V",
                "integration_coord": 100_000 + i * 1000,
                "rep1": mean, "rep2": mean, "rep3": mean,
                "mean": mean,
                "sd": 0.0,
            })
    df = pd.DataFrame(rows)
    p = tmp_path / "mytk.tsv"
    df.to_csv(p, sep="\t", index=False)
    return p


class _MockAdapter:
    """Returns the label + small noise, so correlations are high but < 1."""

    def __init__(self, labels: np.ndarray, noise: float = 1e-3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self._scores = labels + rng.normal(0, noise, size=len(labels)) * labels

    def predict_integrated_expressions(
        self, constructs: Sequence[PromoterIntegrationConstruct]
    ) -> np.ndarray:
        return self._scores


def test_mock_satisfies_protocol():
    assert isinstance(
        _MockAdapter(np.array([1.0, 2.0])),
        PromoterIntegrationExpressionPredictor,
    )


# ── __init__ ──────────────────────────────────────────────────


class TestInit:
    def test_loads_constructs(self, mytk_data):
        bench = MytkBenchmark(mytk_data, fasta_path="/dev/null", info=INFO)
        assert len(bench.constructs) == N_LOCI * len(PROMOTERS)
        assert len(bench.labels) == N_LOCI * len(PROMOTERS)
        c = bench.constructs[0]
        assert isinstance(c, PromoterIntegrationConstruct)
        assert c.locus_id and c.promoter in PROMOTERS
        assert c.chrom == "V"

    def test_fasta_path_property(self, mytk_data, tmp_path):
        fa = tmp_path / "ref.fa"
        fa.write_text(">V\nACGT\n")
        bench = MytkBenchmark(mytk_data, fasta_path=fa, info=INFO)
        assert bench.fasta_path == fa

    def test_missing_column_asserts(self, tmp_path):
        bad = tmp_path / "bad.tsv"
        pd.DataFrame({"locus_id": ["ura3"], "promoter": ["pTDH3"]}).to_csv(
            bad, sep="\t", index=False
        )
        with pytest.raises(AssertionError):
            MytkBenchmark(bad, fasta_path="/dev/null", info=INFO)


# ── evaluate / metrics ────────────────────────────────────────


class TestEvaluate:
    def test_per_promoter_and_pooled(self, mytk_data):
        bench = MytkBenchmark(mytk_data, fasta_path="/dev/null", info=INFO)
        results = bench.evaluate(_MockAdapter(bench.labels))
        # one result per promoter, each over the 11 loci
        assert set(results.per_promoter) == set(PROMOTERS)
        for res in results.per_promoter.values():
            assert res.n == N_LOCI
            assert res.spearman_rho > 0.9  # near-perfect mock
        assert results.pooled.n == N_LOCI * len(PROMOTERS)
        assert results.mean_per_promoter_spearman > 0.9

    def test_length_mismatch_asserts(self, mytk_data):
        bench = MytkBenchmark(mytk_data, fasta_path="/dev/null", info=INFO)

        class _Wrong:
            def predict_integrated_expressions(self, constructs):
                return np.zeros(len(constructs) - 1)

        with pytest.raises(AssertionError):
            bench.evaluate(_Wrong())

    def test_nan_scores_tolerated(self, mytk_data):
        bench = MytkBenchmark(mytk_data, fasta_path="/dev/null", info=INFO)
        # Rows are grouped per locus as [pTDH3, pRPL18B, pREV1]; drop the
        # first two of locus 0 so two different promoters each lose one.
        nan_idx = [0, 1]
        dropped = {bench.constructs[i].promoter for i in nan_idx}

        class _SomeNan:
            def __init__(self, labels):
                self.s = labels.copy()
                self.s[nan_idx] = np.nan

            def predict_integrated_expressions(self, constructs):
                return self.s

        results = bench.evaluate(_SomeNan(bench.labels))
        # NaNs are masked, not crashed on: pooled drops exactly 2 rows,
        # and only the two affected promoters lose one each.
        assert results.pooled.n == N_LOCI * len(PROMOTERS) - len(nan_idx)
        for prom, res in results.per_promoter.items():
            assert res.n == (N_LOCI - 1 if prom in dropped else N_LOCI)
        assert np.isfinite(results.mean_per_promoter_spearman)


# ── save / load roundtrip ─────────────────────────────────────


class TestSaveLoad:
    def test_roundtrip(self, mytk_data, tmp_path):
        bench = MytkBenchmark(mytk_data, fasta_path="/dev/null", info=INFO)
        results = bench.evaluate(_MockAdapter(bench.labels))
        out = tmp_path / "out"
        bench.save_results(results, out)
        loaded = bench.load_results(out)
        np.testing.assert_array_almost_equal(loaded.scores, results.scores)
        np.testing.assert_array_almost_equal(loaded.labels, results.labels)
        assert loaded.pooled.n == results.pooled.n
        assert (
            loaded.mean_per_promoter_spearman
            == pytest.approx(results.mean_per_promoter_spearman)
        )

    def test_summary_dict(self, mytk_data):
        bench = MytkBenchmark(mytk_data, fasta_path="/dev/null", info=INFO)
        results = bench.evaluate(_MockAdapter(bench.labels))
        s = bench.summary_dict(results)
        assert s["n_rows_total"] == N_LOCI * len(PROMOTERS)
        assert "mean_per_promoter_spearman_rho" in s
        assert "pooled_spearman_rho" in s
        for prom in PROMOTERS:
            assert f"{prom}_spearman_rho" in s

    def test_plot(self, mytk_data, tmp_path):
        bench = MytkBenchmark(mytk_data, fasta_path="/dev/null", info=INFO)
        results = bench.evaluate(_MockAdapter(bench.labels))
        bench.plot(results, tmp_path / "plots")
        assert (tmp_path / "plots" / "scatter.png").exists()

    def test_headline(self, mytk_data):
        bench = MytkBenchmark(mytk_data, fasta_path="/dev/null", info=INFO)
        results = bench.evaluate(_MockAdapter(bench.labels))
        h = bench.headline(results)
        assert "per-promoter" in h and "pooled" in h


# ── registry ──────────────────────────────────────────────────


class TestRegistry:
    def test_task_registered(self):
        assert "mytk_ints_promoter" in TASKS

    def test_factory_builds_benchmark(self, mytk_data, tmp_path):
        fa = tmp_path / "ref.fa"
        fa.write_text(">V\nACGT\n")
        task = TASKS["mytk_ints_promoter"](data_path=mytk_data, fasta_path=fa)
        assert isinstance(task, MytkBenchmark)
        assert task.adapter_protocol is PromoterIntegrationExpressionPredictor
