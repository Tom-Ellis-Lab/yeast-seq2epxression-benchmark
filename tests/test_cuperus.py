"""Tests for the Cuperus 5'-UTR benchmark (model-independent, mock adapter)."""
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from yeastbench.adapters.protocols import FivePrimeUtrReporterExpressionPredictor
from yeastbench.benchmarks.base import BenchmarkInfo
from yeastbench.benchmarks.cuperus import CuperusUTRBenchmark

BUCKET_EDGES = [10, 30, 60, 101]


def _make_utrs(n: int, length: int = 50, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    bases = np.array(list("ACGT"))
    return ["".join(rng.choice(bases, size=length)) for _ in range(n)]


@pytest.fixture
def dist(tmp_path: Path) -> tuple[Path, Path]:
    # random library: 50 rows, t0 spread so ~30 land in the clean bucket (t0>=101);
    # growth_rate is a deterministic function of the UTR so a mock that recomputes
    # it correlates perfectly.
    utrs = _make_utrs(50, 50, seed=1)
    t0 = np.array([5 + 5 * i for i in range(50)])  # 5..250; >=101 for i>=20
    gr = np.array([float(s.count("G")) for s in utrs])
    buckets = np.digitize(t0, BUCKET_EDGES) + 1
    rand = pd.DataFrame({
        "UTR": utrs, "growth_rate": gr, "t0": t0, "t1": t0,
        "depth_bucket": buckets, "is_paper_top5": t0 >= 101,
    })
    rpath = tmp_path / "random_utrs.tsv"
    rand.to_csv(rpath, sep="\t", index=False)

    # native library: 20 variable-length fragments, a couple with t0<10
    nutrs = _make_utrs(12, 50, seed=2) + _make_utrs(8, 7, seed=3)  # 8 short (7 bp)
    nt0 = np.array([3, 7] + [50 + 20 * i for i in range(18)])  # two noisy (<10)
    ngr = np.array([float(s.count("C")) for s in nutrs])
    nat = pd.DataFrame({
        "UTR_name": [f"GENE{i}:50:0" for i in range(20)],
        "UTR": nutrs, "growth_rate": ngr, "t0": nt0, "t1": nt0,
    })
    npath = tmp_path / "native_utrs.tsv"
    nat.to_csv(npath, sep="\t", index=False)
    return rpath, npath


class _PerfectRandomAdapter:
    def predict_utr_expressions(self, utrs: Sequence[str]) -> np.ndarray:
        return np.array([float(s.count("G")) for s in utrs])


class _PerfectNativeAdapter:
    def predict_utr_expressions(self, utrs: Sequence[str]) -> np.ndarray:
        return np.array([float(s.count("C")) for s in utrs])


class _NaNAdapter:
    def predict_utr_expressions(self, utrs: Sequence[str]) -> np.ndarray:
        out = np.array([float(s.count("G")) for s in utrs])
        out[: max(1, len(out) // 10)] = np.nan
        return out


def _bench(dist) -> CuperusUTRBenchmark:
    rpath, npath = dist
    return CuperusUTRBenchmark(
        random_path=rpath, native_path=npath,
        info=BenchmarkInfo("cuperus_utr", "v1", "Cuperus 5'-UTR", ""),
    )


class TestProtocol:
    def test_mock_satisfies_protocol(self):
        assert isinstance(_PerfectRandomAdapter(), FivePrimeUtrReporterExpressionPredictor)


class TestEvaluate:
    def test_perfect_random(self, dist):
        b = _bench(dist)
        res = b.evaluate(_PerfectRandomAdapter())
        assert res.random.metric1["overall"]["spearman"] == pytest.approx(1.0)
        # depth buckets present and the clean bucket (5) is populated
        assert 5 in res.random.metric1["by_bucket"]
        assert res.random.metric1["by_bucket"][5]["n"] > 0
        # metric 2 ran on the clean bucket
        assert res.random.metric2["n"] > 0
        assert np.isfinite(res.random.metric2["partial_spearman"])

    def test_native_split_and_variable_length(self, dist):
        b = _bench(dist)
        res = b.evaluate(_PerfectNativeAdapter())
        assert res.native.metric1["overall"]["spearman"] == pytest.approx(1.0)
        # noisy (1) and rest (2) split both present
        assert set(res.native.metric1["by_bucket"]) == {1, 2}

    def test_nan_scores_dropped(self, dist):
        b = _bench(dist)
        res = b.evaluate(_NaNAdapter())
        assert res.random.metric1["overall"]["n"] < len(res.random.scores)


class TestPersistence:
    def test_save_load_roundtrip(self, dist, tmp_path):
        b = _bench(dist)
        res = b.evaluate(_PerfectRandomAdapter())
        out = tmp_path / "out"
        b.save_results(res, out)
        loaded = b.load_results(out)
        np.testing.assert_allclose(
            loaded.random.scores, res.random.scores, equal_nan=True
        )
        np.testing.assert_allclose(
            loaded.random.labels, res.random.labels, equal_nan=True
        )
        assert (
            loaded.random.metric1["overall"]["spearman"]
            == pytest.approx(res.random.metric1["overall"]["spearman"])
        )


class TestSummaryHeadlinePlot:
    def test_summary_keys(self, dist):
        b = _bench(dist)
        s = b.summary_dict(b.evaluate(_PerfectRandomAdapter()))
        for k in ("random_spearman", "random_metric2_partial_spearman",
                  "native_spearman", "random_bucket5_spearman"):
            assert k in s

    def test_headline_is_str(self, dist):
        b = _bench(dist)
        assert isinstance(b.headline(b.evaluate(_PerfectRandomAdapter())), str)

    def test_plot_writes_file(self, dist, tmp_path):
        b = _bench(dist)
        res = b.evaluate(_PerfectRandomAdapter())
        b.plot(res, tmp_path / "p")
        assert (tmp_path / "p" / "cuperus.png").exists()
