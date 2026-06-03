"""Tests for the Cuperus 5'-UTR metrics + Kozak feature core."""
import numpy as np
import pytest

from yeastbench.benchmarks._cuperus_metrics import (
    zero_shot_correlation,
    kozak_design,
    partial_correlation,
)


class TestKozakDesign:
    def test_shape(self):
        X = kozak_design(["A" * 50, "C" * 50])
        assert X.shape == (2, 15)

    def test_all_A_is_reference_row(self):
        # 'A' is the dropped reference at every position -> all-zero
        X = kozak_design(["A" * 50])
        assert not X.any()

    def test_position_encoding(self):
        # last 5 nt are C G T A C  (-5..-1 = C,G,T,A,C)
        seq = "A" * 45 + "CGTAC"
        X = kozak_design([seq])[0]
        # -1='C'->col0; -2='A'->ref; -3='T'->col 6+2=8; -4='G'->col 9+1=10; -5='C'->col12
        nonzero = set(np.flatnonzero(X).tolist())
        assert nonzero == {0, 8, 10, 12}

    def test_short_fragment_skips_missing_positions(self):
        # native fragments can be < 5 bp; missing positions stay all-zero
        X = kozak_design(["CG"])[0]  # -1='G'->col1; -2='C'->col3; -3..-5 missing
        assert set(np.flatnonzero(X).tolist()) == {1, 3}


class TestZeroShotCorrelation:
    def test_perfect_predictor(self):
        E = np.linspace(-2, 2, 50)
        r = zero_shot_correlation(E.copy(), E)
        assert r["overall"]["spearman"] == pytest.approx(1.0)
        assert r["overall"]["pearson"] == pytest.approx(1.0)

    def test_buckets(self):
        E = np.linspace(-2, 2, 60)
        g = E.copy()
        buckets = np.repeat([1, 2, 3], 20)
        r = zero_shot_correlation(g, E, buckets=buckets)
        assert set(r["by_bucket"]) == {1, 2, 3}
        for b in (1, 2, 3):
            assert r["by_bucket"][b]["n"] == 20
            assert r["by_bucket"][b]["spearman"] == pytest.approx(1.0)

    def test_drops_nan(self):
        g = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
        E = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert zero_shot_correlation(g, E)["overall"]["n"] == 4

    def test_constant_input_gives_nan_corr_not_dropped(self):
        # a constant score (or label) -> undefined correlation -> NaN, but the
        # rows are NOT dropped (n is the finite-pair count).
        r = zero_shot_correlation(np.ones(5), np.arange(5.0))["overall"]
        assert r["n"] == 5
        assert np.isnan(r["pearson"])
        assert np.isnan(r["spearman"])


class TestPartialCorrelation:
    def test_recovers_partial_signal(self):
        rng = np.random.default_rng(0)
        n = 3000
        f = rng.standard_normal(n)
        g = rng.standard_normal(n)
        E = 2 * f + 3 * g + 0.5 * rng.standard_normal(n)  # g genuinely contributes
        res = partial_correlation(g, E, f)
        assert res["n"] == n
        assert res["partial_pearson"] > 0.8
        assert res["partial_spearman"] > 0.8   # headline partial-corr number, value-pinned
        assert res["incremental_r2"] > 0.3

    def test_null_when_g_irrelevant(self):
        rng = np.random.default_rng(1)
        n = 3000
        f = rng.standard_normal(n)
        g = rng.standard_normal(n)            # independent of E
        E = 2 * f + 0.5 * rng.standard_normal(n)
        res = partial_correlation(g, E, f)
        assert abs(res["partial_pearson"]) < 0.1
        assert abs(res["partial_spearman"]) < 0.1
        assert res["incremental_r2"] < 0.02   # adding irrelevant g doesn't help OOF

    def test_multifeature_f(self):
        rng = np.random.default_rng(2)
        n = 2000
        f = rng.standard_normal((n, 3))
        g = rng.standard_normal(n)
        E = f @ np.array([1.0, -2.0, 0.5]) + 2 * g + 0.5 * rng.standard_normal(n)
        res = partial_correlation(g, E, f)
        assert res["partial_pearson"] > 0.7
        assert res["incremental_r2"] > 0.2

    def test_too_few_samples(self):
        res = partial_correlation(np.arange(4.0), np.arange(4.0), np.arange(4.0))
        assert res["n"] == 4
        assert np.isnan(res["partial_pearson"])

    def test_guard_boundary_n6_nan_n7_finite(self):
        # guard is n < n_folds + 2 (=7 at default): n=6 -> NaN, n=7 -> computed.
        rng = np.random.default_rng(3)
        for n, want_nan in ((6, True), (7, False)):
            f = rng.standard_normal(n)
            g = rng.standard_normal(n)
            E = f + g
            res = partial_correlation(g, E, f)
            assert res["n"] == n
            assert bool(np.isnan(res["partial_pearson"])) == want_nan

    def test_guard_fires_after_nan_mask(self):
        # 8 rows, 2 NaN -> 6 finite -> below the 7-sample floor -> NaN.
        g = np.array([1.0, 2, 3, 4, 5, 6, np.nan, np.nan])
        E = np.arange(8.0)
        f = np.arange(8.0)
        res = partial_correlation(g, E, f)
        assert res["n"] == 6
        assert np.isnan(res["partial_pearson"])
