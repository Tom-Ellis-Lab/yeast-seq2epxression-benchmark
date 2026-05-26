"""Tests for the Hong et al. IGR-insertion benchmark + scaffold."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pysam
import pytest

from yeastbench.adapters._hong_scaffold import (
    MCHERRY_CDS_LEN,
    MCHERRY_CDS_START_IN_PAYLOAD,
    PAYLOAD_LEN,
    HongLocus,
    build_insertion_context,
    load_cassette_payload,
)
from yeastbench.adapters.protocols import IGRInsertionExpressionPredictor
from yeastbench.benchmarks.base import BenchmarkInfo
from yeastbench.benchmarks.hong_igr import (
    DiagnosticB,
    HongIGRInsertionBenchmark,
    MRNA_FLUO_CEILING_SPCC_PUBLISHED,
    _topk_enrichment,
    _select_diagnostic_b,
)
from yeastbench.registry import TASKS

INFO = BenchmarkInfo(name="test_hong", version="test", description="t", distribution_uri="")
REAL_CASSETTE = Path("data/tasks/hong/expression_cassette.fasta")


# ── Cassette payload ──────────────────────────────────────────


class TestCassettePayload:
    def test_loads_real_payload(self):
        p = load_cassette_payload(REAL_CASSETTE)
        assert len(p) == PAYLOAD_LEN == 1595
        assert p[MCHERRY_CDS_START_IN_PAYLOAD : MCHERRY_CDS_START_IN_PAYLOAD + 3] == "ATG"
        # mCherry CDS ends with a stop codon
        stop = p[
            MCHERRY_CDS_START_IN_PAYLOAD + MCHERRY_CDS_LEN - 3
            : MCHERRY_CDS_START_IN_PAYLOAD + MCHERRY_CDS_LEN
        ]
        assert stop in ("TAA", "TAG", "TGA")

    def test_bad_length_raises(self, tmp_path):
        bad = tmp_path / "bad.fasta"
        bad.write_text(">x\nACGT\n")
        with pytest.raises(ValueError, match="expected 1595"):
            load_cassette_payload(bad)

    def test_bad_mcherry_start_raises(self, tmp_path):
        bad = tmp_path / "bad.fasta"
        bad.write_text(">x\n" + "T" * PAYLOAD_LEN + "\n")
        with pytest.raises(ValueError, match="ATG"):
            load_cassette_payload(bad)


# ── build_insertion_context ───────────────────────────────────


@pytest.fixture
def mini_genome(tmp_path: Path) -> pysam.FastaFile:
    """One 60 kb chromosome of deterministic non-ACGT-cassette filler."""
    rng = np.random.default_rng(0)
    chrom = "".join(rng.choice(list("ACGT"), 60_000))
    fa = tmp_path / "mini.fa"
    fa.write_text(">I\n" + "\n".join(chrom[i : i + 60] for i in range(0, len(chrom), 60)) + "\n")
    pysam.faidx(str(fa))
    return pysam.FastaFile(str(fa))


class TestBuildInsertionContext:
    PARAMS = dict(
        seq_len=16384, crop_bp_each_side=1024, bin_width=16, output_bins=896
    )

    def test_centers_cassette(self, mini_genome):
        payload = load_cassette_payload(REAL_CASSETTE)
        locus = HongLocus("test1", "IntTrain", "I", 30_000)
        ctx = build_insertion_context(locus, payload, mini_genome, **self.PARAMS)
        assert ctx is not None
        sl = self.PARAMS["seq_len"]
        assert len(ctx.window_seq) == sl
        assert ctx.mcherry_bins.size > 0
        # Cassette centered ⇒ cassette midpoint within ±1 bp of window midpoint
        cassette_mid_in_window = ctx.payload_start_in_window + PAYLOAD_LEN // 2
        assert abs(cassette_mid_in_window - sl // 2) <= 1

    def test_payload_written_forward(self, mini_genome):
        """Cassette is always integrated on + strand → payload sits as-is."""
        payload = load_cassette_payload(REAL_CASSETTE)
        locus = HongLocus("test2", "IntProp", "I", 30_000)
        ctx = build_insertion_context(locus, payload, mini_genome, **self.PARAMS)
        assert ctx is not None
        ps = ctx.payload_start_in_window
        assert ctx.window_seq[ps : ps + PAYLOAD_LEN] == payload
        # mCherry start lands at payload_start + offset
        assert ctx.mcherry_start_in_window == ps + MCHERRY_CDS_START_IN_PAYLOAD
        atg = payload[MCHERRY_CDS_START_IN_PAYLOAD : MCHERRY_CDS_START_IN_PAYLOAD + 3]
        assert ctx.window_seq[ctx.mcherry_start_in_window : ctx.mcherry_start_in_window + 3] == atg

    def test_native_flank_preserved(self, mini_genome):
        """Native bases immediately up- and downstream of the cut site
        appear in the window adjacent to the cassette."""
        payload = load_cassette_payload(REAL_CASSETTE)
        cut = 30_000
        locus = HongLocus("test3", "IntTrain", "I", cut)
        ctx = build_insertion_context(locus, payload, mini_genome, **self.PARAMS)
        assert ctx is not None
        chrom_seq = mini_genome.fetch("I").upper()
        # Native base immediately before the cassette = 1-based position `cut`
        ps = ctx.payload_start_in_window
        assert ctx.window_seq[ps - 1] == chrom_seq[cut - 1]
        # Native base immediately after the cassette = 1-based position cut+1
        assert ctx.window_seq[ps + PAYLOAD_LEN] == chrom_seq[cut]

    def test_short_chromosome_returns_none(self, tmp_path):
        payload = load_cassette_payload(REAL_CASSETTE)
        rng = np.random.default_rng(1)
        chrom = "".join(rng.choice(list("ACGT"), 5_000))
        fa = tmp_path / "short.fa"
        fa.write_text(">I\n" + "\n".join(
            chrom[i : i + 60] for i in range(0, len(chrom), 60)
        ) + "\n")
        pysam.faidx(str(fa))
        g = pysam.FastaFile(str(fa))
        locus = HongLocus("short1", "IntTrain", "I", 2_000)
        ctx = build_insertion_context(locus, payload, g, **self.PARAMS)
        assert ctx is None


# ── Top-k enrichment helper ───────────────────────────────────


class TestTopKEnrichment:
    def test_perfect_predictor(self):
        labels = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
        scores = labels.copy()  # perfect ranking
        # Top-2 = labels [2.0, 1.5] = mean 1.75; overall mean = 1.02
        e = _topk_enrichment(scores, labels, k=2)
        assert e == pytest.approx(1.75 / 1.02, rel=1e-3)

    def test_random_floor(self):
        labels = np.array([1.0] * 10)  # all equal
        scores = np.random.default_rng(0).standard_normal(10)
        assert _topk_enrichment(scores, labels, k=3) == pytest.approx(1.0)

    def test_k_too_large_returns_nan(self):
        labels = np.array([1.0, 2.0])
        scores = np.array([0.5, 1.5])
        assert np.isnan(_topk_enrichment(scores, labels, k=5))


# ── Benchmark ─────────────────────────────────────────────────


@pytest.fixture
def hong_data(tmp_path: Path) -> Path:
    """Minimal Hong distribution TSV: 4 IntTrain + 3 IntProp."""
    rows = [
        ("IntTrain1", "IntTrain", "I", 10_000, 0.7, "AAAAAAAAAAAAAAAAAAAA"),
        ("IntTrain2", "IntTrain", "I", 20_000, 0.9, "CCCCCCCCCCCCCCCCCCCC"),
        ("IntTrain3", "IntTrain", "II", 30_000, 1.0, "GGGGGGGGGGGGGGGGGGGG"),
        ("IntTrain4", "IntTrain", "II", 40_000, 1.2, "TTTTTTTTTTTTTTTTTTTT"),
        ("IntProp1", "IntProp", "I", 50_000, 0.6, "ACACACACACACACACACAC"),
        ("IntProp2", "IntProp", "II", 60_000, 0.85, "GTGTGTGTGTGTGTGTGTGT"),
        ("IntProp3", "IntProp", "II", 70_000, 1.1, "ATATATATATATATATATAT"),
    ]
    df = pd.DataFrame(rows, columns=[
        "locus_id", "set", "chrom", "integration_coord",
        "fluorescence_norm_intrain92", "gRNA",
    ])
    p = tmp_path / "hong_mini.tsv"
    df.to_csv(p, sep="\t", index=False)
    return p


class _MockAdapter:
    """Perfect predictor: returns each locus's measured fluorescence.
    Does NOT implement predict_diagnostic_readouts — used to test the
    back-compat path."""

    def __init__(self, locus_to_label: dict[str, float]):
        self.l2l = locus_to_label

    def predict_expressions(self, loci: Sequence[HongLocus]) -> np.ndarray:
        return np.array([self.l2l[lc.locus_id] for lc in loci], dtype=float)


class _DiagAdapter:
    """Adapter that exposes multiple readouts. ``primary`` is anti-
    correlated with the label (ρ = −1); the ``"chromatin × flank"``
    readout is the perfect predictor (ρ = +1). The benchmark should
    pick the chromatin one as Diagnostic B."""

    def __init__(self, locus_to_label: dict[str, float], seed: int = 0):
        self.l2l = locus_to_label
        self.seed = seed

    def predict_expressions(self, loci):
        perfect = np.array([self.l2l[lc.locus_id] for lc in loci], dtype=float)
        return -perfect  # anti-correlated → ρ = -1

    def predict_diagnostic_readouts(self, loci):
        perfect = np.array(
            [self.l2l[lc.locus_id] for lc in loci], dtype=float,
        )
        rng = np.random.default_rng(self.seed + 1)
        return {
            "primary": -perfect,                   # ρ = -1
            "chromatin × flank": perfect,          # ρ = +1
            "noise × cassette": rng.standard_normal(len(loci)),
        }


assert isinstance(_MockAdapter({}), IGRInsertionExpressionPredictor)
assert isinstance(_DiagAdapter({}), IGRInsertionExpressionPredictor)


class TestHongIGRInsertionBenchmark:
    def _bench(self, hong_data) -> HongIGRInsertionBenchmark:
        return HongIGRInsertionBenchmark(
            labels_path=hong_data,
            cassette_seq=REAL_CASSETTE,
            fasta_path="/dev/null",
            info=INFO,
        )

    def test_init_loads_loci(self, hong_data):
        b = self._bench(hong_data)
        assert len(b.loci) == 7
        assert sum(lc.set == "IntTrain" for lc in b.loci) == 4
        assert sum(lc.set == "IntProp" for lc in b.loci) == 3
        assert b.loci[0].locus_id == "IntTrain1"
        assert b.loci[0].chrom == "I"
        assert b.loci[0].integration_coord == 10_000

    def test_evaluate_perfect_predictor(self, hong_data):
        b = self._bench(hong_data)
        l2l = {lc.locus_id: b.labels[i] for i, lc in enumerate(b.loci)}
        res = b.evaluate(_MockAdapter(l2l))
        # Perfect predictor → ρ = 1.0 per tier
        assert res.metrics["IntTrain"]["spearman_rho"] == pytest.approx(1.0)
        assert res.metrics["IntProp"]["spearman_rho"] == pytest.approx(1.0)
        assert res.metrics["pooled"]["spearman_rho"] == pytest.approx(1.0)
        assert res.metrics["IntTrain"]["n"] == 4
        assert res.metrics["IntProp"]["n"] == 3

    def test_save_load_roundtrip(self, hong_data, tmp_path):
        b = self._bench(hong_data)
        l2l = {lc.locus_id: b.labels[i] for i, lc in enumerate(b.loci)}
        res = b.evaluate(_MockAdapter(l2l))
        out = tmp_path / "out"
        b.save_results(res, out)
        loaded = b.load_results(out)
        np.testing.assert_array_almost_equal(loaded.scores, res.scores)
        np.testing.assert_array_equal(loaded.sets, res.sets)
        assert loaded.locus_ids == res.locus_ids
        assert loaded.metrics["IntProp"]["spearman_rho"] == pytest.approx(
            res.metrics["IntProp"]["spearman_rho"]
        )

    def test_plot_and_summary_and_headline(self, hong_data, tmp_path):
        b = self._bench(hong_data)
        l2l = {lc.locus_id: b.labels[i] for i, lc in enumerate(b.loci)}
        res = b.evaluate(_MockAdapter(l2l))
        b.plot(res, tmp_path / "p")
        for png in ("scatter_primary.png", "top_k_enrichment.png"):
            assert (tmp_path / "p" / png).exists(), png

        s = b.summary_dict(res)
        assert s["n_rows_total"] == 7
        assert s["mrna_fluo_ceiling_spcc_published"] == MRNA_FLUO_CEILING_SPCC_PUBLISHED
        assert s["IntProp_spearman_rho"] == pytest.approx(1.0)
        assert s["IntTrain_spearman_rho"] == pytest.approx(1.0)

        h = b.headline(res)
        assert "Primary" in h and "IntProp ρ" in h and "IntTrain ρ" in h
        # Without diagnostic readouts, headline notes Diag B unavailable
        assert "Diag B" in h


# ── Diagnostic B ─────────────────────────────────────────────


class TestDiagnosticB:
    def _bench(self, hong_data):
        return HongIGRInsertionBenchmark(
            labels_path=hong_data,
            cassette_seq=REAL_CASSETTE,
            fasta_path="/dev/null",
            info=INFO,
        )

    def test_select_diagnostic_b_picks_best_intrain(self):
        """_select_diagnostic_b should pick the candidate that maximises
        signed Spearman ρ on IntTrain rows."""
        labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        sets = np.array(['IntTrain', 'IntTrain', 'IntTrain', 'IntProp', 'IntProp', 'IntProp'])
        readouts = {
            "primary": np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.0]),  # anti-correlated
            "perfect_train": labels.copy(),                       # ρ=1 on IntTrain, ρ=1 on IntProp
            "noise":         np.array([0.7, 0.1, 0.9, 0.4, 0.3, 0.2]),
        }
        diag = _select_diagnostic_b(readouts, labels, sets)
        assert diag is not None
        assert diag.readout_name == "perfect_train"
        assert diag.intrain_rho == pytest.approx(1.0)
        assert diag.intprop_rho == pytest.approx(1.0)
        assert diag.n_candidates == 2  # "perfect_train" + "noise"

    def test_select_returns_none_if_no_candidates(self):
        labels = np.array([1.0, 2.0, 3.0])
        sets = np.array(['IntTrain'] * 3)
        assert _select_diagnostic_b({"primary": labels}, labels, sets) is None

    def test_evaluate_with_diagnostic_adapter(self, hong_data):
        b = self._bench(hong_data)
        l2l = {lc.locus_id: b.labels[i] for i, lc in enumerate(b.loci)}
        res = b.evaluate(_DiagAdapter(l2l))
        # Primary is anti-correlated → ρ = -1
        assert res.metrics["IntProp"]["spearman_rho"] == pytest.approx(-1.0)
        # Diagnostic B picks the perfect chromatin × flank readout
        assert res.diagnostic_b is not None
        assert res.diagnostic_b.readout_name == "chromatin × flank"
        assert res.diagnostic_b.intrain_rho == pytest.approx(1.0)
        assert res.diagnostic_b.intprop_rho == pytest.approx(1.0)

    def test_evaluate_without_diagnostic_adapter(self, hong_data):
        """Adapters that don't implement predict_diagnostic_readouts
        still work — Diag B is None."""
        b = self._bench(hong_data)
        l2l = {lc.locus_id: b.labels[i] for i, lc in enumerate(b.loci)}
        res = b.evaluate(_MockAdapter(l2l))
        assert res.diagnostic_b is None

    def test_save_load_with_diagnostic_b(self, hong_data, tmp_path):
        b = self._bench(hong_data)
        l2l = {lc.locus_id: b.labels[i] for i, lc in enumerate(b.loci)}
        res = b.evaluate(_DiagAdapter(l2l))
        assert res.diagnostic_b is not None
        out = tmp_path / "out"
        b.save_results(res, out)
        # Files written
        assert (out / "diagnostic_b_scores.npy").exists()
        loaded = b.load_results(out)
        assert loaded.diagnostic_b is not None
        assert loaded.diagnostic_b.readout_name == res.diagnostic_b.readout_name
        assert loaded.diagnostic_b.intrain_rho == pytest.approx(res.diagnostic_b.intrain_rho)
        assert loaded.diagnostic_b.intprop_rho == pytest.approx(res.diagnostic_b.intprop_rho)
        np.testing.assert_array_almost_equal(
            loaded.diagnostic_b.scores, res.diagnostic_b.scores
        )

    def test_summary_and_headline_include_diag_b(self, hong_data, tmp_path):
        b = self._bench(hong_data)
        l2l = {lc.locus_id: b.labels[i] for i, lc in enumerate(b.loci)}
        res = b.evaluate(_DiagAdapter(l2l))
        s = b.summary_dict(res)
        assert s["diag_b_readout_name"] == "chromatin × flank"
        assert s["diag_b_intprop_spearman_rho"] == pytest.approx(1.0)
        h = b.headline(res)
        assert "Diag B [chromatin × flank]" in h

    def test_plot_includes_diag_b_panel(self, hong_data, tmp_path):
        b = self._bench(hong_data)
        l2l = {lc.locus_id: b.labels[i] for i, lc in enumerate(b.loci)}
        res = b.evaluate(_DiagAdapter(l2l))
        b.plot(res, tmp_path / "p")
        assert (tmp_path / "p" / "scatter_diagnostic_b.png").exists()


# ── Registry ──────────────────────────────────────────────────


class TestHongRegistry:
    def test_task_registered(self):
        assert "hong_igr" in TASKS

    def test_factory_builds_benchmark(self, hong_data):
        task = TASKS["hong_igr"](
            labels_path=hong_data,
            cassette_seq=REAL_CASSETTE,
            fasta_path="/dev/null",
        )
        assert isinstance(task, HongIGRInsertionBenchmark)
        assert task.adapter_protocol is IGRInsertionExpressionPredictor


# ── Real distribution sanity ──────────────────────────────────


REAL_DISTRIBUTION = Path("data/tasks/hong/hong_igr_v1.tsv")


@pytest.mark.skipif(
    not REAL_DISTRIBUTION.exists(),
    reason="real Hong distribution missing (run scripts/hong/build_hong_distribution.py)",
)
class TestRealDistribution:
    def test_shape(self):
        df = pd.read_csv(REAL_DISTRIBUTION, sep="\t")
        assert len(df) == 150
        assert (df["set"] == "IntTrain").sum() == 98
        assert (df["set"] == "IntProp").sum() == 52

    def test_intrain92_is_reference(self):
        df = pd.read_csv(REAL_DISTRIBUTION, sep="\t")
        row = df[df["locus_id"] == "IntTrain92"].iloc[0]
        assert row["fluorescence_norm_intrain92"] == pytest.approx(1.0)
        assert row["chrom"] == "XVI"
