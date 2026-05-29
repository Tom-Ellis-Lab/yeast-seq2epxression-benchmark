"""Tests for the Hong et al. IGR-insertion benchmark + scaffold."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pysam
import pytest

from yeastbench.adapters._hong_scaffold import (
    DIAGNOSTIC_REGION_NAMES,
    MCHERRY_CDS_LEN,
    MCHERRY_CDS_START_IN_PAYLOAD,
    PAYLOAD_LEN,
    HongLocus,
    aggregate_diagnostic_base_readouts,
    aggregate_diagnostic_readouts,
    build_insertion_context,
    load_cassette_payload,
    readout_region_base_positions,
    readout_region_bins,
)
from yeastbench.adapters.protocols import IGRInsertionExpressionPredictor
from yeastbench.benchmarks.base import BenchmarkInfo
from yeastbench.benchmarks.hong_igr import (
    HongIGRInsertionBenchmark,
    MRNA_FLUO_CEILING_SPCC_PUBLISHED,
    _topk_enrichment,
    _select_inttrain_fitted,
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

    def test_near_chromosome_end_shifts_cassette_off_center(self, mini_genome):
        """When a locus is too close to a chromosome end to centre the
        cassette, `_cassette_scaffold.build_insertion_context` clamps
        ``window_start``. The cassette is then no longer at the window
        midpoint, and ``mcherry_bins`` shifts accordingly. Adapters that
        derive readout-region bins must do so per-locus, not once from
        an arbitrary context (regression for the IntProp5 bug).
        """
        payload = load_cassette_payload(REAL_CASSETTE)
        chrom_len = mini_genome.get_reference_length("I")
        sl = self.PARAMS["seq_len"]
        centered = build_insertion_context(
            HongLocus("centered", "IntTrain", "I", 30_000),
            payload, mini_genome, **self.PARAMS,
        )
        # mini_genome is 60 kb; with seq_len=16384 any cut closer than
        # seq_len/2 to the right end forces the clamp.
        near_end = build_insertion_context(
            HongLocus("nearend", "IntProp", "I", chrom_len - 4_000),
            payload, mini_genome, **self.PARAMS,
        )
        assert centered is not None
        assert near_end is not None
        centered_mid = centered.payload_start_in_window + PAYLOAD_LEN // 2
        near_end_mid = near_end.payload_start_in_window + PAYLOAD_LEN // 2
        assert abs(centered_mid - sl // 2) <= 1
        # Window clamped against the chromosome's right edge → window
        # starts later in spliced coords, so the cassette ends up
        # *further along* (larger index) inside the window.
        assert near_end_mid > centered_mid
        # Therefore the readout bins land on different output positions.
        assert not np.array_equal(near_end.mcherry_bins, centered.mcherry_bins)
        rb_c = readout_region_bins(
            centered,
            self.PARAMS["crop_bp_each_side"],
            self.PARAMS["bin_width"],
            self.PARAMS["output_bins"],
        )
        rb_e = readout_region_bins(
            near_end,
            self.PARAMS["crop_bp_each_side"],
            self.PARAMS["bin_width"],
            self.PARAMS["output_bins"],
        )
        assert not np.array_equal(rb_c["cassette CDS"], rb_e["cassette CDS"])


# ── Per-locus diagnostic-readout aggregation ──────────────────


class TestAggregateDiagnosticReadouts:
    """The aggregator must compute readout-region bins per-locus. If it
    naively used one bin set across all loci, near-end (clamped) loci
    would score against the wrong window positions (the IntProp5 bug)."""

    PARAMS = dict(
        seq_len=16384, crop_bp_each_side=1024, bin_width=16, output_bins=896
    )

    def _two_contexts(self, mini_genome):
        payload = load_cassette_payload(REAL_CASSETTE)
        chrom_len = mini_genome.get_reference_length("I")
        centered = build_insertion_context(
            HongLocus("c", "IntTrain", "I", 30_000),
            payload, mini_genome, **self.PARAMS,
        )
        near_end = build_insertion_context(
            HongLocus("e", "IntProp", "I", chrom_len - 4_000),
            payload, mini_genome, **self.PARAMS,
        )
        assert centered is not None and near_end is not None
        return centered, near_end

    def test_picks_per_locus_bins_not_shared(self, mini_genome):
        centered, near_end = self._two_contexts(mini_genome)
        # Sanity: clamping really did move the cassette.
        assert not np.array_equal(centered.mcherry_bins, near_end.mcherry_bins)

        n = 2
        out_bins = self.PARAMS["output_bins"]
        # Per-locus coverage: row 0 has a peak at *centered*'s cassette
        # CDS bins; row 1 has a peak at *near_end*'s cassette CDS bins.
        # If the aggregator reused row 0's bins for row 1, row 1's
        # "cassette CDS" sum would be 0 (no signal at those positions).
        cov = np.zeros((n, out_bins), dtype=np.float64)
        cov[0, centered.mcherry_bins] = 1.0
        cov[1, near_end.mcherry_bins] = 1.0
        group_cov = {"G": cov}
        contexts = [(0, centered), (1, near_end)]
        track_groups = [("G", [0], +1)]

        readouts = aggregate_diagnostic_readouts(
            group_cov, contexts, track_groups,
            n,
            self.PARAMS["crop_bp_each_side"],
            self.PARAMS["bin_width"],
            self.PARAMS["output_bins"],
        )

        # Both loci should see their full per-locus peak — same value.
        cds = readouts["G × cassette CDS"]
        assert cds[0] == pytest.approx(float(centered.mcherry_bins.size))
        assert cds[1] == pytest.approx(float(near_end.mcherry_bins.size))
        # Cross-check the bug: if we summed row 1 at row 0's bins,
        # the result would be 0 (centered's bins fall outside near_end's
        # cassette region). Confirm the bins really do disagree:
        misapplied = float(cov[1, centered.mcherry_bins].sum())
        assert misapplied == 0.0

    def test_invalid_loci_stay_nan(self, mini_genome):
        centered, _ = self._two_contexts(mini_genome)
        n = 3  # 3 loci total, only one valid context (row 1)
        cov = np.zeros((n, self.PARAMS["output_bins"]), dtype=np.float64)
        cov[1, centered.mcherry_bins] = 1.0
        readouts = aggregate_diagnostic_readouts(
            {"G": cov}, [(1, centered)], [("G", [0], +1)],
            n,
            self.PARAMS["crop_bp_each_side"],
            self.PARAMS["bin_width"],
            self.PARAMS["output_bins"],
        )
        cds = readouts["G × cassette CDS"]
        assert np.isnan(cds[0]) and np.isnan(cds[2])
        assert cds[1] == pytest.approx(float(centered.mcherry_bins.size))

    def test_applies_sign(self, mini_genome):
        centered, _ = self._two_contexts(mini_genome)
        cov = np.zeros((1, self.PARAMS["output_bins"]), dtype=np.float64)
        cov[0, centered.mcherry_bins] = 1.0
        readouts = aggregate_diagnostic_readouts(
            {"H3": cov}, [(0, centered)], [("H3", [0], -1)],
            1,
            self.PARAMS["crop_bp_each_side"],
            self.PARAMS["bin_width"],
            self.PARAMS["output_bins"],
        )
        assert readouts["H3 × cassette CDS"][0] == pytest.approx(
            -float(centered.mcherry_bins.size)
        )

    def test_keys_match_region_names(self, mini_genome):
        centered, _ = self._two_contexts(mini_genome)
        cov = np.zeros((1, self.PARAMS["output_bins"]), dtype=np.float64)
        readouts = aggregate_diagnostic_readouts(
            {"G": cov}, [(0, centered)], [("G", [0], +1)],
            1,
            self.PARAMS["crop_bp_each_side"],
            self.PARAMS["bin_width"],
            self.PARAMS["output_bins"],
        )
        for region_name in DIAGNOSTIC_REGION_NAMES:
            assert f"G × {region_name}" in readouts


# ── Per-base (untransformed) readout ──────────────────────────


class _StubPerbaseModel:
    """Stand-in for the model wrapper exposing the per-base forwards the
    Hong adapters call. Base position ``p`` carries value ``p`` on every
    track, so a base-sum over a region is a known arithmetic series and the
    cross-track mean (Yorzoi) is trivial to predict."""

    def __init__(self, out_len: int, n_tracks: int = 162):
        import torch

        self.device = torch.device("cpu")
        self._ol = out_len
        self._nt = n_tracks

    def forward_tracks_perbase(self, x):  # Yorzoi path
        import torch

        base = torch.arange(self._ol, dtype=torch.float32)
        return base[None, None, :].expand(x.shape[0], self._nt, self._ol).contiguous()

    def forward_track_mean_perbase(self, x, track_subset):  # Shorkie path
        import torch

        base = torch.arange(self._ol, dtype=torch.float32)
        return base[None, :].expand(x.shape[0], self._ol).contiguous()


def _write_mini_genome_path(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    chrom = "".join(rng.choice(list("ACGT"), 60_000))
    fa = tmp_path / "mini.fa"
    fa.write_text(">I\n" + "\n".join(chrom[i : i + 60] for i in range(0, len(chrom), 60)) + "\n")
    pysam.faidx(str(fa))
    return fa


class TestHongPerBaseReadout:
    PARAMS = dict(
        seq_len=16384, crop_bp_each_side=1024, bin_width=16, output_bins=896
    )

    def test_mcherry_base_positions_are_exact_cds_span(self, mini_genome):
        payload = load_cassette_payload(REAL_CASSETTE)
        ctx = build_insertion_context(
            HongLocus("c", "IntTrain", "I", 30_000), payload, mini_genome, **self.PARAMS,
        )
        assert ctx is not None
        # centered cassette → CDS fully inside the crop → exactly RFP_CDS_LEN
        assert ctx.mcherry_base_positions.size == MCHERRY_CDS_LEN
        # contiguous span starting at mcherry_start_in_window - crop
        lo = ctx.mcherry_start_in_window - self.PARAMS["crop_bp_each_side"]
        np.testing.assert_array_equal(
            ctx.mcherry_base_positions, np.arange(lo, lo + MCHERRY_CDS_LEN)
        )

    def test_base_readout_tighter_than_bins(self, mini_genome):
        payload = load_cassette_payload(REAL_CASSETTE)
        ctx = build_insertion_context(
            HongLocus("c", "IntTrain", "I", 30_000), payload, mini_genome, **self.PARAMS,
        )
        assert ctx is not None
        # 711 is not a multiple of 16, so the whole-bin footprint overhangs.
        assert ctx.mcherry_base_positions.size == MCHERRY_CDS_LEN
        assert ctx.mcherry_bins.size * self.PARAMS["bin_width"] > MCHERRY_CDS_LEN

    def test_region_base_positions_are_per_locus(self, mini_genome):
        # Clamped (near-end) and centered loci put the cassette at different
        # window offsets → different base positions (per-locus, not shared).
        payload = load_cassette_payload(REAL_CASSETTE)
        chrom_len = mini_genome.get_reference_length("I")
        centered = build_insertion_context(
            HongLocus("c", "IntTrain", "I", 30_000), payload, mini_genome, **self.PARAMS,
        )
        near_end = build_insertion_context(
            HongLocus("e", "IntProp", "I", chrom_len - 4_000), payload, mini_genome, **self.PARAMS,
        )
        assert centered is not None and near_end is not None
        out_len = self.PARAMS["output_bins"] * self.PARAMS["bin_width"]
        rb_c = readout_region_base_positions(centered, self.PARAMS["crop_bp_each_side"], out_len)
        rb_e = readout_region_base_positions(near_end, self.PARAMS["crop_bp_each_side"], out_len)
        assert not np.array_equal(rb_c["cassette CDS"], rb_e["cassette CDS"])
        # cassette CDS region == the context's mcherry base positions
        np.testing.assert_array_equal(rb_c["cassette CDS"], centered.mcherry_base_positions)

    def test_aggregate_base_readouts_per_locus(self, mini_genome):
        # Base-path analogue of the IntProp5 per-locus regression.
        payload = load_cassette_payload(REAL_CASSETTE)
        chrom_len = mini_genome.get_reference_length("I")
        centered = build_insertion_context(
            HongLocus("c", "IntTrain", "I", 30_000), payload, mini_genome, **self.PARAMS,
        )
        near_end = build_insertion_context(
            HongLocus("e", "IntProp", "I", chrom_len - 4_000), payload, mini_genome, **self.PARAMS,
        )
        assert centered is not None and near_end is not None
        out_len = self.PARAMS["output_bins"] * self.PARAMS["bin_width"]
        cov = np.zeros((2, out_len), dtype=np.float64)
        cov[0, centered.mcherry_base_positions] = 1.0
        cov[1, near_end.mcherry_base_positions] = 1.0
        readouts = aggregate_diagnostic_base_readouts(
            {"G": cov}, [(0, centered), (1, near_end)], [("G", [0], +1)],
            2, self.PARAMS["crop_bp_each_side"], out_len,
        )
        cds = readouts["G × cassette CDS"]
        assert cds[0] == pytest.approx(float(centered.mcherry_base_positions.size))
        assert cds[1] == pytest.approx(float(near_end.mcherry_base_positions.size))

    def test_yorzoi_hong_sums_raw_base_positions(self, tmp_path):
        pytest.importorskip("torch")
        from yeastbench.adapters._yorzoi_constants import (
            BIN_WIDTH, CROP_BP_EACH_SIDE, OUTPUT_BINS, SEQ_LEN,
        )
        from yeastbench.adapters.yorzoi_hong import YorzoiHongPredictor

        fa = _write_mini_genome_path(tmp_path)
        payload = load_cassette_payload(REAL_CASSETTE)
        locus = HongLocus("c", "IntTrain", "I", 30_000)
        ctx = build_insertion_context(
            locus, payload, pysam.FastaFile(str(fa)),
            SEQ_LEN, CROP_BP_EACH_SIDE, BIN_WIDTH, OUTPUT_BINS,
        )
        model = _StubPerbaseModel(OUTPUT_BINS * BIN_WIDTH, n_tracks=162)
        adapter = YorzoiHongPredictor(
            model, fasta_path=fa, batch_size=4, track_annotation_path="/nonexistent",
        )
        out = adapter.predict_expressions([locus])
        # all + tracks identical → mean == base-sum over mcherry base positions
        assert out[0] == pytest.approx(float(ctx.mcherry_base_positions.sum()))

    def test_shorkie_hong_sums_raw_base_positions(self, tmp_path):
        pytest.importorskip("torch")
        from yeastbench.adapters._shorkie_constants import (
            BIN_WIDTH, CROP_BP_EACH_SIDE, OUTPUT_BINS, SEQ_LEN,
        )
        from yeastbench.adapters.shorkie_hong import ShorkieHongPredictor

        fa = _write_mini_genome_path(tmp_path)
        payload = load_cassette_payload(REAL_CASSETTE)
        locus = HongLocus("c", "IntTrain", "I", 30_000)
        ctx = build_insertion_context(
            locus, payload, pysam.FastaFile(str(fa)),
            SEQ_LEN, CROP_BP_EACH_SIDE, BIN_WIDTH, OUTPUT_BINS,
        )
        model = _StubPerbaseModel(OUTPUT_BINS * BIN_WIDTH)
        adapter = ShorkieHongPredictor(
            model, fasta_path=fa, batch_size=4, targets_path="/nonexistent",
        )
        out = adapter.predict_expressions([locus])
        assert out[0] == pytest.approx(float(ctx.mcherry_base_positions.sum()))


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
    pick the chromatin one as IntTrain-fitted."""

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
        # Without diagnostic readouts, headline notes IntTrain-fitted unavailable
        assert "IntTrain-fitted" in h


# ── IntTrain-fitted IntProp ρ ────────────────────────────────


class TestIntTrainFitted:
    def _bench(self, hong_data):
        return HongIGRInsertionBenchmark(
            labels_path=hong_data,
            cassette_seq=REAL_CASSETTE,
            fasta_path="/dev/null",
            info=INFO,
        )

    def test_select_inttrain_fitted_picks_best_intrain(self):
        """_select_inttrain_fitted should pick the candidate that maximises
        signed Spearman ρ on IntTrain rows."""
        labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        sets = np.array(['IntTrain', 'IntTrain', 'IntTrain', 'IntProp', 'IntProp', 'IntProp'])
        readouts = {
            "primary": np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.0]),  # anti-correlated
            "perfect_train": labels.copy(),                       # ρ=1 on IntTrain, ρ=1 on IntProp
            "noise":         np.array([0.7, 0.1, 0.9, 0.4, 0.3, 0.2]),
        }
        fit = _select_inttrain_fitted(readouts, labels, sets)
        assert fit is not None
        assert fit.readout_name == "perfect_train"
        assert fit.intrain_rho == pytest.approx(1.0)
        assert fit.intprop_rho == pytest.approx(1.0)
        assert fit.n_candidates == 2  # "perfect_train" + "noise"

    def test_select_returns_none_if_no_candidates(self):
        labels = np.array([1.0, 2.0, 3.0])
        sets = np.array(['IntTrain'] * 3)
        assert _select_inttrain_fitted({"primary": labels}, labels, sets) is None

    def test_evaluate_with_diagnostic_adapter(self, hong_data):
        b = self._bench(hong_data)
        l2l = {lc.locus_id: b.labels[i] for i, lc in enumerate(b.loci)}
        res = b.evaluate(_DiagAdapter(l2l))
        # Primary is anti-correlated → ρ = -1
        assert res.metrics["IntProp"]["spearman_rho"] == pytest.approx(-1.0)
        # IntTrain-fitted picks the perfect chromatin × flank readout
        assert res.inttrain_fitted is not None
        assert res.inttrain_fitted.readout_name == "chromatin × flank"
        assert res.inttrain_fitted.intrain_rho == pytest.approx(1.0)
        assert res.inttrain_fitted.intprop_rho == pytest.approx(1.0)

    def test_evaluate_without_diagnostic_adapter(self, hong_data):
        """Adapters that don't implement predict_diagnostic_readouts
        still work — IntTrain-fitted is None."""
        b = self._bench(hong_data)
        l2l = {lc.locus_id: b.labels[i] for i, lc in enumerate(b.loci)}
        res = b.evaluate(_MockAdapter(l2l))
        assert res.inttrain_fitted is None

    def test_save_load_with_inttrain_fitted(self, hong_data, tmp_path):
        b = self._bench(hong_data)
        l2l = {lc.locus_id: b.labels[i] for i, lc in enumerate(b.loci)}
        res = b.evaluate(_DiagAdapter(l2l))
        assert res.inttrain_fitted is not None
        out = tmp_path / "out"
        b.save_results(res, out)
        # Files written
        assert (out / "inttrain_fitted_scores.npy").exists()
        loaded = b.load_results(out)
        assert loaded.inttrain_fitted is not None
        assert loaded.inttrain_fitted.readout_name == res.inttrain_fitted.readout_name
        assert loaded.inttrain_fitted.intrain_rho == pytest.approx(res.inttrain_fitted.intrain_rho)
        assert loaded.inttrain_fitted.intprop_rho == pytest.approx(res.inttrain_fitted.intprop_rho)
        np.testing.assert_array_almost_equal(
            loaded.inttrain_fitted.scores, res.inttrain_fitted.scores
        )

    def test_summary_and_headline_include_inttrain_fitted(self, hong_data, tmp_path):
        b = self._bench(hong_data)
        l2l = {lc.locus_id: b.labels[i] for i, lc in enumerate(b.loci)}
        res = b.evaluate(_DiagAdapter(l2l))
        s = b.summary_dict(res)
        assert s["inttrain_fitted_readout_name"] == "chromatin × flank"
        assert s["inttrain_fitted_intprop_spearman_rho"] == pytest.approx(1.0)
        h = b.headline(res)
        assert "IntTrain-fitted [chromatin × flank]" in h

    def test_plot_includes_inttrain_fitted_panel(self, hong_data, tmp_path):
        b = self._bench(hong_data)
        l2l = {lc.locus_id: b.labels[i] for i, lc in enumerate(b.loci)}
        res = b.evaluate(_DiagAdapter(l2l))
        b.plot(res, tmp_path / "p")
        assert (tmp_path / "p" / "scatter_inttrain_fitted.png").exists()


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
