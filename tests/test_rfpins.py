"""Tests for the Wu et al. RFP-insertion benchmark + scaffold."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pysam
import pytest

from yeastbench.adapters._genome import Gene
from yeastbench.adapters._wu_scaffold import (
    CASSETTE_FEATURES,
    D1_SEQ,
    D2_SEQ,
    DEFAULT_BARCODES_TSV,
    PAYLOAD_LEN,
    RFP_CDS_LEN,
    RFP_CDS_START_IN_PAYLOAD,
    U1_SEQ,
    U2_SEQ,
    WuLocus,
    build_insertion_context,
    inject_barcodes,
    load_barcodes,
    load_cassette_payload,
    payload_feature_window_span,
    resolve_loci,
    reverse_complement,
    span_to_bins,
)
from yeastbench.adapters.protocols import CassetteExpressionPredictor
from yeastbench.benchmarks.base import BenchmarkInfo
from yeastbench.benchmarks.rfpins import (
    CLASS_NAMES,
    EXTREME_HIGH_CUTOFF,
    EXTREME_LOW_CUTOFF,
    RFPInsertionBenchmark,
    _metrics,
)
from yeastbench.registry import TASKS

INFO = BenchmarkInfo(name="test_wu", version="test", description="t", distribution_uri="")
REAL_CASSETTE = Path("data/tasks/wu_rfpins/expression_cassette.fasta")


# ── Cassette payload ──────────────────────────────────────────


class TestCassettePayload:
    def test_loads_real_payload(self):
        p = load_cassette_payload(REAL_CASSETTE)
        assert len(p) == PAYLOAD_LEN == 3522
        assert p[RFP_CDS_START_IN_PAYLOAD : RFP_CDS_START_IN_PAYLOAD + 3] == "ATG"

    def test_bad_length_raises(self, tmp_path):
        bad = tmp_path / "bad.fasta"
        bad.write_text(">x\nACGT\n")
        with pytest.raises(ValueError, match="expected 3522"):
            load_cassette_payload(bad)

    def test_bad_rfp_start_raises(self, tmp_path):
        bad = tmp_path / "bad.fasta"
        bad.write_text(">x\n" + "T" * PAYLOAD_LEN + "\n")
        with pytest.raises(ValueError, match="ATG"):
            load_cassette_payload(bad)


class TestReverseComplement:
    def test_basic(self):
        assert reverse_complement("ACGT") == "ACGT"
        assert reverse_complement("AAAAC") == "GTTTT"

    def test_preserves_n(self):
        assert reverse_complement("NACN") == "NGTN"

    def test_involution_on_payload(self):
        p = load_cassette_payload(REAL_CASSETTE)
        assert reverse_complement(reverse_complement(p)) == p


# ── resolve_loci ──────────────────────────────────────────────


class TestResolveLoci:
    def test_resolves_and_drops(self):
        genes = {
            "YAL001C": Gene("I", "-", 200, 100, 200, ((100, 200),)),
            "YBR002W": Gene("II", "+", 300, 300, 500, ((300, 500),)),
        }
        loci, dropped = resolve_loci(["YAL001C", "MISSING", "YBR002W"], genes)
        assert dropped == ["MISSING"]
        assert loci[1] is None
        assert loci[0] == WuLocus("YAL001C", "I", "-", 100, 200)
        assert loci[2] == WuLocus("YBR002W", "II", "+", 300, 500)


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

    def test_plus_strand_stop_at_downstream_crop_edge(self, mini_genome):
        payload = load_cassette_payload(REAL_CASSETTE)
        locus = WuLocus("G+", "I", "+", 30_000, 30_300)
        ctx = build_insertion_context(locus, payload, mini_genome, **self.PARAMS)
        assert ctx is not None
        assert len(ctx.window_seq) == self.PARAMS["seq_len"]
        assert ctx.rfp_bins.size > 0
        sl, crop = self.PARAMS["seq_len"], self.PARAMS["crop_bp_each_side"]
        # mCherry stop at the right crop edge → CDS 5' at (sl-crop)-CDS_LEN
        exp = (sl - crop) - RFP_CDS_LEN
        assert ctx.rfp_start_in_window == exp
        atg = payload[RFP_CDS_START_IN_PAYLOAD : RFP_CDS_START_IN_PAYLOAD + 30]
        assert ctx.window_seq[exp : exp + 30] == atg
        # payload placed forward; oriented payload starts where claimed
        assert ctx.payload_rc is False
        ps = ctx.payload_start_in_window
        assert ctx.window_seq[ps : ps + 18] == payload[:18]  # U1

    def test_minus_strand_stop_at_upstream_genomic_crop_edge(self, mini_genome):
        payload = load_cassette_payload(REAL_CASSETTE)
        locus = WuLocus("G-", "I", "-", 30_000, 30_300)
        ctx = build_insertion_context(locus, payload, mini_genome, **self.PARAMS)
        assert ctx is not None
        assert len(ctx.window_seq) == self.PARAMS["seq_len"]
        assert ctx.rfp_bins.size > 0
        crop = self.PARAMS["crop_bp_each_side"]
        # − strand: transcription stop at the LEFT crop edge → CDS 5' = crop
        assert ctx.rfp_start_in_window == crop
        assert ctx.payload_rc is True
        rc = reverse_complement(payload)
        rfp0 = len(payload) - RFP_CDS_START_IN_PAYLOAD - RFP_CDS_LEN
        assert ctx.window_seq[crop : crop + 30] == rc[rfp0 : rfp0 + 30]

    def test_rfp_base_positions_are_exact_cds_span(self, mini_genome):
        # Per-base readout: the mCherry CDS maps to exactly RFP_CDS_LEN
        # contiguous base positions, clamped to the output crop. For a +
        # strand locus the stop sits at the downstream crop edge, so the CDS
        # base positions are the last RFP_CDS_LEN of the [0, out_len) output.
        payload = load_cassette_payload(REAL_CASSETTE)
        locus = WuLocus("G+", "I", "+", 30_000, 30_300)
        ctx = build_insertion_context(locus, payload, mini_genome, **self.PARAMS)
        assert ctx is not None
        out_len = self.PARAMS["output_bins"] * self.PARAMS["bin_width"]  # 14336
        assert ctx.rfp_base_positions.size == RFP_CDS_LEN  # 711, exact
        np.testing.assert_array_equal(
            ctx.rfp_base_positions, np.arange(out_len - RFP_CDS_LEN, out_len)
        )

    def test_base_readout_is_tighter_than_bin_readout(self, mini_genome):
        # RFP_CDS_LEN (711) is not a multiple of bin_width (16), so the
        # whole-bin readout overhangs the CDS edges: base count < bins*width.
        payload = load_cassette_payload(REAL_CASSETTE)
        locus = WuLocus("G+", "I", "+", 30_000, 30_300)
        ctx = build_insertion_context(locus, payload, mini_genome, **self.PARAMS)
        assert ctx is not None
        bin_footprint = ctx.rfp_bins.size * self.PARAMS["bin_width"]
        assert ctx.rfp_base_positions.size == RFP_CDS_LEN
        assert bin_footprint > RFP_CDS_LEN  # bins round outward; bases are exact

    def test_minus_strand_base_positions_match_bins_span(self, mini_genome):
        # − strand: stop at the LEFT crop edge → CDS base positions start at 0.
        payload = load_cassette_payload(REAL_CASSETTE)
        locus = WuLocus("G-", "I", "-", 30_000, 30_300)
        ctx = build_insertion_context(locus, payload, mini_genome, **self.PARAMS)
        assert ctx is not None
        np.testing.assert_array_equal(
            ctx.rfp_base_positions, np.arange(0, RFP_CDS_LEN)
        )

    def test_short_chromosome_returns_none(self, tmp_path):
        # Chromosome shorter than SEQ_LEN → native+payload can't fill a
        # full window even with all available flank.
        payload = load_cassette_payload(REAL_CASSETTE)
        rng = np.random.default_rng(1)
        chrom = "".join(rng.choice(list("ACGT"), 5_000))
        fa = tmp_path / "short.fa"
        fa.write_text(">I\n" + "\n".join(
            chrom[i : i + 60] for i in range(0, len(chrom), 60)
        ) + "\n")
        pysam.faidx(str(fa))
        g = pysam.FastaFile(str(fa))
        locus = WuLocus("Gshort", "I", "+", 2_000, 2_300)
        ctx = build_insertion_context(locus, payload, g, **self.PARAMS)
        assert ctx is None


# ── Per-base adapter wiring (stub model, no GPU) ──────────────


class _StubPerbaseModel:
    """Minimal stand-in for the model wrapper exposing the per-base forwards
    the Wu adapters call. Base position ``p`` carries value ``p`` on every
    track, so a base-sum over the readout is a known arithmetic series and
    the strand-matched track mean is trivial to predict."""

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


def _write_mini_genome(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    chrom = "".join(rng.choice(list("ACGT"), 60_000))
    fa = tmp_path / "mini.fa"
    fa.write_text(">I\n" + "\n".join(chrom[i : i + 60] for i in range(0, len(chrom), 60)) + "\n")
    pysam.faidx(str(fa))
    return fa


def _write_mini_gtf(tmp_path: Path) -> Path:
    gtf = tmp_path / "mini.gtf"
    gtf.write_text(
        'I\tsgd\tgene\t30000\t30300\t.\t+\t.\tgene_id "G+";\n'
        'I\tsgd\texon\t30000\t30300\t.\t+\t.\tgene_id "G+";\n'
    )
    return gtf


def _write_mini_barcodes(tmp_path: Path, gene_ids: Sequence[str]) -> Path:
    """A barcodes.tsv with valid (distinct) 20 bp tags for the given loci."""
    bc = tmp_path / "barcodes.tsv"
    lines = ["ORF_name\tuptag\tdntag\tuptag_source\tdntag_source"]
    for k, g in enumerate(gene_ids):
        up = ("ACGT" * 5)[:20]
        dn = (("TGCA" * 5)[:19] + "ACGT"[k % 4])  # distinct per locus
        lines.append(f"{g}\t{up}\t{dn}\tdesigned\tdesigned")
    bc.write_text("\n".join(lines) + "\n")
    return bc


class TestWuPerBaseAdapter:
    """The adapters call the per-base forward and read the cassette CDS by
    base position; with an identity-per-position stub the score is the exact
    sum of the readout base positions (raw counts, not transformed bins)."""

    def test_yorzoi_wu_sums_raw_base_positions(self, tmp_path):
        pytest.importorskip("torch")
        from yeastbench.adapters._yorzoi_constants import (
            BIN_WIDTH, CROP_BP_EACH_SIDE, OUTPUT_BINS, SEQ_LEN,
        )
        from yeastbench.adapters.yorzoi_wu import YorzoiWuPredictor

        fa, gtf = _write_mini_genome(tmp_path), _write_mini_gtf(tmp_path)
        payload = load_cassette_payload(REAL_CASSETTE)
        locus = WuLocus("G+", "I", "+", 30_000, 30_300)
        ctx = build_insertion_context(
            locus, payload, pysam.FastaFile(str(fa)),
            SEQ_LEN, CROP_BP_EACH_SIDE, BIN_WIDTH, OUTPUT_BINS,
        )
        model = _StubPerbaseModel(OUTPUT_BINS * BIN_WIDTH, n_tracks=162)
        adapter = YorzoiWuPredictor(
            model, fasta_path=fa, gtf_path=gtf,
            barcodes_path=_write_mini_barcodes(tmp_path, ["G+"]), batch_size=4,
        )
        out = adapter.predict_expressions([locus])
        # barcodes land outside the mCherry CDS, so the readout positions
        # (hence the stub's position-sum score) are unchanged by injection.
        assert out[0] == pytest.approx(float(ctx.rfp_base_positions.sum()))

    def test_shorkie_wu_sums_raw_base_positions(self, tmp_path):
        pytest.importorskip("torch")
        from yeastbench.adapters._shorkie_constants import (
            BIN_WIDTH, CROP_BP_EACH_SIDE, OUTPUT_BINS, SEQ_LEN,
        )
        from yeastbench.adapters.shorkie_wu import ShorkieWuPredictor

        fa, gtf = _write_mini_genome(tmp_path), _write_mini_gtf(tmp_path)
        payload = load_cassette_payload(REAL_CASSETTE)
        locus = WuLocus("G+", "I", "+", 30_000, 30_300)
        ctx = build_insertion_context(
            locus, payload, pysam.FastaFile(str(fa)),
            SEQ_LEN, CROP_BP_EACH_SIDE, BIN_WIDTH, OUTPUT_BINS,
        )
        model = _StubPerbaseModel(OUTPUT_BINS * BIN_WIDTH)
        adapter = ShorkieWuPredictor(
            model, fasta_path=fa, gtf_path=gtf,
            barcodes_path=_write_mini_barcodes(tmp_path, ["G+"]), batch_size=4,
        )
        out = adapter.predict_expressions([locus])
        assert out[0] == pytest.approx(float(ctx.rfp_base_positions.sum()))


# ── Barcode injection + data ──────────────────────────────────

REAL_BARCODES = Path("data/tasks/wu_rfpins/barcodes.tsv")
REAL_LABELS = Path("data/tasks/wu_rfpins/table_s2_fluorescence_1044_loci.csv")


class TestBarcodeInjection:
    UP = "AAAACCCCGGGGTTTTACGT"  # 20 bp
    DN = "TTTTGGGGCCCCAAAATGCA"  # 20 bp

    def test_replaces_slots_and_preserves_flanks(self):
        p = load_cassette_payload(REAL_CASSETTE)
        new = inject_barcodes(p, self.UP, self.DN)
        assert len(new) == PAYLOAD_LEN
        assert new[18:38] == self.UP and new[3485:3505] == self.DN
        assert new[0:18] == U1_SEQ and new[38:56] == U2_SEQ
        assert new[3466:3485] == D2_SEQ and new[3505:3522] == D1_SEQ
        # nothing between/around the slots changes (only the 2×20 bp slots)
        assert new[56:3466] == p[56:3466]
        assert new[3522:] == p[3522:]

    def test_removes_all_placeholder_Ns(self):
        # The frozen scaffold has exactly the two 20×N slots; injecting real
        # tags must leave zero N (the all-zero one-hot OOD artifact gone).
        p = load_cassette_payload(REAL_CASSETTE)
        assert p.count("N") == 40
        assert inject_barcodes(p, self.UP, self.DN).count("N") == 0

    def test_rejects_bad_tags(self):
        p = load_cassette_payload(REAL_CASSETTE)
        with pytest.raises(ValueError):
            inject_barcodes(p, "ACGT", self.DN)            # too short
        with pytest.raises(ValueError):
            inject_barcodes(p, "N" * 20, self.DN)          # N not allowed
        with pytest.raises(ValueError):
            inject_barcodes(p, self.UP, "ACGTX" + "A" * 15)  # non-ACGT char

    def test_rejects_bad_payload(self):
        with pytest.raises(ValueError):
            inject_barcodes("ACGT", self.UP, self.DN)       # wrong length
        with pytest.raises(ValueError):
            inject_barcodes("A" * PAYLOAD_LEN, self.UP, self.DN)  # flanks absent

    def test_reinjection_is_idempotent(self):
        p = load_cassette_payload(REAL_CASSETTE)
        once = inject_barcodes(p, self.UP, self.DN)
        assert inject_barcodes(once, self.UP, self.DN) == once


class TestBarcodesData:
    def test_loads_real_barcodes(self):
        bc = load_barcodes(REAL_BARCODES)
        assert len(bc) == 1044
        for up, dn in bc.values():
            assert len(up) == 20 and len(dn) == 20
            assert set(up) <= set("ACGT") and set(dn) <= set("ACGT")

    def test_default_path_matches_real_file(self):
        assert DEFAULT_BARCODES_TSV == REAL_BARCODES.resolve() or \
            DEFAULT_BARCODES_TSV.name == "barcodes.tsv"

    def test_covers_every_labelled_orf(self):
        # The adapter indexes barcodes by gene_id with no fallback, so the
        # table must cover every ORF in the labels CSV or scoring KeyErrors.
        bc = load_barcodes(REAL_BARCODES)
        orfs = pd.read_csv(REAL_LABELS)["ORF_name"].astype(str).tolist()
        assert [o for o in orfs if o not in bc] == []

    def test_rejects_table_with_N_tag(self, tmp_path):
        bad = tmp_path / "bad.tsv"
        bad.write_text(
            "ORF_name\tuptag\tdntag\tuptag_source\tdntag_source\n"
            "YAL001C\t" + "N" * 20 + "\t" + "ACGT" * 5 + "\tdesigned\tdesigned\n"
        )
        with pytest.raises(ValueError, match="non-20bp/non-ACGT"):
            load_barcodes(bad)


# ── Annotation helpers ────────────────────────────────────────


class TestAnnotationHelpers:
    def test_cassette_features_consistent(self):
        names = [f[0] for f in CASSETTE_FEATURES]
        assert "mCherry" in names and "pURA3" in names
        for _, _, a, b in CASSETTE_FEATURES:
            assert 0 <= a < b <= PAYLOAD_LEN
        mch = next(f for f in CASSETTE_FEATURES if f[0] == "mCherry")
        assert mch[2] == RFP_CDS_START_IN_PAYLOAD
        assert mch[3] == RFP_CDS_START_IN_PAYLOAD + RFP_CDS_LEN

    def test_feature_window_span_forward(self):
        lo, hi = payload_feature_window_span(10, 20, payload_start_in_window=100, payload_rc=False)
        assert (lo, hi) == (110, 120)

    def test_feature_window_span_rc(self):
        # forward [10,20) → rc maps to [PAYLOAD_LEN-20, PAYLOAD_LEN-10)
        lo, hi = payload_feature_window_span(10, 20, payload_start_in_window=0, payload_rc=True)
        assert (lo, hi) == (PAYLOAD_LEN - 20, PAYLOAD_LEN - 10)

    def test_span_to_bins(self):
        # window [1024+0, 1024+32) with crop 1024, 16 bp/bin → bins [0,2)
        assert span_to_bins(1024, 1056, 1024, 16, 896) == (0, 2)
        # entirely in the cropped-out region → None
        assert span_to_bins(0, 100, 1024, 16, 896) is None


# ── Metrics ───────────────────────────────────────────────────


class TestMetrics:
    def test_cutoffs(self):
        assert EXTREME_LOW_CUTOFF == 5.0
        assert EXTREME_HIGH_CUTOFF == 8.0

    def test_metrics_perfect_predictor(self):
        labels = np.linspace(1.0, 12.0, 200)
        scores = labels * 3.0 + 7.0  # monotone → perfect ranking
        m = _metrics(scores, labels)
        assert m["n_scored"] == 200
        assert m["pearson_r"] > 0.999
        assert m["spearman_rho"] > 0.999
        # tails perfectly separable by a monotone predictor
        assert m["low_n_pos"] == int((labels < 5.0).sum())
        assert m["high_n_pos"] == int((labels >= 8.0).sum())
        assert m["low_auroc"] > 0.999 and m["low_auprc"] > 0.999
        assert m["high_auroc"] > 0.999 and m["high_auprc"] > 0.999

    def test_metrics_direction(self):
        # adversarial (anti-correlated) predictor → AUROC ≈ 0
        labels = np.linspace(1.0, 12.0, 200)
        scores = -labels
        m = _metrics(scores, labels)
        assert m["low_auroc"] < 0.001 and m["high_auroc"] < 0.001

    def test_metrics_one_class_is_nan(self):
        labels = np.array([6.0, 6.5, 7.0, 6.2])  # no extreme-low/high
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        m = _metrics(scores, labels)
        assert m["low_n_pos"] == 0 and m["high_n_pos"] == 0
        assert np.isnan(m["low_auroc"]) and np.isnan(m["high_auroc"])

    def test_metrics_ignores_nan(self):
        labels = np.array([1.0, 5.0, 7.0, np.nan, 9.0])
        scores = np.array([0.1, np.nan, 0.3, 0.4, 0.9])
        m = _metrics(scores, labels)
        assert m["n_scored"] == 3  # rows 0, 2, 4


# ── Benchmark ─────────────────────────────────────────────────


@pytest.fixture
def wu_data(tmp_path: Path) -> tuple[Path, Path]:
    """Mini labels CSV (8 ORFs, 1 unresolvable) + matching GTF."""
    orfs = [f"YAL{i:03d}W" for i in range(7)] + ["YZZ999W"]  # last not in GTF
    vals = [1.0, 4.0, 5.5, 6.5, 7.5, 8.5, 9.0, 11.0]
    df = pd.DataFrame({
        "No.": [f"TH{i:05d}" for i in range(8)],
        "ORF_name": orfs,
        "Relative_Fluorescence_Average": vals,
        "Relative_Fluorescence_Error": [0.1] * 8,
    })
    csv = tmp_path / "labels.csv"
    df.to_csv(csv, index=False)

    gtf_lines = []
    for i in range(7):
        s = 1000 + i * 1000
        e = s + 300
        attr = f'gene_id "YAL{i:03d}W"; gene_biotype "protein_coding";'
        gtf_lines.append(f"I\tsgd\tgene\t{s}\t{e}\t.\t+\t.\t{attr}")
        gtf_lines.append(f"I\tsgd\texon\t{s}\t{e}\t.\t+\t.\t{attr}")
    gtf = tmp_path / "mini.gtf"
    gtf.write_text("\n".join(gtf_lines) + "\n")
    return csv, gtf


class _MockAdapter:
    """Returns each locus's label by gene_id (perfect predictor)."""

    def __init__(self, gene_to_label: dict[str, float]):
        self.g2l = gene_to_label

    def predict_expressions(self, loci: Sequence[WuLocus]) -> np.ndarray:
        return np.array([self.g2l[lc.gene_id] for lc in loci], dtype=float)


assert isinstance(_MockAdapter({}), CassetteExpressionPredictor)


class TestRFPInsertionBenchmark:
    def _bench(self, wu_data) -> RFPInsertionBenchmark:
        csv, gtf = wu_data
        return RFPInsertionBenchmark(
            cassette_seq=REAL_CASSETTE, labels_path=csv,
            fasta_path="/dev/null", gtf_path=gtf, info=INFO,
        )

    def test_init_resolves_and_drops(self, wu_data):
        b = self._bench(wu_data)
        assert len(b.labels) == 8
        assert b.dropped == ["YZZ999W"]
        assert sum(lc is not None for lc in b.loci) == 7

    def test_evaluate_perfect(self, wu_data):
        b = self._bench(wu_data)
        g2l = {b.gene_ids[i]: b.labels[i] for i in range(len(b.labels))}
        res = b.evaluate(_MockAdapter(g2l))
        assert res.n_scored == 7  # 8 rows − 1 unresolved
        assert res.pearson_r > 0.999
        assert np.isnan(res.scores[7])  # the dropped ORF row

    def test_save_load_roundtrip(self, wu_data, tmp_path):
        b = self._bench(wu_data)
        g2l = {b.gene_ids[i]: b.labels[i] for i in range(len(b.labels))}
        res = b.evaluate(_MockAdapter(g2l))
        out = tmp_path / "out"
        b.save_results(res, out)
        loaded = b.load_results(out)
        np.testing.assert_array_almost_equal(
            np.nan_to_num(loaded.scores, nan=-1), np.nan_to_num(res.scores, nan=-1)
        )
        assert loaded.dropped_ids == res.dropped_ids
        assert loaded.n_scored == res.n_scored

    def test_plot_and_summary_and_headline(self, wu_data, tmp_path):
        b = self._bench(wu_data)
        g2l = {b.gene_ids[i]: b.labels[i] for i in range(len(b.labels))}
        res = b.evaluate(_MockAdapter(g2l))
        b.plot(res, tmp_path / "p")
        for png in ("scatter.png", "measured_classes.png",
                    "roc_pr_extreme_low.png", "roc_pr_extreme_high.png"):
            assert (tmp_path / "p" / png).exists(), png
        s = b.summary_dict(res)
        assert s["n_rows_total"] == 8
        assert s["n_rows_scored"] == 7
        assert s["n_dropped_unresolved"] == 1
        assert s["extreme_low_n_pos"] == 2 and s["extreme_high_n_pos"] == 2
        # perfect mock predictor separates both tails
        assert s["extreme_low_auroc"] > 0.999
        assert s["extreme_high_auroc"] > 0.999
        h = b.headline(res)
        assert "Pearson" in h and "extreme-low AUROC" in h


class TestWuRegistry:
    def test_task_registered(self):
        assert "wu_rfpins" in TASKS

    def test_factory_builds_benchmark(self, wu_data):
        csv, gtf = wu_data
        task = TASKS["wu_rfpins"](
            cassette_seq=REAL_CASSETTE, labels_path=csv,
            fasta_path="/dev/null", gtf_path=gtf,
        )
        assert isinstance(task, RFPInsertionBenchmark)
        assert task.adapter_protocol is CassetteExpressionPredictor
        assert len(CLASS_NAMES) == 5
