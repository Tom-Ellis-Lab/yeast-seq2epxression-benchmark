"""Tests for the Wu et al. RFP-insertion benchmark + scaffold."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pysam
import pytest

from yeastbench.adapters._genome import Gene, parse_gene_annotations
from yeastbench.adapters._wu_scaffold import (
    PAYLOAD_LEN,
    RFP_CDS_LEN,
    RFP_CDS_START_IN_PAYLOAD,
    WuLocus,
    build_insertion_context,
    load_cassette_payload,
    resolve_loci,
    reverse_complement,
)
from yeastbench.adapters.protocols import CassetteExpressionPredictor
from yeastbench.benchmarks.base import BenchmarkInfo
from yeastbench.benchmarks.rfpins import (
    CLASS_NAMES,
    RFPInsertionBenchmark,
    _classify_absolute,
    _metrics,
    _rank_match_classes,
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

    def test_plus_strand_centers_atg(self, mini_genome):
        payload = load_cassette_payload(REAL_CASSETTE)
        locus = WuLocus("G+", "I", "+", 30_000, 30_300)  # ORF 30000..30300
        ctx = build_insertion_context(locus, payload, mini_genome, **self.PARAMS)
        assert ctx is not None
        assert len(ctx.window_seq) == self.PARAMS["seq_len"]
        assert ctx.rfp_bins.size > 0
        # mCherry start codon centred (mid-chromosome → no clamping)
        atg30 = payload[RFP_CDS_START_IN_PAYLOAD : RFP_CDS_START_IN_PAYLOAD + 30]
        idx = ctx.window_seq.find(atg30)
        assert idx != -1
        assert abs(idx - self.PARAMS["seq_len"] // 2) <= 1

    def test_minus_strand_inserts_revcomp(self, mini_genome):
        payload = load_cassette_payload(REAL_CASSETTE)
        locus = WuLocus("G-", "I", "-", 30_000, 30_300)
        ctx = build_insertion_context(locus, payload, mini_genome, **self.PARAMS)
        assert ctx is not None
        assert len(ctx.window_seq) == self.PARAMS["seq_len"]
        assert ctx.rfp_bins.size > 0
        # cassette is reverse-complemented for − strand ORFs
        rc_chunk = reverse_complement(payload)[:30]
        assert rc_chunk in ctx.window_seq

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


# ── Metrics ───────────────────────────────────────────────────


class TestMetrics:
    def test_classify_absolute(self):
        v = np.array([0.98, 4.99, 5.0, 6.5, 7.99, 8.0, 12.9])
        np.testing.assert_array_equal(
            _classify_absolute(v), np.array([0, 0, 1, 2, 3, 4, 4])
        )

    def test_rank_match_preserves_class_sizes(self):
        true_cls = np.array([0, 0, 1, 1, 1, 2, 3, 4])
        pred = np.array([0.1, 0.2, 9, 8, 7, 3, 4, 5], dtype=float)
        pc = _rank_match_classes(pred, true_cls)
        for c in range(5):
            assert (pc == c).sum() == (true_cls == c).sum()
        # lowest two predictions → class 0
        assert set(pc[[0, 1]]) == {0}

    def test_metrics_perfect_correlation(self):
        labels = np.linspace(1.0, 12.0, 200)
        scores = labels * 3.0 + 7.0  # monotone → r = ρ = 1
        m = _metrics(scores, labels)
        assert m["n_scored"] == 200
        assert m["pearson_r"] > 0.999
        assert m["spearman_rho"] > 0.999
        assert m["qwk"] > 0.999
        assert m["confusion"].sum() == 200

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
        assert (tmp_path / "p" / "scatter.png").exists()
        assert (tmp_path / "p" / "measured_classes.png").exists()
        s = b.summary_dict(res)
        assert s["n_rows_total"] == 8
        assert s["n_rows_scored"] == 7
        assert s["n_dropped_unresolved"] == 1
        assert len(s["confusion"]) == 5
        assert "Pearson" in b.headline(res) and "Spearman" in b.headline(res)


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
