"""Tests for the Cuperus construct assembly + window placement."""
from pathlib import Path

import numpy as np
import pysam
import pytest

from yeastbench.adapters._cuperus_scaffold import (
    CuperusBackground,
    CuperusConstruct,
    build_context,
)

PROMOTER = "AC" * 150                       # 300 bp
HIS3 = "ATG" + "GAT" * 219 + "TAG"          # 663 bp, starts ATG / ends TAG
TERM = "GT" * 125                           # 250 bp
CONSTRUCT = CuperusConstruct(PROMOTER, HIS3, TERM)

PARAMS = dict(seq_len=16384, crop_bp_each_side=1024, bin_width=16, output_bins=896)
OUT_LEN = PARAMS["output_bins"] * PARAMS["bin_width"]  # 14336


@pytest.fixture
def mini_genome(tmp_path: Path) -> pysam.FastaFile:
    rng = np.random.default_rng(0)
    chrom = "".join(rng.choice(list("ACGT"), 60_000))
    fa = tmp_path / "mini.fa"
    fa.write_text(">I\n" + "\n".join(chrom[i : i + 60] for i in range(0, len(chrom), 60)) + "\n")
    pysam.faidx(str(fa))
    return pysam.FastaFile(str(fa))


BG = CuperusBackground("test", "I", 30_000, 30_300)


class TestAssemble:
    def test_layout(self):
        utr = "G" * 50
        cas = CONSTRUCT.assemble(utr)
        assert cas.payload == PROMOTER + utr + HIS3 + TERM
        assert cas.readout_start == len(PROMOTER) + len(utr)
        assert cas.readout_len == len(HIS3)
        # readout slice is the HIS3 ORF
        assert cas.payload[cas.readout_start : cas.readout_start + cas.readout_len] == HIS3

    def test_lowercase_utr_normalised(self):
        cas = CONSTRUCT.assemble("acgt" * 12 + "ac")  # 50 bp lowercase
        assert cas.payload.isupper()


class TestBuildContext:
    def test_window_and_readout(self, mini_genome):
        utr = "GGGGCCCCAAAATTTT" * 3 + "GG"  # 50 bp
        ctx = build_context(CONSTRUCT, utr, BG, mini_genome, **PARAMS)
        assert ctx is not None
        assert len(ctx.window_seq) == PARAMS["seq_len"]
        # the assembled construct sits at payload_start_in_window, forward
        payload = PROMOTER + utr + HIS3 + TERM
        ps = ctx.payload_start_in_window
        assert ctx.window_seq[ps : ps + len(payload)] == payload
        assert ctx.payload_rc is False
        # readout points at the HIS3 ATG
        rs = ctx.readout_start_in_window
        assert rs == ps + len(PROMOTER) + len(utr)
        assert ctx.window_seq[rs : rs + 3] == "ATG"
        # per-base readout = exactly the HIS3 ORF span, within the crop
        assert ctx.readout_base_positions.size == len(HIS3)

    def test_centered(self, mini_genome):
        utr = "A" * 50
        ctx = build_context(CONSTRUCT, utr, BG, mini_genome, **PARAMS)
        payload_len = len(PROMOTER) + len(utr) + len(HIS3) + len(TERM)
        center = ctx.payload_start_in_window + payload_len // 2
        assert abs(center - PARAMS["seq_len"] // 2) <= 1

    def test_variable_length_utr_shifts_readout(self, mini_genome):
        short = "ACG" * 5  # 15 bp (native-style short fragment)
        ctx = build_context(CONSTRUCT, short, BG, mini_genome, **PARAMS)
        assert ctx is not None
        # invariant: HIS3 starts right after promoter + UTR within the payload
        assert ctx.readout_start_in_window == ctx.payload_start_in_window + len(PROMOTER) + len(short)
        assert ctx.window_seq[ctx.readout_start_in_window : ctx.readout_start_in_window + 3] == "ATG"
        assert ctx.readout_base_positions.size == len(HIS3)

    def test_short_chromosome_returns_none(self, tmp_path):
        rng = np.random.default_rng(1)
        chrom = "".join(rng.choice(list("ACGT"), 5_000))  # < seq_len
        fa = tmp_path / "short.fa"
        fa.write_text(">I\n" + "\n".join(chrom[i : i + 60] for i in range(0, len(chrom), 60)) + "\n")
        pysam.faidx(str(fa))
        g = pysam.FastaFile(str(fa))
        bg = CuperusBackground("t", "I", 2_500, 2_800)
        assert build_context(CONSTRUCT, "A" * 50, bg, g, **PARAMS) is None
