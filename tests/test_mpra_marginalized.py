"""Tests for the marginalized / native-position MPRA infrastructure."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from yeastbench.adapters._genome import Gene, one_hot_encode_channels_first
from yeastbench.adapters._marginalized_mpra import (
    HOST_GENES,
    INSERT_LEN,
    InsertionContext,
    build_alt_one_hot,
    extract_insert,
    reverse_complement,
)
from yeastbench.benchmarks.base import BenchmarkInfo
from yeastbench.benchmarks.mpra import MPRAMarginalizedBenchmark


# ── extract_insert ────────────────────────────────────────────


POLYT = "TGCATTTTTTTCACATC"
POLYA = "GGTTACGGCTGTT"
N80 = "A" * 80


class TestExtractInsert:
    def test_returns_full_110bp(self):
        seq = POLYT + N80 + POLYA
        assert len(seq) == 110
        assert extract_insert(seq) == seq

    def test_wrong_length_raises(self):
        with pytest.raises(AssertionError):
            extract_insert("ACGT" * 10)


class TestInsertLen:
    def test_is_110(self):
        assert INSERT_LEN == 110


# ── reverse_complement ────────────────────────────────────────


class TestReverseComplement:
    def test_basic(self):
        assert reverse_complement("ACGT") == "ACGT"
        assert reverse_complement("AAAA") == "TTTT"
        assert reverse_complement("GCTA") == "TAGC"

    def test_roundtrip(self):
        seq = "ACGTACGTACGT"
        assert reverse_complement(reverse_complement(seq)) == seq


# ── build_alt_one_hot ─────────────────────────────────────────


class TestBuildAltOneHot:
    def test_channels_first_positive_strand(self):
        ref = np.zeros((4, 300), dtype=np.float32)
        insert = "A" * INSERT_LEN
        alt = build_alt_one_hot(ref, insert, insert_start_in_window=50, gene_strand="+")
        assert alt[0, 50 : 50 + INSERT_LEN].sum() == INSERT_LEN
        assert alt[:, :50].sum() == 0
        assert alt[:, 50 + INSERT_LEN :].sum() == 0

    def test_channels_first_negative_strand(self):
        ref = np.zeros((4, 300), dtype=np.float32)
        insert = "A" * INSERT_LEN  # RC → "T" * INSERT_LEN
        alt = build_alt_one_hot(ref, insert, insert_start_in_window=50, gene_strand="-")
        assert alt[3, 50 : 50 + INSERT_LEN].sum() == INSERT_LEN
        assert alt[0, 50 : 50 + INSERT_LEN].sum() == 0

    def test_channels_last(self):
        ref = np.zeros((300, 4), dtype=np.float32)
        insert = "C" * INSERT_LEN
        alt = build_alt_one_hot(ref, insert, insert_start_in_window=10, gene_strand="+")
        assert alt[10 : 10 + INSERT_LEN, 1].sum() == INSERT_LEN

    def test_does_not_mutate_ref(self):
        ref = np.zeros((4, 300), dtype=np.float32)
        ref_copy = ref.copy()
        build_alt_one_hot(ref, "G" * INSERT_LEN, 0, "+")
        np.testing.assert_array_equal(ref, ref_copy)


# ── HOST_GENES ────────────────────────────────────────────────


class TestHostGenes:
    def test_count(self):
        assert len(HOST_GENES) == 22

    def test_systematic_names(self):
        for gene_id in HOST_GENES:
            assert gene_id.startswith("Y"), f"{gene_id} is not a systematic yeast gene ID"


# ── MPRAMarginalizedBenchmark ─────────────────────────────────

INFO = BenchmarkInfo(
    name="test", version="test", description="test", distribution_uri=""
)


class TestMPRAMarginalizedBenchmark:
    def test_has_genome_paths(self, mpra_distribution, tmp_path):
        fasta = tmp_path / "ref.fa"
        gtf = tmp_path / "ref.gtf"
        fasta.write_text(">chrI\nACGT\n")
        gtf.write_text("")
        bench = MPRAMarginalizedBenchmark(mpra_distribution, fasta, gtf, INFO)
        assert bench.fasta_path == fasta
        assert bench.gtf_path == gtf

    def test_inherits_evaluate(self, mpra_distribution, tmp_path):
        """MPRAMarginalizedBenchmark should inherit evaluate from parent."""
        fasta = tmp_path / "ref.fa"
        gtf = tmp_path / "ref.gtf"
        fasta.write_text(">chrI\nACGT\n")
        gtf.write_text("")
        bench = MPRAMarginalizedBenchmark(mpra_distribution, fasta, gtf, INFO)
        assert hasattr(bench, "evaluate")
        assert hasattr(bench, "plot")
        assert hasattr(bench, "save_results")


# ── Registry ──────────────────────────────────────────────────


class TestMarginalizedRegistry:
    def test_task_registered(self):
        from yeastbench.registry import TASKS
        assert "rafi_mpra_marginalized" in TASKS

    def test_factory_produces_marginalized_benchmark(self, mpra_distribution, tmp_path):
        from yeastbench.registry import TASKS
        fasta = tmp_path / "ref.fa"
        gtf = tmp_path / "ref.gtf"
        fasta.write_text(">chrI\nACGT\n")
        gtf.write_text("")
        task = TASKS["rafi_mpra_marginalized"](
            data_dir=mpra_distribution, fasta_path=fasta, gtf_path=gtf
        )
        assert isinstance(task, MPRAMarginalizedBenchmark)
        assert hasattr(task, "fasta_path")
        assert hasattr(task, "gtf_path")
