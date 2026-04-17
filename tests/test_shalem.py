"""Tests for Shalem MPRA marginalized benchmark infrastructure."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from yeastbench.adapters._shalem_scaffold import (
    CYC1_CDS_TAIL_LEN,
    FILLER_LEN,
    INSERT_LEN,
    MUTANT_UTR_LEN,
    REPLACE_LEN,
    HostGeneSpec,
    assemble_replacement,
    load_host_genes,
    reverse_complement,
    splice_into_one_hot,
)
from yeastbench.adapters.protocols import TerminatorMarginalizedExpressionPredictor
from yeastbench.benchmarks.base import BenchmarkInfo
from yeastbench.benchmarks.shalem import ShalemMPRAMarginalizedBenchmark
from yeastbench.registry import TASKS


INFO = BenchmarkInfo(name="test_shalem", version="test", description="t", distribution_uri="")

POLYT5 = "GGGGACCAGGTGCCGTAAG"   # 19 bp SexAI flank
POLYA3 = "GCGATCCTAGGGCGATCA"    # 18 bp AvrII flank


# ── Constants ─────────────────────────────────────────────────


class TestConstants:
    def test_lengths(self):
        assert INSERT_LEN == 150
        assert CYC1_CDS_TAIL_LEN == 100
        assert MUTANT_UTR_LEN == 200
        assert FILLER_LEN == 300
        assert REPLACE_LEN == 450


# ── reverse_complement ────────────────────────────────────────


class TestReverseComplement:
    def test_basic(self):
        assert reverse_complement("ACGT") == "ACGT"
        assert reverse_complement("AAAA") == "TTTT"
        assert reverse_complement("GCTA") == "TAGC"

    def test_preserves_N(self):
        assert reverse_complement("NACN") == "NGTN"


# ── assemble_replacement ──────────────────────────────────────


class TestAssembleReplacement:
    def test_plus_strand(self):
        insert = "A" * INSERT_LEN
        filler = "C" * FILLER_LEN
        rep = assemble_replacement(insert, filler, "+")
        assert len(rep) == REPLACE_LEN
        # Insert comes first, filler second
        assert rep[:INSERT_LEN] == "A" * INSERT_LEN
        assert rep[INSERT_LEN:] == "C" * FILLER_LEN

    def test_minus_strand(self):
        insert = "A" * INSERT_LEN
        filler = "C" * FILLER_LEN
        rep = assemble_replacement(insert, filler, "-")
        assert len(rep) == REPLACE_LEN
        # RC'd: filler (C's) becomes G's on 5' end, insert (A's) becomes T's on 3' end
        assert rep[:FILLER_LEN] == "G" * FILLER_LEN
        assert rep[FILLER_LEN:] == "T" * INSERT_LEN

    def test_asserts_lengths(self):
        with pytest.raises(AssertionError):
            assemble_replacement("A" * 100, "C" * FILLER_LEN, "+")


# ── splice_into_one_hot ───────────────────────────────────────


class TestSpliceIntoOneHot:
    def test_channels_first(self):
        ref = np.zeros((4, 2000), dtype=np.float32)
        replacement = "A" * REPLACE_LEN
        alt = splice_into_one_hot(ref, replacement, replace_start_in_window=100)
        # 'A' is channel 0
        assert alt[0, 100 : 100 + REPLACE_LEN].sum() == REPLACE_LEN
        assert alt[:, :100].sum() == 0
        assert alt[:, 100 + REPLACE_LEN:].sum() == 0

    def test_channels_last(self):
        ref = np.zeros((2000, 4), dtype=np.float32)
        replacement = "C" * REPLACE_LEN
        alt = splice_into_one_hot(ref, replacement, replace_start_in_window=50)
        # 'C' is channel 1
        assert alt[50 : 50 + REPLACE_LEN, 1].sum() == REPLACE_LEN

    def test_does_not_mutate_ref(self):
        ref = np.zeros((4, 2000), dtype=np.float32)
        ref_copy = ref.copy()
        splice_into_one_hot(ref, "G" * REPLACE_LEN, 0)
        np.testing.assert_array_equal(ref, ref_copy)


# ── load_host_genes ───────────────────────────────────────────


class TestLoadHostGenes:
    def test_loads_committed_artifact(self):
        genes = load_host_genes()
        assert len(genes) == 22
        n_pos = sum(1 for g in genes if g.strand == "+")
        n_neg = sum(1 for g in genes if g.strand == "-")
        assert n_pos == 10
        assert n_neg == 12
        for g in genes:
            assert g.gene_id.startswith("Y"), g.gene_id

    def test_custom_path(self, tmp_path):
        payload = {
            "genes": [
                {"gene_id": "YAL001C", "strand": "-", "chrom": "I", "start_1based": 1, "end_1based": 100, "cds_len": 100, "median_tpm": 5.0},
                {"gene_id": "YBL001C", "strand": "+", "chrom": "II", "start_1based": 1, "end_1based": 100, "cds_len": 100, "median_tpm": 10.0},
            ],
        }
        p = tmp_path / "custom.json"
        p.write_text(json.dumps(payload))
        genes = load_host_genes(p)
        assert len(genes) == 2
        assert genes[0] == HostGeneSpec(gene_id="YAL001C", strand="-")


# ── Mock adapter for the benchmark ────────────────────────────


class _MockAdapter:
    def __init__(self, labels: np.ndarray, noise_std: float = 0.01, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.predictions = labels + rng.normal(0, noise_std, size=len(labels))

    def predict_terminator_marginalized(self, seqs: Sequence[str]) -> np.ndarray:
        return self.predictions


assert isinstance(_MockAdapter(np.array([0.0])), TerminatorMarginalizedExpressionPredictor)


# ── ShalemMPRAMarginalizedBenchmark ───────────────────────────


@pytest.fixture
def shalem_data(tmp_path: Path) -> Path:
    """Synthetic mini Shalem TSV with 20 rows (5 with NA Expression)."""
    rows = []
    for i in range(20):
        oligo = POLYT5 + ("A" * 102) + ("C" * 11) + POLYA3  # 19 + 102 + 11 + 18 = 150
        assert len(oligo) == 150
        expr = float(i) if i < 15 else ""  # last 5 NA
        rows.append([i + 1, i + 1, expr, "Context|Name=testctx", oligo])
    df = pd.DataFrame(rows, columns=["Design ID", "Lib ID", "Expression", "Description", "Oligo Sequence"])
    p = tmp_path / "mini.tsv"
    df.to_csv(p, sep="\t", index=False)
    return p


class TestShalemBenchmark:
    def test_init_drops_no_rows(self, shalem_data):
        bench = ShalemMPRAMarginalizedBenchmark(
            shalem_data, fasta_path="/dev/null", gtf_path="/dev/null", info=INFO,
        )
        assert len(bench.sequences) == 20
        assert len(bench.labels) == 20
        # 5 are NaN
        assert np.isnan(bench.labels).sum() == 5

    def test_fasta_gtf_paths(self, shalem_data, tmp_path):
        fa = tmp_path / "fa.fa"
        gt = tmp_path / "gt.gtf"
        fa.write_text(">X\nACGT\n")
        gt.write_text("")
        bench = ShalemMPRAMarginalizedBenchmark(shalem_data, fa, gt, INFO)
        assert bench.fasta_path == fa
        assert bench.gtf_path == gt

    def test_evaluate_drops_na(self, shalem_data):
        bench = ShalemMPRAMarginalizedBenchmark(
            shalem_data, fasta_path="/dev/null", gtf_path="/dev/null", info=INFO,
        )
        adapter = _MockAdapter(bench.labels)  # NaN-aware: adapter mock returns labels directly (NaN ok)
        results = bench.evaluate(adapter)
        # 15 non-NaN rows contribute to overall
        assert results.overall.n == 15
        # Near-perfect correlation
        assert results.overall.pearson_r > 0.99

    def test_summary_dict(self, shalem_data):
        bench = ShalemMPRAMarginalizedBenchmark(
            shalem_data, fasta_path="/dev/null", gtf_path="/dev/null", info=INFO,
        )
        adapter = _MockAdapter(bench.labels)
        results = bench.evaluate(adapter)
        s = bench.summary_dict(results)
        assert s["n_rows_total"] == 20
        assert s["n_rows_scored"] == 15
        assert "overall_pearson_r" in s
        assert "overall_spearman_rho" in s

    def test_save_load_roundtrip(self, shalem_data, tmp_path):
        bench = ShalemMPRAMarginalizedBenchmark(
            shalem_data, fasta_path="/dev/null", gtf_path="/dev/null", info=INFO,
        )
        adapter = _MockAdapter(bench.labels)
        results = bench.evaluate(adapter)
        out = tmp_path / "out"
        bench.save_results(results, out)
        loaded = bench.load_results(out)
        np.testing.assert_array_almost_equal(loaded.scores, results.scores)
        np.testing.assert_array_almost_equal(
            np.nan_to_num(loaded.labels, nan=-9999),
            np.nan_to_num(results.labels, nan=-9999),
        )
        assert loaded.overall.n == results.overall.n

    def test_plot(self, shalem_data, tmp_path):
        bench = ShalemMPRAMarginalizedBenchmark(
            shalem_data, fasta_path="/dev/null", gtf_path="/dev/null", info=INFO,
        )
        adapter = _MockAdapter(bench.labels)
        results = bench.evaluate(adapter)
        plot_dir = tmp_path / "plots"
        bench.plot(results, plot_dir)
        assert (plot_dir / "scatter.png").exists()

    def test_headline(self, shalem_data):
        bench = ShalemMPRAMarginalizedBenchmark(
            shalem_data, fasta_path="/dev/null", gtf_path="/dev/null", info=INFO,
        )
        adapter = _MockAdapter(bench.labels)
        results = bench.evaluate(adapter)
        h = bench.headline(results)
        assert "Pearson" in h and "Spearman" in h


# ── Registry ──────────────────────────────────────────────────


class TestShalemRegistry:
    def test_task_registered(self):
        assert "shalem_mpra_marginalized" in TASKS

    def test_factory_builds_benchmark(self, shalem_data, tmp_path):
        fa = tmp_path / "fa.fa"
        gt = tmp_path / "gt.gtf"
        fa.write_text(">X\nACGT\n")
        gt.write_text("")
        task = TASKS["shalem_mpra_marginalized"](
            data_path=shalem_data, fasta_path=fa, gtf_path=gt,
        )
        assert isinstance(task, ShalemMPRAMarginalizedBenchmark)
        assert task.adapter_protocol is TerminatorMarginalizedExpressionPredictor
