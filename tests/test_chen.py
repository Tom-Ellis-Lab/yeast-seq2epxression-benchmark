"""Tests for the Chen synonymous-mutation benchmark + CAI baseline.

These cover the wiring (TSV → benchmark → adapter → per-replicate
Pearson) using a synthetic adapter and the real distribution TSVs.
The model adapters (Shorkie, Yorzoi) and the CodonTransformer
baseline aren't tested here — they need GPU + model weights and are
exercised by the real run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yeastbench.adapters.baselines.cai import CAIBaselinePredictor
from yeastbench.adapters.protocols import LocalCodingVariantPredictor
from yeastbench.benchmarks.base import BenchmarkInfo
from yeastbench.benchmarks.chen import ChenSynonymousBenchmark

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "tasks" / "chen_synonymous"
FASTA_PATH = DATA_DIR / "construct_chrII_gfp.fa"
GTF_PATH = DATA_DIR / "construct_gfp.gtf"
LOCI_PATH = DATA_DIR / "library_loci.json"


REQUIRED_FILES = [
    DATA_DIR / "gfp_r1.tsv",
    DATA_DIR / "gfp_r2.tsv",
    DATA_DIR / "tdh3.tsv",
]
DISTRIBUTION_AVAILABLE = all(p.exists() for p in REQUIRED_FILES)


pytestmark = pytest.mark.skipif(
    not DISTRIBUTION_AVAILABLE,
    reason="Chen distribution TSVs not built (scripts/chen/build_distribution_tsvs.py)",
)


class _ConstantScorer(LocalCodingVariantPredictor):
    """Returns 0.0 for every variant. Pearson is undefined (constant
    prediction); used to verify the benchmark handles degenerate cases."""

    def predict_local_variants(self, library_ids, variant_seqs):
        return np.zeros(len(variant_seqs), dtype=float)


def _benchmark(library: str, fasta=FASTA_PATH, gtf=GTF_PATH) -> ChenSynonymousBenchmark:
    tsv = DATA_DIR / f"{library}.tsv"
    ceiling = {"gfp_r1": 0.83, "gfp_r2": 0.73, "tdh3": 0.72}[library]
    return ChenSynonymousBenchmark(
        library=library,
        data_path=tsv,
        fasta_path=fasta,
        gtf_path=gtf,
        library_loci_path=LOCI_PATH,
        replicate_ceiling_pearson=ceiling,
        info=BenchmarkInfo(
            name=f"chen_{library}",
            version="v1-test",
            description=f"Chen {library} (test)",
            distribution_uri="",
        ),
    )


def test_two_replicate_library_reports_per_replicate_pearson():
    bench = _benchmark("gfp_r1")
    results = bench.evaluate(_ConstantScorer())
    summary = bench.summary_dict(results)
    # GFP r1 has two replicate columns → 2 Pearsons + 2 Spearmans + 2 n's
    assert "pearson_rep1" in summary
    assert "pearson_rep2" in summary
    assert "spearman_rep1" in summary
    assert "spearman_rep2" in summary
    assert results.label_columns == ("log2mRNA_rep1", "log2mRNA_rep2")
    assert results.labels.shape == (1124, 2)
    # Pearson of a constant predictor is NaN
    assert np.isnan(summary["pearson_rep1"])
    assert np.isnan(summary["pearson_rep2"])


def test_single_replicate_library_reports_one_pearson():
    bench = _benchmark(
        "tdh3",
        fasta=DATA_DIR / "construct_chrII_tdh3.fa",
        gtf=DATA_DIR / "construct_tdh3.gtf",
    )
    results = bench.evaluate(_ConstantScorer())
    summary = bench.summary_dict(results)
    assert "pearson" in summary
    assert "spearman" in summary
    assert "pearson_rep1" not in summary
    assert results.label_columns == ("log2mRNA",)
    assert results.labels.shape == (523, 1)


def test_compare_task_name_is_per_library():
    """Each Chen library is its own compare group (see ChenSynonymousBenchmark
    docstring); v1 ships separate compare panels per library."""
    for lib in ["gfp_r1", "gfp_r2", "tdh3"]:
        fasta = DATA_DIR / (
            "construct_chrII_tdh3.fa" if lib == "tdh3" else "construct_chrII_gfp.fa"
        )
        gtf = DATA_DIR / ("construct_tdh3.gtf" if lib == "tdh3" else "construct_gfp.gtf")
        b = _benchmark(lib, fasta=fasta, gtf=gtf)
        assert b.compare_task_name == f"chen_{lib}"


def test_cai_baseline_matches_chen_column_on_gfp_r1():
    """CAI baseline must return Chen's precomputed CAI column verbatim,
    so its Pearson against itself is exactly 1.0."""
    from types import SimpleNamespace

    df = pd.read_csv(DATA_DIR / "gfp_r1.tsv", sep="\t")
    fake_task = SimpleNamespace(
        library="gfp_r1", data_path=DATA_DIR / "gfp_r1.tsv",
    )
    adapter = CAIBaselinePredictor.from_task(task=fake_task)
    scores = adapter.predict_local_variants(
        ["gfp_r1"] * len(df),
        df["variable_seq"].astype(str).str.upper().tolist(),
    )
    np.testing.assert_allclose(scores, df["CAI"].to_numpy(), rtol=1e-12)


def test_cai_baseline_pearson_is_sensible():
    """End-to-end: Pearson of Chen's CAI vs measured log2mRNA on the
    full TDH3 library. The paper observed roughly ρ ≈ 0.3 for CAI on
    these libraries — we just check non-trivial (|r| > 0.1) and that
    nothing NaNs out."""
    from types import SimpleNamespace

    bench = _benchmark(
        "tdh3",
        fasta=DATA_DIR / "construct_chrII_tdh3.fa",
        gtf=DATA_DIR / "construct_tdh3.gtf",
    )
    fake_task = SimpleNamespace(
        library="tdh3", data_path=DATA_DIR / "tdh3.tsv",
    )
    adapter = CAIBaselinePredictor.from_task(task=fake_task)
    results = bench.evaluate(adapter)
    summary = bench.summary_dict(results)
    assert np.isfinite(summary["pearson"])
    assert abs(summary["pearson"]) > 0.1
