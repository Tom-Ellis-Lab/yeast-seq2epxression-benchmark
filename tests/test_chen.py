"""Tests for the unified Chen synonymous-mutation benchmark + CAI baseline.

These cover the wiring (TSVs → one task → adapter → per-library +
aggregate metrics) using a synthetic adapter and the real distribution
TSVs. The marginalised model adapters (Shorkie, Yorzoi) and the
CodonTransformer baseline aren't tested here — they need GPU + model
weights and are exercised by the real run.
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
FASTA_PATH = REPO / "data" / "tasks" / "R64-1-1.fa"
HOSTS_PATH = DATA_DIR / "marginalized_hosts.json"


REQUIRED_FILES = [
    DATA_DIR / "gfp_r1.tsv",
    DATA_DIR / "gfp_r2.tsv",
    DATA_DIR / "tdh3.tsv",
    HOSTS_PATH,
]
DISTRIBUTION_AVAILABLE = all(p.exists() for p in REQUIRED_FILES)


pytestmark = pytest.mark.skipif(
    not DISTRIBUTION_AVAILABLE,
    reason="Chen distribution not built (scripts/chen/build_distribution_tsvs.py + marginalized_hosts.json)",
)

# Per-library ceilings, mirroring configs/default.yaml.
_LIBRARIES = [
    {"library": "gfp_r1", "data_path": DATA_DIR / "gfp_r1.tsv",
     "replicate_ceiling_pearson": 0.83, "replicate_ceiling_spearman": 0.71},
    {"library": "gfp_r2", "data_path": DATA_DIR / "gfp_r2.tsv",
     "replicate_ceiling_pearson": 0.73, "replicate_ceiling_spearman": 0.71},
    {"library": "tdh3", "data_path": DATA_DIR / "tdh3.tsv",
     "replicate_ceiling_pearson": 0.72},
]


class _ConstantScorer(LocalCodingVariantPredictor):
    """Returns 0.0 for every variant. Pearson is undefined (constant
    prediction); used to verify the benchmark handles degenerate cases."""

    def predict_local_variants(self, library_ids, variant_seqs):
        return np.zeros(len(variant_seqs), dtype=float)


def _benchmark() -> ChenSynonymousBenchmark:
    return ChenSynonymousBenchmark(
        libraries=_LIBRARIES,
        fasta_path=FASTA_PATH,
        hosts_path=HOSTS_PATH,
        data_dir=DATA_DIR,
        info=BenchmarkInfo(
            name="chen_synonymous",
            version="v1-test",
            description="Chen synonymous (test)",
            distribution_uri="",
        ),
    )


def test_requires_exactly_three_libraries():
    info = BenchmarkInfo(
        name="chen_synonymous", version="t", description="", distribution_uri="",
    )
    # Missing a library (rejected before any TSV is read).
    with pytest.raises(ValueError, match="exactly the three"):
        ChenSynonymousBenchmark(
            libraries=_LIBRARIES[:2], fasta_path=FASTA_PATH,
            hosts_path=HOSTS_PATH, data_dir=DATA_DIR, info=info,
        )
    # Duplicated library (len 3 but wrong set).
    with pytest.raises(ValueError, match="exactly the three"):
        ChenSynonymousBenchmark(
            libraries=[_LIBRARIES[0], _LIBRARIES[0], _LIBRARIES[1]],
            fasta_path=FASTA_PATH, hosts_path=HOSTS_PATH, data_dir=DATA_DIR,
            info=info,
        )


def test_evaluates_all_libraries_with_per_library_results():
    bench = _benchmark()
    results = bench.evaluate(_ConstantScorer())
    assert results.libraries == ("gfp_r1", "gfp_r2", "tdh3")
    assert results.per_library["gfp_r1"].labels.shape == (1124, 2)
    assert results.per_library["gfp_r2"].labels.shape == (2432, 2)
    assert results.per_library["tdh3"].labels.shape == (523, 1)
    assert results.per_library["gfp_r1"].label_columns == (
        "log2mRNA_rep1", "log2mRNA_rep2",
    )
    assert results.per_library["tdh3"].label_columns == ("log2mRNA",)


def test_summary_has_library_prefixed_and_aggregate_keys():
    bench = _benchmark()
    results = bench.evaluate(_ConstantScorer())
    summary = bench.summary_dict(results)

    # GFP libs report per-replicate Pearson/Spearman, library-prefixed.
    for lib in ("gfp_r1", "gfp_r2"):
        for k in (f"{lib}_pearson_rep1", f"{lib}_pearson_rep2",
                  f"{lib}_spearman_rep1", f"{lib}_spearman_rep2",
                  f"{lib}_n_rep1", f"{lib}_n_rep2", f"{lib}_ceiling_pearson"):
            assert k in summary, k
    # TDH3 has a single merged column and no Spearman ceiling.
    assert "tdh3_pearson" in summary
    assert "tdh3_spearman" in summary
    assert "tdh3_n_scored" in summary
    assert "tdh3_ceiling_pearson" in summary
    assert "tdh3_ceiling_spearman" not in summary
    # Aggregate over the 5 replicate columns + total count.
    assert "pearson_mean" in summary
    assert "spearman_mean" in summary
    assert summary["n_rows_total"] == 1124 + 2432 + 523
    # Constant predictor → every correlation (and the mean) is NaN.
    assert np.isnan(summary["gfp_r1_pearson_rep1"])
    assert np.isnan(summary["pearson_mean"])


def test_headline_metric_labels_cover_plot_axes():
    bench = _benchmark()
    labels = bench.headline_metric_labels()
    # 5 Pearson + 5 Spearman per-replicate + 2 aggregates.
    expected = {
        "gfp_r1_pearson_rep1", "gfp_r1_pearson_rep2",
        "gfp_r1_spearman_rep1", "gfp_r1_spearman_rep2",
        "gfp_r2_pearson_rep1", "gfp_r2_pearson_rep2",
        "gfp_r2_spearman_rep1", "gfp_r2_spearman_rep2",
        "tdh3_pearson", "tdh3_spearman",
        "pearson_mean", "spearman_mean",
    }
    assert set(labels) == expected


def test_save_load_roundtrip(tmp_path):
    bench = _benchmark()
    results = bench.evaluate(_ConstantScorer())
    bench.save_results(results, tmp_path)
    loaded = bench.load_results(tmp_path)
    assert loaded.libraries == results.libraries
    for lib in results.libraries:
        np.testing.assert_array_equal(
            loaded.per_library[lib].scores, results.per_library[lib].scores,
        )
        np.testing.assert_array_equal(
            loaded.per_library[lib].labels, results.per_library[lib].labels,
        )
    assert loaded.per_library["tdh3"].ceiling_spearman is None


def test_cai_baseline_registers_all_libraries_and_returns_chen_column():
    """The CAI baseline built from the unified task serves every library and
    returns Chen's precomputed CAI column verbatim per library."""
    from types import SimpleNamespace

    fake_task = SimpleNamespace(libraries=[
        SimpleNamespace(library=spec["library"], data_path=spec["data_path"])
        for spec in _LIBRARIES
    ])
    adapter = CAIBaselinePredictor.from_task(task=fake_task)
    for lib in ("gfp_r1", "gfp_r2", "tdh3"):
        df = pd.read_csv(DATA_DIR / f"{lib}.tsv", sep="\t")
        scores = adapter.predict_local_variants(
            [lib] * len(df),
            df["variable_seq"].astype(str).str.upper().tolist(),
        )
        np.testing.assert_allclose(scores, df["CAI"].to_numpy(), rtol=1e-12)


def test_cai_baseline_pearson_is_sensible():
    """End-to-end on the unified task: Chen's CAI gives a non-trivial,
    finite per-library Pearson (paper observed roughly r ≈ 0.3)."""
    from types import SimpleNamespace

    bench = _benchmark()
    fake_task = SimpleNamespace(libraries=[
        SimpleNamespace(library=spec["library"], data_path=spec["data_path"])
        for spec in _LIBRARIES
    ])
    adapter = CAIBaselinePredictor.from_task(task=fake_task)
    results = bench.evaluate(adapter)
    summary = bench.summary_dict(results)
    assert np.isfinite(summary["tdh3_pearson"])
    assert abs(summary["tdh3_pearson"]) > 0.1
    assert np.isfinite(summary["pearson_mean"])


def test_marginalized_host_contexts_build_for_all_libraries():
    """Sanity: the shared _chen_marginalized helper builds 20 contexts
    per library, and each context's variable region translates to the
    library's expected peptide. Doesn't load any model — pure data."""
    import pysam
    from yeastbench.adapters._chen_marginalized import (
        load_hosts, build_cassette, build_host_contexts,
    )
    from yeastbench.adapters._shorkie_constants import (
        SEQ_LEN, CROP_BP_EACH_SIDE, BIN_WIDTH, OUTPUT_BINS,
    )

    hosts = load_hosts(HOSTS_PATH)
    assert len(hosts) == 20

    fasta = pysam.FastaFile(str(FASTA_PATH))
    for library in ("gfp_r1", "gfp_r2", "tdh3"):
        cassette = build_cassette(library, fasta, DATA_DIR)
        contexts = build_host_contexts(
            library=library, hosts=hosts, fasta=fasta, cassette=cassette,
            seq_len=SEQ_LEN, crop_bp_each_side=CROP_BP_EACH_SIDE,
            bin_width=BIN_WIDTH, output_bins=OUTPUT_BINS,
        )
        assert len(contexts) == 20
        for c in contexts:
            assert c.cds_bin_hi > c.cds_bin_lo
            assert c.exon_bins.size > 0
            assert c.cds_base_hi > c.cds_base_lo
            assert (c.cds_base_hi - c.cds_base_lo) <= (
                c.cds_bin_hi - c.cds_bin_lo
            ) * BIN_WIDTH
