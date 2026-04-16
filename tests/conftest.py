"""Shared fixtures for yeastbench tests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from yeastbench.adapters.protocols import Variant, VariantEffectScorer


# ── Mock adapter ──────────────────────────────────────────────


class DeterministicScorer:
    """A fake VariantEffectScorer that returns deterministic scores.

    Positive variants (even indices in the input list) get higher
    absolute scores than negatives (odd indices), so AUROC > 0.5.
    """

    def score_variants(self, variants: Sequence[Variant]) -> np.ndarray:
        rng = np.random.default_rng(42)
        scores = np.empty(len(variants), dtype=float)
        for i, v in enumerate(variants):
            # Use a seed derived from the variant to be deterministic
            local_rng = np.random.default_rng(hash((v.chrom, v.pos, v.ref, v.alt)) % 2**31)
            scores[i] = local_rng.normal(0, 1)
        return scores


class PerfectScorer:
    """Returns |score|=1.0 for positives (even indices) and 0.0 for negatives."""

    def score_variants(self, variants: Sequence[Variant]) -> np.ndarray:
        scores = np.zeros(len(variants), dtype=float)
        for i in range(0, len(variants), 2):
            scores[i] = 1.0
        return scores


assert isinstance(DeterministicScorer(), VariantEffectScorer)
assert isinstance(PerfectScorer(), VariantEffectScorer)


# ── Synthetic distribution ────────────────────────────────────

NEGSET_COLUMNS = [
    "pair_id",
    "pos_chrom",
    "pos_pos",
    "pos_ref",
    "pos_alt",
    "pos_gene",
    "pos_gene_strand",
    "pos_distance_to_tss",
    "neg_chrom",
    "neg_pos",
    "neg_ref",
    "neg_alt",
    "neg_gene",
    "neg_gene_strand",
    "neg_distance_to_tss",
]


def _make_negset(n_pairs: int, seed: int) -> pd.DataFrame:
    """Generate a synthetic negset TSV DataFrame."""
    rng = np.random.default_rng(seed)
    chroms = [str(rng.integers(1, 17)) for _ in range(n_pairs * 2)]
    bases = list("ACGT")
    rows = []
    for i in range(n_pairs):
        pos_pos = int(rng.integers(1000, 200_000))
        neg_pos = int(rng.integers(1000, 200_000))
        ref, alt = rng.choice(bases, 2, replace=False)
        neg_ref, neg_alt = rng.choice(bases, 2, replace=False)
        rows.append({
            "pair_id": i,
            "pos_chrom": chroms[2 * i],
            "pos_pos": pos_pos,
            "pos_ref": ref,
            "pos_alt": alt,
            "pos_gene": f"YAL{i:03d}W",
            "pos_gene_strand": rng.choice(["+", "-"]),
            "pos_distance_to_tss": int(rng.integers(50, 25000)),
            "neg_chrom": chroms[2 * i + 1],
            "neg_pos": neg_pos,
            "neg_ref": neg_ref,
            "neg_alt": neg_alt,
            "neg_gene": f"YBR{i:03d}C",
            "neg_gene_strand": rng.choice(["+", "-"]),
            "neg_distance_to_tss": int(rng.integers(50, 25000)),
        })
    return pd.DataFrame(rows, columns=NEGSET_COLUMNS)


@pytest.fixture
def synthetic_distribution(tmp_path: Path) -> Path:
    """Create a temporary distribution directory with 3 negset TSV files."""
    dist_dir = tmp_path / "caudal_eqtl_v1"
    dist_dir.mkdir()
    ref_dir = dist_dir / "reference"
    ref_dir.mkdir()
    # Dummy reference files (not used by evaluate, only by adapters)
    (ref_dir / "R64-1-1.fa").write_text(">chrI\nACGT\n")
    (ref_dir / "R64-1-1.115.gtf").write_text("")

    for i in range(1, 4):
        df = _make_negset(n_pairs=50, seed=i * 100)
        df.to_csv(dist_dir / f"negset_{i}.tsv", sep="\t", index=False)

    return dist_dir
