from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Variant:
    chrom: str
    pos: int  # 1-based, R64-1-1
    ref: str
    alt: str
    gene_id: str  # Ensembl ID


@runtime_checkable
class VariantEffectScorer(Protocol):
    def score_variants(self, variants: Sequence[Variant]) -> np.ndarray: ...


@runtime_checkable
class SequenceExpressionPredictor(Protocol):
    """Predict the expression of each input sequence in a fixed construct
    context (e.g., embedded in the DREAM MPRA plasmid scaffold)."""

    def predict_expressions(self, seqs: Sequence[str]) -> np.ndarray: ...


@runtime_checkable
class CassetteExpressionPredictor(Protocol):
    """Predict expression of a reporter in one constant cassette that is
    integrated at varying genomic loci (Wu et al. position-effect task).
    ``loci`` is a sequence of resolved ``WuLocus`` records; returns one
    scalar per locus, aligned to input order."""

    def predict_expressions(self, loci: Sequence) -> np.ndarray: ...


@runtime_checkable
class MarginalizedSequenceExpressionPredictor(Protocol):
    """Predict the marginalized effect of each input sequence across native
    host-gene contexts, inserted **upstream** of the host-gene TSS
    (promoter-MPRA flavour, e.g. Rafi/deBoer).  Returns mean logSED across
    a committed list of host genes."""

    def predict_marginalized_expressions(self, seqs: Sequence[str]) -> np.ndarray: ...


@runtime_checkable
class CoverageTrackPredictor(Protocol):
    """Predict an RNA-seq-like coverage profile for a single construct
    on a given strand. Used by the Brooks SCRaMBLE benchmark
    (sequence-in / coverage-out, not variant-effect).

    **Output contract:** ``predict_coverage`` must return a 1D numpy
    array of length ``seq_len - 2 * crop_bp_each_side``, in **raw
    per-base predicted-count units** — adapters are responsible for
    inverting any model-specific training transform (e.g. Borzoi/Yorzoi's
    `x^0.75 + sqrt`-squash + 4 bp binning) and unbinning back to per-base
    before returning, so the benchmark can compute LFC / Pearson / JSD
    directly against raw per-base pileups in matching units.

    Adapters expose ``seq_len`` and ``crop_bp_each_side`` so the
    benchmark can slice the true per-base coverage to the predicted
    central region and map CDS coordinates correctly."""

    seq_len: int
    crop_bp_each_side: int

    def predict_coverage(self, construct_seq: str, strand: str) -> np.ndarray: ...


@runtime_checkable
class TerminatorMarginalizedExpressionPredictor(Protocol):
    """Predict the marginalized effect of each input sequence across native
    host-gene contexts, inserted **downstream** of the host-gene stop codon
    with a non-terminating filler (terminator-MPRA flavour, e.g. Shalem).
    Returns mean logSED across a committed list of host genes.

    Distinct from ``MarginalizedSequenceExpressionPredictor`` because the
    insertion *site* and the surrounding scaffold are semantically
    different (promoter-region insertion vs terminator-region insertion),
    so the two protocols cleanly disambiguate in the registry."""

    def predict_terminator_marginalized(self, seqs: Sequence[str]) -> np.ndarray: ...
