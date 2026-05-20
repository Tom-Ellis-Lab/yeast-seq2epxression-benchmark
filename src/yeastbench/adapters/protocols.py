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
    """Predict an RNA-seq-like coverage profile for a batch of constructs.
    Used by the Brooks SCRaMBLE benchmark (sequence-in / coverage-out,
    not variant-effect).

    **Output contract:** ``predict_coverage_batch`` returns a 2D numpy
    array of shape ``(B, seq_len - 2 * crop_bp_each_side)``, in **raw
    per-base predicted-count units** — adapters are responsible for
    inverting any model-specific training transform (e.g. Borzoi/Yorzoi's
    `x^0.75 + sqrt`-squash + 4 bp binning) and unbinning back to per-base
    before returning, so the benchmark can compute LFC / Pearson / JSD
    directly against raw per-base pileups in matching units.

    Adapters expose ``seq_len`` and ``crop_bp_each_side`` so the
    benchmark can slice the true per-base coverage to the predicted
    central region and map CDS coordinates correctly. Adapters also
    expose ``batch_size`` — the maximum batch the benchmark may pass
    in one call (the benchmark chunks larger sets internally).

    **Optional per-sample ``strains``.** For track-based models (e.g.
    Yorzoi), the benchmark may pass per-sample strain identifiers so
    the adapter routes each prediction to the matching experimental
    tracks (Brooks SCRaMBLE strain S → use S's Nanopore tracks;
    native construct → use JS94's deep-WT tracks). Adapters without
    per-condition track selection may ignore them.

    **``varies_by_strain``** (class-level attribute, default ``True``).
    Set to ``False`` for adapters whose predictions are identical for
    different strain values (e.g. Shorkie, which has no Brooks-
    specific output tracks). The Brooks benchmark uses this to skip
    redundant native predictions when computing per-replicate LFCs —
    if ``False``, the native is predicted once and broadcast across
    the JS94 replicate axis."""

    seq_len: int
    crop_bp_each_side: int
    batch_size: int
    varies_by_strain: bool

    def predict_coverage_batch(
        self,
        seqs: Sequence[str],
        strands: Sequence[str],
        strains: Sequence[str | None] | None = None,
    ) -> np.ndarray: ...


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
