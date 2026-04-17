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
class MarginalizedSequenceExpressionPredictor(Protocol):
    """Predict the marginalized effect of each input sequence across native
    host-gene contexts, inserted **upstream** of the host-gene TSS
    (promoter-MPRA flavour, e.g. Rafi/deBoer).  Returns mean logSED across
    a committed list of host genes."""
    def predict_marginalized_expressions(self, seqs: Sequence[str]) -> np.ndarray: ...


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
