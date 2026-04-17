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
    host-gene contexts (mean logSED over 22 host genes at a fixed
    upstream insertion offset)."""
    def predict_marginalized_expressions(self, seqs: Sequence[str]) -> np.ndarray: ...
