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


@dataclass(frozen=True)
class Region:
    chrom: str
    start: int  # 1-based inclusive
    end: int  # 1-based inclusive
    strand: str  # '+' or '-'


@runtime_checkable
class VariantEffectScorer(Protocol):
    def score_variants(self, variants: Sequence[Variant]) -> np.ndarray: ...


@runtime_checkable
class TrackPredictor(Protocol):
    def predict_tracks(self, regions: Sequence[Region]) -> list[np.ndarray]: ...


@runtime_checkable
class SequenceExpressionPredictor(Protocol):
    def predict_expressions(self, seqs: Sequence[str]) -> np.ndarray: ...
