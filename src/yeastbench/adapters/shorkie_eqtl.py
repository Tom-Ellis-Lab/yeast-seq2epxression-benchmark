from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from yeastbench.adapters.protocols import Variant, VariantEffectScorer

if TYPE_CHECKING:
    from yeastbench.models.shorkie import Shorkie


SHORKIE_1011_RNA_SEQ_TRACK_IDS: list[int] = []


class ShorkieVariantScorer(VariantEffectScorer):
    def __init__(
        self,
        model: "Shorkie",
        fasta_path: str,
        gtf_path: str,
        track_subset: list[int],
        batch_size: int = 8,
    ) -> None:
        self.model = model
        self.fasta_path = fasta_path
        self.gtf_path = gtf_path
        self.track_subset = track_subset
        self.batch_size = batch_size

    def score_variants(self, variants: Sequence[Variant]) -> np.ndarray:
        raise NotImplementedError
