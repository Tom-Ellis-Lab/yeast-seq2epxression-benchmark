from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from yeastbench.adapters.protocols import Variant, VariantEffectScorer


class YorzoiVariantScorer(VariantEffectScorer):
    def __init__(
        self,
        model: Any,
        fasta_path: str,
        gtf_path: str,
        batch_size: int = 16,
    ) -> None:
        self.model = model
        self.fasta_path = fasta_path
        self.gtf_path = gtf_path
        self.batch_size = batch_size

    def score_variants(self, variants: Sequence[Variant]) -> np.ndarray:
        raise NotImplementedError
