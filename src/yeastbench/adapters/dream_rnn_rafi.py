"""DREAM-RNN supervised baseline adapter for the Rafi/deBoer MPRA.

This is the **in-distribution supervised reference**, reported separately from
the zero-shot foundation models. Unlike the marginalized adapters (Shorkie,
Yorzoi), it does **not** insert each sequence at native host-gene loci. It runs
DREAM-RNN directly on the insert in its own reporter plasmid context — the
substrate the model was trained on — and returns the predicted reporter
expression. The benchmark then correlates that scalar against the measured
expression per stratum, identically to the foundation models' mean-logSED
scalar. See ``benchmarks/rafi_mpra_promoter.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from yeastbench.adapters.protocols import SequenceExpressionScorer
from yeastbench.models.dream_rnn import DreamRnn, DreamRnnExpressionPredictor


class DreamRnnRafiPredictor(SequenceExpressionScorer):
    def __init__(self, predictor: DreamRnnExpressionPredictor) -> None:
        self._predictor = predictor

    @classmethod
    def from_weights(
        cls,
        weights_path: str | Path,
        plasmid_path: str | Path,
        device: str = "cuda",
        batch_size: int = 1024,
    ) -> "DreamRnnRafiPredictor":
        import torch

        model = DreamRnn()
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)  # strict: confirms the architecture match
        predictor = DreamRnnExpressionPredictor(
            model, plasmid_path, device=device, batch_size=batch_size
        )
        return cls(predictor)

    def predict_expression_scores(self, seqs: Sequence[str]) -> np.ndarray:
        return np.asarray(self._predictor.predict(list(seqs)), dtype=float)
