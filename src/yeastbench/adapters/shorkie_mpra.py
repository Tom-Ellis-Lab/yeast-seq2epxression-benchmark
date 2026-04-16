"""Shorkie fixed-context adapter for the Rafi / deBoer MPRA promoter benchmark.

Implements ``SequenceExpressionPredictor``:
  1. Embed each 110 bp test sequence into the 5,000 bp plasmid construct.
  2. Centre-pad to 16,384 bp with N's (all-zero one-hot).
  3. Forward (+RC) through the 8-fold Shorkie ensemble.
  4. Aggregate: cross-track mean over RNA-seq tracks, then sum over the
     YFP-CDS output bins → scalar expression prediction per sequence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._deboer_scaffold import YFP_END, YFP_START, build_construct
from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters.protocols import SequenceExpressionPredictor
from yeastbench.adapters.shorkie_eqtl import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    SHORKIE_1011_RNA_SEQ_TRACK_IDS,
)

if TYPE_CHECKING:
    import torch
    from yeastbench.models.shorkie import Shorkie

# The 5,000 bp construct is centre-padded to 16,384 bp.
_PAD_LEFT = (SEQ_LEN - 5000) // 2   # 5692
_PAD_RIGHT = SEQ_LEN - 5000 - _PAD_LEFT  # 5692

# YFP CDS position in the *padded* 16,384 bp sequence.
_YFP_START_PADDED = _PAD_LEFT + YFP_START  # 6700
_YFP_END_PADDED = _PAD_LEFT + YFP_END      # 7417

# Output-bin indices covering the YFP CDS (after 1024 bp crop).
_YFP_BIN_START = (_YFP_START_PADDED - CROP_BP_EACH_SIDE) // BIN_WIDTH  # 354
_YFP_BIN_END = (_YFP_END_PADDED - 1 - CROP_BP_EACH_SIDE) // BIN_WIDTH + 1  # 400
YFP_BINS = np.arange(_YFP_BIN_START, _YFP_BIN_END)  # 46 bins


class ShorkieMPRAPredictor(SequenceExpressionPredictor):
    def __init__(
        self,
        models: list["Shorkie"],
        track_subset: list[int] = SHORKIE_1011_RNA_SEQ_TRACK_IDS,
        device: "str | torch.device" = "cuda",
        batch_size: int = 8,
        use_rc: bool = True,
    ) -> None:
        import torch as _torch

        if not models:
            raise ValueError("Must provide at least one model fold")
        self.models = models
        self.track_subset = list(track_subset)
        self.device = _torch.device(device)
        self.batch_size = batch_size
        self.use_rc = use_rc
        for m in self.models:
            m.to(self.device).eval()

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        track_subset: list[int] = SHORKIE_1011_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 8,
        use_rc: bool = True,
    ) -> "ShorkieMPRAPredictor":
        from yeastbench.models.shorkie import Shorkie

        with open(params_path) as f:
            config = json.load(f)
        models = [
            Shorkie.from_tf_checkpoint(config["model"], str(p))
            for p in checkpoint_paths
        ]
        return cls(models, list(track_subset), device, batch_size, use_rc=use_rc)

    @staticmethod
    def _encode(seq_110bp: str) -> np.ndarray:
        """110 bp → centre-padded 16,384 bp → (4, 16384) one-hot."""
        construct = build_construct(seq_110bp).upper()
        padded = "N" * _PAD_LEFT + construct + "N" * _PAD_RIGHT
        assert len(padded) == SEQ_LEN
        return one_hot_encode_channels_first(padded)

    def predict_expressions(self, seqs: Sequence[str]) -> np.ndarray:
        import torch as _torch

        n = len(seqs)
        scores = np.empty(n, dtype=np.float64)
        track_idx_t = _torch.tensor(
            self.track_subset, device=self.device, dtype=_torch.long
        )
        yfp_bins_t = _torch.from_numpy(YFP_BINS).to(self.device)
        n_tracks = len(self.track_subset)

        for batch_start in tqdm(range(0, n, self.batch_size), desc="Shorkie MPRA"):
            batch_end = min(batch_start + self.batch_size, n)
            batch_seqs = seqs[batch_start:batch_end]
            B = len(batch_seqs)

            arrs = [self._encode(s) for s in batch_seqs]
            x = _torch.from_numpy(np.stack(arrs, axis=0)).to(self.device)

            with _torch.no_grad():
                acc = _torch.zeros(
                    B, OUTPUT_BINS, n_tracks,
                    device=self.device, dtype=_torch.float32,
                )
                x_rc = x.flip(dims=[1, 2]) if self.use_rc else None
                for m in self.models:
                    out = m(x).index_select(2, track_idx_t)
                    if self.use_rc:
                        out_rc = m(x_rc).index_select(2, track_idx_t).flip(dims=[1])
                        out = 0.5 * (out + out_rc)
                    acc.add_(out)
                acc.div_(len(self.models))

            # Cross-track mean → sum over YFP bins → scalar per sequence
            cov = acc.mean(dim=2)  # (B, 896)
            yfp_signal = cov.index_select(1, yfp_bins_t).sum(dim=1)  # (B,)
            scores[batch_start:batch_end] = yfp_signal.cpu().numpy()

        return scores
