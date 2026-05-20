"""Yorzoi fixed-context adapter for the Rafi / deBoer MPRA promoter benchmark.

Implements ``SequenceExpressionPredictor``:
  1. Embed each 110 bp test sequence into the 5,000 bp plasmid construct.
  2. Trim 8 bp from the downstream end to reach 4,992 bp (Yorzoi's input
     length).  The removed bases are from the tail of ``cen_downstream`` —
     far from the promoter and YFP regions.
  3. Forward (+RC with strand-track swap) through the Yorzoi model.
  4. Aggregate: cross-track mean over plus-strand RNA-seq tracks, then sum
     over YFP-CDS output bins → scalar expression prediction per sequence.

Forward, device, autocast, and RC averaging live in
`yeastbench.models.yorzoi.Yorzoi`; this adapter only handles task-
specific input construction and output aggregation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._deboer_scaffold import YFP_END, YFP_START, build_construct
from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    YORZOI_PLUS_TRACK_IDS,
)
from yeastbench.adapters.protocols import SequenceExpressionPredictor
from yeastbench.models.yorzoi import Yorzoi

if TYPE_CHECKING:
    import torch

# The 5,000 bp construct is trimmed to 4,992 bp by removing 8 bp from the
# downstream end.  YFP and the variable region are unchanged.
_TRIM_RIGHT = 5000 - SEQ_LEN  # 8

# YFP CDS position within the 4,992 bp trimmed construct (0-indexed).
# Same as in the full 5,000 bp construct — trimming happens at the far end.
_YFP_START = YFP_START  # 1008
_YFP_END = YFP_END      # 1725

# Output-bin indices covering the YFP CDS (after 996 bp crop).
_YFP_BIN_START = (_YFP_START - CROP_BP_EACH_SIDE) // BIN_WIDTH  # 1
_YFP_BIN_END = (_YFP_END - 1 - CROP_BP_EACH_SIDE) // BIN_WIDTH + 1  # 73
YFP_BINS = np.arange(_YFP_BIN_START, _YFP_BIN_END)  # 72 bins


class YorzoiMPRAPredictor(SequenceExpressionPredictor):
    def __init__(
        self,
        model: Yorzoi,
        track_subset: list[int] = YORZOI_PLUS_TRACK_IDS,
        batch_size: int = 16,
    ) -> None:
        self.model = model
        self.track_subset = list(track_subset)
        self.batch_size = batch_size
        assert self.track_subset == YORZOI_PLUS_TRACK_IDS, (
            "YorzoiMPRAPredictor currently only supports the default "
            "plus-strand track subset; non-default subsets need RC "
            "averaging logic the wrapper doesn't expose."
        )

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        track_subset: list[int] = YORZOI_PLUS_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 16,
        use_rc: bool = True,
        autocast: bool = True,
    ) -> "YorzoiMPRAPredictor":
        return cls(
            Yorzoi.from_pretrained(
                hf_repo, device=device, use_rc=use_rc, autocast=autocast,
            ),
            track_subset=list(track_subset),
            batch_size=batch_size,
        )

    @staticmethod
    def _encode(seq_110bp: str) -> np.ndarray:
        """110 bp → 5,000 bp construct → trim to 4,992 bp → (4992, 4) one-hot."""
        construct = build_construct(seq_110bp).upper()
        trimmed = construct[:-_TRIM_RIGHT]
        assert len(trimmed) == SEQ_LEN
        return one_hot_encode_channels_first(trimmed).T  # (L, 4) channels-last

    def predict_expressions(self, seqs: Sequence[str]) -> np.ndarray:
        import torch as _torch

        n = len(seqs)
        scores = np.empty(n, dtype=np.float64)
        yfp_bins_t = _torch.from_numpy(YFP_BINS).to(self.model.device)
        plus_idx_t = _torch.tensor(
            self.track_subset, device=self.model.device, dtype=_torch.long
        )

        for batch_start in tqdm(range(0, n, self.batch_size), desc="Yorzoi MPRA"):
            batch_end = min(batch_start + self.batch_size, n)
            batch_seqs = seqs[batch_start:batch_end]

            arrs = [self._encode(s) for s in batch_seqs]
            x = _torch.from_numpy(np.stack(arrs, axis=0)).to(self.model.device)

            with _torch.no_grad():
                # Full-track forward with RC averaging (handled by wrapper).
                pred = self.model.forward_tracks_binned(x).float()  # (B, 162, 300)
            pred_sub = pred.index_select(1, plus_idx_t)              # (B, 81, 300)

            # Cross-track mean → sum over YFP bins → scalar per sequence
            cov = pred_sub.mean(dim=1)  # (B, 300)
            yfp_signal = cov.index_select(1, yfp_bins_t).sum(dim=1)  # (B,)
            scores[batch_start:batch_end] = yfp_signal.cpu().numpy()

        return scores
