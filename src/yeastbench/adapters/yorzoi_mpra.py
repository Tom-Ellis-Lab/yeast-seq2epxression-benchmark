"""Yorzoi fixed-context adapter for the Rafi / deBoer MPRA promoter benchmark.

Implements ``SequenceExpressionPredictor``:
  1. Embed each 110 bp test sequence into the 5,000 bp plasmid construct.
  2. Trim 8 bp from the downstream end to reach 4,992 bp (Yorzoi's input
     length).  The removed bases are from the tail of ``cen_downstream`` —
     far from the promoter and YFP regions.
  3. Forward (+RC with strand-track swap) through the Yorzoi model.
  4. Aggregate: cross-track mean over plus-strand RNA-seq tracks, then sum
     over YFP-CDS output bins → scalar expression prediction per sequence.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._deboer_scaffold import YFP_END, YFP_START, build_construct
from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters.protocols import SequenceExpressionPredictor
from yeastbench.adapters.yorzoi_eqtl import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    YORZOI_MINUS_TRACK_IDS,
    YORZOI_PLUS_TRACK_IDS,
)

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
        model: Any,  # yorzoi.model.borzoi.Borzoi
        track_subset: list[int] = YORZOI_PLUS_TRACK_IDS,
        device: "str | torch.device" = "cuda",
        batch_size: int = 16,
        use_rc: bool = True,
        autocast: bool = True,
    ) -> None:
        import torch as _torch

        self.model = model
        self.track_subset = list(track_subset)
        self.device = _torch.device(device)
        self.batch_size = batch_size
        self.use_rc = use_rc
        self.autocast = autocast
        self.model.to(self.device).eval()

        self._is_plus_only = self.track_subset == YORZOI_PLUS_TRACK_IDS

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
        from yorzoi.model.borzoi import Borzoi

        model = Borzoi.from_pretrained(hf_repo)
        return cls(model, list(track_subset), device, batch_size, use_rc=use_rc, autocast=autocast)

    @staticmethod
    def _encode(seq_110bp: str) -> np.ndarray:
        """110 bp → 5,000 bp construct → trim to 4,992 bp → (4992, 4) one-hot."""
        construct = build_construct(seq_110bp).upper()
        trimmed = construct[:-_TRIM_RIGHT]
        assert len(trimmed) == SEQ_LEN
        return one_hot_encode_channels_first(trimmed).T  # (L, 4) channels-last

    def _forward_with_rc(self, x: "torch.Tensor") -> "torch.Tensor":
        """Run forward (+optional RC) → (B, len(track_subset), OUTPUT_BINS)."""
        import torch as _torch

        track_idx_t = _torch.tensor(
            self.track_subset, device=self.device, dtype=_torch.long
        )
        ctx = (
            _torch.autocast(device_type="cuda")
            if self.autocast and self.device.type == "cuda"
            else _torch.amp.autocast(device_type="cpu", enabled=False)
        )
        with ctx:
            out_fwd = self.model(x)  # (B, 162, 300)
        if not self.use_rc:
            return out_fwd.index_select(1, track_idx_t)

        x_rc = x.flip(dims=[1, 2])
        with ctx:
            out_rc = self.model(x_rc)

        if self._is_plus_only:
            swap_idx = _torch.tensor(
                YORZOI_MINUS_TRACK_IDS, device=self.device, dtype=_torch.long
            )
            out_rc_aligned = out_rc.index_select(1, swap_idx).flip(dims=[2])
            out_fwd_sub = out_fwd.index_select(1, track_idx_t)
        else:
            raise NotImplementedError(
                "RC averaging for non-default Yorzoi track subsets not implemented."
            )

        return 0.5 * (out_fwd_sub + out_rc_aligned)

    def predict_expressions(self, seqs: Sequence[str]) -> np.ndarray:
        import torch as _torch

        n = len(seqs)
        scores = np.empty(n, dtype=np.float64)
        yfp_bins_t = _torch.from_numpy(YFP_BINS).to(self.device)

        for batch_start in tqdm(range(0, n, self.batch_size), desc="Yorzoi MPRA"):
            batch_end = min(batch_start + self.batch_size, n)
            batch_seqs = seqs[batch_start:batch_end]
            B = len(batch_seqs)

            arrs = [self._encode(s) for s in batch_seqs]
            x = _torch.from_numpy(np.stack(arrs, axis=0)).to(self.device)

            with _torch.no_grad():
                pred = self._forward_with_rc(x).float()  # (B, n_tracks, 300)

            # Cross-track mean → sum over YFP bins → scalar per sequence
            cov = pred.mean(dim=1)  # (B, 300)
            yfp_signal = cov.index_select(1, yfp_bins_t).sum(dim=1)  # (B,)
            scores[batch_start:batch_end] = yfp_signal.cpu().numpy()

        return scores
