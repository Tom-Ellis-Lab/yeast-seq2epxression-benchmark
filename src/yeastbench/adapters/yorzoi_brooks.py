"""Yorzoi adapter for the Brooks SCRaMBLE benchmark.

Sequence-in / per-bin-coverage-out: predict an RNA-seq-like coverage
profile over a 4992 bp construct, strand-matched to the gene's strand,
RC-averaged with the standard plus↔minus track swap.

Exposes ``bin_width``, ``crop_bp_each_side``, ``output_bins`` so the
benchmark can align per-base truth to the per-bin output.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    YORZOI_MINUS_TRACK_IDS,
    YORZOI_PLUS_TRACK_IDS,
)
from yeastbench.adapters.protocols import CoverageTrackPredictor

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

N_TRACKS_TOTAL = 162


class YorzoiBrooksPredictor(CoverageTrackPredictor):
    bin_width: ClassVar[int] = BIN_WIDTH
    crop_bp_each_side: ClassVar[int] = CROP_BP_EACH_SIDE
    output_bins: ClassVar[int] = OUTPUT_BINS

    def __init__(
        self,
        model: Any,
        device: "str | torch.device" = "cuda",
        use_rc: bool = True,
        autocast: bool = True,
    ) -> None:
        import torch as _torch

        self.model = model
        self.device = _torch.device(device)
        self.use_rc = use_rc
        self.autocast = autocast
        self.model.to(self.device).eval()

        plus = _torch.tensor(YORZOI_PLUS_TRACK_IDS, device=self.device,
                             dtype=_torch.long)
        minus = _torch.tensor(YORZOI_MINUS_TRACK_IDS, device=self.device,
                              dtype=_torch.long)
        swap = _torch.empty(N_TRACKS_TOTAL, dtype=_torch.long,
                            device=self.device)
        swap[plus] = minus
        swap[minus] = plus
        self._full_swap_idx = swap

    @classmethod
    def from_pretrained(
        cls, hf_repo: str, device: str = "cuda",
        use_rc: bool = True, autocast: bool = True,
    ) -> "YorzoiBrooksPredictor":
        from yorzoi.model.borzoi import Borzoi
        return cls(Borzoi.from_pretrained(hf_repo),
                   device=device, use_rc=use_rc, autocast=autocast)

    def _forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """(1, SEQ_LEN, 4) → (1, 162, OUTPUT_BINS), RC-averaged with swap."""
        import torch as _torch

        amp = (
            _torch.autocast(device_type="cuda")
            if self.autocast and self.device.type == "cuda"
            else _torch.amp.autocast(device_type="cpu", enabled=False)
        )
        with amp:
            out_fwd = self.model(x)
        if not self.use_rc:
            return out_fwd
        x_rc = x.flip(dims=[1, 2])
        with amp:
            out_rc = self.model(x_rc)
        out_rc_aligned = out_rc.index_select(1, self._full_swap_idx).flip(dims=[2])
        return 0.5 * (out_fwd + out_rc_aligned)

    def predict_coverage(self, construct_seq: str, strand: str) -> np.ndarray:
        import torch as _torch

        assert len(construct_seq) == SEQ_LEN, (
            f"Brooks construct must be {SEQ_LEN} bp; got {len(construct_seq)}"
        )
        x = _torch.from_numpy(
            one_hot_encode_channels_first(construct_seq).T  # (L, 4) channels-last
        ).unsqueeze(0).to(self.device)
        with _torch.no_grad():
            pred = self._forward(x).float()  # (1, 162, bins)
        ts, te = (0, 81) if strand == "+" else (81, 162)
        return pred[0, ts:te].mean(dim=0).cpu().numpy()  # (OUTPUT_BINS,)
