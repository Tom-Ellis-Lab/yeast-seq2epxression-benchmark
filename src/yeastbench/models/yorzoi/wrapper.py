"""Benchmark-side wrapper around the third-party Yorzoi (Borzoi-lineage)
model. Owns device, autocast, RC averaging with the plus↔minus strand
swap, and the batched forward path.

Adapters take a `Yorzoi` instance and call `forward_tracks_binned` —
they no longer need to know how to load checkpoints, build a swap
index, or construct an autocast context. Task-specific aggregation
(track-subset means, CDS-bin sums, marginalised LFCs, ...) stays on
the adapter.

This is intentionally a thin wrapper: the body of
`forward_tracks_binned` is lifted verbatim from the
`_forward_full_tracks` method that used to be duplicated across
`yorzoi_shalem.py`, `yorzoi_wu.py`, and `yorzoi_brooks.py` (and
inlined in the simpler `yorzoi_eqtl.py` / `yorzoi_mpra.py`). Net
behavior is bit-identical."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    import torch

# Architecture (verified by loading the HF checkpoint):
# input 4,992 bp → output (B, 162 tracks, 300 bins), 10 bp/bin,
# post-crop output covers the central 3,000 bp of input.
SEQ_LEN: int = 4992
OUTPUT_BINS: int = 300
BIN_WIDTH: int = 10
CROP_BP_EACH_SIDE: int = 996

# Track layout from yorzoi/track_annotation.json:
#   indices 0..80   → '+' (forward) strand, 81 tracks
#   indices 81..161 → '-' (reverse) strand, 81 tracks (same samples)
N_PLUS_TRACKS: int = 81
N_TRACKS_TOTAL: int = 162
PLUS_TRACK_IDS: list[int] = list(range(0, 81))
MINUS_TRACK_IDS: list[int] = list(range(81, 162))


class Yorzoi:
    """Benchmark-side wrapper around `yorzoi.model.borzoi.Borzoi`."""

    # Geometry constants exposed for adapters
    SEQ_LEN: ClassVar[int] = SEQ_LEN
    OUTPUT_BINS: ClassVar[int] = OUTPUT_BINS
    BIN_WIDTH: ClassVar[int] = BIN_WIDTH
    CROP_BP_EACH_SIDE: ClassVar[int] = CROP_BP_EACH_SIDE
    N_PLUS_TRACKS: ClassVar[int] = N_PLUS_TRACKS
    N_TRACKS_TOTAL: ClassVar[int] = N_TRACKS_TOTAL

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

        # Precompute the plus↔minus swap index used to align RC-flipped
        # outputs with the forward outputs.
        plus = _torch.tensor(PLUS_TRACK_IDS, device=self.device,
                             dtype=_torch.long)
        minus = _torch.tensor(MINUS_TRACK_IDS, device=self.device,
                              dtype=_torch.long)
        swap = _torch.empty(N_TRACKS_TOTAL, dtype=_torch.long,
                            device=self.device)
        swap[plus] = minus
        swap[minus] = plus
        self._full_swap_idx = swap

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        device: "str | torch.device" = "cuda",
        use_rc: bool = True,
        autocast: bool = True,
    ) -> "Yorzoi":
        from yorzoi.model.borzoi import Borzoi
        return cls(
            Borzoi.from_pretrained(hf_repo),
            device=device, use_rc=use_rc, autocast=autocast,
        )

    def _autocast_ctx(self):
        """Autocast context matching the model's device. Construct
        fresh each call (avoids subtle reuse issues across forwards)."""
        import torch as _torch

        if self.autocast and self.device.type == "cuda":
            return _torch.autocast(device_type="cuda")
        return _torch.amp.autocast(device_type="cpu", enabled=False)

    def forward_tracks_binned(self, x: "torch.Tensor") -> "torch.Tensor":
        """Single forward through the model with RC averaging + the
        plus↔minus strand swap. Returns ``(B, 162, OUTPUT_BINS)`` —
        still binned at ``BIN_WIDTH`` bp, still in Borzoi's transformed
        output space. Adapter is responsible for any inverse transform
        / unbin / track-subset selection downstream."""
        with self._autocast_ctx():
            out_fwd = self.model(x)
        if not self.use_rc:
            return out_fwd
        x_rc = x.flip(dims=[1, 2])
        with self._autocast_ctx():
            out_rc = self.model(x_rc)
        out_rc_aligned = out_rc.index_select(1, self._full_swap_idx).flip(dims=[2])
        return 0.5 * (out_fwd + out_rc_aligned)
