"""Yorzoi adapter for the Brooks SCRaMBLE benchmark.

Sequence-in / **per-base, untransformed** coverage-out. Yorzoi was
trained with the Borzoi-style target transform
``y = transform(bin_4bp(x))`` where
``transform(x) = min(x^0.75, 384 + sqrt(x^0.75 - 384))`` is a piecewise
power+sqrt squash (`yorzoi/yorzoi/utils.py`). Predictions therefore live
in a transformed binned space — log-ratios / shape metrics computed
directly on them do not match what you'd compute on raw pileups. This
adapter inverts the transform and unbins back to a per-base predicted
count vector so the benchmark can compare apples-to-apples with the
raw Nanopore per-base truth in the distribution.

Geometry exposed: ``seq_len`` (input length, 4992 bp) and
``crop_bp_each_side`` (996 bp). The returned vector covers the central
``seq_len - 2 * crop_bp_each_side`` = 3000 bp, base-by-base.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    BROOKS_NANOPORE_TRACK_IDS_PLUS_ALL,
    BROOKS_NANOPORE_TRACKS_BY_STRAIN,
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
N_PLUS_TRACKS = 81  # minus-track index = plus-track index + N_PLUS_TRACKS

TrackMode = Literal["all", "nanopore_all", "matched"]


def _borzoi_inv_transform(y: np.ndarray) -> np.ndarray:
    """Numpy port of yorzoi.utils._borzoi_transform_inv (per-bin)."""
    expd = np.where(y <= 384.0, y, 384.0 + (y - 384.0) ** 2)
    return np.power(np.clip(expd, 0.0, None), 1.0 / 0.75)


def _unbin_per_base(binned: np.ndarray, bin_width: int) -> np.ndarray:
    """Spread per-bin totals back to per-base values (each bin's total ÷
    bin_width, repeated bin_width times). Sum across the bin's bases
    recovers the original bin total — what we want for downstream sums
    over CDS intervals."""
    return np.repeat(binned, bin_width) / float(bin_width)


class YorzoiBrooksPredictor(CoverageTrackPredictor):
    # Geometry exposed to the benchmark
    seq_len: ClassVar[int] = SEQ_LEN
    crop_bp_each_side: ClassVar[int] = CROP_BP_EACH_SIDE

    def __init__(
        self,
        model: Any,
        device: "str | torch.device" = "cuda",
        use_rc: bool = True,
        autocast: bool = True,
        track_mode: TrackMode = "matched",
    ) -> None:
        """``track_mode`` controls which output tracks the prediction
        averages across.

        - ``"all"`` (v1): mean over all 81 strand-matched Yorzoi tracks,
          mixing protocols (Nanopore / Illumina / clone alignments).
        - ``"nanopore_all"``: mean over the 63 Brooks Nanopore direct-RNA
          tracks only (drops 11 non-Nanopore tracks; keeps every strain).
        - ``"matched"`` (default): per call, average only over the
          Nanopore tracks belonging to the supplied ``strain`` — i.e.
          the model's most in-distribution prediction for that strain.
          Requires the benchmark to pass ``strain=`` to
          ``predict_coverage``. If ``strain`` is missing or absent from
          ``BROOKS_NANOPORE_TRACKS_BY_STRAIN`` the call falls back to
          ``"nanopore_all"`` and emits a debug log line.
        """
        import torch as _torch

        self.model = model
        self.device = _torch.device(device)
        self.use_rc = use_rc
        self.autocast = autocast
        self.track_mode = track_mode
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
        track_mode: TrackMode = "matched",
    ) -> "YorzoiBrooksPredictor":
        from yorzoi.model.borzoi import Borzoi
        return cls(Borzoi.from_pretrained(hf_repo),
                   device=device, use_rc=use_rc, autocast=autocast,
                   track_mode=track_mode)

    def _plus_axis_indices(self, strain: str | None) -> list[int]:
        """Indices on the 81-track plus axis for this prediction.
        Minus-axis indices are derived by adding N_PLUS_TRACKS."""
        if self.track_mode == "all":
            return YORZOI_PLUS_TRACK_IDS
        if self.track_mode == "nanopore_all":
            return BROOKS_NANOPORE_TRACK_IDS_PLUS_ALL
        # "matched"
        if strain is None or strain not in BROOKS_NANOPORE_TRACKS_BY_STRAIN:
            log.debug("track_mode='matched' but strain=%r not in map; "
                      "falling back to nanopore_all", strain)
            return BROOKS_NANOPORE_TRACK_IDS_PLUS_ALL
        return BROOKS_NANOPORE_TRACKS_BY_STRAIN[strain]

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

    def predict_coverage(self, construct_seq: str, strand: str,
                         strain: str | None = None) -> np.ndarray:
        """Per-base predicted Nanopore-like coverage over the central
        ``seq_len - 2*crop`` = 3000 bp of the input window, **already
        untransformed and unbinned** so the benchmark can compute LFC,
        Pearson and JSD directly against raw per-base pileups in the
        same units (predicted-count scale).

        ``strain`` routes to per-strain Nanopore tracks when
        ``track_mode == "matched"`` (the default). See ``__init__``."""
        import torch as _torch

        assert len(construct_seq) == SEQ_LEN, (
            f"Brooks construct must be {SEQ_LEN} bp; got {len(construct_seq)}"
        )
        x = _torch.from_numpy(
            one_hot_encode_channels_first(construct_seq).T  # (L, 4) channels-last
        ).unsqueeze(0).to(self.device)
        with _torch.no_grad():
            pred = self._forward(x).float()                  # (1, 162, OUTPUT_BINS)

        plus_axis = self._plus_axis_indices(strain)
        if strand == "+":
            channels = plus_axis
        else:
            channels = [i + N_PLUS_TRACKS for i in plus_axis]
        idx = _torch.tensor(channels, device=pred.device, dtype=_torch.long)
        binned = pred[0].index_select(0, idx).mean(dim=0).cpu().numpy()
        # 1. Invert Yorzoi's training transform (per-bin).
        # 2. Spread each bin total back to per-base values (length 3000).
        raw_binned = _borzoi_inv_transform(binned)            # raw counts per bin
        return _unbin_per_base(raw_binned, BIN_WIDTH)         # (3000,)
