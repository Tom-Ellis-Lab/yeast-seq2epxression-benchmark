"""Shorkie adapter for the Brooks SCRaMBLE benchmark.

Sequence-in / **per-base, raw-count** coverage-out. Shorkie was trained
with a Poisson loss + softplus head, so its outputs are already in raw
predicted-count units — no inverse-transform needed (unlike Yorzoi).
The adapter:

1. One-hot encodes the 16,384 bp construct.
2. Forwards through the 8-fold Shorkie ensemble with optional RC
   averaging; averages predictions across folds.
3. Index-selects the T0 RNA-seq tracks
   (``SHORKIE_T0_RNA_SEQ_TRACK_IDS``, 384 unstranded coverage tracks)
   and means across them.
4. Spreads each 16 bp bin's total across its 16 bp uniformly to recover
   a per-base vector of length ``seq_len - 2 * crop_bp_each_side`` =
   14,336 bp.

Shorkie has no Brooks-specific output tracks, so the prediction does
not vary by strain — ``varies_by_strain = False`` tells the Brooks
benchmark to call the adapter once for the native and broadcast the
result across the JS94 replicate axis.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Sequence

import numpy as np

from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._shorkie_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    SEQ_LEN,
    SHORKIE_T0_RNA_SEQ_TRACK_IDS,
)
from yeastbench.adapters.protocols import CoverageTrackPredictor
from yeastbench.models.shorkie import Shorkie

log = logging.getLogger(__name__)


def _unbin_per_base(binned: np.ndarray, bin_width: int) -> np.ndarray:
    """Spread per-bin totals back to per-base (each bin's total /
    bin_width, repeated bin_width times). Sum over a CDS interval on
    the per-base vector matches the bin-level sum that the model
    actually outputs."""
    return np.repeat(binned, bin_width) / float(bin_width)


class ShorkieBrooksPredictor(CoverageTrackPredictor):
    # Geometry exposed to the benchmark
    seq_len: ClassVar[int] = SEQ_LEN
    crop_bp_each_side: ClassVar[int] = CROP_BP_EACH_SIDE
    # Shorkie has no Brooks-specific tracks → ignores the strain hint
    # and always returns the same prediction. Tells the benchmark to
    # skip redundant native forwards.
    varies_by_strain: ClassVar[bool] = False
    # Smaller batch than Yorzoi: 8-fold ensemble × 16,384 bp ×
    # all-track activation tensors fits in ~24 GB at batch 4 with RC.
    batch_size: ClassVar[int] = 4

    def __init__(
        self,
        model: Shorkie,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        batch_size: int = 4,
    ) -> None:
        self.model = model
        self.track_subset = list(track_subset)
        self.batch_size = int(batch_size)

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        use_rc: bool = True,
        batch_size: int = 4,
    ) -> "ShorkieBrooksPredictor":
        return cls(
            Shorkie.from_checkpoints(
                params_path, checkpoint_paths, device=device, use_rc=use_rc,
            ),
            track_subset=list(track_subset),
            batch_size=batch_size,
        )

    def predict_coverage_batch(
        self,
        seqs: Sequence[str],
        strands: Sequence[str],
        strains: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Batched per-base predicted RNA-seq-like coverage over the
        central ``seq_len − 2*crop`` = 14,336 bp of each input window,
        averaged across the T0 RNA-seq tracks and the model ensemble.
        Already in raw predicted-count units (Shorkie's Poisson/softplus
        head outputs raw counts directly — no inverse transform needed).

        ``strains`` and ``strands`` are accepted to satisfy the
        protocol but ignored — Shorkie has no Brooks-specific output
        tracks (RNA-seq tracks are unstranded coverage bigwigs; the
        same subset works for both ``+`` and ``-`` host genes).
        Returns shape ``(B, 14336)``."""
        import torch as _torch

        B = len(seqs)
        for s in seqs:
            assert len(s) == SEQ_LEN, (
                f"Brooks/Shorkie construct must be {SEQ_LEN} bp; got "
                f"{len(s)}"
            )

        arrs = np.stack(
            [one_hot_encode_channels_first(s) for s in seqs], axis=0
        )                                                  # (B, 4, SEQ_LEN)
        x = _torch.from_numpy(arrs).to(self.model.device)
        track_idx_t = _torch.tensor(
            self.track_subset, device=self.model.device, dtype=_torch.long
        )

        with _torch.no_grad():
            # (B, OUTPUT_BINS) — ensemble + RC + track-mean (Pattern B:
            # per-fold track mean folded into the ensemble accumulator).
            acc = self.model.forward_track_mean_binned(x, track_idx_t)

        binned = acc.cpu().numpy()                          # (B, OUTPUT_BINS)
        return np.stack(
            [_unbin_per_base(binned[i], BIN_WIDTH) for i in range(B)],
            axis=0,
        )                                                   # (B, 14336)
