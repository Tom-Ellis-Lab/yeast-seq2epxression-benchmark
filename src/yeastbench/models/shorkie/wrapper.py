"""Benchmark-side wrapper around the 8-fold Shorkie ensemble. Owns
device handling, the ensemble loop, RC averaging, and the batched
forward path.

Adapters take a `Shorkie` instance and call either:
  * `forward_tracks_binned(x, track_subset)` — accumulates the
    per-track output across folds, returning a (B, OUTPUT_BINS,
    n_tracks) tensor. The adapter does any track-mean / track-mean →
    bin-sum aggregation downstream.
  * `forward_track_mean_binned(x, track_subset)` — accumulates the
    *track-mean* across folds, returning (B, OUTPUT_BINS). Use when
    the adapter immediately collapses tracks via `mean(dim=2)`; saves
    memory at the cost of slightly different floating-point ordering
    vs `forward_tracks_binned(...).mean(dim=2)` (the inter-fold mean
    and inter-track mean commute mathematically but not bit-exactly).

Two methods are kept rather than one because the existing adapters
use both orderings; preserving each adapter's exact pattern keeps the
refactor bit-identical on the way through."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Sequence

if TYPE_CHECKING:
    import torch

    from yeastbench.models.shorkie.nn import ShorkieModule

# Architecture (derived from data/models/shorkie/params.json):
# 16,384 bp input → 896 output bins × 16 bp/bin, post-crop covering
# the central 14,336 bp of input.
SEQ_LEN: int = 16384
OUTPUT_BINS: int = 896
BIN_WIDTH: int = 16
CROP_BP_EACH_SIDE: int = 1024


class Shorkie:
    """8-fold Shorkie ensemble wrapper."""

    # Geometry constants exposed for adapters
    SEQ_LEN: ClassVar[int] = SEQ_LEN
    OUTPUT_BINS: ClassVar[int] = OUTPUT_BINS
    BIN_WIDTH: ClassVar[int] = BIN_WIDTH
    CROP_BP_EACH_SIDE: ClassVar[int] = CROP_BP_EACH_SIDE

    def __init__(
        self,
        folds: list["ShorkieModule"],
        device: "str | torch.device" = "cuda",
        use_rc: bool = True,
    ) -> None:
        import torch as _torch

        if not folds:
            raise ValueError("Must provide at least one Shorkie fold")
        self.folds = folds
        self.device = _torch.device(device)
        self.use_rc = use_rc
        for m in self.folds:
            m.to(self.device).eval()

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        device: "str | torch.device" = "cuda",
        use_rc: bool = True,
    ) -> "Shorkie":
        from yeastbench.models.shorkie.nn import ShorkieModule

        with open(params_path) as fh:
            config = json.load(fh)
        folds = [
            ShorkieModule.from_tf_checkpoint(config["model"], str(p))
            for p in checkpoint_paths
        ]
        return cls(folds, device=device, use_rc=use_rc)

    def forward_tracks_binned(
        self,
        x: "torch.Tensor",
        track_subset: "torch.Tensor",
    ) -> "torch.Tensor":
        """Ensemble + RC averaged, accumulating the per-track output.
        Returns ``(B, OUTPUT_BINS, len(track_subset))``.

        ``track_subset`` is a long tensor of track indices, on the same
        device as the model. Adapters that need the cross-track mean
        of the result should then call ``.mean(dim=2)``."""
        import torch as _torch

        B = x.shape[0]
        n_tracks = int(track_subset.numel())
        acc = _torch.zeros(
            B, OUTPUT_BINS, n_tracks,
            device=self.device, dtype=_torch.float32,
        )
        x_rc = x.flip(dims=[1, 2]) if self.use_rc else None
        for m in self.folds:
            out = m(x).index_select(2, track_subset)
            if self.use_rc:
                out_rc = m(x_rc).index_select(2, track_subset).flip(dims=[1])
                out = 0.5 * (out + out_rc)
            acc.add_(out)
        acc.div_(len(self.folds))
        return acc

    def forward_track_mean_binned(
        self,
        x: "torch.Tensor",
        track_subset: "torch.Tensor",
    ) -> "torch.Tensor":
        """Ensemble + RC averaged with the per-fold track mean folded
        in (memory-cheaper variant of ``forward_tracks_binned(...)
        .mean(dim=2)``). Returns ``(B, OUTPUT_BINS)``.

        Bit-not-identical to ``forward_tracks_binned(...).mean(dim=2)``
        because the inter-fold and inter-track means happen in
        different orders. Use whichever ordering matches the adapter's
        existing code path."""
        import torch as _torch

        B = x.shape[0]
        acc = _torch.zeros(
            B, OUTPUT_BINS, device=self.device, dtype=_torch.float32,
        )
        x_rc = x.flip(dims=[1, 2]) if self.use_rc else None
        for m in self.folds:
            out = m(x).index_select(2, track_subset)
            if self.use_rc:
                out_rc = m(x_rc).index_select(2, track_subset).flip(dims=[1])
                out = 0.5 * (out + out_rc)
            acc.add_(out.mean(dim=2))
        acc.div_(len(self.folds))
        return acc
