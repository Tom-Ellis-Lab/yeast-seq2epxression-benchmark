"""Yorzoi model package.

`Yorzoi` is the benchmark-side wrapper around
`yorzoi.model.borzoi.Borzoi` (the third-party pytorch nn module). It
owns device handling, autocast, RC averaging with the plus↔minus
strand swap, and the batched forward path. Adapters consume
`forward_tracks_binned` and do their own task-specific aggregation
on top."""
from yeastbench.models.yorzoi.wrapper import Yorzoi

__all__ = ["Yorzoi"]
