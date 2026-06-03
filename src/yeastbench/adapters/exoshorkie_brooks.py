"""ExoShorkie adapter for the Brooks SCRaMBLE benchmark.

This is ExoShorkie's native task: the model literally predicts RNA-seq coverage.
Sequence-in / per-base count-space coverage-out, mirroring
``ShorkieBrooksPredictor`` but with the ExoShorkie wrapper, which inverts the
log-z normalization (``count = expm1(z*sigma + mu)`` per student) before the
6-student / RC averaging — so the returned per-base values are count-space
coverage, directly comparable to Shorkie's.

ExoShorkie has no Brooks-specific tracks and a single coverage output, so the
prediction does not vary by strain — ``varies_by_strain = False``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Sequence

import numpy as np

from yeastbench.adapters._exoshorkie_constants import (
    CROP_BP_EACH_SIDE,
    SEQ_LEN,
)
from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters.protocols import CoverageTrackPredictor
from yeastbench.models.exoshorkie import ExoShorkie

log = logging.getLogger(__name__)


class ExoShorkieBrooksPredictor(CoverageTrackPredictor):
    # Geometry exposed to the benchmark
    seq_len: ClassVar[int] = SEQ_LEN
    crop_bp_each_side: ClassVar[int] = CROP_BP_EACH_SIDE
    varies_by_strain: ClassVar[bool] = False
    # 6 students × 16,384 bp × (RC) fits comfortably; matches Shorkie's batch.
    batch_size: ClassVar[int] = 4

    def __init__(self, model: ExoShorkie, batch_size: int = 4) -> None:
        self.model = model
        self.batch_size = int(batch_size)

    @classmethod
    def from_students(
        cls,
        students_dir: str | Path | None = None,
        device: str = "cuda",
        use_rc: bool = True,
        batch_size: int = 4,
    ) -> "ExoShorkieBrooksPredictor":
        kwargs = {} if students_dir is None else {"students_dir": students_dir}
        return cls(
            ExoShorkie.from_students(device=device, use_rc=use_rc, **kwargs),
            batch_size=batch_size,
        )

    def predict_coverage_batch(
        self,
        seqs: Sequence[str],
        strands: Sequence[str],
        strains: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        """Batched per-base count-space coverage over the central
        ``seq_len − 2*crop`` = 14,336 bp of each window, ensemble + RC averaged.
        ``strands``/``strains`` are accepted to satisfy the protocol but ignored
        (single coverage output, no strain-specific tracks). Returns
        ``(B, 14336)``."""
        import torch as _torch

        for s in seqs:
            assert len(s) == SEQ_LEN, (
                f"Brooks/ExoShorkie construct must be {SEQ_LEN} bp; got {len(s)}"
            )

        out = np.empty((len(seqs), (SEQ_LEN - 2 * CROP_BP_EACH_SIDE)), dtype=np.float32)
        for bs in range(0, len(seqs), self.batch_size):
            chunk = seqs[bs : bs + self.batch_size]
            arrs = np.stack(
                [one_hot_encode_channels_first(s) for s in chunk], axis=0
            )                                              # (b, 4, SEQ_LEN)
            x = _torch.from_numpy(arrs).float().to(self.model.device)
            with _torch.no_grad():
                perbase = self.model.forward_perbase(x)    # (b, 14336)
            out[bs : bs + len(chunk)] = perbase.cpu().numpy()
        return out
