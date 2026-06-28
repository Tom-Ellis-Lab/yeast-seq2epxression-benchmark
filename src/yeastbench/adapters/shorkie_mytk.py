"""Shorkie adapter for the MYTK promoter × integration-site task.

For each ``(locus, promoter)`` construct, build a 16 kb input = native
R64 with the full per-promoter payload (``ConS-[promoter]-mScarlet-
tTDH1-ConE``) spliced in at the integration midpoint (always ``+``
strand, centred), forward-pass through the 8-fold ensemble with optional
RC averaging, take the cross-track mean over the T0 RNA-seq tracks, and
sum over the mScarlet-CDS output bases. Absolute readout (no REF
baseline) — the signal of interest is the absolute mScarlet level as a
function of promoter × integration position, exactly as in the Hong
adapter.

The only difference from ``shorkie_hong`` is that the payload varies per
promoter (resolved from the 4-record payload FASTA) and the mScarlet
readout span is located by substring rather than a fixed offset. See
``docs/benchmarks/mytk_ints_promoter.md``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._mytk_scaffold import (
    DEFAULT_PAYLOADS_FASTA,
    build_mytk_context,
    load_payload_geometries,
)
from yeastbench.adapters.protocols import (
    PromoterIntegrationConstruct,
    PromoterIntegrationExpressionPredictor,
)
from yeastbench.adapters._shorkie_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    SHORKIE_T0_RNA_SEQ_TRACK_IDS,
)
from yeastbench.models.shorkie import Shorkie

log = logging.getLogger(__name__)


class ShorkieMytkPredictor(PromoterIntegrationExpressionPredictor):
    def __init__(
        self,
        model: Shorkie,
        fasta_path: str | Path,
        payloads_fasta: str | Path | None = None,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        batch_size: int = 16,
    ) -> None:
        import pysam
        import torch as _torch

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.geometries = load_payload_geometries(
            payloads_fasta if payloads_fasta is not None else DEFAULT_PAYLOADS_FASTA
        )
        self.track_subset = list(track_subset)
        self.batch_size = batch_size
        self._track_idx_t = _torch.tensor(
            self.track_subset, device=self.model.device, dtype=_torch.long
        )

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        fasta_path: str | Path,
        payloads_fasta: str | Path | None = None,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 16,
        use_rc: bool = True,
    ) -> "ShorkieMytkPredictor":
        return cls(
            Shorkie.from_checkpoints(
                params_path, checkpoint_paths, device=device, use_rc=use_rc,
            ),
            fasta_path=fasta_path,
            payloads_fasta=payloads_fasta,
            track_subset=list(track_subset),
            batch_size=batch_size,
        )

    def predict_integrated_expressions(
        self, constructs: Sequence[PromoterIntegrationConstruct]
    ) -> np.ndarray:
        import torch as _torch

        scores = np.full(len(constructs), np.nan, dtype=np.float64)
        contexts = [
            (i, ctx)
            for i, c in enumerate(constructs)
            if (ctx := build_mytk_context(
                c, self.geometries, self.fasta,
                SEQ_LEN, CROP_BP_EACH_SIDE, BIN_WIDTH, OUTPUT_BINS,
            )) is not None
        ]

        for bs in tqdm(
            range(0, len(contexts), self.batch_size), desc="Shorkie MYTK"
        ):
            batch = contexts[bs : bs + self.batch_size]
            x = _torch.from_numpy(
                np.stack([
                    one_hot_encode_channels_first(c.window_seq) for _, c in batch
                ])
            ).to(self.model.device)
            with _torch.no_grad():
                # (B, OUTPUT_BINS*BIN_WIDTH) per-base raw counts.
                cov = self.model.forward_track_mean_perbase(x, self._track_idx_t)
            for j, (row_idx, ctx) in enumerate(batch):
                base_idx = ctx.readout_base_positions
                if base_idx.size == 0:
                    continue  # readout outside crop → leave NaN
                base_t = _torch.as_tensor(
                    base_idx, device=self.model.device, dtype=_torch.long
                )
                scores[row_idx] = float(cov[j].index_select(0, base_t).sum().item())

        return scores
