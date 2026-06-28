"""Yorzoi adapter for the MYTK promoter × integration-site task.

Same construct as the Shorkie MYTK adapter (native R64 spliced with the
full per-promoter payload at the integration midpoint, mScarlet-CDS
readout, no REF baseline) but with Yorzoi's 4992 bp input and 300 output
bins. The payload is always integrated on the chromosome's ``+`` strand,
so we always use the ``+``-strand track subset (tracks 0–80).

The only difference from ``yorzoi_hong`` is that the payload varies per
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
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
)
from yeastbench.models.yorzoi import Yorzoi

log = logging.getLogger(__name__)

# Payload is always + strand → always use the + tracks (0..80).
YORZOI_PLUS_TRACK_START = 0
YORZOI_PLUS_TRACK_END = 81


class YorzoiMytkPredictor(PromoterIntegrationExpressionPredictor):
    def __init__(
        self,
        model: Yorzoi,
        fasta_path: str | Path,
        payloads_fasta: str | Path | None = None,
        batch_size: int = 32,
    ) -> None:
        import pysam

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.geometries = load_payload_geometries(
            payloads_fasta if payloads_fasta is not None else DEFAULT_PAYLOADS_FASTA
        )
        self.batch_size = batch_size

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        fasta_path: str | Path,
        payloads_fasta: str | Path | None = None,
        device: str = "cuda",
        batch_size: int = 32,
        use_rc: bool = True,
        autocast: bool = True,
    ) -> "YorzoiMytkPredictor":
        return cls(
            Yorzoi.from_pretrained(
                hf_repo, device=device, use_rc=use_rc, autocast=autocast,
            ),
            fasta_path=fasta_path,
            payloads_fasta=payloads_fasta,
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
            range(0, len(contexts), self.batch_size), desc="Yorzoi MYTK"
        ):
            batch = contexts[bs : bs + self.batch_size]
            x = _torch.from_numpy(
                np.stack([
                    one_hot_encode_channels_first(c.window_seq).T for _, c in batch
                ])
            ).to(self.model.device)
            with _torch.no_grad():
                # (B, 162, OUTPUT_BINS*BIN_WIDTH) per-base raw counts.
                perbase = self.model.forward_tracks_perbase(x)
            for j, (row_idx, ctx) in enumerate(batch):
                base_idx = ctx.readout_base_positions
                if base_idx.size == 0:
                    continue  # readout outside crop → leave NaN
                base_t = _torch.as_tensor(
                    base_idx, device=self.model.device, dtype=_torch.long
                )
                per_track = perbase[j].index_select(1, base_t).sum(dim=1)  # (162,)
                scores[row_idx] = float(
                    per_track[YORZOI_PLUS_TRACK_START:YORZOI_PLUS_TRACK_END].mean().item()
                )

        return scores
