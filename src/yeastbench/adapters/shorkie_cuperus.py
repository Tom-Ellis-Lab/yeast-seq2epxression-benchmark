"""Shorkie adapter for the Cuperus 5'-UTR reporter task.

For each UTR, assemble the ``CYC1``pr + UTR + ``HIS3`` + ``CYC1``term construct,
embed it at the background locus (default ``HIS3``'s own, for in-distribution
window flanks), forward-pass (8 folds, optional RC), take the cross-track mean
over the T0 RNA-seq tracks, and sum over the ``HIS3``-ORF per-base positions.
Absolute readout — no REF baseline (Spearman is the scale-free headline).

Contexts are built and forwarded in streaming batches: the random library is
~489k UTRs, so materializing every 16 kb window upfront would blow memory. The
construct is scored in the single fixed ``HIS3`` context (no marginalization).
See ``benchmarks/cuperus_mpra_5utr.md``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._cuperus_scaffold import (
    HIS3_BACKGROUND,
    CuperusConstruct,
    build_context,
)
from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._shorkie_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    SHORKIE_T0_RNA_SEQ_TRACK_IDS,
)
from yeastbench.adapters.protocols import FivePrimeUtrReporterExpressionPredictor
from yeastbench.models.shorkie import Shorkie

log = logging.getLogger(__name__)


class ShorkieCuperusPredictor(FivePrimeUtrReporterExpressionPredictor):
    def __init__(
        self,
        model: Shorkie,
        fasta_path: str | Path,
        construct_json: str | Path | None = None,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        batch_size: int = 16,
    ) -> None:
        import pysam
        import torch as _torch

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.construct = CuperusConstruct.from_json(construct_json) if construct_json else CuperusConstruct.from_json()
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
        construct_json: str | Path | None = None,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 16,
        use_rc: bool = True,
    ) -> "ShorkieCuperusPredictor":
        return cls(
            Shorkie.from_checkpoints(
                params_path, checkpoint_paths, device=device, use_rc=use_rc,
            ),
            fasta_path=fasta_path,
            construct_json=construct_json,
            track_subset=list(track_subset),
            batch_size=batch_size,
        )

    def _contexts(self, utrs: Sequence[str]) -> Iterator[tuple[int, object]]:
        for ui, utr in enumerate(utrs):
            ctx = build_context(
                self.construct, utr, HIS3_BACKGROUND, self.fasta,
                seq_len=SEQ_LEN, crop_bp_each_side=CROP_BP_EACH_SIDE,
                bin_width=BIN_WIDTH, output_bins=OUTPUT_BINS,
            )
            if ctx is not None and ctx.readout_base_positions.size:
                yield ui, ctx

    def predict_utr_expressions(self, utrs: Sequence[str]) -> np.ndarray:
        import torch as _torch

        n = len(utrs)
        out = np.full(n, np.nan, dtype=np.float64)  # UTRs with no usable context stay NaN

        def flush(buf: list) -> None:
            x = _torch.from_numpy(
                np.stack([one_hot_encode_channels_first(c.window_seq) for _, c in buf])
            ).to(self.model.device)
            with _torch.no_grad():
                cov = self.model.forward_track_mean_perbase(x, self._track_idx_t)
            for j, (ui, ctx) in enumerate(buf):
                base_t = _torch.as_tensor(
                    ctx.readout_base_positions, device=self.model.device, dtype=_torch.long
                )
                out[ui] = float(cov[j].index_select(0, base_t).sum().item())

        buf: list = []
        for item in tqdm(self._contexts(utrs), total=n, desc="Shorkie Cuperus"):
            buf.append(item)
            if len(buf) >= self.batch_size:
                flush(buf)
                buf = []
        if buf:
            flush(buf)
        return out
