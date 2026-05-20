"""Shorkie adapter for the Wu et al. RFP-insertion position-effect task.

For each deletion locus, build a 16 kb input = native R64-1-1 sequence
with the ORF replaced by the constant cassette payload, forward-pass
(8 folds, optional RC averaging), take the cross-track mean over the T0
RNA-seq tracks, and sum over the mCherry-CDS bins.  This is an
**absolute** readout — there is no REF baseline (no "reference" without
the cassette).  See ``benchmarks/wu_rfpins.md``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._genome import (
    one_hot_encode_channels_first,
    parse_gene_annotations,
)
from yeastbench.adapters._wu_scaffold import (
    DEFAULT_CASSETTE_FASTA,
    WuInsertionContext,
    WuLocus,
    build_insertion_context,
    load_cassette_payload,
)
from yeastbench.adapters.protocols import CassetteExpressionPredictor
from yeastbench.adapters._shorkie_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    SHORKIE_T0_RNA_SEQ_TRACK_IDS,
)
from yeastbench.models.shorkie import Shorkie

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


class ShorkieWuPredictor(CassetteExpressionPredictor):
    def __init__(
        self,
        model: Shorkie,
        fasta_path: str | Path,
        gtf_path: str | Path,
        cassette_fasta: str | Path | None = None,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        batch_size: int = 16,
    ) -> None:
        import pysam
        import torch as _torch

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.genes = parse_gene_annotations(gtf_path)
        self.payload = load_cassette_payload(
            cassette_fasta if cassette_fasta is not None else DEFAULT_CASSETTE_FASTA
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
        gtf_path: str | Path,
        cassette_fasta: str | Path | None = None,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 16,
        use_rc: bool = True,
    ) -> "ShorkieWuPredictor":
        return cls(
            Shorkie.from_checkpoints(
                params_path, checkpoint_paths, device=device, use_rc=use_rc,
            ),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            cassette_fasta=cassette_fasta,
            track_subset=list(track_subset),
            batch_size=batch_size,
        )

    def predict_expressions(self, loci: Sequence[WuLocus]) -> np.ndarray:
        import torch as _torch

        scores = np.full(len(loci), np.nan, dtype=np.float64)
        contexts: list[tuple[int, WuInsertionContext]] = []
        for i, locus in enumerate(loci):
            ctx = build_insertion_context(
                locus, self.payload, self.fasta,
                SEQ_LEN, CROP_BP_EACH_SIDE, BIN_WIDTH, OUTPUT_BINS,
            )
            if ctx is not None:
                contexts.append((i, ctx))

        for bs in tqdm(
            range(0, len(contexts), self.batch_size), desc="Shorkie Wu"
        ):
            batch = contexts[bs : bs + self.batch_size]
            x = _torch.from_numpy(
                np.stack([
                    one_hot_encode_channels_first(c.window_seq) for _, c in batch
                ])
            ).to(self.model.device)
            with _torch.no_grad():
                # (B, OUTPUT_BINS) — ensemble + RC + per-fold track mean
                cov = self.model.forward_track_mean_binned(x, self._track_idx_t)
            for j, (row_idx, ctx) in enumerate(batch):
                bins_t = _torch.as_tensor(
                    ctx.rfp_bins, device=self.model.device, dtype=_torch.long
                )
                scores[row_idx] = float(cov[j].index_select(0, bins_t).sum().item())

        return scores
