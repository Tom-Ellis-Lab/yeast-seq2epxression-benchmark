"""Yorzoi adapter for the Wu et al. RFP-insertion position-effect task.

Same construct as the Shorkie Wu adapter (native genome with the ORF
replaced by the constant cassette, mCherry-CDS readout, no REF baseline)
but with Yorzoi's 4992 bp input, 300 output bins, and strand-matched
track aggregation (+ strand ORF → tracks 0–80, − strand ORF → 81–161).
RC averaging swaps strand tracks as in the other Yorzoi adapters.
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
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
)
from yeastbench.models.yorzoi import Yorzoi

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


class YorzoiWuPredictor(CassetteExpressionPredictor):
    def __init__(
        self,
        model: Yorzoi,
        fasta_path: str | Path,
        gtf_path: str | Path,
        cassette_fasta: str | Path | None = None,
        batch_size: int = 32,
    ) -> None:
        import pysam

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.genes = parse_gene_annotations(gtf_path)
        self.payload = load_cassette_payload(
            cassette_fasta if cassette_fasta is not None else DEFAULT_CASSETTE_FASTA
        )
        self.batch_size = batch_size

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        fasta_path: str | Path,
        gtf_path: str | Path,
        cassette_fasta: str | Path | None = None,
        device: str = "cuda",
        batch_size: int = 32,
        use_rc: bool = True,
        autocast: bool = True,
    ) -> "YorzoiWuPredictor":
        return cls(
            Yorzoi.from_pretrained(
                hf_repo, device=device, use_rc=use_rc, autocast=autocast,
            ),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            cassette_fasta=cassette_fasta,
            batch_size=batch_size,
        )

    def predict_expressions(self, loci: Sequence[WuLocus]) -> np.ndarray:
        import torch as _torch

        scores = np.full(len(loci), np.nan, dtype=np.float64)
        contexts: list[tuple[int, WuInsertionContext, str]] = []
        for i, locus in enumerate(loci):
            ctx = build_insertion_context(
                locus, self.payload, self.fasta,
                SEQ_LEN, CROP_BP_EACH_SIDE, BIN_WIDTH, OUTPUT_BINS,
            )
            if ctx is not None:
                contexts.append((i, ctx, locus.strand))

        for bs in tqdm(
            range(0, len(contexts), self.batch_size), desc="Yorzoi Wu"
        ):
            batch = contexts[bs : bs + self.batch_size]
            x = _torch.from_numpy(
                np.stack([
                    one_hot_encode_channels_first(c.window_seq).T
                    for _, c, _ in batch
                ])
            ).to(self.model.device)
            with _torch.no_grad():
                pred = self.model.forward_tracks_binned(x).float()  # (B, 162, bins)
            for j, (row_idx, ctx, strand) in enumerate(batch):
                bins_t = _torch.as_tensor(
                    ctx.rfp_bins, device=self.model.device, dtype=_torch.long
                )
                per_track = pred[j].index_select(1, bins_t).sum(dim=1)  # (162,)
                ts, te = (0, 81) if strand == "+" else (81, 162)
                scores[row_idx] = float(per_track[ts:te].mean().item())

        return scores
