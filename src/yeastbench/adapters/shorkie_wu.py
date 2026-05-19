"""Shorkie adapter for the Wu et al. RFP-insertion position-effect task.

For each deletion locus, build a 16 kb input = native R64-1-1 sequence
with the ORF replaced by the constant cassette payload, forward-pass
(8 folds, optional RC averaging), take the cross-track mean over the T0
RNA-seq tracks, and sum over the mCherry-CDS bins.  This is an
**absolute** readout — there is no REF baseline (no "reference" without
the cassette).  See ``benchmarks/wu_rfpins.md``.
"""
from __future__ import annotations

import json
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
from yeastbench.adapters.shorkie_eqtl import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
)
from yeastbench.adapters.shorkie_mpra_marginalized import (
    SHORKIE_T0_RNA_SEQ_TRACK_IDS,
)

if TYPE_CHECKING:
    import torch
    from yeastbench.models.shorkie import Shorkie

log = logging.getLogger(__name__)


class ShorkieWuPredictor(CassetteExpressionPredictor):
    def __init__(
        self,
        models: list["Shorkie"],
        fasta_path: str | Path,
        gtf_path: str | Path,
        cassette_fasta: str | Path | None = None,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: "str | torch.device" = "cuda",
        batch_size: int = 16,
        use_rc: bool = True,
    ) -> None:
        import pysam
        import torch as _torch

        if not models:
            raise ValueError("Must provide at least one model fold")
        self.models = models
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.genes = parse_gene_annotations(gtf_path)
        self.payload = load_cassette_payload(
            cassette_fasta if cassette_fasta is not None else DEFAULT_CASSETTE_FASTA
        )
        self.track_subset = list(track_subset)
        self.device = _torch.device(device)
        self.batch_size = batch_size
        self.use_rc = use_rc
        for m in self.models:
            m.to(self.device).eval()
        self._track_idx_t = _torch.tensor(
            self.track_subset, device=self.device, dtype=_torch.long
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
        from yeastbench.models.shorkie import Shorkie

        with open(params_path) as f:
            config = json.load(f)
        models = [
            Shorkie.from_tf_checkpoint(config["model"], str(p))
            for p in checkpoint_paths
        ]
        return cls(
            models, fasta_path, gtf_path, cassette_fasta,
            list(track_subset), device, batch_size, use_rc=use_rc,
        )

    def _forward_avg(self, x: "torch.Tensor") -> "torch.Tensor":
        """(B, 4, SEQ_LEN) → (B, OUTPUT_BINS): cross-track-mean per bin,
        averaged across folds (and forward/RC)."""
        import torch as _torch

        B = x.shape[0]
        acc = _torch.zeros(B, OUTPUT_BINS, device=self.device, dtype=_torch.float32)
        x_rc = x.flip(dims=[1, 2]) if self.use_rc else None
        for m in self.models:
            out = m(x).index_select(2, self._track_idx_t)  # (B, bins, n_tracks)
            if self.use_rc:
                out_rc = m(x_rc).index_select(2, self._track_idx_t).flip(dims=[1])
                out = 0.5 * (out + out_rc)
            acc.add_(out.mean(dim=2))
        acc.div_(len(self.models))
        return acc

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
            ).to(self.device)
            with _torch.no_grad():
                cov = self._forward_avg(x)  # (B, OUTPUT_BINS)
            for j, (row_idx, ctx) in enumerate(batch):
                bins_t = _torch.as_tensor(
                    ctx.rfp_bins, device=self.device, dtype=_torch.long
                )
                scores[row_idx] = float(cov[j].index_select(0, bins_t).sum().item())

        return scores
