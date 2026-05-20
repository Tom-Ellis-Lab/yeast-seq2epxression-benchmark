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
from typing import TYPE_CHECKING, Any, Sequence

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
    YORZOI_MINUS_TRACK_IDS,
    YORZOI_PLUS_TRACK_IDS,
)

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

N_TRACKS_TOTAL = 162


class YorzoiWuPredictor(CassetteExpressionPredictor):
    def __init__(
        self,
        model: Any,  # yorzoi.model.borzoi.Borzoi
        fasta_path: str | Path,
        gtf_path: str | Path,
        cassette_fasta: str | Path | None = None,
        device: "str | torch.device" = "cuda",
        batch_size: int = 32,
        use_rc: bool = True,
        autocast: bool = True,
    ) -> None:
        import pysam
        import torch as _torch

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.genes = parse_gene_annotations(gtf_path)
        self.payload = load_cassette_payload(
            cassette_fasta if cassette_fasta is not None else DEFAULT_CASSETTE_FASTA
        )
        self.device = _torch.device(device)
        self.batch_size = batch_size
        self.use_rc = use_rc
        self.autocast = autocast
        self.model.to(self.device).eval()

        plus_ids = _torch.tensor(
            YORZOI_PLUS_TRACK_IDS, device=self.device, dtype=_torch.long
        )
        minus_ids = _torch.tensor(
            YORZOI_MINUS_TRACK_IDS, device=self.device, dtype=_torch.long
        )
        full_swap = _torch.empty(
            N_TRACKS_TOTAL, dtype=_torch.long, device=self.device
        )
        full_swap[plus_ids] = minus_ids
        full_swap[minus_ids] = plus_ids
        self._full_swap_idx = full_swap

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
        from yorzoi.model.borzoi import Borzoi

        model = Borzoi.from_pretrained(hf_repo)
        return cls(
            model, fasta_path, gtf_path, cassette_fasta,
            device, batch_size, use_rc=use_rc, autocast=autocast,
        )

    def _forward_full_tracks(self, x: "torch.Tensor") -> "torch.Tensor":
        """(B, SEQ_LEN, 4) → (B, 162, OUTPUT_BINS), RC-averaged with
        strand swap."""
        import torch as _torch

        ctx = (
            _torch.autocast(device_type="cuda")
            if self.autocast and self.device.type == "cuda"
            else _torch.amp.autocast(device_type="cpu", enabled=False)
        )
        with ctx:
            out_fwd = self.model(x)
        if not self.use_rc:
            return out_fwd
        x_rc = x.flip(dims=[1, 2])
        with ctx:
            out_rc = self.model(x_rc)
        out_rc_aligned = out_rc.index_select(1, self._full_swap_idx).flip(dims=[2])
        return 0.5 * (out_fwd + out_rc_aligned)

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
            ).to(self.device)
            with _torch.no_grad():
                pred = self._forward_full_tracks(x).float()  # (B, 162, bins)
            for j, (row_idx, ctx, strand) in enumerate(batch):
                bins_t = _torch.as_tensor(
                    ctx.rfp_bins, device=self.device, dtype=_torch.long
                )
                per_track = pred[j].index_select(1, bins_t).sum(dim=1)  # (162,)
                ts, te = (0, 81) if strand == "+" else (81, 162)
                scores[row_idx] = float(per_track[ts:te].mean().item())

        return scores
