"""Shorkie marginalized / native-position adapter for Rafi MPRA.

Inserts the 110 bp MPRA sequence at native yeast genome positions upstream
of 22 host genes.  For each (gene, offset) context:
  REF = native genomic window (precomputed once)
  ALT = same window with 110 bp replaced at the insertion site
  logSED = log2(alt_exon_sum + 1) - log2(ref_exon_sum + 1)

Aggregation: cross-track mean (T0 RNA-seq tracks, unstranded in Shorkie's
targets sheet) → mean across offsets → mean across host genes.

REF one-hots are cached on GPU so only ALT forward passes run per
sequence.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._marginalized_mpra import (
    INSERT_LEN,
    compute_insertion_contexts,
    extract_insert,
    reverse_complement,
)
from yeastbench.adapters._shorkie_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    SHORKIE_T0_RNA_SEQ_TRACK_IDS,
)
from yeastbench.adapters.protocols import MarginalizedSequenceExpressionPredictor
from yeastbench.models.shorkie import Shorkie

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)



class ShorkieMPRAMarginalizedPredictor(MarginalizedSequenceExpressionPredictor):
    def __init__(
        self,
        model: Shorkie,
        fasta_path: str | Path,
        gtf_path: str | Path,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        batch_size: int = 32,
        n_sample: int | None = None,
        seed: int = 42,
    ) -> None:
        import pysam
        import torch as _torch

        from yeastbench.adapters._genome import parse_gene_annotations

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.track_subset = list(track_subset)
        self.batch_size = batch_size
        self.n_sample = n_sample
        self.seed = seed

        self.genes = parse_gene_annotations(gtf_path)
        self.contexts = compute_insertion_contexts(
            gtf_path, self.fasta,
            seq_len=SEQ_LEN,
            crop_bp_each_side=CROP_BP_EACH_SIDE,
            bin_width=BIN_WIDTH,
            output_bins=OUTPUT_BINS,
        )
        log.info(
            "Marginalized MPRA (Shorkie): %d contexts across %d genes",
            len(self.contexts),
            len({c.gene_id for c in self.contexts}),
        )

        self._gene_ids = sorted({c.gene_id for c in self.contexts})
        self._gene_contexts: dict[str, list[int]] = {g: [] for g in self._gene_ids}
        for i, c in enumerate(self.contexts):
            self._gene_contexts[c.gene_id].append(i)

        # Cache REF one-hots as a single GPU tensor (n_ctx, 4, SEQ_LEN) — channels-first
        ref_np = np.zeros((len(self.contexts), 4, SEQ_LEN), dtype=np.float32)
        for i, ctx in enumerate(self.contexts):
            gene = self.genes[ctx.gene_id]
            seq = self.fasta.fetch(
                gene.chrom_roman, ctx.window_start, ctx.window_start + SEQ_LEN,
            ).upper()
            ref_np[i] = one_hot_encode_channels_first(seq)
        self._ref_ohs_gpu = _torch.from_numpy(ref_np).to(self.model.device)

        self._track_idx_t = _torch.tensor(
            self.track_subset, device=self.model.device, dtype=_torch.long
        )

        self._ref_exon_sums = self._precompute_baselines()  # GPU tensor (n_ctx, n_tracks)

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        fasta_path: str | Path,
        gtf_path: str | Path,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 32,
        use_rc: bool = True,
        n_sample: int | None = None,
        seed: int = 42,
    ) -> "ShorkieMPRAMarginalizedPredictor":
        return cls(
            Shorkie.from_checkpoints(
                params_path, checkpoint_paths, device=device, use_rc=use_rc,
            ),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            track_subset=list(track_subset),
            batch_size=batch_size,
            n_sample=n_sample,
            seed=seed,
        )

    def _forward_avg(self, x: "torch.Tensor") -> "torch.Tensor":
        """Wrapper-level shortcut: ensemble + RC + per-fold track mean →
        (B, OUTPUT_BINS)."""
        return self.model.forward_track_mean_binned(x, self._track_idx_t)

    def _precompute_baselines(self) -> "torch.Tensor":
        """Returns (n_ctx,) GPU tensor of REF exon-bin sums (cross-track averaged)."""
        import torch as _torch

        n_ctx = len(self.contexts)
        ref_sums = _torch.zeros(n_ctx, device=self.model.device, dtype=_torch.float32)

        for batch_start in tqdm(
            range(0, n_ctx, self.batch_size), desc="Shorkie REF baseline"
        ):
            batch_end = min(batch_start + self.batch_size, n_ctx)
            x = self._ref_ohs_gpu[batch_start:batch_end]

            with _torch.no_grad():
                cov = self._forward_avg(x)  # (B, OUTPUT_BINS)

            for i in range(batch_end - batch_start):
                ctx = self.contexts[batch_start + i]
                if ctx.exon_bins.size == 0:
                    continue
                bins_t = _torch.as_tensor(ctx.exon_bins, device=self.model.device, dtype=_torch.long)
                ref_sums[batch_start + i] = cov[i].index_select(0, bins_t).sum()

        return ref_sums

    def _build_alt_batch_gpu(
        self, insert_fwd_oh: "torch.Tensor", insert_rc_oh: "torch.Tensor",
        batch_start: int, batch_end: int,
    ) -> "torch.Tensor":
        """Construct ALT one-hots on GPU by copying REFs and splicing the insert.

        Channels-first layout: (B, 4, SEQ_LEN).
        """
        alt = self._ref_ohs_gpu[batch_start:batch_end].clone()
        for i in range(batch_end - batch_start):
            ctx = self.contexts[batch_start + i]
            s = ctx.insert_start_in_window
            oh = insert_rc_oh if ctx.gene_strand == "-" else insert_fwd_oh
            alt[i, :, s : s + INSERT_LEN] = oh
        return alt

    def _score_one_sequence(self, insert_seq: str) -> float:
        import torch as _torch

        insert_fwd_np = one_hot_encode_channels_first(insert_seq.upper())  # (4, L)
        insert_rc_np = one_hot_encode_channels_first(
            reverse_complement(insert_seq).upper()
        )
        insert_fwd_oh = _torch.from_numpy(insert_fwd_np).to(self.model.device)
        insert_rc_oh = _torch.from_numpy(insert_rc_np).to(self.model.device)

        n_ctx = len(self.contexts)
        alt_exon_sums = _torch.zeros(n_ctx, device=self.model.device, dtype=_torch.float32)

        for batch_start in range(0, n_ctx, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_ctx)
            x = self._build_alt_batch_gpu(insert_fwd_oh, insert_rc_oh, batch_start, batch_end)

            with _torch.no_grad():
                cov = self._forward_avg(x)

            for i in range(batch_end - batch_start):
                ctx = self.contexts[batch_start + i]
                if ctx.exon_bins.size == 0:
                    continue
                bins_t = _torch.as_tensor(ctx.exon_bins, device=self.model.device, dtype=_torch.long)
                alt_exon_sums[batch_start + i] = cov[i].index_select(0, bins_t).sum()

        # logSED per context (already cross-track-averaged by _forward_avg)
        logsed_per_ctx = (
            _torch.log2(alt_exon_sums + 1.0) - _torch.log2(self._ref_exon_sums + 1.0)
        )

        gene_means: list[float] = []
        for gene_id in self._gene_ids:
            idx = self._gene_contexts[gene_id]
            gene_means.append(float(logsed_per_ctx[idx].mean()))

        return float(np.mean(gene_means))

    def predict_marginalized_expressions(self, seqs: Sequence[str]) -> np.ndarray:
        n = len(seqs)
        scores = np.full(n, np.nan, dtype=np.float64)

        if self.n_sample is not None and self.n_sample < n:
            rng = np.random.default_rng(self.seed)
            sample_idx = rng.choice(n, size=self.n_sample, replace=False)
        else:
            sample_idx = np.arange(n)

        for idx in tqdm(sample_idx, desc="Shorkie marginalized"):
            insert = extract_insert(seqs[idx])
            scores[idx] = self._score_one_sequence(insert)

        return scores
