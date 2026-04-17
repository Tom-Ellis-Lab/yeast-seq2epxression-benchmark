"""Shorkie marginalized adapter for the Shalem MPRA terminator benchmark.

For each 150 bp test oligo, insert it + a 300 bp CYC1 no-termination filler
immediately downstream of each of 22 host genes' stop codons, predict
expression (mean logSED over T0 RNA-seq tracks, summed over host-gene
exon bins), and return the mean logSED across host genes as the per-
sequence prediction.

REF baselines (native context, no replacement) are pre-computed at init
and cached on GPU so only ALT forward passes run per test sequence.
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
from yeastbench.adapters._shalem_scaffold import (
    INSERT_LEN,
    REPLACE_LEN,
    ShalemInsertionContext,
    assemble_replacement,
    build_filler,
    compute_insertion_contexts,
    load_host_genes,
)
from yeastbench.adapters.protocols import TerminatorMarginalizedExpressionPredictor
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


class ShorkieShalemPredictor(TerminatorMarginalizedExpressionPredictor):
    def __init__(
        self,
        models: list["Shorkie"],
        fasta_path: str | Path,
        gtf_path: str | Path,
        host_genes_json: str | Path | None = None,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: "str | torch.device" = "cuda",
        batch_size: int = 32,
        use_rc: bool = True,
        n_sample: int | None = None,
        seed: int = 42,
    ) -> None:
        import pysam
        import torch as _torch

        if not models:
            raise ValueError("Must provide at least one model fold")
        self.models = models
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.track_subset = list(track_subset)
        self.device = _torch.device(device)
        self.batch_size = batch_size
        self.use_rc = use_rc
        self.n_sample = n_sample
        self.seed = seed
        for m in self.models:
            m.to(self.device).eval()

        self.genes = parse_gene_annotations(gtf_path)
        host_genes = load_host_genes(
            Path(host_genes_json) if host_genes_json is not None
            else __import__("yeastbench.adapters._shalem_scaffold", fromlist=["DEFAULT_HOST_GENES_JSON"]).DEFAULT_HOST_GENES_JSON
        )
        self.contexts: list[ShalemInsertionContext] = compute_insertion_contexts(
            host_genes, self.genes, self.fasta,
            seq_len=SEQ_LEN,
            crop_bp_each_side=CROP_BP_EACH_SIDE,
            bin_width=BIN_WIDTH,
            output_bins=OUTPUT_BINS,
        )
        log.info(
            "Shalem marginalized (Shorkie): %d host-gene contexts", len(self.contexts)
        )

        # 300 bp CYC1 no-termination filler (same across all test seqs)
        self.filler: str = build_filler(self.fasta, self.genes)
        assert len(self.filler) == 300

        # Cache REF one-hots on GPU: (n_ctx, 4, SEQ_LEN) channels-first
        ref_np = np.zeros((len(self.contexts), 4, SEQ_LEN), dtype=np.float32)
        for i, ctx in enumerate(self.contexts):
            gene = self.genes[ctx.gene_id]
            seq = self.fasta.fetch(
                gene.chrom_roman, ctx.window_start, ctx.window_start + SEQ_LEN,
            ).upper()
            ref_np[i] = one_hot_encode_channels_first(seq)
        self._ref_ohs_gpu = _torch.from_numpy(ref_np).to(self.device)

        self._track_idx_t = _torch.tensor(
            self.track_subset, device=self.device, dtype=_torch.long
        )

        # Precompute REF cross-track-mean exon sums
        self._ref_exon_sums = self._precompute_baselines()  # (n_ctx,) GPU tensor

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        fasta_path: str | Path,
        gtf_path: str | Path,
        host_genes_json: str | Path | None = None,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 32,
        use_rc: bool = True,
        n_sample: int | None = None,
        seed: int = 42,
    ) -> "ShorkieShalemPredictor":
        from yeastbench.models.shorkie import Shorkie

        with open(params_path) as f:
            config = json.load(f)
        models = [
            Shorkie.from_tf_checkpoint(config["model"], str(p))
            for p in checkpoint_paths
        ]
        return cls(
            models, fasta_path, gtf_path, host_genes_json, list(track_subset),
            device, batch_size, use_rc=use_rc,
            n_sample=n_sample, seed=seed,
        )

    # ── Forward pass ─────────────────────────────────────────

    def _forward_avg(self, x: "torch.Tensor") -> "torch.Tensor":
        """Run all folds (+ optional RC) on *x*, return (B, OUTPUT_BINS):
        cross-track-mean prediction per bin, averaged across folds."""
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

    def _precompute_baselines(self) -> "torch.Tensor":
        import torch as _torch

        n_ctx = len(self.contexts)
        ref_sums = _torch.zeros(n_ctx, device=self.device, dtype=_torch.float32)

        for batch_start in tqdm(
            range(0, n_ctx, self.batch_size), desc="Shorkie REF baseline"
        ):
            batch_end = min(batch_start + self.batch_size, n_ctx)
            x = self._ref_ohs_gpu[batch_start:batch_end]
            with _torch.no_grad():
                cov = self._forward_avg(x)  # (B, OUTPUT_BINS)
            for i in range(batch_end - batch_start):
                ctx = self.contexts[batch_start + i]
                bins_t = _torch.as_tensor(ctx.exon_bins, device=self.device, dtype=_torch.long)
                ref_sums[batch_start + i] = cov[i].index_select(0, bins_t).sum()

        return ref_sums

    def _build_alt_batch_gpu(
        self, insert_oh_fwd: "torch.Tensor", insert_oh_rc: "torch.Tensor",
        batch_start: int, batch_end: int,
    ) -> "torch.Tensor":
        """Clone REF batch and splice the 450 bp replacement per context.

        ``insert_oh_fwd`` is the one-hot of (insert + filler) in + strand
        orientation; ``insert_oh_rc`` is its reverse complement (used for
        - strand host genes).
        """
        alt = self._ref_ohs_gpu[batch_start:batch_end].clone()  # (B, 4, SEQ_LEN)
        for i in range(batch_end - batch_start):
            ctx = self.contexts[batch_start + i]
            s = ctx.replace_start_in_window
            oh = insert_oh_rc if ctx.gene_strand == "-" else insert_oh_fwd
            alt[i, :, s : s + REPLACE_LEN] = oh
        return alt

    def _score_one_sequence(self, oligo_150bp: str) -> float:
        import torch as _torch

        # (insert + filler) in + strand orientation
        rep_fwd = assemble_replacement(oligo_150bp, self.filler, "+")
        rep_rc = assemble_replacement(oligo_150bp, self.filler, "-")
        insert_oh_fwd = _torch.from_numpy(
            one_hot_encode_channels_first(rep_fwd)
        ).to(self.device)
        insert_oh_rc = _torch.from_numpy(
            one_hot_encode_channels_first(rep_rc)
        ).to(self.device)

        n_ctx = len(self.contexts)
        alt_exon_sums = _torch.zeros(n_ctx, device=self.device, dtype=_torch.float32)

        for batch_start in range(0, n_ctx, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_ctx)
            x = self._build_alt_batch_gpu(
                insert_oh_fwd, insert_oh_rc, batch_start, batch_end,
            )
            with _torch.no_grad():
                cov = self._forward_avg(x)
            for i in range(batch_end - batch_start):
                ctx = self.contexts[batch_start + i]
                bins_t = _torch.as_tensor(ctx.exon_bins, device=self.device, dtype=_torch.long)
                alt_exon_sums[batch_start + i] = cov[i].index_select(0, bins_t).sum()

        # logSED per context, then mean across 22 host genes
        logsed_per_ctx = (
            _torch.log2(alt_exon_sums + 1.0) - _torch.log2(self._ref_exon_sums + 1.0)
        )
        return float(logsed_per_ctx.mean().item())

    def predict_terminator_marginalized(
        self, seqs: Sequence[str]
    ) -> np.ndarray:
        n = len(seqs)
        scores = np.full(n, np.nan, dtype=np.float64)

        if self.n_sample is not None and self.n_sample < n:
            rng = np.random.default_rng(self.seed)
            sample_idx = rng.choice(n, size=self.n_sample, replace=False)
        else:
            sample_idx = np.arange(n)

        for idx in tqdm(sample_idx, desc="Shorkie Shalem"):
            scores[idx] = self._score_one_sequence(seqs[idx])

        return scores
