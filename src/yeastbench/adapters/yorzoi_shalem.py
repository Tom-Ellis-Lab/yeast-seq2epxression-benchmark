"""Yorzoi marginalized adapter for the Shalem MPRA terminator benchmark.

Same protocol as the Shorkie version but with Yorzoi's 4992 bp input,
300 output bins at 10 bp/bin, and strand-matched track aggregation
(+ strand host genes → tracks 0–80, − strand host genes → tracks 81–161).
RC averaging swaps strand tracks as in the Rafi / eQTL Yorzoi adapters.
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
from yeastbench.adapters._shalem_scaffold import (
    REPLACE_LEN,
    ShalemInsertionContext,
    assemble_replacement,
    build_filler,
    compute_insertion_contexts,
    load_host_genes,
)
from yeastbench.adapters.protocols import TerminatorMarginalizedExpressionPredictor
from yeastbench.adapters.yorzoi_eqtl import (
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


class YorzoiShalemPredictor(TerminatorMarginalizedExpressionPredictor):
    def __init__(
        self,
        model: Any,  # yorzoi.model.borzoi.Borzoi
        fasta_path: str | Path,
        gtf_path: str | Path,
        host_genes_json: str | Path | None = None,
        device: "str | torch.device" = "cuda",
        batch_size: int = 64,
        use_rc: bool = True,
        autocast: bool = True,
        n_sample: int | None = None,
        seed: int = 42,
    ) -> None:
        import pysam
        import torch as _torch

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.device = _torch.device(device)
        self.batch_size = batch_size
        self.use_rc = use_rc
        self.autocast = autocast
        self.n_sample = n_sample
        self.seed = seed
        self.model.to(self.device).eval()

        self.genes = parse_gene_annotations(gtf_path)
        host_genes = load_host_genes(
            Path(host_genes_json) if host_genes_json is not None
            else __import__(
                "yeastbench.adapters._shalem_scaffold",
                fromlist=["DEFAULT_HOST_GENES_JSON"],
            ).DEFAULT_HOST_GENES_JSON
        )
        self.contexts: list[ShalemInsertionContext] = compute_insertion_contexts(
            host_genes, self.genes, self.fasta,
            seq_len=SEQ_LEN,
            crop_bp_each_side=CROP_BP_EACH_SIDE,
            bin_width=BIN_WIDTH,
            output_bins=OUTPUT_BINS,
        )
        log.info(
            "Shalem marginalized (Yorzoi): %d host-gene contexts", len(self.contexts)
        )

        self.filler: str = build_filler(self.fasta, self.genes)
        assert len(self.filler) == 300

        # Full RC-swap index (swap + ↔ − strand tracks) for RC averaging
        plus_ids = _torch.tensor(YORZOI_PLUS_TRACK_IDS, device=self.device, dtype=_torch.long)
        minus_ids = _torch.tensor(YORZOI_MINUS_TRACK_IDS, device=self.device, dtype=_torch.long)
        full_swap = _torch.empty(N_TRACKS_TOTAL, dtype=_torch.long, device=self.device)
        full_swap[plus_ids] = minus_ids
        full_swap[minus_ids] = plus_ids
        self._full_swap_idx = full_swap

        # Per-context strand-matched track slice
        self._ctx_track_start: list[int] = []
        self._ctx_track_end: list[int] = []
        for c in self.contexts:
            if c.gene_strand == "+":
                self._ctx_track_start.append(0)
                self._ctx_track_end.append(81)
            else:
                self._ctx_track_start.append(81)
                self._ctx_track_end.append(162)

        # Cache REF one-hots on GPU (n_ctx, SEQ_LEN, 4) channels-last
        ref_np = np.zeros((len(self.contexts), SEQ_LEN, 4), dtype=np.float32)
        for i, ctx in enumerate(self.contexts):
            gene = self.genes[ctx.gene_id]
            seq = self.fasta.fetch(
                gene.chrom_roman, ctx.window_start, ctx.window_start + SEQ_LEN,
            ).upper()
            ref_np[i] = one_hot_encode_channels_first(seq).T
        self._ref_ohs_gpu = _torch.from_numpy(ref_np).to(self.device)

        # (n_ctx, 162) REF exon sums
        self._ref_exon_sums = self._precompute_baselines()

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        fasta_path: str | Path,
        gtf_path: str | Path,
        host_genes_json: str | Path | None = None,
        device: str = "cuda",
        batch_size: int = 64,
        use_rc: bool = True,
        autocast: bool = True,
        n_sample: int | None = None,
        seed: int = 42,
    ) -> "YorzoiShalemPredictor":
        from yorzoi.model.borzoi import Borzoi

        model = Borzoi.from_pretrained(hf_repo)
        return cls(
            model, fasta_path, gtf_path, host_genes_json,
            device, batch_size, use_rc=use_rc, autocast=autocast,
            n_sample=n_sample, seed=seed,
        )

    # ── Forward pass ────────────────────────────────────────

    def _forward_full_tracks(self, x: "torch.Tensor") -> "torch.Tensor":
        """Run forward (+ RC averaging with strand swap). Returns (B, 162, OUTPUT_BINS)."""
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

    def _precompute_baselines(self) -> "torch.Tensor":
        import torch as _torch

        n_ctx = len(self.contexts)
        ref_sums = _torch.zeros(n_ctx, N_TRACKS_TOTAL, device=self.device, dtype=_torch.float32)

        for batch_start in tqdm(
            range(0, n_ctx, self.batch_size), desc="Yorzoi REF baseline"
        ):
            batch_end = min(batch_start + self.batch_size, n_ctx)
            x = self._ref_ohs_gpu[batch_start:batch_end]
            with _torch.no_grad():
                pred = self._forward_full_tracks(x).float()  # (B, 162, bins)
            for i in range(batch_end - batch_start):
                ctx = self.contexts[batch_start + i]
                bins_t = _torch.as_tensor(ctx.exon_bins, device=self.device, dtype=_torch.long)
                ref_sums[batch_start + i] = pred[i].index_select(1, bins_t).sum(dim=1)

        return ref_sums

    def _build_alt_batch_gpu(
        self, insert_oh_fwd: "torch.Tensor", insert_oh_rc: "torch.Tensor",
        batch_start: int, batch_end: int,
    ) -> "torch.Tensor":
        """Channels-last (L, 4) ALT batch for Yorzoi."""
        alt = self._ref_ohs_gpu[batch_start:batch_end].clone()  # (B, L, 4)
        for i in range(batch_end - batch_start):
            ctx = self.contexts[batch_start + i]
            s = ctx.replace_start_in_window
            oh = insert_oh_rc if ctx.gene_strand == "-" else insert_oh_fwd
            alt[i, s : s + REPLACE_LEN, :] = oh
        return alt

    def _score_one_sequence(self, oligo_150bp: str) -> float:
        import torch as _torch

        rep_fwd = assemble_replacement(oligo_150bp, self.filler, "+")
        rep_rc = assemble_replacement(oligo_150bp, self.filler, "-")
        # Yorzoi needs channels-last (L, 4)
        insert_oh_fwd = _torch.from_numpy(
            one_hot_encode_channels_first(rep_fwd).T
        ).to(self.device)
        insert_oh_rc = _torch.from_numpy(
            one_hot_encode_channels_first(rep_rc).T
        ).to(self.device)

        n_ctx = len(self.contexts)
        alt_exon_sums = _torch.zeros(n_ctx, N_TRACKS_TOTAL, device=self.device, dtype=_torch.float32)

        for batch_start in range(0, n_ctx, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_ctx)
            x = self._build_alt_batch_gpu(
                insert_oh_fwd, insert_oh_rc, batch_start, batch_end,
            )
            with _torch.no_grad():
                pred = self._forward_full_tracks(x).float()
            for i in range(batch_end - batch_start):
                ctx = self.contexts[batch_start + i]
                bins_t = _torch.as_tensor(ctx.exon_bins, device=self.device, dtype=_torch.long)
                alt_exon_sums[batch_start + i] = pred[i].index_select(1, bins_t).sum(dim=1)

        # Cross-track mean over strand-matched 81 tracks BEFORE log (logSED_agg convention)
        alt_mean = _torch.zeros(n_ctx, device=self.device, dtype=_torch.float32)
        ref_mean = _torch.zeros(n_ctx, device=self.device, dtype=_torch.float32)
        for i in range(n_ctx):
            ts, te = self._ctx_track_start[i], self._ctx_track_end[i]
            alt_mean[i] = alt_exon_sums[i, ts:te].mean()
            ref_mean[i] = self._ref_exon_sums[i, ts:te].mean()

        logsed_per_ctx = _torch.log2(alt_mean + 1.0) - _torch.log2(ref_mean + 1.0)
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

        for idx in tqdm(sample_idx, desc="Yorzoi Shalem"):
            scores[idx] = self._score_one_sequence(seqs[idx])

        return scores
