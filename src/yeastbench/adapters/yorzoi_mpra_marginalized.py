"""Yorzoi marginalized / native-position adapter for Rafi MPRA.

Inserts the 110 bp MPRA sequence at native yeast genome positions
upstream of 22 host genes and computes logSED (log fold-change in
predicted expression) between the native REF and the edited ALT.

Yorzoi's 162 output tracks split into strand pairs (0-80 = ``+`` strand,
81-161 = ``-`` strand; same 81 samples measured on each strand).  For
each host gene we aggregate over the **strand-matched** 81 tracks: a
positive-strand host gene uses tracks 0-80, a negative-strand host gene
uses tracks 81-161.  This matters because Yorzoi's training made tracks
strand-specific — using ``+`` tracks for a ``-`` strand gene would pick
up antisense/noise signal that can anti-correlate with the real
regulatory effect.

REF one-hots are cached on the GPU so that only ALT forward passes
happen per test sequence.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._genome import one_hot_encode_channels_first, parse_gene_annotations
from yeastbench.adapters._marginalized_mpra import (
    INSERT_LEN,
    compute_insertion_contexts,
    extract_insert,
    reverse_complement,
)
from yeastbench.adapters.protocols import MarginalizedSequenceExpressionPredictor
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

N_TRACKS_TOTAL = 162  # 81 plus-strand + 81 minus-strand


class YorzoiMPRAMarginalizedPredictor(MarginalizedSequenceExpressionPredictor):
    def __init__(
        self,
        model: Any,  # yorzoi.model.borzoi.Borzoi
        fasta_path: str | Path,
        gtf_path: str | Path,
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
        self.contexts = compute_insertion_contexts(
            gtf_path, self.fasta,
            seq_len=SEQ_LEN,
            crop_bp_each_side=CROP_BP_EACH_SIDE,
            bin_width=BIN_WIDTH,
            output_bins=OUTPUT_BINS,
        )
        log.info(
            "Marginalized MPRA (Yorzoi): %d contexts across %d genes",
            len(self.contexts),
            len({c.gene_id for c in self.contexts}),
        )

        self._gene_ids = sorted({c.gene_id for c in self.contexts})
        self._gene_contexts: dict[str, list[int]] = {g: [] for g in self._gene_ids}
        for i, c in enumerate(self.contexts):
            self._gene_contexts[c.gene_id].append(i)

        # Precomputed strand masks for full RC swap (all 162 tracks)
        plus_ids = _torch.tensor(YORZOI_PLUS_TRACK_IDS, device=self.device, dtype=_torch.long)
        minus_ids = _torch.tensor(YORZOI_MINUS_TRACK_IDS, device=self.device, dtype=_torch.long)
        # full_swap_idx[i] gives the paired-strand track for track i
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

        # Cache REF one-hots as one GPU tensor (n_ctx, SEQ_LEN, 4)
        ref_np = np.zeros((len(self.contexts), SEQ_LEN, 4), dtype=np.float32)
        for i, ctx in enumerate(self.contexts):
            gene = self.genes[ctx.gene_id]
            seq = self.fasta.fetch(
                gene.chrom_roman, ctx.window_start, ctx.window_start + SEQ_LEN,
            ).upper()
            ref_np[i] = one_hot_encode_channels_first(seq).T
        self._ref_ohs_gpu = _torch.from_numpy(ref_np).to(self.device)

        # Precompute REF exon sums — full (n_ctx, 162) — strand-matched selection
        # happens at aggregation time.
        self._ref_exon_sums = self._precompute_baselines()  # GPU tensor

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        fasta_path: str | Path,
        gtf_path: str | Path,
        device: str = "cuda",
        batch_size: int = 64,
        use_rc: bool = True,
        autocast: bool = True,
        n_sample: int | None = None,
        seed: int = 42,
    ) -> "YorzoiMPRAMarginalizedPredictor":
        from yorzoi.model.borzoi import Borzoi

        model = Borzoi.from_pretrained(hf_repo)
        return cls(
            model, fasta_path, gtf_path,
            device, batch_size, use_rc=use_rc, autocast=autocast,
            n_sample=n_sample, seed=seed,
        )

    def _forward_full_tracks(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward (+ RC averaging with strand swap over ALL 162 tracks).

        Returns (B, 162, OUTPUT_BINS).
        """
        import torch as _torch

        ctx = (
            _torch.autocast(device_type="cuda")
            if self.autocast and self.device.type == "cuda"
            else _torch.amp.autocast(device_type="cpu", enabled=False)
        )
        with ctx:
            out_fwd = self.model(x)  # (B, 162, OUTPUT_BINS)
        if not self.use_rc:
            return out_fwd

        x_rc = x.flip(dims=[1, 2])
        with ctx:
            out_rc = self.model(x_rc)
        out_rc_aligned = out_rc.index_select(1, self._full_swap_idx).flip(dims=[2])
        return 0.5 * (out_fwd + out_rc_aligned)

    def _precompute_baselines(self) -> "torch.Tensor":
        """Returns (n_contexts, 162) tensor of REF exon-bin sums on GPU."""
        import torch as _torch

        n_ctx = len(self.contexts)
        ref_sums = _torch.zeros(n_ctx, N_TRACKS_TOTAL, device=self.device, dtype=_torch.float32)

        for batch_start in tqdm(
            range(0, n_ctx, self.batch_size), desc="Yorzoi REF baseline"
        ):
            batch_end = min(batch_start + self.batch_size, n_ctx)
            x = self._ref_ohs_gpu[batch_start:batch_end]

            with _torch.no_grad():
                pred = self._forward_full_tracks(x).float()

            for i in range(batch_end - batch_start):
                ctx = self.contexts[batch_start + i]
                if ctx.exon_bins.size == 0:
                    continue
                bins_t = _torch.as_tensor(ctx.exon_bins, device=self.device, dtype=_torch.long)
                ref_sums[batch_start + i] = pred[i].index_select(1, bins_t).sum(dim=1)

        return ref_sums

    def _build_alt_batch_gpu(
        self, insert_fwd_oh: "torch.Tensor", insert_rc_oh: "torch.Tensor",
        batch_start: int, batch_end: int,
    ) -> "torch.Tensor":
        """Construct ALT one-hots on GPU by copying REFs and splicing the insert."""
        alt = self._ref_ohs_gpu[batch_start:batch_end].clone()  # (B, L, 4)
        for i in range(batch_end - batch_start):
            ctx = self.contexts[batch_start + i]
            s = ctx.insert_start_in_window
            oh = insert_rc_oh if ctx.gene_strand == "-" else insert_fwd_oh
            alt[i, s : s + INSERT_LEN, :] = oh
        return alt

    def _score_one_sequence(self, insert_seq: str) -> float:
        import torch as _torch

        # Encode insert once (CPU → GPU), both orientations
        insert_fwd_np = one_hot_encode_channels_first(insert_seq.upper()).T  # (L, 4)
        insert_rc_np = one_hot_encode_channels_first(
            reverse_complement(insert_seq).upper()
        ).T
        insert_fwd_oh = _torch.from_numpy(insert_fwd_np).to(self.device)
        insert_rc_oh = _torch.from_numpy(insert_rc_np).to(self.device)

        n_ctx = len(self.contexts)
        alt_exon_sums = _torch.zeros(n_ctx, N_TRACKS_TOTAL, device=self.device, dtype=_torch.float32)

        for batch_start in range(0, n_ctx, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_ctx)
            x = self._build_alt_batch_gpu(insert_fwd_oh, insert_rc_oh, batch_start, batch_end)

            with _torch.no_grad():
                pred = self._forward_full_tracks(x).float()

            for i in range(batch_end - batch_start):
                ctx = self.contexts[batch_start + i]
                if ctx.exon_bins.size == 0:
                    continue
                bins_t = _torch.as_tensor(ctx.exon_bins, device=self.device, dtype=_torch.long)
                alt_exon_sums[batch_start + i] = pred[i].index_select(1, bins_t).sum(dim=1)

        # Cross-track mean of exon sums BEFORE log (logSED_agg convention,
        # matches the eQTL adapter and the Shorkie paper's "mean → sum → log").
        # Strand-matched tracks per context: + gene → [0:81], - gene → [81:162].
        alt_mean = _torch.zeros(n_ctx, device=self.device, dtype=_torch.float32)
        ref_mean = _torch.zeros(n_ctx, device=self.device, dtype=_torch.float32)
        for i in range(n_ctx):
            ts, te = self._ctx_track_start[i], self._ctx_track_end[i]
            alt_mean[i] = alt_exon_sums[i, ts:te].mean()
            ref_mean[i] = self._ref_exon_sums[i, ts:te].mean()

        logsed_per_ctx = _torch.log2(alt_mean + 1.0) - _torch.log2(ref_mean + 1.0)

        # Mean across offsets per gene, then across genes
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

        for idx in tqdm(sample_idx, desc="Yorzoi marginalized"):
            insert = extract_insert(seqs[idx])
            scores[idx] = self._score_one_sequence(insert)

        return scores
