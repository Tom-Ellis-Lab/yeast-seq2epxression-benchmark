"""Shorkie variant-effect scorer for the Caudal eQTL benchmark.

Implements the ``logSED_agg`` scoring procedure documented in
``benchmarks/caudal_eqtl.md``: window-placement constraint solve, strict
ref-allele check, ref/alt one-hot, 8-fold ensemble, panel-track slice,
cross-track mean → exon-bin sum → log2 fold change.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from yeastbench.adapters._genome import (
    ARABIC_TO_ROMAN,
    Gene,
    gene_exon_bins,
    one_hot_encode_channels_first,
    parse_gene_annotations,
    place_window,
)
from yeastbench.adapters._shorkie_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    SHORKIE_1011_RNA_SEQ_TRACK_IDS,
)
from yeastbench.adapters.protocols import Variant, VariantEffectScorer
from yeastbench.models.shorkie import Shorkie

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class _ScoringJob:
    """Per-variant pre-computed metadata for batched scoring."""
    chrom_roman: str
    window_start: int  # 0-based
    var_idx_in_window: int  # 0-based
    ref: str
    alt: str
    bin_idx: np.ndarray  # exon-overlapping output bins for the target gene


class ShorkieVariantScorer(VariantEffectScorer):
    def __init__(
        self,
        model: Shorkie,
        fasta_path: str | Path,
        gtf_path: str | Path,
        track_subset: list[int] = SHORKIE_1011_RNA_SEQ_TRACK_IDS,
        batch_size: int = 8,
    ) -> None:
        import pysam

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.genes = parse_gene_annotations(gtf_path)
        self.track_subset = list(track_subset)
        self.batch_size = batch_size

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        fasta_path: str | Path,
        gtf_path: str | Path,
        track_subset: list[int] = SHORKIE_1011_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 8,
        use_rc: bool = True,
    ) -> "ShorkieVariantScorer":
        return cls(
            Shorkie.from_checkpoints(
                params_path, checkpoint_paths, device=device, use_rc=use_rc,
            ),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            track_subset=list(track_subset),
            batch_size=batch_size,
        )

    def _prepare_jobs(self, variants: Sequence[Variant]) -> list[_ScoringJob]:
        """Resolve windows, verify ref base when the variant is inside the
        window, and pre-compute the exon-bin index per variant.

        When the variant is too far from the gene's center to fit alongside
        it inside one ``SEQ_LEN`` input window, the window falls back to
        gene-centered (matches the canonical Shorkie scoring script) and
        the variant ends up outside the input. In that case we keep the
        ref-pass prediction but skip the alt mutation, mirroring the
        upstream behavior — the resulting variant effect score is ≈ 0,
        which is the correct behavior for a variant the architecture
        cannot in principle see.
        """
        jobs: list[_ScoringJob] = []
        for v in variants:
            gene = self.genes[v.gene_id]
            chrom_roman = ARABIC_TO_ROMAN[v.chrom]
            if chrom_roman != gene.chrom_roman:
                raise ValueError(
                    f"variant chrom {v.chrom} ({chrom_roman}) does not match "
                    f"gene {v.gene_id} chrom {gene.chrom_roman}"
                )
            chrom_len = self.fasta.get_reference_length(chrom_roman)
            start0 = place_window(
                v.pos, gene.gene_center, chrom_len, SEQ_LEN, CROP_BP_EACH_SIDE
            )
            var_idx = v.pos - 1 - start0
            if 0 <= var_idx < SEQ_LEN:
                seq = self.fasta.fetch(
                    chrom_roman, start0, start0 + SEQ_LEN
                ).upper()
                ref_in_fasta = seq[var_idx]
                if ref_in_fasta != v.ref.upper():
                    raise ValueError(
                        f"REF mismatch at {v.chrom}:{v.pos}: "
                        f"FASTA={ref_in_fasta}, claimed={v.ref}"
                    )
            jobs.append(
                _ScoringJob(
                    chrom_roman=chrom_roman,
                    window_start=start0,
                    var_idx_in_window=var_idx,
                    ref=v.ref.upper(),
                    alt=v.alt.upper(),
                    bin_idx=gene_exon_bins(
                        gene, start0, CROP_BP_EACH_SIDE, BIN_WIDTH, OUTPUT_BINS
                    ),
                )
            )
        return jobs

    def _build_one_hot_pair(self, job: _ScoringJob) -> tuple[np.ndarray, np.ndarray]:
        seq = self.fasta.fetch(
            job.chrom_roman, job.window_start, job.window_start + SEQ_LEN
        ).upper()
        ref_oh = one_hot_encode_channels_first(seq)
        if 0 <= job.var_idx_in_window < SEQ_LEN:
            i = job.var_idx_in_window
            alt_seq = seq[:i] + job.alt + seq[i + len(job.ref):]
            if len(alt_seq) != SEQ_LEN:
                alt_seq = (alt_seq + "N" * SEQ_LEN)[:SEQ_LEN]
            alt_oh = one_hot_encode_channels_first(alt_seq)
        else:
            alt_oh = ref_oh  # variant outside window → no mutation, score ≈ 0
        return ref_oh, alt_oh

    def score_variants(self, variants: Sequence[Variant]) -> np.ndarray:
        import torch as _torch

        jobs = self._prepare_jobs(variants)
        n = len(jobs)
        scores = np.empty(n, dtype=np.float64)
        track_idx_t = _torch.tensor(
            self.track_subset, device=self.model.device, dtype=_torch.long
        )

        for batch_start in range(0, n, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n)
            batch_jobs = jobs[batch_start:batch_end]
            B = len(batch_jobs)

            ref_arrs, alt_arrs = [], []
            for job in batch_jobs:
                r, a = self._build_one_hot_pair(job)
                ref_arrs.append(r)
                alt_arrs.append(a)
            x = _torch.from_numpy(
                np.concatenate(
                    [np.stack(ref_arrs, axis=0), np.stack(alt_arrs, axis=0)], axis=0
                )
            ).to(self.model.device)

            with _torch.no_grad():
                # (2B, OUTPUT_BINS, n_tracks) — ensemble + RC averaged
                acc = self.model.forward_tracks_binned(x, track_idx_t)
            cov = acc.mean(dim=2)  # cross-track mean → (2B, 896)

            for i, job in enumerate(batch_jobs):
                bins = job.bin_idx
                if bins.size == 0:
                    scores[batch_start + i] = 0.0
                    continue
                bins_t = _torch.from_numpy(bins).to(self.model.device)
                ref_sum = cov[i].index_select(0, bins_t).sum().item()
                alt_sum = cov[i + B].index_select(0, bins_t).sum().item()
                scores[batch_start + i] = float(
                    np.log2(alt_sum + 1.0) - np.log2(ref_sum + 1.0)
                )

        return scores
