"""Yorzoi variant-effect scorer for the Caudal eQTL benchmark.

Implements the logSED-agg scoring procedure documented in
``benchmarks/caudal_eqtl.md``'s Yorzoi section:
- 4,992 bp input window (channels-last), 300 output bins × 10 bp/bin,
  covering the central 3,000 bp.
- ``+`` (forward) strand tracks only (indices 0..80 of the 162-track
  output), regardless of the target gene's strand. Locked by the spec.
- Aggregation: cross-track mean → exon-bin sum → log2 fold change.
- Optional RC averaging, with proper track-strand swap on the RC pass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from yeastbench.adapters._genome import (
    ARABIC_TO_ROMAN,
    Gene,
    gene_exon_bins,
    one_hot_encode_channels_first,
    parse_gene_annotations,
    place_window,
)
from yeastbench.adapters.protocols import Variant, VariantEffectScorer

if TYPE_CHECKING:
    import torch


# Yorzoi architecture constants (verified by loading the HF checkpoint):
# input 4,992 bp → output (B, 162 tracks, 300 bins), resolution 10 bp/bin,
# post-crop output covers the central 3,000 bp of input.
SEQ_LEN = 4992
OUTPUT_BINS = 300
BIN_WIDTH = 10
CROP_BP_EACH_SIDE = 996  # (4992 - 3000) // 2

# Track layout from yorzoi/track_annotation.json:
#   indices 0..80   → '+' (forward) strand, 81 tracks
#   indices 81..161 → '-' (reverse) strand, 81 tracks (same samples)
YORZOI_PLUS_TRACK_IDS: list[int] = list(range(0, 81))
YORZOI_MINUS_TRACK_IDS: list[int] = list(range(81, 162))


@dataclass(frozen=True)
class _ScoringJob:
    chrom_roman: str
    window_start: int  # 0-based
    var_idx_in_window: int
    ref: str
    alt: str
    bin_idx: np.ndarray


class YorzoiVariantScorer(VariantEffectScorer):
    def __init__(
        self,
        model: Any,  # yorzoi.model.borzoi.Borzoi
        fasta_path: str | Path,
        gtf_path: str | Path,
        track_subset: list[int] = YORZOI_PLUS_TRACK_IDS,
        device: "str | torch.device" = "cuda",
        batch_size: int = 16,
        use_rc: bool = True,
        autocast: bool = True,
    ) -> None:
        import pysam
        import torch as _torch

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.genes = parse_gene_annotations(gtf_path)
        self.track_subset = list(track_subset)
        self.device = _torch.device(device)
        self.batch_size = batch_size
        self.use_rc = use_rc
        self.autocast = autocast
        self.model.to(self.device).eval()

        self._is_plus_only_subset = self.track_subset == YORZOI_PLUS_TRACK_IDS
        self._plus_subset_t = None
        self._minus_subset_t = None

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        fasta_path: str | Path,
        gtf_path: str | Path,
        track_subset: list[int] = YORZOI_PLUS_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 16,
        use_rc: bool = True,
        autocast: bool = True,
    ) -> "YorzoiVariantScorer":
        from yorzoi.model.borzoi import Borzoi

        model = Borzoi.from_pretrained(hf_repo)
        return cls(
            model, fasta_path, gtf_path, list(track_subset),
            device, batch_size, use_rc=use_rc, autocast=autocast,
        )

    def _prepare_jobs(self, variants: Sequence[Variant]) -> list[_ScoringJob]:
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
        """Returns (ref, alt) one-hot arrays in channels-LAST layout ``(L, 4)``,
        which is what Yorzoi expects as input."""
        seq = self.fasta.fetch(
            job.chrom_roman, job.window_start, job.window_start + SEQ_LEN
        ).upper()
        ref_oh = one_hot_encode_channels_first(seq).T  # (L, 4)
        if 0 <= job.var_idx_in_window < SEQ_LEN:
            i = job.var_idx_in_window
            alt_seq = seq[:i] + job.alt + seq[i + len(job.ref):]
            if len(alt_seq) != SEQ_LEN:
                alt_seq = (alt_seq + "N" * SEQ_LEN)[:SEQ_LEN]
            alt_oh = one_hot_encode_channels_first(alt_seq).T
        else:
            alt_oh = ref_oh  # variant outside window → no mutation
        return ref_oh, alt_oh

    def _forward_with_rc(self, x: "torch.Tensor") -> "torch.Tensor":
        """Run forward (+optional RC) and return (B, len(track_subset), OUTPUT_BINS).

        For stranded Yorzoi tracks, RC averaging requires swapping the +
        and - track groups on the RC pass (since feeding RC(X) to the
        model means its '+' output tracks predict what was X's '-'
        strand). For the spec-default ``plus_only`` subset that reduces
        to: forward → tracks 0..80; RC → tracks 81..161 flipped along the
        bin axis; average the two.
        """
        import torch as _torch

        track_idx_t = _torch.tensor(
            self.track_subset, device=self.device, dtype=_torch.long
        )
        ctx = (
            _torch.autocast(device_type="cuda")
            if self.autocast and self.device.type == "cuda"
            else _torch.amp.autocast(device_type="cpu", enabled=False)
        )
        with ctx:
            out_fwd = self.model(x)  # (B, 162, 300)
        if not self.use_rc:
            return out_fwd.index_select(1, track_idx_t)

        # Reverse-complement input: x is (B, L, 4) channels-last.
        # Flip position axis (dim=1) and channel axis (dim=2) to complement.
        x_rc = x.flip(dims=[1, 2])
        with ctx:
            out_rc = self.model(x_rc)  # (B, 162, 300)

        if self._is_plus_only_subset:
            swap_idx = _torch.tensor(
                YORZOI_MINUS_TRACK_IDS, device=self.device, dtype=_torch.long
            )
            out_rc_aligned = out_rc.index_select(1, swap_idx).flip(dims=[2])
            out_fwd_sub = out_fwd.index_select(1, track_idx_t)
        else:
            # Generic: for each requested track, its RC counterpart is
            # the strand-paired track (± 81). Require the track subset to
            # be either all-plus (0..80) or all-minus (81..161) for now —
            # mixed subsets would need per-track mapping.
            raise NotImplementedError(
                "RC averaging for non-default Yorzoi track subsets not implemented yet."
            )

        return 0.5 * (out_fwd_sub + out_rc_aligned)

    def score_variants(self, variants: Sequence[Variant]) -> np.ndarray:
        import torch as _torch

        jobs = self._prepare_jobs(variants)
        n = len(jobs)
        scores = np.empty(n, dtype=np.float64)

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
            ).to(self.device)

            with _torch.no_grad():
                pred = self._forward_with_rc(x).float()  # (2B, n_tracks, 300)

            cov = pred.mean(dim=1)  # cross-track mean → (2B, 300)

            for i, job in enumerate(batch_jobs):
                bins = job.bin_idx
                if bins.size == 0:
                    scores[batch_start + i] = 0.0
                    continue
                bins_t = _torch.from_numpy(bins).to(self.device)
                ref_sum = cov[i].index_select(0, bins_t).sum().item()
                alt_sum = cov[i + B].index_select(0, bins_t).sum().item()
                scores[batch_start + i] = float(
                    np.log2(alt_sum + 1.0) - np.log2(ref_sum + 1.0)
                )

        return scores
