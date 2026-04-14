"""Shorkie variant-effect scorer for the Caudal eQTL benchmark.

Implements the ``logSED_agg`` scoring procedure documented in
``benchmarks/caudal_eqtl.md``: window-placement constraint solve, strict
ref-allele check, ref/alt one-hot, 8-fold ensemble, panel-track slice,
cross-track mean → exon-bin sum → log2 fold change.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from yeastbench.adapters.protocols import Variant, VariantEffectScorer

if TYPE_CHECKING:
    import torch
    from yeastbench.models.shorkie import Shorkie


SHORKIE_1011_RNA_SEQ_TRACK_IDS: list[int] = list(range(4201, 5215))
"""Output-track indices for the '1000-RNA-Seq' group from the Shorkie
canonical targets sheet (https://storage.googleapis.com/seqnn-share/shorkie/targets.txt).

The benchmark spec names this set '1011 RNA-seq tracks'. The actual count
in the released targets sheet is **1014** (indices 4201..5214 inclusive,
contiguous). Discrepancy is upstream's; we pin to the released file."""


# Architecture constants derived from data/models/shorkie/params.json:
# 16,384 bp input, ResTower (pool 2, repeat 7) → 128× downsample, then 3
# UNet upsamples (2× each) → 16× net downsample, then Cropping1D(64 bins
# each side) → 896 output bins covering the central 14,336 bp.
SEQ_LEN = 16384
OUTPUT_BINS = 896
BIN_WIDTH = 16
CROP_BP_EACH_SIDE = 1024  # 64 bins × 16 bp/bin


# Spec uses Arabic chrom names ('1'..'16'); FASTA and GTF use Roman.
ARABIC_TO_ROMAN = {
    "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V",
    "6": "VI", "7": "VII", "8": "VIII", "9": "IX", "10": "X",
    "11": "XI", "12": "XII", "13": "XIII", "14": "XIV", "15": "XV",
    "16": "XVI",
}


_BASES = "ACGT"
_BASE_TO_IDX = {b: i for i, b in enumerate(_BASES)}


def one_hot_encode(seq: str) -> np.ndarray:
    """DNA string → (4, len(seq)) float32. N / non-ACGT becomes all-zeros."""
    out = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        idx = _BASE_TO_IDX.get(b)
        if idx is not None:
            out[idx, i] = 1.0
    return out


@dataclass(frozen=True)
class Gene:
    chrom_roman: str
    strand: str  # '+' or '-'
    tss: int
    gene_start: int  # 1-based inclusive
    gene_end: int    # 1-based inclusive
    exons: tuple[tuple[int, int], ...]  # 1-based inclusive

    @property
    def gene_center(self) -> int:
        return (self.gene_start + self.gene_end) // 2


_GENE_ID_RE = re.compile(r'gene_id\s+"([^"]+)"')


def parse_gene_annotations(gtf_path: str | Path) -> dict[str, Gene]:
    raw: dict[str, dict] = {}
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            chrom, _, ftype, start, end, _, strand, _, attrs = f
            ftype = ftype.lower()
            m = _GENE_ID_RE.search(attrs)
            if not m:
                continue
            gid = m.group(1)
            entry = raw.setdefault(gid, {"exons": []})
            if ftype == "gene":
                entry.update(
                    chrom=chrom,
                    strand=strand,
                    start=int(start),
                    end=int(end),
                    tss=int(start) if strand == "+" else int(end),
                )
            elif ftype == "exon":
                entry["exons"].append((int(start), int(end)))

    out: dict[str, Gene] = {}
    for gid, e in raw.items():
        if "chrom" not in e:
            continue
        out[gid] = Gene(
            chrom_roman=e["chrom"],
            strand=e["strand"],
            tss=e["tss"],
            gene_start=e["start"],
            gene_end=e["end"],
            exons=tuple(sorted(set(e["exons"]))),
        )
    return out


def place_window(var_pos: int, gene_center: int, chrom_length: int) -> int:
    """Pick a 0-based window start so the variant lies inside the model
    input AND the gene center lies inside the output crop. If both
    constraints are unsatisfiable, fall back to gene-centered (variant may
    be near the input edge but the gene's prediction stays well-defined)."""
    var0 = var_pos - 1
    gc0 = gene_center - 1
    var_min = var0 - SEQ_LEN + 1
    var_max = var0
    gc_min = gc0 - SEQ_LEN + CROP_BP_EACH_SIDE + 1
    gc_max = gc0 - CROP_BP_EACH_SIDE
    lo = max(var_min, gc_min, 0)
    hi = min(var_max, gc_max, chrom_length - SEQ_LEN)
    if lo <= hi:
        return (lo + hi) // 2
    return max(0, min(gc0 - SEQ_LEN // 2, chrom_length - SEQ_LEN))


def gene_exon_bins(gene: Gene, window_start_0based: int) -> np.ndarray:
    """Output-bin indices (in [0, OUTPUT_BINS)) overlapping any of the
    gene's annotated exons within this window."""
    bin_start_bp = window_start_0based + CROP_BP_EACH_SIDE
    out: set[int] = set()
    for ex_start, ex_end in gene.exons:
        ex_lo = ex_start - 1
        ex_hi = ex_end
        b_lo = max(0, (ex_lo - bin_start_bp) // BIN_WIDTH)
        b_hi = min(OUTPUT_BINS, (ex_hi - bin_start_bp + BIN_WIDTH - 1) // BIN_WIDTH)
        if b_hi > b_lo:
            out.update(range(b_lo, b_hi))
    if not out:
        return np.array([], dtype=np.int64)
    return np.array(sorted(out), dtype=np.int64)


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
        models: list["Shorkie"],
        fasta_path: str | Path,
        gtf_path: str | Path,
        track_subset: list[int] = SHORKIE_1011_RNA_SEQ_TRACK_IDS,
        device: "str | torch.device" = "cuda",
        batch_size: int = 8,
    ) -> None:
        import pysam
        import torch as _torch

        if not models:
            raise ValueError("Must provide at least one model fold")
        self.models = models
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.genes = parse_gene_annotations(gtf_path)
        self.track_subset = list(track_subset)
        self.device = _torch.device(device)
        self.batch_size = batch_size
        for m in self.models:
            m.to(self.device).eval()

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
    ) -> "ShorkieVariantScorer":
        from yeastbench.models.shorkie import Shorkie

        with open(params_path) as f:
            config = json.load(f)
        models = [
            Shorkie.from_tf_checkpoint(config["model"], str(p))
            for p in checkpoint_paths
        ]
        return cls(models, fasta_path, gtf_path, list(track_subset), device, batch_size)

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
            start0 = place_window(v.pos, gene.gene_center, chrom_len)
            seq = self.fasta.fetch(chrom_roman, start0, start0 + SEQ_LEN).upper()
            if len(seq) != SEQ_LEN:
                raise ValueError(
                    f"FASTA fetch on {chrom_roman} returned {len(seq)} bp "
                    f"(expected {SEQ_LEN}) at start={start0}"
                )
            var_idx = v.pos - 1 - start0
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
                    bin_idx=gene_exon_bins(gene, start0),
                )
            )
        return jobs

    def _build_one_hot_pair(self, job: _ScoringJob) -> tuple[np.ndarray, np.ndarray]:
        seq = self.fasta.fetch(
            job.chrom_roman, job.window_start, job.window_start + SEQ_LEN
        ).upper()
        ref_oh = one_hot_encode(seq)
        alt_seq = seq[: job.var_idx_in_window] + job.alt + seq[job.var_idx_in_window + len(job.ref):]
        if len(alt_seq) != SEQ_LEN:
            alt_seq = (alt_seq + "N" * SEQ_LEN)[:SEQ_LEN]
        alt_oh = one_hot_encode(alt_seq)
        return ref_oh, alt_oh

    def score_variants(self, variants: Sequence[Variant]) -> np.ndarray:
        import torch as _torch

        jobs = self._prepare_jobs(variants)
        n = len(jobs)
        scores = np.empty(n, dtype=np.float64)
        track_idx_t = _torch.tensor(
            self.track_subset, device=self.device, dtype=_torch.long
        )
        n_tracks = len(self.track_subset)

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
                acc = _torch.zeros(
                    2 * B, OUTPUT_BINS, n_tracks,
                    device=self.device, dtype=_torch.float32,
                )
                for m in self.models:
                    out = m(x)
                    acc.add_(out.index_select(2, track_idx_t))
                acc.div_(len(self.models))

            cov = acc.mean(dim=2)  # cross-track mean → (2B, 896)

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
