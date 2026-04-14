"""Shared genome-access utilities for variant-effect adapters.

Parameterized so model adapters with different input / output / binning
geometries can reuse the same window-placement and exon-bin-selection
logic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ARABIC_TO_ROMAN = {
    "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V",
    "6": "VI", "7": "VII", "8": "VIII", "9": "IX", "10": "X",
    "11": "XI", "12": "XII", "13": "XIII", "14": "XIV", "15": "XV",
    "16": "XVI",
}


_BASES = "ACGT"
_BASE_TO_IDX = {b: i for i, b in enumerate(_BASES)}


def one_hot_encode_channels_first(seq: str) -> np.ndarray:
    """DNA string → ``(4, L)`` float32. N / non-ACGT becomes all-zeros."""
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
    tss: int  # 1-based
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


def place_window(
    var_pos: int,
    gene_center: int,
    chrom_length: int,
    seq_len: int,
    crop_bp_each_side: int,
) -> int:
    """Pick a 0-based window start so the variant lies inside the model
    input AND the gene center lies inside the output crop. If both
    constraints are unsatisfiable, fall back to gene-centered (variant
    may be outside the input; the caller is responsible for handling
    that case — typically by skipping the alt mutation)."""
    var0 = var_pos - 1
    gc0 = gene_center - 1
    var_min = var0 - seq_len + 1
    var_max = var0
    gc_min = gc0 - seq_len + crop_bp_each_side + 1
    gc_max = gc0 - crop_bp_each_side
    lo = max(var_min, gc_min, 0)
    hi = min(var_max, gc_max, chrom_length - seq_len)
    if lo <= hi:
        return (lo + hi) // 2
    return max(0, min(gc0 - seq_len // 2, chrom_length - seq_len))


def gene_exon_bins(
    gene: Gene,
    window_start_0based: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
) -> np.ndarray:
    """Output-bin indices in ``[0, output_bins)`` overlapping any of the
    gene's annotated exons within this window."""
    bin_start_bp = window_start_0based + crop_bp_each_side
    out: set[int] = set()
    for ex_start, ex_end in gene.exons:
        ex_lo = ex_start - 1
        ex_hi = ex_end
        b_lo = max(0, (ex_lo - bin_start_bp) // bin_width)
        b_hi = min(output_bins, (ex_hi - bin_start_bp + bin_width - 1) // bin_width)
        if b_hi > b_lo:
            out.update(range(b_lo, b_hi))
    if not out:
        return np.array([], dtype=np.int64)
    return np.array(sorted(out), dtype=np.int64)
