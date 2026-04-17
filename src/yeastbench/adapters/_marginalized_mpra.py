"""Shared infrastructure for marginalized / native-position MPRA adapters.

Both Shorkie and Yorzoi marginalized adapters insert the 110 bp MPRA
sequence (polyT overhang + 80 bp random + polyA overhang) at native yeast
genome positions upstream of 22 host genes and compute logSED (log
fold-change in predicted expression).

The Shorkie paper's `1_prepare_tsv.py` feeds the **full 110 bp sequence**
from the test CSVs — including the polyT/polyA overhangs — into the
scoring pipeline; we match that protocol.

Offset convention: ``offset`` = genomic distance between the TSS and the
*nearest* edge of the insert.  The insert occupies INSERT_LEN bp further
upstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from yeastbench.adapters._genome import (
    Gene,
    gene_exon_bins,
    one_hot_encode_channels_first,
    parse_gene_annotations,
    place_window,
)

if TYPE_CHECKING:
    import pysam

# ── Constants ─────────────────────────────────────────────────

INSERT_LEN = 110  # full MPRA sequence: 17 bp polyT + 80 bp random + 13 bp polyA

# Shorkie paper (Figure S20A) selects the 180 bp upstream position for the
# final marginalized-MPRA analysis after scanning multiple positions.  We
# default to that single offset; pass an explicit ``offsets`` list to
# ``compute_insertion_contexts`` to scan a range instead.
DEFAULT_OFFSETS: tuple[int, ...] = (180,)

# The 22 host genes used in the Shorkie paper's marginalized MPRA protocol.
# Positive-strand (10) and negative-strand (12).
HOST_GENES: dict[str, str] = {
    # Positive strand
    "YOL056W": "GPM3",
    "YGR212W": "SLI1",
    "YDR484W": "VPS52",
    "YMR160W": "YMR160W",
    "YDR337W": "MRPS28",
    "YLL055W": "YCT1",
    "YOR286W": "RDL2",
    "YJL097W": "PHS1",
    "YHR087W": "RTC3",
    "YKL062W": "MSN4",
    # Negative strand
    "YLR218C": "COA4",
    "YPL096C-A": "ERI1",
    "YIL093C": "RSM25",
    "YDR414C": "ERD1",
    "YGL136C": "MRM2",
    "YGL131C": "SNT2",
    "YOL007C": "CSI2",
    "YJL121C": "RPE1",
    "YBL105C": "PKC1",
    "YER093C-A": "AIM11",
    "YKL029C": "MAE1",
    "YDR116C": "MRPL1",
}

_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def extract_insert(seq_110bp: str) -> str:
    """Return the 110 bp MPRA sequence unchanged.

    The Shorkie paper protocol inserts the full 110 bp (polyT + 80 N +
    polyA).  This helper is kept so callers don't special-case the
    length and to allow swapping to a core-80 variant later.
    """
    assert len(seq_110bp) == INSERT_LEN, (
        f"expected {INSERT_LEN} bp, got {len(seq_110bp)}"
    )
    return seq_110bp


# ── Insertion context ─────────────────────────────────────────


@dataclass(frozen=True)
class InsertionContext:
    """Pre-computed metadata for one (gene, offset) insertion site."""
    gene_id: str
    gene_strand: str         # '+' or '-'
    offset_bp: int
    window_start: int        # 0-based genomic coordinate
    insert_start_in_window: int  # 0-based position within the model-input window
    exon_bins: np.ndarray    # output-bin indices overlapping gene exons


def compute_insertion_contexts(
    gtf_path: str | Path,
    fasta: "pysam.FastaFile",
    seq_len: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
    offsets: "tuple[int, ...] | list[int]" = DEFAULT_OFFSETS,
) -> list[InsertionContext]:
    """Derive all valid (gene, offset) insertion sites for the 22 host genes.

    An offset is valid when:
      1. The INSERT_LEN bp insert fits fully within the model-input window.
      2. The gene's exons produce at least one output bin.
      3. The insertion site is within chromosome bounds.
    """
    genes = parse_gene_annotations(gtf_path)
    contexts: list[InsertionContext] = []

    for gene_id in HOST_GENES:
        if gene_id not in genes:
            continue
        gene = genes[gene_id]
        chrom_len = fasta.get_reference_length(gene.chrom_roman)

        for offset in offsets:
            # Nearest-edge-at-offset convention:
            #   + strand: insert occupies [tss - offset - LEN, tss - offset)  (0-based)
            #   - strand: insert occupies [tss + offset - 1, tss + offset - 1 + LEN)
            if gene.strand == "+":
                insert_end = gene.tss - offset
                insert_start = insert_end - INSERT_LEN
            else:
                insert_start = gene.tss + offset - 1
                insert_end = insert_start + INSERT_LEN

            if insert_start < 0 or insert_end > chrom_len:
                continue

            insert_center_1based = (insert_start + insert_end) // 2 + 1
            window_start = place_window(
                var_pos=insert_center_1based,
                gene_center=gene.gene_center,
                chrom_length=chrom_len,
                seq_len=seq_len,
                crop_bp_each_side=crop_bp_each_side,
            )

            if insert_start < window_start or insert_end > window_start + seq_len:
                continue

            exon_bins = gene_exon_bins(
                gene, window_start, crop_bp_each_side, bin_width, output_bins
            )
            if exon_bins.size == 0:
                continue

            contexts.append(InsertionContext(
                gene_id=gene_id,
                gene_strand=gene.strand,
                offset_bp=offset,
                window_start=window_start,
                insert_start_in_window=insert_start - window_start,
                exon_bins=exon_bins,
            ))

    return contexts


def build_alt_one_hot(
    ref_oh: np.ndarray,
    insert_seq: str,
    insert_start_in_window: int,
    gene_strand: str,
) -> np.ndarray:
    """Splice ``insert_seq`` into a REF one-hot array.

    For negative-strand genes the insert is reverse-complemented so it is
    in the transcription direction of the host gene.

    Parameters
    ----------
    ref_oh : (4, L) or (L, 4) array — copied, not mutated
    insert_seq : the INSERT_LEN bp sequence
    insert_start_in_window : 0-based position in the window
    gene_strand : '+' or '-'

    Returns
    -------
    ALT one-hot array with the same shape as *ref_oh*.
    """
    channels_first = ref_oh.shape[0] == 4

    seq = insert_seq.upper()
    if gene_strand == "-":
        seq = reverse_complement(seq)

    insert_oh = one_hot_encode_channels_first(seq)  # (4, L_insert)
    L_insert = insert_oh.shape[1]

    alt = ref_oh.copy()
    s = insert_start_in_window
    if channels_first:
        alt[:, s : s + L_insert] = insert_oh
    else:
        alt[s : s + L_insert, :] = insert_oh.T
    return alt
