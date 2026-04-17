"""Shared infrastructure for the Shalem MPRA terminator marginalized adapter.

Each test sequence (150 bp) is inserted immediately downstream of each of
22 host-gene stop codons, followed by a fixed 300 bp no-termination filler
(100 bp CYC1 CDS tail + 200 bp cycl-512 mutant 3′-UTR, a la Guo 1995
PNAS 92:4211).  For each (sequence, host gene) pair the adapter computes
logSED over the host's exon bins; the final per-sequence prediction is
the mean logSED across the 22 genes.

Host-gene list is committed at
``data/processed/shalem_mpra/host_genes.json``.  Filler construction
happens once at adapter init from CYC1 (YJR048W) fetched from the R64-1-1
FASTA.
"""
from __future__ import annotations

import json
import logging
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

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HOST_GENES_JSON = REPO_ROOT / "data" / "processed" / "shalem_mpra" / "host_genes.json"

# ── Construct lengths ─────────────────────────────────────────

INSERT_LEN = 150           # full Shalem oligo (Primer5 + variable + barcode + Primer3)
CYC1_CDS_TAIL_LEN = 100    # last 100 bp of CYC1 CDS
MUTANT_UTR_LEN = 200       # first 200 bp of cycl-512 mutant UTR
FILLER_LEN = CYC1_CDS_TAIL_LEN + MUTANT_UTR_LEN  # 300
REPLACE_LEN = INSERT_LEN + FILLER_LEN            # 450

# CYC1 = YJR048W (Ensembl yeast)
CYC1_GENE_ID = "YJR048W"

# The two TATTTA motifs flank the 3' UTR efficiency-element region that
# is deleted in the cycl-512 mutant.
TATTTA = "TATTTA"

_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


# ── Host-gene list ────────────────────────────────────────────


@dataclass(frozen=True)
class HostGeneSpec:
    gene_id: str
    strand: str  # '+' or '-'


def load_host_genes(path: Path = DEFAULT_HOST_GENES_JSON) -> list[HostGeneSpec]:
    with open(path) as fh:
        data = json.load(fh)
    return [HostGeneSpec(gene_id=g["gene_id"], strand=g["strand"]) for g in data["genes"]]


# ── cycl-512 mutant filler construction ───────────────────────


def build_filler(fasta: "pysam.FastaFile", gtf_genes: dict[str, Gene]) -> str:
    """Return the 300 bp CYC1-CDS-tail + cycl-512-mutant-UTR filler.

    Implements the deletion by locating the two ``TATTTA`` motifs that
    flank the efficiency-element region in the CYC1 3'-UTR and removing
    everything between the first motif's end and the second motif's end.

    In R64-1-1 this removes a 40 bp region (Guo 1995 reported 38 bp on
    their strain — the structural lesion is identical).
    """
    cyc1 = gtf_genes.get(CYC1_GENE_ID)
    if cyc1 is None:
        raise RuntimeError(
            f"CYC1 ({CYC1_GENE_ID}) not found in GTF — cannot build filler"
        )
    if cyc1.strand != "+":
        raise RuntimeError(f"Expected CYC1 on + strand, got {cyc1.strand}")

    # Fetch CDS + 500 bp downstream (plenty for locating the UTR motifs)
    fetch_start = cyc1.gene_start - 1  # 0-based
    fetch_end = cyc1.gene_end + 500
    seq = fasta.fetch(cyc1.chrom_roman, fetch_start, fetch_end).upper()

    cds_len = cyc1.gene_end - cyc1.gene_start + 1  # stop codon inclusive
    cds = seq[:cds_len]
    utr = seq[cds_len:]

    # Locate the first and second TATTTA motifs in the UTR
    first = utr.find(TATTTA)
    if first < 0:
        raise RuntimeError("Could not locate first TATTTA in CYC1 UTR")
    second = utr.find(TATTTA, first + len(TATTTA))
    if second < 0:
        raise RuntimeError("Could not locate second TATTTA in CYC1 UTR")

    # Delete everything between first_TATTTA_end and second_TATTTA_end
    del_start = first + len(TATTTA)
    del_end = second + len(TATTTA)
    mutant_utr = utr[:del_start] + utr[del_end:]

    cds_tail = cds[-CYC1_CDS_TAIL_LEN:]
    utr_slice = mutant_utr[:MUTANT_UTR_LEN]
    filler = cds_tail + utr_slice
    if len(filler) != FILLER_LEN:
        raise RuntimeError(
            f"Filler length {len(filler)} != {FILLER_LEN}; "
            "check CYC1 UTR length in the FASTA"
        )
    return filler


# ── Insertion contexts (per host gene × window placement) ─────


@dataclass(frozen=True)
class ShalemInsertionContext:
    """Pre-computed metadata for one host-gene insertion site."""
    gene_id: str
    gene_strand: str
    window_start: int         # 0-based genomic
    replace_start_in_window: int  # 0-based position of the 450 bp replacement
    exon_bins: np.ndarray     # output-bin indices overlapping gene exons


def compute_insertion_contexts(
    host_genes: list[HostGeneSpec],
    gtf_genes: dict[str, Gene],
    fasta: "pysam.FastaFile",
    seq_len: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
) -> list[ShalemInsertionContext]:
    """Build one window-placement context per host gene.

    Replace-region geometry (0-based genomic, transcription direction):
      + strand:  [stop_end_1based,          stop_end_1based + REPLACE_LEN)
      − strand:  [gene_start_0based - REPLACE_LEN, gene_start_0based)

    (+ strand: replacement starts at the base just past the last CDS base.
     − strand: replacement ends just before the first CDS base.)
    """
    contexts: list[ShalemInsertionContext] = []
    for h in host_genes:
        gene = gtf_genes.get(h.gene_id)
        if gene is None:
            raise RuntimeError(f"Host gene {h.gene_id} not found in GTF")
        if gene.strand != h.strand:
            raise RuntimeError(
                f"Strand mismatch for {h.gene_id}: host_genes.json says "
                f"{h.strand}, GTF says {gene.strand}"
            )
        chrom_len = fasta.get_reference_length(gene.chrom_roman)

        if gene.strand == "+":
            replace_start = gene.gene_end  # 0-based position just past last CDS base
        else:
            replace_start = gene.gene_start - 1 - REPLACE_LEN
        replace_end = replace_start + REPLACE_LEN
        if replace_start < 0 or replace_end > chrom_len:
            raise RuntimeError(
                f"Replacement region for {h.gene_id} falls outside chromosome "
                f"{gene.chrom_roman} (len {chrom_len}): "
                f"[{replace_start}, {replace_end})"
            )

        # Window placement: pick a window so the replacement-centre is inside
        # the input and the gene centre is inside the output crop.
        replace_center_1based = (replace_start + replace_end) // 2 + 1
        window_start = place_window(
            var_pos=replace_center_1based,
            gene_center=gene.gene_center,
            chrom_length=chrom_len,
            seq_len=seq_len,
            crop_bp_each_side=crop_bp_each_side,
        )
        if (
            replace_start < window_start
            or replace_end > window_start + seq_len
        ):
            raise RuntimeError(
                f"Replacement region for {h.gene_id} doesn't fit in the chosen "
                f"window (shouldn't happen with the filtered host-gene list)"
            )

        exon_bins = gene_exon_bins(
            gene, window_start, crop_bp_each_side, bin_width, output_bins
        )
        if exon_bins.size == 0:
            raise RuntimeError(
                f"No exon bins in output crop for {h.gene_id} — selection "
                "filter should have rejected this"
            )

        contexts.append(ShalemInsertionContext(
            gene_id=h.gene_id,
            gene_strand=gene.strand,
            window_start=window_start,
            replace_start_in_window=replace_start - window_start,
            exon_bins=exon_bins,
        ))
    return contexts


def assemble_replacement(insert_seq: str, filler: str, gene_strand: str) -> str:
    """Return the 450 bp replacement string in genomic (forward) orientation.

    For − strand host genes the entire (insert + filler) is reverse-
    complemented so that in the gene's *transcription* direction it reads:
        5' → [insert] [filler] → 3'
    """
    assert len(insert_seq) == INSERT_LEN, f"insert must be {INSERT_LEN} bp"
    assert len(filler) == FILLER_LEN, f"filler must be {FILLER_LEN} bp"
    rep = insert_seq.upper() + filler.upper()
    if gene_strand == "-":
        rep = reverse_complement(rep)
    assert len(rep) == REPLACE_LEN
    return rep


def splice_into_one_hot(
    ref_oh: np.ndarray, replacement_seq: str, replace_start_in_window: int,
) -> np.ndarray:
    """Splice a replacement sequence into a (channels-first or -last)
    one-hot REF array, returning a new array.
    """
    channels_first = ref_oh.shape[0] == 4
    rep_oh = one_hot_encode_channels_first(replacement_seq)  # (4, 450)
    L = rep_oh.shape[1]
    alt = ref_oh.copy()
    s = replace_start_in_window
    if channels_first:
        alt[:, s : s + L] = rep_oh
    else:
        alt[s : s + L, :] = rep_oh.T
    return alt


__all__ = [
    "INSERT_LEN",
    "FILLER_LEN",
    "REPLACE_LEN",
    "CYC1_GENE_ID",
    "HostGeneSpec",
    "ShalemInsertionContext",
    "load_host_genes",
    "build_filler",
    "compute_insertion_contexts",
    "assemble_replacement",
    "splice_into_one_hot",
    "reverse_complement",
]
