"""Shared infrastructure for marginalized Chen-synonymous-MPRA adapters.

The variant-effect signal in Chen 2017 is **CDS-intrinsic codon usage**
(established by the PGAL1 investigation notebooks: both Shorkie and
Yorzoi correctly predict native GAL1 in glucose as near-zero coverage
but predict the GFP-CDS-containing construct as 11-29× higher at the
same locus — the prediction is driven by the CDS, not by the
promoter). Single-locus scoring at PGAL1 is therefore evaluating the
codon signal through a locus the model knows is silent in its
training conditions, which adds locus-specific calibration noise.

Marginalising over 20 strong-and-active-in-YPD host loci removes that
noise. We splice the variant gene's CDS + TADH1 into each host's CDS
location (keeping host promoter and downstream context), score each
variant as ``mean_host(log2(alt_sum + 1) − log2(ref_sum + 1))``, and
report Pearson + Spearman vs Chen's measured log2(mRNA).

This module:

- Loads the curated 20-host JSON (``data/tasks/chen_synonymous/
  marginalized_hosts.json``) — see spec for selection criteria.
- Builds the per-library cassette (variant gene CDS + TADH1):
    GFP libraries → preferred-yeast-codon GFP with row-0 variants at
                     codons 41-52 and 156-167, + TADH1
    TDH3 library  → native R64-1-1 TDH3 CDS + TADH1
- For each host, computes the modified chromosome (host CDS replaced
  by cassette, strand-aware), the model-input window centred on the
  cassette, and the output-bin indices spanning the variant CDS.
- Provides a single ``ChenHostContext`` per host with everything an
  adapter needs to score a variant: cached REF one-hot, position of
  the 36 nt variable block inside the window, whether to RC the
  variant block before splicing (for - strand hosts), output bins to
  sum over for the cassette CDS.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from yeastbench.adapters._genome import one_hot_encode_channels_first, place_window

if TYPE_CHECKING:
    import pysam


# Codon table — duplicated locally so this module has no dependency on the
# build-script's CODON_TABLE constant.
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

PREFERRED_CODON = {
    "A": "GCT", "C": "TGT", "D": "GAC", "E": "GAA", "F": "TTC",
    "G": "GGT", "H": "CAC", "I": "ATC", "K": "AAG", "L": "TTG",
    "M": "ATG", "N": "AAC", "P": "CCA", "Q": "CAA", "R": "AGA",
    "S": "TCT", "T": "ACT", "V": "GTT", "W": "TGG", "Y": "TAC",
    "*": "TAA",
}

# WT A. victoria GFP, 238 aa (Prasher 1992; UniProt P42212).
GFP_PROTEIN = (  # TODO: check correctness of coding sequence
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGV"
    "QCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNI"
    "LGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQ"
    "SALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)

# TADH1 from R64-1-1 chrXV 159446-159548 (- strand): 103 nt of clean
# intergenic between ADH1's stop and MHF1's CDS, reverse-complemented
# to mRNA-sense. Same source as scripts/chen/build_construct_reference.py.
TADH1_CHROM = "XV"
TADH1_START_1BASED = 159446
TADH1_END_1BASED = 159548

# 0-based protein positions of the 12-codon variable region per library.
# Verified by translating Chen's supp variant_seqs against these slices.
LIBRARY_PROTEIN_POS = {"gfp_r1": 41, "gfp_r2": 156, "tdh3": 56}
LIBRARY_EXPECTED_PEPTIDE = {
    "gfp_r1": "LTLKFICTTGKL",
    "gfp_r2": "QKNGIKVNFKIR",
    "tdh3":   "EVSHDDKHIIVD",
}

# Native TDH3 (YGR192C) on chrVII − strand.
TDH3_CHROM = "VII"
TDH3_START_1BASED = 882815
TDH3_END_1BASED = 883810

VAR_LEN = 36  # nt


_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")
def revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _translate(dna: str) -> str:
    return "".join(CODON_TABLE[dna[i : i + 3]] for i in range(0, len(dna), 3))


# ─── Host metadata ──────────────────────────────────────────────────


@dataclass(frozen=True)
class HostMeta:
    gene_id: str
    gene_name: str
    chrom: str
    strand: str        # '+' or '-'
    cds_start: int     # 1-based inclusive
    cds_end: int       # 1-based inclusive
    tier: str          # 'low' / 'medium' / 'medium_high' / 'high'
    dee2_tpm: float


def load_hosts(path: str | Path) -> list[HostMeta]:
    raw = json.loads(Path(path).read_text())["hosts"]
    return [HostMeta(
        gene_id=h["gene_id"], gene_name=h["gene_name"],
        chrom=h["chrom"], strand=h["strand"],
        cds_start=int(h["cds_start"]), cds_end=int(h["cds_end"]),
        tier=h["tier"], dee2_tpm=float(h["dee2_tpm"]),
    ) for h in raw]


# ─── Cassette construction ──────────────────────────────────────────


def _pull_native_cds(fasta: "pysam.FastaFile", chrom: str, start: int, end: int, strand: str) -> str:
    """1-based inclusive coords. Returns mRNA-sense (revcomp'd if − strand)."""
    seq = fasta.fetch(chrom, start - 1, end).upper()
    return revcomp(seq) if strand == "-" else seq


def _pull_tadh1(fasta: "pysam.FastaFile") -> str:
    return _pull_native_cds(fasta, TADH1_CHROM, TADH1_START_1BASED, TADH1_END_1BASED, "-")


def _build_gfp_cds(ref_r1: str, ref_r2: str) -> str:
    """GFP CDS encoded with preferred yeast codons everywhere except the
    two variable regions (which take row-0 variants from each library)."""
    codons = [PREFERRED_CODON[aa] for aa in GFP_PROTEIN]
    for j, off in enumerate(range(41, 53)):
        codons[off] = ref_r1[3 * j : 3 * j + 3]
    for j, off in enumerate(range(156, 168)):
        codons[off] = ref_r2[3 * j : 3 * j + 3]
    cds = "".join(codons) + PREFERRED_CODON["*"]
    prot = _translate(cds).rstrip("*")
    if prot != GFP_PROTEIN:
        raise RuntimeError("synthesised GFP CDS does not translate to WT GFP")
    return cds


@dataclass(frozen=True)
class Cassette:
    sequence: str           # mRNA-sense, variant_cds + tadh1
    variant_cds_len: int    # nt
    tadh1_len: int
    var_offset_in_cassette: int  # 0-based nt offset of the 36 nt variable region


def build_cassette(
    library: str,
    fasta: "pysam.FastaFile",
    data_dir: Path | str,
) -> Cassette:
    import pandas as pd
    data_dir = Path(data_dir)
    tadh1 = _pull_tadh1(fasta)

    if library in ("gfp_r1", "gfp_r2"):
        ref_r1 = pd.read_csv(data_dir / "gfp_r1.tsv", sep="\t").iloc[0]["variable_seq"].upper()
        ref_r2 = pd.read_csv(data_dir / "gfp_r2.tsv", sep="\t").iloc[0]["variable_seq"].upper()
        variant_cds = _build_gfp_cds(ref_r1, ref_r2)
    elif library == "tdh3":
        variant_cds = _pull_native_cds(fasta, TDH3_CHROM, TDH3_START_1BASED, TDH3_END_1BASED, "-")
        # Sanity-check that the variable region in the native TDH3 CDS translates
        # to the published peptide.
        p = LIBRARY_PROTEIN_POS["tdh3"]
        block = variant_cds[3 * p : 3 * p + VAR_LEN]
        if _translate(block) != LIBRARY_EXPECTED_PEPTIDE["tdh3"]:
            raise RuntimeError(f"TDH3 variable block doesn't translate to "
                               f"{LIBRARY_EXPECTED_PEPTIDE['tdh3']!r}")
    else:
        raise ValueError(f"unknown library {library!r}")

    cassette_seq = variant_cds + tadh1
    var_offset = 3 * LIBRARY_PROTEIN_POS[library]
    return Cassette(
        sequence=cassette_seq,
        variant_cds_len=len(variant_cds),
        tadh1_len=len(tadh1),
        var_offset_in_cassette=var_offset,
    )


# ─── Per-host context ───────────────────────────────────────────────


@dataclass(frozen=True)
class ChenHostContext:
    host: HostMeta
    # All offsets below are 0-based, in modified-chromosome coordinates.
    window_start: int                # start of model-input window
    seq_len: int                     # window length (model architecture)
    window_seq: str                  # the actual modified chrom slice, length seq_len
    cds_bin_lo: int                  # output bin (closed) — variant CDS
    cds_bin_hi: int                  # output bin (open)
    cds_base_lo: int                 # output base position (closed) — variant CDS
    cds_base_hi: int                 # output base position (open)
    var_start_in_window: int         # +strand position of the 36-nt variable block
    var_needs_revcomp: bool          # True if the host is − strand → alt block must be revcomp'd

    @property
    def exon_bins(self) -> np.ndarray:
        return np.arange(self.cds_bin_lo, self.cds_bin_hi, dtype=np.int64)


def build_host_contexts(
    library: str,
    hosts: list[HostMeta],
    fasta: "pysam.FastaFile",
    cassette: Cassette,
    seq_len: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
) -> list[ChenHostContext]:
    """For each host: build the modified chromosome (host CDS replaced
    by ``cassette``, RC if − strand), centre the model window on the
    inserted cassette's CDS, and pre-compute everything an adapter
    needs to score one variant against that host.
    """
    contexts: list[ChenHostContext] = []
    expected_peptide = LIBRARY_EXPECTED_PEPTIDE[library]

    for host in hosts:
        if host.chrom not in fasta.references:
            raise KeyError(f"chromosome {host.chrom!r} not in FASTA")
        chrom_native = fasta.fetch(host.chrom).upper()

        # Splice: replace host CDS span with cassette (or revcomp'd cassette)
        # on the genomic + strand.
        cassette_seq = cassette.sequence if host.strand == "+" else revcomp(cassette.sequence)
        modified = (
            chrom_native[: host.cds_start - 1]
            + cassette_seq
            + chrom_native[host.cds_end :]
        )

        # In modified +strand 0-based coords, the inserted cassette occupies
        # [host.cds_start - 1, host.cds_start - 1 + len(cassette_seq)).
        cassette_lo = host.cds_start - 1                      # 0-based incl.
        cassette_hi = cassette_lo + cassette.variant_cds_len + cassette.tadh1_len  # 0-based excl.

        # Variant CDS portion of the cassette (mRNA-sense reads
        # variant_cds first, TADH1 second):
        if host.strand == "+":
            cds_lo_pos = cassette_lo
            cds_hi_pos = cassette_lo + cassette.variant_cds_len
            var_start_in_chrom = cassette_lo + cassette.var_offset_in_cassette
            var_needs_revcomp = False
        else:
            # Cassette stored revcomp'd on +strand. TADH1 ends up first
            # (low coord side), variant CDS ends up last.
            cds_lo_pos = cassette_lo + cassette.tadh1_len
            cds_hi_pos = cassette_lo + cassette.tadh1_len + cassette.variant_cds_len
            # Variable region (mRNA-sense offset → revcomp +strand offset):
            #   mRNA pos j → revcomp pos cassette_len - 1 - j
            # mRNA-sense 36 nt block at [var_offset_in_cassette, +36)
            # → on +strand at [cassette_len - VAR_LEN - var_offset_in_cassette,
            #                  cassette_len - var_offset_in_cassette)
            cassette_total = cassette.variant_cds_len + cassette.tadh1_len
            var_start_in_chrom = (
                cassette_lo + cassette_total - VAR_LEN - cassette.var_offset_in_cassette
            )
            var_needs_revcomp = True

        # Centre window on the cassette CDS midpoint.
        cassette_cds_center_1based = (cds_lo_pos + cds_hi_pos) // 2 + 1
        window_start = place_window(
            var_pos=cassette_cds_center_1based,
            gene_center=cassette_cds_center_1based,
            chrom_length=len(modified),
            seq_len=seq_len,
            crop_bp_each_side=crop_bp_each_side,
        )

        if (cds_lo_pos < window_start or
            cds_hi_pos > window_start + seq_len):
            raise RuntimeError(
                f"host {host.gene_name}: cassette CDS outside model window — "
                f"window=[{window_start},{window_start + seq_len}), "
                f"cds=[{cds_lo_pos},{cds_hi_pos})"
            )

        var_start_in_window = var_start_in_chrom - window_start
        if not (0 <= var_start_in_window <= seq_len - VAR_LEN):
            raise RuntimeError(
                f"host {host.gene_name}: variable region offset "
                f"{var_start_in_window} outside window"
            )

        # Sanity check: read the 36 nt at var_start_in_window from
        # modified +strand; revcomp if − strand host; verify peptide.
        observed_block = modified[var_start_in_chrom : var_start_in_chrom + VAR_LEN]
        observed_mrna = revcomp(observed_block) if host.strand == "-" else observed_block
        observed_peptide = _translate(observed_mrna)
        if observed_peptide != expected_peptide:
            raise RuntimeError(
                f"{library} at host {host.gene_name}: variable region "
                f"translates to {observed_peptide!r}, expected {expected_peptide!r}"
            )

        # Output bins overlapping the cassette CDS.
        bin_start_bp = window_start + crop_bp_each_side
        bin_lo = max(0, (cds_lo_pos - bin_start_bp) // bin_width)
        bin_hi = min(output_bins, (cds_hi_pos - bin_start_bp + bin_width - 1) // bin_width)
        if bin_hi <= bin_lo:
            raise RuntimeError(
                f"host {host.gene_name}: cassette CDS produces no output bins"
            )
        # Exact per-base CDS span in the cropped output (same base_start_bp
        # as the bin calc; contiguous because the cassette CDS is one span).
        base_lo = max(0, cds_lo_pos - bin_start_bp)
        base_hi = min(output_bins * bin_width, cds_hi_pos - bin_start_bp)

        window_seq = modified[window_start : window_start + seq_len]
        contexts.append(ChenHostContext(
            host=host,
            window_start=window_start,
            seq_len=seq_len,
            window_seq=window_seq,
            cds_bin_lo=int(bin_lo),
            cds_bin_hi=int(bin_hi),
            cds_base_lo=int(base_lo),
            cds_base_hi=int(base_hi),
            var_start_in_window=int(var_start_in_window),
            var_needs_revcomp=var_needs_revcomp,
        ))

    return contexts


def alt_block_oh(variant_seq: str, needs_revcomp: bool) -> np.ndarray:
    """One-hot a 36 nt variant block, RC if the host is − strand. Returns
    channels-first (4, VAR_LEN)."""
    seq = variant_seq.upper()
    if len(seq) != VAR_LEN:
        raise ValueError(f"variant_seq must be {VAR_LEN} nt, got {len(seq)}")
    if needs_revcomp:
        seq = revcomp(seq)
    return one_hot_encode_channels_first(seq)
