"""Shared infrastructure for the Wu et al. RFP-insertion benchmark.

Position-effect task: one *constant* RFP cassette is integrated at the
YKO *kanMX* deletion locus of 1044 single-ORF deletion strains; only the
genomic neighbourhood varies.  For each locus an adapter builds a model
input window = native R64-1-1 sequence with the deleted ORF replaced by
the verified cassette payload, then reads out the mCherry CDS.

Cassette payload (frozen, see ``benchmarks/wu_rfpins.md`` and
``scripts/wu/build_cassette_fasta.py``):

    U1(18) + UPTAG(20=N) + U2(18) + RFP-TU-core(3410) + D2(19) + DNTAG(20=N) + D1(17)
    = 3522 bp.  mCherry CDS at payload offset 554, length 711.

There is **no REF baseline** (no "reference" without the cassette); the
readout is the *absolute* cross-track-mean coverage summed over the
mCherry-CDS bins.  Orientation: the cassette transcribes in the deleted
ORF's direction, so for − strand ORFs the payload is reverse-complemented.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from yeastbench.adapters._genome import Gene

if TYPE_CHECKING:
    import pysam

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASSETTE_FASTA = (
    REPO_ROOT / "data" / "tasks" / "wu_rfpins" / "expression_cassette.fasta"
)

# ── Cassette geometry ─────────────────────────────────────────
PAYLOAD_LEN = 3522
RFP_CDS_START_IN_PAYLOAD = 554   # 0-based; = U1+UPTAG+U2 (56) + core offset 498
RFP_CDS_LEN = 711                # mCherry CDS incl. stop codon

_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def load_cassette_payload(fasta_path: str | Path) -> str:
    """Read the single-record cassette FASTA → uppercase payload string.

    Asserts the frozen geometry so a regenerated/edited FASTA that no
    longer matches the spec fails loudly instead of silently shifting the
    readout window.
    """
    lines = Path(fasta_path).read_text().splitlines()
    seq = "".join(ln.strip() for ln in lines if ln and not ln.startswith(">")).upper()
    if len(seq) != PAYLOAD_LEN:
        raise ValueError(
            f"cassette payload is {len(seq)} bp, expected {PAYLOAD_LEN} "
            f"({fasta_path}); regenerate with scripts/wu/build_cassette_fasta.py"
        )
    rfp = seq[RFP_CDS_START_IN_PAYLOAD : RFP_CDS_START_IN_PAYLOAD + RFP_CDS_LEN]
    if not rfp.startswith("ATG"):
        raise ValueError(
            "mCherry CDS does not start with ATG at the expected payload "
            f"offset {RFP_CDS_START_IN_PAYLOAD} — cassette FASTA layout drift"
        )
    return seq


# Cassette sub-feature spans in forward-payload coordinates (0-based,
# half-open). Derived from the GenBank record (core = GB 210..3619 →
# payload offset 56; verified by scripts/wu/verify_cassette.py).
CASSETTE_FEATURES: tuple[tuple[str, str, int, int], ...] = (
    ("U1", "scar", 0, 18),
    ("UPTAG", "barcode", 18, 38),
    ("U2", "scar", 38, 56),
    ("tCYC1", "terminator", 56, 298),
    ("pURA3", "promoter", 304, 540),
    ("mCherry", "reporter_cds", RFP_CDS_START_IN_PAYLOAD,
     RFP_CDS_START_IN_PAYLOAD + RFP_CDS_LEN),
    ("tADH1", "terminator", 1288, 1616),
    ("pLEU2", "promoter", 1624, 1997),
    ("LEU2", "marker_cds", 1997, 3092),
    ("tLEU2", "terminator", 3092, 3434),
    ("D2", "scar", 3466, 3485),
    ("DNTAG", "barcode", 3485, 3505),
    ("D1", "scar", 3505, 3522),
)


def payload_feature_window_span(
    a: int, b: int, payload_start_in_window: int, payload_rc: bool
) -> tuple[int, int]:
    """Forward-payload span ``[a, b)`` → window-coordinate span,
    accounting for − strand reverse-complementing."""
    if payload_rc:
        lo = payload_start_in_window + (PAYLOAD_LEN - b)
        hi = payload_start_in_window + (PAYLOAD_LEN - a)
    else:
        lo = payload_start_in_window + a
        hi = payload_start_in_window + b
    return lo, hi


def span_to_bins(
    win_lo: int,
    win_hi: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
) -> tuple[int, int] | None:
    """Window span → ``[b_lo, b_hi)`` output-bin range, or ``None`` if it
    does not overlap the readable crop."""
    b_lo = max(0, (win_lo - crop_bp_each_side) // bin_width)
    b_hi = min(
        output_bins, (win_hi - crop_bp_each_side + bin_width - 1) // bin_width
    )
    return (b_lo, b_hi) if b_hi > b_lo else None


# ── Loci ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class WuLocus:
    """One integration locus = a deleted ORF, resolved to R64-1-1."""
    gene_id: str
    chrom: str          # roman
    strand: str         # '+' / '-'
    gene_start: int     # 1-based inclusive (ORF span removed and replaced)
    gene_end: int       # 1-based inclusive


def resolve_loci(
    orf_names: list[str], gtf_genes: dict[str, Gene]
) -> tuple[list[WuLocus | None], list[str]]:
    """Map each ORF systematic name → WuLocus (None if unresolved).

    Returns ``(loci_aligned_to_input, dropped_ids)``.  Alignment to the
    input order is preserved; unresolved ORFs are ``None`` so the
    benchmark can keep label/score arrays row-aligned and report drops.
    """
    loci: list[WuLocus | None] = []
    dropped: list[str] = []
    for gid in orf_names:
        g = gtf_genes.get(gid)
        if g is None:
            loci.append(None)
            dropped.append(gid)
            continue
        loci.append(
            WuLocus(
                gene_id=gid,
                chrom=g.chrom_roman,
                strand=g.strand,
                gene_start=g.gene_start,
                gene_end=g.gene_end,
            )
        )
    return loci, dropped


# ── Insertion context (one model-input window per locus) ──────


@dataclass(frozen=True)
class WuInsertionContext:
    gene_id: str
    window_seq: str          # SEQ_LEN, forward genomic orientation
    rfp_bins: np.ndarray     # output-bin indices overlapping the mCherry CDS
    # ── placement metadata (for annotation / inspection) ──
    locus: WuLocus
    up_avail: int                  # native bp upstream of the ORF in `spliced`
    window_start_in_spliced: int   # window offset within native_up+payload+native_down
    payload_rc: bool               # payload reverse-complemented (− strand ORF)
    payload_start_in_window: int   # window index where the oriented payload begins
    rfp_start_in_window: int       # window index of the mCherry CDS 5' (genomic +)


def _rfp_output_bins(
    rfp_start_in_window: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
) -> np.ndarray:
    """Output-bin indices overlapping the mCherry CDS, same convention as
    ``_genome.gene_exon_bins`` (window-relative coords)."""
    lo = rfp_start_in_window
    hi = rfp_start_in_window + RFP_CDS_LEN
    b_lo = max(0, (lo - crop_bp_each_side) // bin_width)
    b_hi = min(
        output_bins,
        (hi - crop_bp_each_side + bin_width - 1) // bin_width,
    )
    if b_hi <= b_lo:
        return np.array([], dtype=np.int64)
    return np.arange(b_lo, b_hi, dtype=np.int64)


def build_insertion_context(
    locus: WuLocus,
    payload: str,
    fasta: "pysam.FastaFile",
    seq_len: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
) -> WuInsertionContext | None:
    """Build the SEQ_LEN model input for one locus.

    Splices the cassette payload in place of the ORF ``[gene_start,
    gene_end]`` (nominal SGDP deletion boundary) and positions the
    cassette so the **mCherry stop codon sits at the downstream edge of
    the readable output crop**, maximising the genomic context *upstream*
    of the reporter (where the position-effect signal lives).  For −
    strand ORFs the payload is reverse-complemented, so transcription
    runs genomic-right→left and "downstream" is the genomic-low side; the
    anchor mirrors accordingly.  Returns ``None`` if the locus is too
    close to a chromosome end to form a full window or the readout window
    does not land in the output crop (caller scores it NaN and reports
    it).
    """
    chrom_len = fasta.get_reference_length(locus.chrom)
    Lp = len(payload)

    if locus.strand == "+":
        oriented = payload
        rfp0 = RFP_CDS_START_IN_PAYLOAD
    else:
        oriented = reverse_complement(payload)
        rfp0 = Lp - RFP_CDS_START_IN_PAYLOAD - RFP_CDS_LEN

    up_avail = min(seq_len, locus.gene_start - 1)
    down_avail = min(seq_len, chrom_len - locus.gene_end)
    native_up = fasta.fetch(
        locus.chrom, locus.gene_start - 1 - up_avail, locus.gene_start - 1
    ).upper()
    native_down = fasta.fetch(
        locus.chrom, locus.gene_end, locus.gene_end + down_avail
    ).upper()

    spliced = native_up + oriented + native_down
    if len(spliced) < seq_len:
        return None  # too close to a chromosome end

    cds_lo = up_avail + rfp0                 # mCherry CDS 5' in `spliced` (genomic +)
    cds_hi = cds_lo + RFP_CDS_LEN            # mCherry CDS 3' in `spliced` (genomic +)
    # Put the reporter's transcription STOP codon at the downstream edge
    # of the readable output crop, maximising upstream context.
    #   + strand ORF: transcription is genomic-L→R; stop = cds_hi → right
    #                  crop edge  (window index seq_len − crop).
    #   − strand ORF: payload RC'd, transcription is genomic-R→L; stop =
    #                  cds_lo → left crop edge (window index crop).
    if locus.strand == "+":
        window_start = cds_hi - (seq_len - crop_bp_each_side)
    else:
        window_start = cds_lo - crop_bp_each_side
    window_start = max(0, min(window_start, len(spliced) - seq_len))

    window_seq = spliced[window_start : window_start + seq_len]
    rfp_start_in_window = cds_lo - window_start
    rfp_bins = _rfp_output_bins(
        rfp_start_in_window, crop_bp_each_side, bin_width, output_bins
    )
    if rfp_bins.size == 0:
        return None  # mCherry CDS fell outside the output crop

    return WuInsertionContext(
        gene_id=locus.gene_id,
        window_seq=window_seq,
        rfp_bins=rfp_bins,
        locus=locus,
        up_avail=up_avail,
        window_start_in_spliced=window_start,
        payload_rc=(locus.strand == "-"),
        payload_start_in_window=up_avail - window_start,
        rfp_start_in_window=rfp_start_in_window,
    )


__all__ = [
    "DEFAULT_CASSETTE_FASTA",
    "PAYLOAD_LEN",
    "RFP_CDS_START_IN_PAYLOAD",
    "RFP_CDS_LEN",
    "CASSETTE_FEATURES",
    "reverse_complement",
    "load_cassette_payload",
    "payload_feature_window_span",
    "span_to_bins",
    "WuLocus",
    "WuInsertionContext",
    "resolve_loci",
    "build_insertion_context",
]
