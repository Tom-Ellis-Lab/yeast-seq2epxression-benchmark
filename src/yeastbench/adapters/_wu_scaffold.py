"""Wu et al. RFP-insertion benchmark — cassette geometry + locus type.

Position-effect task: one *constant* RFP cassette is integrated at the
YKO *kanMX* deletion locus of 1044 single-ORF deletion strains; only
the genomic neighbourhood varies. The cassette payload's geometry
(``PAYLOAD_LEN`` / ``RFP_CDS_*``) is fixed by
``scripts/wu/build_cassette_fasta.py``; the splicing + window-placement
machinery lives in ``_cassette_scaffold.py`` and is shared with Hong.

Cassette payload (frozen, see ``benchmarks/wu_rfpins.md``):

    U1(18) + UPTAG(20=N) + U2(18) + RFP-TU-core(3410) + D2(19) + DNTAG(20=N) + D1(17)
    = 3522 bp.  mCherry CDS at payload offset 554, length 711.

Window-anchor strategy: ``readout_at_downstream_edge`` — the mCherry
stop codon sits at the downstream edge of the readable output crop,
maximising upstream context (where the deleted ORF's natural promoter
would have lived).
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from yeastbench.adapters._cassette_scaffold import (
    CassetteGeometry,
    InsertionSite,
    build_insertion_context as _build_insertion_context_generic,
    load_cassette_payload as _load_cassette_payload_generic,
    reverse_complement,
    span_to_bins,
)
from yeastbench.adapters._genome import Gene

if TYPE_CHECKING:
    import pysam

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASSETTE_FASTA = (
    REPO_ROOT / "data" / "tasks" / "wu_rfpins" / "expression_cassette.fasta"
)
DEFAULT_BARCODES_TSV = (
    REPO_ROOT / "data" / "tasks" / "wu_rfpins" / "barcodes.tsv"
)

# ── Cassette geometry ─────────────────────────────────────────
PAYLOAD_LEN = 3522
RFP_CDS_START_IN_PAYLOAD = 554   # 0-based; = U1+UPTAG+U2 (56) + core offset 498
RFP_CDS_LEN = 711                # mCherry CDS incl. stop codon

# ── Barcode slots + flanking universal sites (forward payload coords) ──
# The two strain-specific 20 bp barcodes sit between constant universal
# priming sites: U1-[UPTAG]-U2 near the 5′ end, D2-[DNTAG]-D1 near the 3′
# end.  ``build_cassette_fasta.py`` freezes both slots as 20×N; the
# adapters inject the real per-locus tags here (a run of N one-hot-encodes
# to all-zero columns — out-of-distribution for the models).
TAG_LEN = 20
UPTAG_SLOT = (18, 38)
DNTAG_SLOT = (3485, 3505)
U1_SEQ = "GATGTCCACGAGGTCTCT"
U2_SEQ = "CGTACGCTGCAGGTCGAC"
D2_SEQ = "ATCGATGAATTCGAGCTCG"
D1_SEQ = "CGGTGTCGGTCTCGTAG"


def _is_tag(s: str) -> bool:
    return len(s) == TAG_LEN and set(s) <= set("ACGT")


def load_barcodes(path: str | Path) -> dict[str, tuple[str, str]]:
    """Read ``barcodes.tsv`` → ``{ORF_name: (uptag, dntag)}``.

    Both tags are validated as bare 20 bp ACGT (no N), in cassette
    top-strand orientation (``uptag`` between U1/U2, ``dntag`` between
    D2/D1).  Built by ``scripts/wu/build_barcodes.py``.
    """
    out: dict[str, tuple[str, str]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            orf = row["ORF_name"].strip()
            up = row["uptag"].strip().upper()
            dn = row["dntag"].strip().upper()
            if not _is_tag(up) or not _is_tag(dn):
                raise ValueError(
                    f"barcodes.tsv: {orf} has non-20bp/non-ACGT tag "
                    f"(uptag={up!r}, dntag={dn!r})"
                )
            out[orf] = (up, dn)
    return out


def inject_barcodes(payload: str, uptag: str, dntag: str) -> str:
    """Return ``payload`` with the UPTAG/DNTAG slots replaced by real tags.

    Anchor-validated: the flanking universal sites (U1/U2 around UPTAG,
    D2/D1 around DNTAG) must be intact before and after, so a wrong offset
    or a corrupted scaffold fails loudly rather than silently mis-splicing.
    Length is preserved (20 bp → 20 bp), so the frozen cassette geometry
    (``RFP_CDS_*``) is unchanged.
    """
    if len(payload) != PAYLOAD_LEN:
        raise ValueError(f"payload len {len(payload)} != {PAYLOAD_LEN}")
    if not _is_tag(uptag) or not _is_tag(dntag):
        raise ValueError(f"tags must be 20bp ACGT (uptag={uptag!r}, dntag={dntag!r})")
    if payload[0:18] != U1_SEQ or payload[38:56] != U2_SEQ:
        raise ValueError("UPTAG slot flanks (U1/U2) do not match the cassette")
    if payload[3466:3485] != D2_SEQ or payload[3505:3522] != D1_SEQ:
        raise ValueError("DNTAG slot flanks (D2/D1) do not match the cassette")

    us, ue = UPTAG_SLOT
    ds, de = DNTAG_SLOT
    new = payload[:us] + uptag + payload[ue:ds] + dntag + payload[de:]

    assert len(new) == PAYLOAD_LEN
    assert new[us:ue] == uptag and new[ds:de] == dntag
    assert new[0:18] == U1_SEQ and new[38:56] == U2_SEQ
    assert new[3466:3485] == D2_SEQ and new[3505:3522] == D1_SEQ
    return new


def load_cassette_payload(fasta_path: str | Path) -> str:
    """Read the single-record cassette FASTA → uppercase payload string.

    Asserts the frozen Wu geometry so a regenerated/edited FASTA that no
    longer matches the spec fails loudly. See
    ``_cassette_scaffold.load_cassette_payload`` for the generic
    implementation.
    """
    return _load_cassette_payload_generic(
        fasta_path,
        expected_len=PAYLOAD_LEN,
        expected_readout_start=RFP_CDS_START_IN_PAYLOAD,
        expected_readout_len=RFP_CDS_LEN,
    )


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
    accounting for − strand reverse-complementing. Wu-specific because
    it knows the cassette's ``PAYLOAD_LEN``."""
    if payload_rc:
        lo = payload_start_in_window + (PAYLOAD_LEN - b)
        hi = payload_start_in_window + (PAYLOAD_LEN - a)
    else:
        lo = payload_start_in_window + a
        hi = payload_start_in_window + b
    return lo, hi


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
    """Wu-named view over an ``InsertionContext`` — preserves the
    historical public fields (``gene_id``, ``rfp_bins``, ``locus``,
    ``rfp_start_in_window``) plus ``rfp_base_positions`` for per-base
    (untransformed) scoring."""
    gene_id: str
    window_seq: str
    rfp_bins: np.ndarray
    rfp_base_positions: np.ndarray  # per-base positions over the mCherry CDS
    locus: WuLocus
    up_avail: int
    window_start_in_spliced: int
    payload_rc: bool
    payload_start_in_window: int
    rfp_start_in_window: int


def build_insertion_context(
    locus: WuLocus,
    payload: str,
    fasta: "pysam.FastaFile",
    seq_len: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
) -> WuInsertionContext | None:
    """Build the SEQ_LEN model input for one Wu locus, splicing the
    cassette in place of the deleted ORF span and anchoring the window
    with the mCherry stop codon at the downstream crop edge. Returns
    ``None`` if the locus is too close to a chromosome end to form a
    full window or the readout falls outside the output crop.
    """
    cassette = CassetteGeometry(
        payload=payload,
        readout_start=RFP_CDS_START_IN_PAYLOAD,
        readout_len=RFP_CDS_LEN,
    )
    site = InsertionSite(
        chrom=locus.chrom,
        deletion_start=locus.gene_start,
        deletion_end=locus.gene_end,
        cassette_strand=locus.strand,
    )
    ctx = _build_insertion_context_generic(
        site, cassette, fasta,
        seq_len, crop_bp_each_side, bin_width, output_bins,
        window_anchor="readout_at_downstream_edge",
    )
    if ctx is None:
        return None
    return WuInsertionContext(
        gene_id=locus.gene_id,
        window_seq=ctx.window_seq,
        rfp_bins=ctx.readout_bins,
        rfp_base_positions=ctx.readout_base_positions,
        locus=locus,
        up_avail=ctx.up_avail,
        window_start_in_spliced=ctx.window_start_in_spliced,
        payload_rc=ctx.payload_rc,
        payload_start_in_window=ctx.payload_start_in_window,
        rfp_start_in_window=ctx.readout_start_in_window,
    )


__all__ = [
    "DEFAULT_CASSETTE_FASTA",
    "DEFAULT_BARCODES_TSV",
    "PAYLOAD_LEN",
    "RFP_CDS_START_IN_PAYLOAD",
    "RFP_CDS_LEN",
    "UPTAG_SLOT",
    "DNTAG_SLOT",
    "CASSETTE_FEATURES",
    "reverse_complement",
    "load_cassette_payload",
    "load_barcodes",
    "inject_barcodes",
    "payload_feature_window_span",
    "span_to_bins",
    "WuLocus",
    "WuInsertionContext",
    "resolve_loci",
    "build_insertion_context",
]
