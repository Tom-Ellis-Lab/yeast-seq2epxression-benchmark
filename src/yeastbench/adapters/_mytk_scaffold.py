"""Shared infrastructure for the MYTK promoter × integration-site adapter.

Each ``(integration locus, promoter)`` construct inserts the **full
per-promoter payload** (``ConS-[promoter]-mScarlet-tTDH1-ConE``) at the
locus midpoint with native flanks intact — a point insertion, centred in
the window (the ``center_cassette`` anchor, as in Hong) — and reads out
predicted coverage over the **mScarlet CDS**.

This is the promoter-varying sibling of Hong's fixed cassette, so it
reuses the generic :mod:`yeastbench.adapters._cassette_scaffold`
machinery (splice + window placement + readout bins). The only
MYTK-specific parts live here:

- a 4-record payload FASTA (3 payloads named for their promoter +
  one ``mScarlet`` readout reference), and
- locating the mScarlet readout span inside each payload by exact
  substring match — because the promoter length differs per payload, a
  single hardcoded offset (Hong's approach) wouldn't be correct across
  the three.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from yeastbench.adapters._cassette_scaffold import (
    CassetteGeometry,
    InsertionContext,
    InsertionSite,
    build_insertion_context,
)
from yeastbench.adapters.protocols import PromoterIntegrationConstruct

if TYPE_CHECKING:
    import pysam

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAYLOADS_FASTA = (
    REPO_ROOT / "data" / "tasks" / "mytk_ints_promoter" / "mytk_payloads.fasta"
)

# The dedicated readout-reference record; its sequence is located inside
# each payload by exact substring match to define the readout span.
READOUT_RECORD = "mScarlet"
PROMOTER_IDS: tuple[str, ...] = ("pTDH3", "pRPL18B", "pREV1")


def _parse_fasta(fasta_path: str | Path) -> dict[str, str]:
    """Minimal multi-record FASTA parser → {record_name: uppercase_seq}.
    Record name is the first whitespace-delimited token after ``>``."""
    recs: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    for line in Path(fasta_path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                recs[name] = "".join(chunks).upper()
            name = line[1:].strip().split()[0]
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if name is not None:
        recs[name] = "".join(chunks).upper()
    return recs


def load_payload_geometries(
    fasta_path: str | Path = DEFAULT_PAYLOADS_FASTA,
    promoter_ids: tuple[str, ...] = PROMOTER_IDS,
    readout_record: str = READOUT_RECORD,
) -> dict[str, CassetteGeometry]:
    """Build a :class:`CassetteGeometry` per promoter from the payload FASTA.

    Resolves each promoter id to the (unique) payload record whose name
    contains it, then locates the ``readout_record`` sequence inside that
    payload (must occur exactly once) to set the readout span. Fails loud
    if any expectation is violated, so an edited/regenerated FASTA can't
    silently shift the readout.
    """
    recs = _parse_fasta(fasta_path)
    if readout_record not in recs:
        raise ValueError(
            f"readout reference {readout_record!r} not found in {fasta_path} "
            f"(records: {sorted(recs)})"
        )
    readout = recs[readout_record]

    geometries: dict[str, CassetteGeometry] = {}
    for pid in promoter_ids:
        matches = [
            name for name in recs
            if name != readout_record and pid in name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"promoter {pid!r} matched {len(matches)} payload records "
                f"{matches} in {fasta_path}; expected exactly 1"
            )
        payload = recs[matches[0]]
        hits = payload.count(readout)
        if hits != 1:
            raise ValueError(
                f"readout {readout_record!r} occurs {hits}× in payload "
                f"{matches[0]!r}; expected exactly 1"
            )
        start = payload.find(readout)
        geometries[pid] = CassetteGeometry(
            payload=payload, readout_start=start, readout_len=len(readout),
        )
    return geometries


def build_mytk_context(
    construct: PromoterIntegrationConstruct,
    geometries: dict[str, CassetteGeometry],
    fasta: "pysam.FastaFile",
    seq_len: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
    cassette_strand: str = "+",
) -> InsertionContext | None:
    """Splice the construct's payload in at ``integration_coord`` (point
    insertion, native flanks intact) and return the centred model-input
    window + mScarlet-CDS readout positions. ``None`` if the locus is too
    close to a chromosome end or the readout falls outside the crop.
    """
    cassette = geometries[construct.promoter]
    coord = construct.integration_coord
    site = InsertionSite(
        chrom=construct.chrom,
        deletion_start=coord + 1,  # empty span → point insertion AT coord
        deletion_end=coord,
        cassette_strand=cassette_strand,
    )
    return build_insertion_context(
        site, cassette, fasta,
        seq_len, crop_bp_each_side, bin_width, output_bins,
        window_anchor="center_cassette",
    )


__all__ = [
    "DEFAULT_PAYLOADS_FASTA",
    "READOUT_RECORD",
    "PROMOTER_IDS",
    "load_payload_geometries",
    "build_mytk_context",
]
