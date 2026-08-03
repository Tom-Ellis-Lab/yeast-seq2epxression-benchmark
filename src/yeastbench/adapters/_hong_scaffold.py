"""Hong et al. IGR-insertion benchmark — cassette geometry + locus type.

Position-effect task: one *constant* ``TDH3p-mCherry-ADH1t`` cassette
is CRISPR-Cas9 integrated at 150 intergenic regions in *S. cerevisiae*.
The cassette is always integrated on the chromosome's ``+`` strand
(donor PCR-primer orientation; see ``docs/benchmarks/hong_igr.md``), so the
locus dataclass carries no strand field. The splicing + window-
placement machinery lives in ``_cassette_scaffold.py`` and is shared
with Wu.

Cassette payload (frozen by ``scripts/hong/build_hong_distribution.py``
from Hong Supp Table S3 row 1):

    TDH3p (706 bp) + mCherry CDS (711 bp incl. stop) + ADH1t (198 bp)
    = 1595 bp.  mCherry CDS at payload offset 686.

Window-anchor strategy: ``center_cassette`` — cassette midpoint at
window midpoint, balanced native flank on both sides. The Hong assay
inserts between two intact native genes, so signal can plausibly come
from either direction; centering avoids privileging upstream over
downstream the way Wu does.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from yeastbench.adapters._cassette_scaffold import (
    CassetteGeometry,
    InsertionSite,
    build_insertion_context as _build_insertion_context_generic,
    load_cassette_payload as _load_cassette_payload_generic,
)

if TYPE_CHECKING:
    import pysam

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASSETTE_FASTA = (
    REPO_ROOT / "data" / "tasks" / "hong" / "expression_cassette.fasta"
)
DEFAULT_FASTA = REPO_ROOT / "data" / "tasks" / "R64-5-1.fa"
DEFAULT_LABELS_TSV = REPO_ROOT / "data" / "tasks" / "hong" / "hong_igr_v1.tsv"

# ── Cassette geometry ─────────────────────────────────────────
PAYLOAD_LEN = 1595
MCHERRY_CDS_START_IN_PAYLOAD = 686    # 0-based; = TDH3p length
MCHERRY_CDS_LEN = 711                 # incl. TAA stop


def load_cassette_payload(fasta_path: str | Path) -> str:
    """Read the single-record cassette FASTA → uppercase payload string.

    Asserts the frozen Hong geometry so a regenerated/edited FASTA that
    no longer matches the spec fails loudly.
    """
    return _load_cassette_payload_generic(
        fasta_path,
        expected_len=PAYLOAD_LEN,
        expected_readout_start=MCHERRY_CDS_START_IN_PAYLOAD,
        expected_readout_len=MCHERRY_CDS_LEN,
    )


# ── Loci ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class HongLocus:
    """One integration locus = an intergenic region's experimental
    cut site, resolved to R64-5-1 by gRNA matching."""
    locus_id: str           # "IntTrain42" / "IntProp5" / ...
    set: str                # "IntTrain" or "IntProp"
    chrom: str              # roman, "I".."XVI" (matches R64-5-1.fa)
    integration_coord: int  # 1-based; cassette inserts AT this position


# ── Insertion context (one model-input window per locus) ──────


@dataclass(frozen=True)
class HongInsertionContext:
    """One model-input window for a Hong locus. ``mcherry_bins`` is the
    binned readout; ``mcherry_base_positions`` is the per-base analogue
    (for per-base, untransformed scoring)."""
    locus_id: str
    window_seq: str
    mcherry_bins: np.ndarray
    mcherry_base_positions: np.ndarray
    locus: HongLocus
    up_avail: int
    window_start_in_spliced: int
    payload_start_in_window: int
    mcherry_start_in_window: int


def build_insertion_context(
    locus: HongLocus,
    payload: str,
    fasta: "pysam.FastaFile",
    seq_len: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
) -> HongInsertionContext | None:
    """Build the SEQ_LEN model input for one Hong locus, splicing the
    cassette at the experimental cut site (always ``+`` strand) and
    centering the cassette in the window. Returns ``None`` if the locus
    is too close to a chromosome end to form a full window or the
    readout falls outside the output crop.
    """
    cassette = CassetteGeometry(
        payload=payload,
        readout_start=MCHERRY_CDS_START_IN_PAYLOAD,
        readout_len=MCHERRY_CDS_LEN,
    )
    site = InsertionSite(
        chrom=locus.chrom,
        deletion_start=locus.integration_coord + 1,
        deletion_end=locus.integration_coord,
        cassette_strand="+",
    )
    ctx = _build_insertion_context_generic(
        site, cassette, fasta,
        seq_len, crop_bp_each_side, bin_width, output_bins,
        window_anchor="center_cassette",
    )
    if ctx is None:
        return None
    # Per-base positions of the mCherry CDS in the cropped output. Derived
    # here (not in the shared cassette scaffold) so this conversion leaves
    # _cassette_scaffold.py untouched and independent of the Wu change.
    mcherry_base_positions = _mcherry_base_positions(
        ctx.readout_start_in_window, crop_bp_each_side, output_bins * bin_width
    )
    return HongInsertionContext(
        locus_id=locus.locus_id,
        window_seq=ctx.window_seq,
        mcherry_bins=ctx.readout_bins,
        mcherry_base_positions=mcherry_base_positions,
        locus=locus,
        up_avail=ctx.up_avail,
        window_start_in_spliced=ctx.window_start_in_spliced,
        payload_start_in_window=ctx.payload_start_in_window,
        mcherry_start_in_window=ctx.readout_start_in_window,
    )


def _mcherry_base_positions(
    mcherry_start_in_window: int, crop_bp_each_side: int, out_len: int
) -> np.ndarray:
    """Per-base positions in ``[0, out_len)`` covering the mCherry CDS.
    ``mcherry_start_in_window`` is 0-based, so no ``-1`` offset."""
    lo = max(0, mcherry_start_in_window - crop_bp_each_side)
    hi = min(out_len, mcherry_start_in_window + MCHERRY_CDS_LEN - crop_bp_each_side)
    return np.arange(lo, hi, dtype=np.int64) if hi > lo else np.array([], dtype=np.int64)


# ── Diagnostic-B readout regions ──────────────────────────────


# Names of the candidate readout regions used by the IntTrain-fitted
# IntProp ρ candidate space. The benchmark's IntTrain selection
# iterates over all (track_group × region) combinations the adapter
# exposes; this list of regions is fixed across adapters so cross-
# model results are comparable.
DIAGNOSTIC_REGION_NAMES: tuple[str, ...] = (
    "cassette CDS",
    "cassette full",
    "flank L 1kb",
    "flank R 1kb",
    "flank both 1kb",
    "flank both 3kb",
)


def _region_spans_bp(
    ctx: HongInsertionContext,
) -> dict[str, list[tuple[int, int]]]:
    """Window-coordinate bp spans for each named readout region — the one
    source of region geometry, shared by the bin and per-base helpers so
    they cannot drift. Window positions are derived from the context's
    payload placement: the cassette occupies ``[payload_start_in_window,
    +PAYLOAD_LEN]`` and the mCherry CDS ``[mcherry_start_in_window,
    +MCHERRY_CDS_LEN]``; flanks are the adjacent native sequence."""
    cassette_lo = ctx.payload_start_in_window
    cassette_hi = ctx.payload_start_in_window + PAYLOAD_LEN
    mcherry_lo = ctx.mcherry_start_in_window
    mcherry_hi = ctx.mcherry_start_in_window + MCHERRY_CDS_LEN
    return {
        "cassette CDS":   [(mcherry_lo, mcherry_hi)],
        "cassette full":  [(cassette_lo, cassette_hi)],
        "flank L 1kb":    [(cassette_lo - 1000, cassette_lo)],
        "flank R 1kb":    [(cassette_hi, cassette_hi + 1000)],
        "flank both 1kb": [(cassette_lo - 1000, cassette_lo),
                           (cassette_hi, cassette_hi + 1000)],
        "flank both 3kb": [(cassette_lo - 3000, cassette_lo),
                           (cassette_hi, cassette_hi + 3000)],
    }


def _spans_to_indices(
    spans: list[tuple[int, int]], to_idx
) -> np.ndarray:
    """Map bp spans → output-index array via ``to_idx(lo_bp, hi_bp)``,
    unioning multiple spans (e.g. the two-sided flank regions)."""
    combined = to_idx(*spans[0])
    for lo, hi in spans[1:]:
        combined = np.union1d(combined, to_idx(lo, hi))
    return combined


def readout_region_bins(
    ctx: HongInsertionContext,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
) -> dict[str, np.ndarray]:
    """Output-bin indices for each named readout region (binned scale)."""

    def _bins(lo_bp: int, hi_bp: int) -> np.ndarray:
        b_lo = max(0, (lo_bp - crop_bp_each_side) // bin_width)
        b_hi = min(
            output_bins,
            (hi_bp - crop_bp_each_side + bin_width - 1) // bin_width,
        )
        return np.arange(b_lo, b_hi, dtype=np.int64) if b_hi > b_lo else np.array(
            [], dtype=np.int64
        )

    return {
        name: _spans_to_indices(spans, _bins)
        for name, spans in _region_spans_bp(ctx).items()
    }


def readout_region_base_positions(
    ctx: HongInsertionContext,
    crop_bp_each_side: int,
    out_len: int,
) -> dict[str, np.ndarray]:
    """Per-base analogue of :func:`readout_region_bins`: exact base
    positions in ``[0, out_len)`` for each readout region. ``out_len`` is
    ``output_bins * bin_width``; window positions are 0-based so there is
    no ``-1`` offset."""

    def _bases(lo_bp: int, hi_bp: int) -> np.ndarray:
        lo = max(0, lo_bp - crop_bp_each_side)
        hi = min(out_len, hi_bp - crop_bp_each_side)
        return np.arange(lo, hi, dtype=np.int64) if hi > lo else np.array(
            [], dtype=np.int64
        )

    return {
        name: _spans_to_indices(spans, _bases)
        for name, spans in _region_spans_bp(ctx).items()
    }


def _aggregate_diagnostic(
    region_fn,
    group_cov: dict[str, np.ndarray],
    contexts: Sequence[tuple[int, "HongInsertionContext"]],
    track_groups: Iterable[tuple[str, object, int]],
    n: int,
) -> dict[str, np.ndarray]:
    """Shared core: sum per-locus coverage over each named readout region,
    per track group, with biological sign applied. ``region_fn(ctx)``
    yields ``{region_name: index_array}`` and is computed from *each
    locus's own context* — required when a locus's window is clamped
    against a chromosome end (the cassette is no longer at the window
    midpoint, so a shared index set would land on native sequence)."""
    readouts: dict[str, np.ndarray] = {
        f"{group_name} × {region_name}": np.full(n, np.nan, dtype=np.float64)
        for group_name in group_cov
        for region_name in DIAGNOSTIC_REGION_NAMES
    }
    for row_idx, ctx in contexts:
        regions = region_fn(ctx)
        for group_name, _idx, sign in track_groups:
            if group_name not in group_cov:
                continue
            cov_row = group_cov[group_name][row_idx]
            for region_name, idx in regions.items():
                if len(idx) > 0:
                    readouts[f"{group_name} × {region_name}"][row_idx] = float(
                        sign * cov_row[idx].sum()
                    )
    return readouts


def aggregate_diagnostic_readouts(
    group_cov: dict[str, np.ndarray],
    contexts: Sequence[tuple[int, "HongInsertionContext"]],
    track_groups: Iterable[tuple[str, object, int]],
    n: int,
    crop_bp_each_side: int,
    bin_width: int,
    output_bins: int,
) -> dict[str, np.ndarray]:
    """Binned-scale diagnostic aggregation (see :func:`_aggregate_diagnostic`)."""
    return _aggregate_diagnostic(
        lambda ctx: readout_region_bins(
            ctx, crop_bp_each_side, bin_width, output_bins
        ),
        group_cov, contexts, track_groups, n,
    )


def aggregate_diagnostic_base_readouts(
    group_cov: dict[str, np.ndarray],
    contexts: Sequence[tuple[int, "HongInsertionContext"]],
    track_groups: Iterable[tuple[str, object, int]],
    n: int,
    crop_bp_each_side: int,
    out_len: int,
) -> dict[str, np.ndarray]:
    """Per-base diagnostic aggregation: ``group_cov`` rows are per-base
    (length ``out_len``) and regions are exact base positions."""
    return _aggregate_diagnostic(
        lambda ctx: readout_region_base_positions(ctx, crop_bp_each_side, out_len),
        group_cov, contexts, track_groups, n,
    )


__all__ = [
    "DEFAULT_CASSETTE_FASTA",
    "DEFAULT_FASTA",
    "DEFAULT_LABELS_TSV",
    "PAYLOAD_LEN",
    "MCHERRY_CDS_START_IN_PAYLOAD",
    "MCHERRY_CDS_LEN",
    "DIAGNOSTIC_REGION_NAMES",
    "load_cassette_payload",
    "readout_region_bins",
    "readout_region_base_positions",
    "aggregate_diagnostic_readouts",
    "aggregate_diagnostic_base_readouts",
    "HongLocus",
    "HongInsertionContext",
    "build_insertion_context",
]
