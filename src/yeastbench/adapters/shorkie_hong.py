"""Shorkie adapter for the Hong et al. IGR-insertion position-effect task.

For each locus, build a 16 kb input = native R64-5-1 with the constant
``TDH3p-mCherry-ADH1t`` cassette spliced in at the experimental cut
site (always + strand), forward-pass through the 8-fold ensemble with
optional RC averaging, take the cross-track mean over the T0 RNA-seq
tracks, and sum over the mCherry-CDS output bins. Absolute readout
(no REF baseline; the signal of interest is the absolute mCherry level
as a function of intergenic position).

Diagnostic B: additionally exposes ``predict_diagnostic_readouts(loci)``
returning per-locus signed scores for **all** (track group × readout
region) combinations the adapter supports. The Hong benchmark uses
this to select the best combination on IntTrain and evaluate on
IntProp. Track groups include T0 RNA-seq plus several Chip-MNase
histone-mark tracks; readout regions span the cassette and immediate
native flanks. Biological signs are applied: active marks and RNA-seq
get sign +1; nucleosome density (H3) gets sign −1.

See ``benchmarks/hong_igr.md`` for the full Primary + Diagnostic B
design.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._hong_scaffold import (
    DEFAULT_CASSETTE_FASTA,
    DIAGNOSTIC_REGION_NAMES,
    HongInsertionContext,
    HongLocus,
    build_insertion_context,
    load_cassette_payload,
    readout_region_bins,
)
from yeastbench.adapters.protocols import IGRInsertionExpressionPredictor
from yeastbench.adapters._shorkie_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    SHORKIE_T0_RNA_SEQ_TRACK_IDS,
)
from yeastbench.models.shorkie import Shorkie

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


# Diagnostic-B track groups: each entry is (group_name, track_indices,
# biological_sign). Sign is applied to the signed score so that
# higher signed score is biologically expected to correlate with
# higher fluorescence (positive ρ). Active marks and RNA-seq use +1;
# nucleosome density (H3 total) uses −1 (more nucleosomes → less
# expression).
def _shorkie_diagnostic_track_groups(
    targets_path: Path,
) -> list[tuple[str, list[int], int]]:
    """Load Shorkie's targets.txt and build the per-track-group index
    lists used by Diagnostic B. Called once at adapter construction
    so the index lookups happen on the CPU side, not in the inference
    loop."""
    df = pd.read_csv(targets_path, sep="\t")

    def _ids(prefix: str, group: str) -> list[int]:
        mask = (
            df["identifier"].astype(str).str.upper().str.startswith(prefix.upper())
            & (df["group"] == group)
        )
        return df.loc[mask, "index"].astype(int).tolist()

    return [
        ("RNA-seq T0",              list(SHORKIE_T0_RNA_SEQ_TRACK_IDS), +1),
        ("H3K27ac",                 _ids("H3K27AC_", "Chip-MNase"),     +1),
        ("H3K4me3",                 _ids("H3K4ME3_", "Chip-MNase"),     +1),
        ("H3K9ac",                  _ids("H3K9AC_",  "Chip-MNase"),     +1),
        ("H3K36me3",                _ids("H3K36ME3_", "Chip-MNase"),    +1),
        ("H3 (nucleosome density)", _ids("H3_S",     "Chip-MNase"),     -1),
    ]


# The Primary readout name (used by the Hong benchmark to identify the
# Primary among the diagnostic dict's keys).
PRIMARY_READOUT_NAME = "RNA-seq T0 × cassette CDS"


class ShorkieHongPredictor(IGRInsertionExpressionPredictor):
    def __init__(
        self,
        model: Shorkie,
        fasta_path: str | Path,
        cassette_fasta: str | Path | None = None,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        batch_size: int = 16,
        targets_path: str | Path | None = None,
    ) -> None:
        import pysam
        import torch as _torch

        self.model = model
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.payload = load_cassette_payload(
            cassette_fasta if cassette_fasta is not None else DEFAULT_CASSETTE_FASTA
        )
        self.track_subset = list(track_subset)
        self.batch_size = batch_size
        self._track_idx_t = _torch.tensor(
            self.track_subset, device=self.model.device, dtype=_torch.long
        )
        # Resolve targets.txt path for Diagnostic B groups
        if targets_path is None:
            targets_path = (
                Path(__file__).resolve().parents[3]
                / "data" / "models" / "shorkie" / "targets.txt"
            )
        self._targets_path = Path(targets_path)
        self._diagnostic_track_groups = (
            _shorkie_diagnostic_track_groups(self._targets_path)
            if self._targets_path.exists() else []
        )

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        fasta_path: str | Path,
        cassette_fasta: str | Path | None = None,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 16,
        use_rc: bool = True,
        targets_path: str | Path | None = None,
    ) -> "ShorkieHongPredictor":
        return cls(
            Shorkie.from_checkpoints(
                params_path, checkpoint_paths, device=device, use_rc=use_rc,
            ),
            fasta_path=fasta_path,
            cassette_fasta=cassette_fasta,
            track_subset=list(track_subset),
            batch_size=batch_size,
            targets_path=targets_path,
        )

    # ── Window-building helpers (shared between primary + diagnostic) ─

    def _build_contexts(
        self, loci: Sequence[HongLocus]
    ) -> tuple[list[tuple[int, HongInsertionContext]], np.ndarray]:
        """Build per-locus insertion contexts + the index of loci that
        produced a valid window."""
        contexts: list[tuple[int, HongInsertionContext]] = []
        for i, locus in enumerate(loci):
            ctx = build_insertion_context(
                locus, self.payload, self.fasta,
                SEQ_LEN, CROP_BP_EACH_SIDE, BIN_WIDTH, OUTPUT_BINS,
            )
            if ctx is not None:
                contexts.append((i, ctx))
        valid_idx = np.array([i for i, _ in contexts], dtype=np.int64)
        return contexts, valid_idx

    # ── Primary readout (fast path; no diagnostic groups) ─────────────

    def predict_expressions(self, loci: Sequence[HongLocus]) -> np.ndarray:
        import torch as _torch

        scores = np.full(len(loci), np.nan, dtype=np.float64)
        contexts, _ = self._build_contexts(loci)

        for bs in tqdm(
            range(0, len(contexts), self.batch_size), desc="Shorkie Hong (primary)"
        ):
            batch = contexts[bs : bs + self.batch_size]
            x = _torch.from_numpy(
                np.stack([
                    one_hot_encode_channels_first(c.window_seq) for _, c in batch
                ])
            ).to(self.model.device)
            with _torch.no_grad():
                cov = self.model.forward_track_mean_binned(x, self._track_idx_t)
            for j, (row_idx, ctx) in enumerate(batch):
                bins_t = _torch.as_tensor(
                    ctx.mcherry_bins, device=self.model.device, dtype=_torch.long
                )
                scores[row_idx] = float(cov[j].index_select(0, bins_t).sum().item())

        return scores

    # ── Diagnostic B readouts (all track groups × all regions) ────────

    def predict_diagnostic_readouts(
        self, loci: Sequence[HongLocus]
    ) -> dict[str, np.ndarray]:
        """Score every (track group × readout region) combination + the
        Primary, with biological signs applied. Returns a flat dict
        keyed by descriptive readout names. Includes a ``'primary'`` key
        (same scores as the entry named ``PRIMARY_READOUT_NAME``)."""
        import torch as _torch

        if not self._diagnostic_track_groups:
            # No targets.txt loaded — degrade to primary-only.
            return {"primary": self.predict_expressions(loci)}

        contexts, valid_idx = self._build_contexts(loci)
        n = len(loci)
        # Per-locus per-bin coverage, per track group.
        # cov_per_group[group_name] is (n, OUTPUT_BINS); rows not in
        # valid_idx are filled with NaN later.
        cov_per_group: dict[str, np.ndarray] = {}
        for group_name, track_ids, _sign in self._diagnostic_track_groups:
            cov_per_group[group_name] = np.full(
                (n, OUTPUT_BINS), np.nan, dtype=np.float64
            )

        # Inference loop: for each batch, run forward_track_mean_binned
        # once per track group. Encoding is shared.
        for bs in tqdm(
            range(0, len(contexts), self.batch_size),
            desc="Shorkie Hong (diagnostic)",
        ):
            batch = contexts[bs : bs + self.batch_size]
            x = _torch.from_numpy(
                np.stack([
                    one_hot_encode_channels_first(c.window_seq) for _, c in batch
                ])
            ).to(self.model.device)
            for group_name, track_ids, _sign in self._diagnostic_track_groups:
                track_idx_t = _torch.as_tensor(
                    track_ids, device=self.model.device, dtype=_torch.long,
                )
                with _torch.no_grad():
                    cov = self.model.forward_track_mean_binned(x, track_idx_t)
                arr = cov.cpu().numpy()
                for j, (row_idx, _ctx) in enumerate(batch):
                    cov_per_group[group_name][row_idx] = arr[j]

        # Sum coverage over each readout region (regions are uniform
        # across loci because the cassette is centered identically).
        # Use any valid context to compute the bin sets.
        if not contexts:
            # No valid contexts at all
            return {"primary": np.full(n, np.nan, dtype=np.float64)}
        region_bins = readout_region_bins(
            contexts[0][1], CROP_BP_EACH_SIDE, BIN_WIDTH, OUTPUT_BINS,
        )

        readouts: dict[str, np.ndarray] = {}
        for group_name, _track_ids, sign in self._diagnostic_track_groups:
            cov = cov_per_group[group_name]
            for region_name, bins in region_bins.items():
                if len(bins) == 0:
                    scores = np.full(n, np.nan, dtype=np.float64)
                else:
                    scores = sign * cov[:, bins].sum(axis=1)
                # Loci whose context was invalid keep NaN
                invalid = np.setdiff1d(np.arange(n), valid_idx, assume_unique=False)
                scores[invalid] = np.nan
                readouts[f"{group_name} × {region_name}"] = scores

        # Primary alias
        readouts["primary"] = readouts[PRIMARY_READOUT_NAME]
        return readouts
