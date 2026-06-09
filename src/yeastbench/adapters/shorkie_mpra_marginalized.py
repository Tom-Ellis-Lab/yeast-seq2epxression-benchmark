"""Shorkie marginalized / native-position adapter for Rafi MPRA.

Inserts the 110 bp MPRA sequence at native yeast genome positions upstream
of 22 host genes and scores logSED (log2 fold-change in predicted
expression) between the native REF and the edited ALT: cross-track mean
over T0 RNA-seq tracks → sum over exon bins → mean across offsets → mean
across host genes.

The REF→ALT→logSED orchestration lives in ``MPRAMarginalizedBase`` /
``MarginalizedLogSED``; this class only wires up Shorkie's input geometry
and the :class:`ShorkieCoverage` reduction strategy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from yeastbench.adapters._marginalized_logsed import ShorkieCoverage
from yeastbench.adapters._marginalized_mpra import MPRAMarginalizedBase
from yeastbench.adapters._shorkie_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    SHORKIE_T0_RNA_SEQ_TRACK_IDS,
)
from yeastbench.models.shorkie import Shorkie


class ShorkieMPRAMarginalizedPredictor(MPRAMarginalizedBase):
    def __init__(
        self,
        model: Shorkie,
        fasta_path: str | Path,
        gtf_path: str | Path,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        batch_size: int = 32,
        n_sample: int | None = None,
        seed: int = 42,
    ) -> None:
        self._setup(
            model=model,
            cov=ShorkieCoverage(model, track_subset),
            gtf_path=gtf_path,
            fasta_path=fasta_path,
            seq_len=SEQ_LEN,
            crop_bp_each_side=CROP_BP_EACH_SIDE,
            bin_width=BIN_WIDTH,
            output_bins=OUTPUT_BINS,
            batch_size=batch_size,
            n_sample=n_sample,
            seed=seed,
            desc="Shorkie marginalized",
        )

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        fasta_path: str | Path,
        gtf_path: str | Path,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 32,
        use_rc: bool = True,
        # On by default for this benchmark: bf16 autocast + inductor compile
        # give ~3.6x throughput with the score ranking preserved to
        # Spearman 0.99999 vs fp32 (so reported Pearson/Spearman are
        # unchanged to the 4th-5th decimal). The shared Shorkie ``model_config``
        # can't carry these flags without breaking sibling adapters, so the
        # default lives here rather than in the config. Pass ``autocast=False``
        # / ``compile=False`` to reproduce the exact fp32 path.
        autocast: bool = True,
        compile: bool = True,
        n_sample: int | None = None,
        seed: int = 42,
    ) -> "ShorkieMPRAMarginalizedPredictor":
        return cls(
            Shorkie.from_checkpoints(
                params_path, checkpoint_paths, device=device, use_rc=use_rc,
                autocast=autocast, compile=compile,
            ),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            track_subset=list(track_subset),
            batch_size=batch_size,
            n_sample=n_sample,
            seed=seed,
        )
