"""Shorkie marginalized adapter for the Shalem MPRA terminator benchmark.

For each 150 bp test oligo, insert it + a 300 bp CYC1 no-termination filler
immediately downstream of each of 22 host genes' stop codons, predict
expression (mean logSED over T0 RNA-seq tracks, summed over host-gene
exon bins), and return the mean logSED across host genes as the per-
sequence prediction.

The REF→ALT→logSED orchestration lives in ``ShalemMarginalizedBase`` /
``MarginalizedLogSED``; this class only wires up Shorkie's input geometry
and the :class:`ShorkieCoverage` reduction strategy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from yeastbench.adapters._marginalized_logsed import ShorkieCoverage
from yeastbench.adapters._shalem_scaffold import ShalemMarginalizedBase
from yeastbench.adapters._shorkie_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    SHORKIE_T0_RNA_SEQ_TRACK_IDS,
)
from yeastbench.models.shorkie import Shorkie


class ShorkieShalemPredictor(ShalemMarginalizedBase):
    def __init__(
        self,
        model: Shorkie,
        fasta_path: str | Path,
        gtf_path: str | Path,
        host_genes_json: str | Path | None = None,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        batch_size: int = 32,
        n_sample: int | None = None,
        seed: int = 42,
    ) -> None:
        self._setup(
            model=model,
            cov=ShorkieCoverage(model, track_subset),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            host_genes_json=host_genes_json,
            seq_len=SEQ_LEN,
            crop_bp_each_side=CROP_BP_EACH_SIDE,
            bin_width=BIN_WIDTH,
            output_bins=OUTPUT_BINS,
            batch_size=batch_size,
            n_sample=n_sample,
            seed=seed,
            desc="Shorkie Shalem",
        )

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        fasta_path: str | Path,
        gtf_path: str | Path,
        host_genes_json: str | Path | None = None,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 32,
        use_rc: bool = True,
        n_sample: int | None = None,
        seed: int = 42,
    ) -> "ShorkieShalemPredictor":
        return cls(
            Shorkie.from_checkpoints(
                params_path, checkpoint_paths, device=device, use_rc=use_rc,
            ),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            host_genes_json=host_genes_json,
            track_subset=list(track_subset),
            batch_size=batch_size,
            n_sample=n_sample,
            seed=seed,
        )
