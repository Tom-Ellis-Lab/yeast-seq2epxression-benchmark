"""Shorkie adapter for the Chen synonymous-mutation MPRA benchmark.

Implements ``LocalCodingVariantPredictor`` by splicing each variant's
36 nt block into the construct's variant-gene CDS, running a Shorkie
forward pass over a 16,384 bp window centred on the variant gene, and
returning the logSED change in summed exon-bin coverage relative to
REF (the construct's wild-type variable region):

    logSED = log2(alt_exon_sum + 1) − log2(ref_exon_sum + 1)

The construct has exactly one variant gene per library, so there's only
one REF window per task — we cache its one-hot encoding on the GPU and
its REF exon sum, then per-variant work is just an in-place splice +
forward pass.

Track aggregation: Shorkie's T0 RNA-seq tracks (same subset the Rafi
marginalized adapter uses), cross-track mean → sum over the variant-
gene's CDS bins → log2 fold change.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._genome import (
    Gene,
    gene_exon_bins,
    one_hot_encode_channels_first,
    parse_gene_annotations,
    place_window,
)
from yeastbench.adapters._shorkie_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    SHORKIE_T0_RNA_SEQ_TRACK_IDS,
)
from yeastbench.adapters.protocols import LocalCodingVariantPredictor
from yeastbench.models.shorkie import Shorkie

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


VAR_LEN = 36


class ShorkieChenPredictor(LocalCodingVariantPredictor):
    def __init__(
        self,
        model: Shorkie,
        fasta_path: str | Path,
        gtf_path: str | Path,
        library: str,
        library_loci_path: str | Path,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        batch_size: int = 32,
    ) -> None:
        import pysam
        import torch as _torch

        self.model = model
        self.library = library
        self.track_subset = list(track_subset)
        self.batch_size = batch_size

        loci = json.loads(Path(library_loci_path).read_text())
        if library not in loci:
            raise KeyError(f"library {library!r} not in {library_loci_path}")
        self.locus = loci[library]

        # Inputs may be string paths; pysam wants str.
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.genes = parse_gene_annotations(gtf_path)
        gene_id = self.locus["gene_id"]
        if gene_id not in self.genes:
            raise KeyError(f"gene {gene_id!r} not in GTF {gtf_path}")
        self.gene: Gene = self.genes[gene_id]

        chrom_length = self.fasta.get_reference_length(self.gene.chrom_roman)
        # Window centred so the gene center is inside the output crop
        # AND the variable region is inside the input window.
        self.window_start_0 = place_window(
            var_pos=self.locus["var_start"],   # 1-based chrII coord
            gene_center=self.gene.gene_center,
            chrom_length=chrom_length,
            seq_len=SEQ_LEN,
            crop_bp_each_side=CROP_BP_EACH_SIDE,
        )
        self.exon_bins = gene_exon_bins(
            self.gene, self.window_start_0,
            crop_bp_each_side=CROP_BP_EACH_SIDE,
            bin_width=BIN_WIDTH,
            output_bins=OUTPUT_BINS,
        )
        if self.exon_bins.size == 0:
            raise RuntimeError(
                f"{library}: gene {gene_id} has no exon bins inside the "
                f"Shorkie output crop — check window placement"
            )
        # 0-based offset of the variable-block start inside the input window.
        var_start_in_window = self.locus["var_start"] - 1 - self.window_start_0
        if not (0 <= var_start_in_window <= SEQ_LEN - VAR_LEN):
            raise RuntimeError(
                f"{library}: variable block at offset {var_start_in_window} "
                f"is outside the Shorkie input window [0, {SEQ_LEN - VAR_LEN}]"
            )
        self.var_start_in_window = var_start_in_window

        ref_seq = self.fasta.fetch(
            self.gene.chrom_roman,
            self.window_start_0,
            self.window_start_0 + SEQ_LEN,
        ).upper()
        if len(ref_seq) != SEQ_LEN:
            raise RuntimeError(
                f"{library}: fetched REF window length {len(ref_seq)} != {SEQ_LEN}"
            )
        ref_oh = one_hot_encode_channels_first(ref_seq)                  # (4, SEQ_LEN)
        self._ref_oh_gpu = _torch.from_numpy(ref_oh).to(self.model.device)  # cached
        self._exon_bins_gpu = _torch.as_tensor(
            self.exon_bins, device=self.model.device, dtype=_torch.long,
        )
        self._track_idx_gpu = _torch.tensor(
            self.track_subset, device=self.model.device, dtype=_torch.long,
        )

        # Precompute REF exon sum (single scalar, cross-track averaged).
        with _torch.no_grad():
            cov = self.model.forward_track_mean_binned(
                self._ref_oh_gpu.unsqueeze(0),    # (1, 4, SEQ_LEN)
                self._track_idx_gpu,
            )                                     # (1, OUTPUT_BINS)
        self._ref_exon_sum = float(cov[0].index_select(0, self._exon_bins_gpu).sum())
        log.info(
            "%s: Shorkie REF window [%d, %d), exon_bins=%d, REF exon sum=%.4f",
            library, self.window_start_0, self.window_start_0 + SEQ_LEN,
            self.exon_bins.size, self._ref_exon_sum,
        )

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        fasta_path: str | Path,
        gtf_path: str | Path,
        library: str,
        library_loci_path: str | Path,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 32,
        use_rc: bool = True,
    ) -> "ShorkieChenPredictor":
        return cls(
            Shorkie.from_checkpoints(
                params_path, checkpoint_paths, device=device, use_rc=use_rc,
            ),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            library=library,
            library_loci_path=library_loci_path,
            track_subset=list(track_subset),
            batch_size=batch_size,
        )

    def _splice_alt_batch(
        self, variant_ohs: "torch.Tensor", batch_size: int,
    ) -> "torch.Tensor":
        """variant_ohs: (B, 4, VAR_LEN) → (B, 4, SEQ_LEN) with REF spliced."""
        import torch as _torch

        alt = self._ref_oh_gpu.unsqueeze(0).expand(batch_size, -1, -1).clone()
        s = self.var_start_in_window
        alt[:, :, s : s + VAR_LEN] = variant_ohs
        return alt

    def predict_local_variants(
        self,
        library_ids: Sequence[str],
        variant_seqs: Sequence[str],
    ) -> np.ndarray:
        import torch as _torch

        if any(lib != self.library for lib in library_ids):
            raise ValueError(
                f"ShorkieChenPredictor bound to library {self.library!r}; "
                f"got library_ids {set(library_ids)}"
            )

        n = len(variant_seqs)
        scores = np.full(n, np.nan, dtype=np.float64)
        ref_log = float(np.log2(self._ref_exon_sum + 1.0))

        for batch_start in tqdm(
            range(0, n, self.batch_size), desc=f"Shorkie Chen {self.library}"
        ):
            batch_end = min(batch_start + self.batch_size, n)
            B = batch_end - batch_start
            # One-hot encode the batch's 36-nt blocks
            var_oh_np = np.zeros((B, 4, VAR_LEN), dtype=np.float32)
            for i in range(B):
                seq = variant_seqs[batch_start + i].upper()
                if len(seq) != VAR_LEN:
                    raise ValueError(
                        f"variant_seq {batch_start + i} has {len(seq)} nt"
                    )
                var_oh_np[i] = one_hot_encode_channels_first(seq)
            var_oh_t = _torch.from_numpy(var_oh_np).to(self.model.device)
            alt = self._splice_alt_batch(var_oh_t, B)

            with _torch.no_grad():
                cov = self.model.forward_track_mean_binned(
                    alt, self._track_idx_gpu,
                )                            # (B, OUTPUT_BINS)

            alt_sum = cov.index_select(1, self._exon_bins_gpu).sum(dim=1)  # (B,)
            alt_log = _torch.log2(alt_sum + 1.0)
            logsed = (alt_log - ref_log).detach().cpu().numpy()
            scores[batch_start:batch_end] = logsed

        return scores
