"""Yorzoi adapter for the Chen synonymous-mutation MPRA benchmark.

Mirrors ``ShorkieChenPredictor`` but with Yorzoi's geometry (4,992 bp
input, (162, 300) output, 10 bp bins, 996 bp crop each side) and
strand-aware track aggregation: the construct's variant gene is on
chrII + strand, so we aggregate over the 81 plus-strand tracks
[0:81].

Score per variant: logSED = log2(alt_exon_sum + 1) − log2(ref_exon_sum + 1)
where the "exon sum" is the mean across the 81 plus-strand tracks of
each track's summed coverage over the variant-gene's CDS bins.
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
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    YORZOI_PLUS_TRACK_IDS,
)
from yeastbench.adapters.protocols import LocalCodingVariantPredictor
from yeastbench.models.yorzoi import Yorzoi

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

VAR_LEN = 36


class YorzoiChenPredictor(LocalCodingVariantPredictor):
    def __init__(
        self,
        model: Yorzoi,
        fasta_path: str | Path,
        gtf_path: str | Path,
        library: str,
        library_loci_path: str | Path,
        batch_size: int = 64,
    ) -> None:
        import pysam
        import torch as _torch

        self.model = model
        self.library = library
        self.batch_size = batch_size

        loci = json.loads(Path(library_loci_path).read_text())
        if library not in loci:
            raise KeyError(f"library {library!r} not in {library_loci_path}")
        self.locus = loci[library]

        self.fasta = pysam.FastaFile(str(fasta_path))
        self.genes = parse_gene_annotations(gtf_path)
        gene_id = self.locus["gene_id"]
        if gene_id not in self.genes:
            raise KeyError(f"gene {gene_id!r} not in GTF {gtf_path}")
        self.gene: Gene = self.genes[gene_id]
        if self.gene.strand != "+":
            raise NotImplementedError(
                f"YorzoiChenPredictor only supports + strand variant genes "
                f"(library {library!r}, gene {gene_id} is on {self.gene.strand})"
            )

        chrom_length = self.fasta.get_reference_length(self.gene.chrom_roman)
        self.window_start_0 = place_window(
            var_pos=self.locus["var_start"],
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
                f"{library}: no gene exon bins inside Yorzoi output crop"
            )
        var_start_in_window = self.locus["var_start"] - 1 - self.window_start_0
        if not (0 <= var_start_in_window <= SEQ_LEN - VAR_LEN):
            raise RuntimeError(
                f"{library}: variable block at offset {var_start_in_window} "
                f"is outside the Yorzoi input window [0, {SEQ_LEN - VAR_LEN}]"
            )
        self.var_start_in_window = var_start_in_window

        ref_seq = self.fasta.fetch(
            self.gene.chrom_roman,
            self.window_start_0,
            self.window_start_0 + SEQ_LEN,
        ).upper()
        if len(ref_seq) != SEQ_LEN:
            raise RuntimeError(
                f"{library}: REF window length {len(ref_seq)} != {SEQ_LEN}"
            )
        ref_oh = one_hot_encode_channels_first(ref_seq).T  # (SEQ_LEN, 4)
        self._ref_oh_gpu = _torch.from_numpy(ref_oh).to(self.model.device)
        self._exon_bins_gpu = _torch.as_tensor(
            self.exon_bins, device=self.model.device, dtype=_torch.long,
        )
        self._plus_tracks_gpu = _torch.as_tensor(
            YORZOI_PLUS_TRACK_IDS, device=self.model.device, dtype=_torch.long,
        )

        # Precompute REF exon sum (single scalar, mean over plus-strand tracks).
        with _torch.no_grad():
            pred = self.model.forward_tracks_binned(
                self._ref_oh_gpu.unsqueeze(0)        # (1, SEQ_LEN, 4)
            ).float()                                # (1, 162, OUTPUT_BINS)
        per_track = pred[0].index_select(1, self._exon_bins_gpu).sum(dim=1)  # (162,)
        self._ref_exon_sum = float(per_track[YORZOI_PLUS_TRACK_IDS].mean())
        log.info(
            "%s: Yorzoi REF window [%d, %d), exon_bins=%d, REF exon sum=%.4f",
            library, self.window_start_0, self.window_start_0 + SEQ_LEN,
            self.exon_bins.size, self._ref_exon_sum,
        )

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        fasta_path: str | Path,
        gtf_path: str | Path,
        library: str,
        library_loci_path: str | Path,
        device: str = "cuda",
        batch_size: int = 64,
        use_rc: bool = True,
        autocast: bool = True,
    ) -> "YorzoiChenPredictor":
        return cls(
            Yorzoi.from_pretrained(
                hf_repo, device=device, use_rc=use_rc, autocast=autocast,
            ),
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            library=library,
            library_loci_path=library_loci_path,
            batch_size=batch_size,
        )

    def _splice_alt_batch(
        self, variant_ohs: "torch.Tensor", batch_size: int,
    ) -> "torch.Tensor":
        """variant_ohs: (B, VAR_LEN, 4) → (B, SEQ_LEN, 4) with REF spliced."""
        alt = self._ref_oh_gpu.unsqueeze(0).expand(batch_size, -1, -1).clone()
        s = self.var_start_in_window
        alt[:, s : s + VAR_LEN, :] = variant_ohs
        return alt

    def predict_local_variants(
        self,
        library_ids: Sequence[str],
        variant_seqs: Sequence[str],
    ) -> np.ndarray:
        import torch as _torch

        if any(lib != self.library for lib in library_ids):
            raise ValueError(
                f"YorzoiChenPredictor bound to library {self.library!r}; "
                f"got library_ids {set(library_ids)}"
            )
        n = len(variant_seqs)
        scores = np.full(n, np.nan, dtype=np.float64)
        ref_log = float(np.log2(self._ref_exon_sum + 1.0))

        for batch_start in tqdm(
            range(0, n, self.batch_size), desc=f"Yorzoi Chen {self.library}"
        ):
            batch_end = min(batch_start + self.batch_size, n)
            B = batch_end - batch_start
            # One-hot encode the batch's 36-nt variable blocks → (B, VAR_LEN, 4)
            var_oh_np = np.zeros((B, VAR_LEN, 4), dtype=np.float32)
            for i in range(B):
                seq = variant_seqs[batch_start + i].upper()
                if len(seq) != VAR_LEN:
                    raise ValueError(
                        f"variant_seq {batch_start + i} has {len(seq)} nt"
                    )
                var_oh_np[i] = one_hot_encode_channels_first(seq).T
            var_oh_t = _torch.from_numpy(var_oh_np).to(self.model.device)
            alt = self._splice_alt_batch(var_oh_t, B)

            with _torch.no_grad():
                pred = self.model.forward_tracks_binned(alt).float()   # (B, 162, 300)

            per_track = pred.index_select(2, self._exon_bins_gpu).sum(dim=2)  # (B, 162)
            alt_sum = per_track[:, YORZOI_PLUS_TRACK_IDS].mean(dim=1)         # (B,)
            alt_log = _torch.log2(alt_sum + 1.0)
            logsed = (alt_log - ref_log).detach().cpu().numpy()
            scores[batch_start:batch_end] = logsed

        return scores
