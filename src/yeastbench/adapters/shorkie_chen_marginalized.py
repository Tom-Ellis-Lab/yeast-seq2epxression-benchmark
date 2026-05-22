"""Shorkie adapter for the Chen synonymous-mutation MPRA benchmark.

Marginalises variant scoring over 20 native yeast loci active in YPD
(see ``benchmarks/chen_synonymous.md`` for the motivation and
host-selection criteria). For each variant and each host:

1. Splice the variant gene CDS + TADH1 into the host's CDS span
   (strand-aware).
2. Overwrite the 36 nt variable region with the variant's sequence.
3. Forward through Shorkie's 8-fold ensemble (with RC averaging), track
   subset = T0 RNA-seq, aggregate per-track mean.
4. Sum coverage over the bins spanning the inserted variant CDS.
5. logSED vs the REF (row-0 variant) score for the same host.

The 20 per-host logSEDs are averaged into the final per-variant score.

Batching: REF predictions are computed once per (host) at init. At
predict time, each batch of variants is forward-passed across all 20
hosts in parallel. A single GPU forward sees a batch of ``B * 20``
windows.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._chen_marginalized import (
    VAR_LEN, alt_block_oh, build_cassette, build_host_contexts, load_hosts,
)
from yeastbench.adapters._genome import one_hot_encode_channels_first
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


class ShorkieChenPredictor(LocalCodingVariantPredictor):
    def __init__(
        self,
        model: Shorkie,
        fasta_path: str | Path,
        library: str,
        hosts_path: str | Path,
        data_dir: str | Path,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        batch_size: int = 4,
    ) -> None:
        """``batch_size`` is in *variants*; each variant fans out across all
        host contexts in one forward pass (B * n_hosts windows)."""
        import pysam
        import torch as _torch

        self.model = model
        self.library = library
        self.track_subset = list(track_subset)
        self.batch_size = batch_size

        self.fasta = pysam.FastaFile(str(fasta_path))
        self.hosts = load_hosts(hosts_path)
        self.cassette = build_cassette(library, self.fasta, Path(data_dir))
        self.contexts = build_host_contexts(
            library=library, hosts=self.hosts, fasta=self.fasta,
            cassette=self.cassette,
            seq_len=SEQ_LEN, crop_bp_each_side=CROP_BP_EACH_SIDE,
            bin_width=BIN_WIDTH, output_bins=OUTPUT_BINS,
        )
        log.info("%s: built %d host contexts (Shorkie geometry)",
                 library, len(self.contexts))

        # Cache REF one-hots on GPU. Shape: (n_hosts, 4, SEQ_LEN).
        n = len(self.contexts)
        ref_np = np.zeros((n, 4, SEQ_LEN), dtype=np.float32)
        for i, ctx in enumerate(self.contexts):
            ref_np[i] = one_hot_encode_channels_first(ctx.window_seq)
        self._ref_oh_gpu = _torch.from_numpy(ref_np).to(self.model.device)
        # Cache per-host CDS bins and var-start as a single tensor for fast indexing.
        self._var_starts = np.array([c.var_start_in_window for c in self.contexts], dtype=np.int64)
        self._var_rc = np.array([c.var_needs_revcomp for c in self.contexts], dtype=bool)
        self._track_idx_gpu = _torch.tensor(
            self.track_subset, device=self.model.device, dtype=_torch.long,
        )

        # Precompute REF exon-sum per host. Shape: (n_hosts,)
        self._ref_exon_sums = self._predict_exon_sums(self._ref_oh_gpu).cpu().numpy()  # (n_hosts,)
        self._ref_log = np.log2(self._ref_exon_sums + 1.0)
        log.info(
            "%s: REF exon-sums per host (min / max / median): %.2f / %.2f / %.2f",
            library,
            float(self._ref_exon_sums.min()), float(self._ref_exon_sums.max()),
            float(np.median(self._ref_exon_sums)),
        )

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        fasta_path: str | Path,
        library: str,
        hosts_path: str | Path,
        data_dir: str | Path,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 4,
        use_rc: bool = True,
    ) -> "ShorkieChenPredictor":
        return cls(
            Shorkie.from_checkpoints(
                params_path, checkpoint_paths, device=device, use_rc=use_rc,
            ),
            fasta_path=fasta_path,
            library=library,
            hosts_path=hosts_path,
            data_dir=data_dir,
            track_subset=list(track_subset),
            batch_size=batch_size,
        )

    # ── Internals ────────────────────────────────────────────────────

    def _predict_exon_sums(self, batch_oh: "torch.Tensor") -> "torch.Tensor":
        """``batch_oh`` shape (B, 4, SEQ_LEN). Returns (B,) — CDS-bin sum
        per row, with row i interpreted as host index i % n_hosts.

        This means the caller must arrange ``batch_oh`` so the *i*-th
        row maps to host ``i % n_hosts``."""
        import torch as _torch

        n = len(self.contexts)
        B = batch_oh.shape[0]
        if B % n != 0:
            raise ValueError(f"batch size {B} must be a multiple of n_hosts={n}")

        with _torch.no_grad():
            cov = self.model.forward_track_mean_binned(batch_oh, self._track_idx_gpu)  # (B, OUT_BINS)

        out = _torch.zeros(B, device=cov.device, dtype=cov.dtype)
        # Could be vectorised but n_hosts is small; per-host bin slice is clearest.
        for h, ctx in enumerate(self.contexts):
            mask_rows = _torch.arange(h, B, n, device=cov.device)
            out[mask_rows] = cov[mask_rows, ctx.cds_bin_lo:ctx.cds_bin_hi].sum(dim=1)
        return out

    def _splice_batch(
        self, variant_seqs: Sequence[str],
    ) -> "torch.Tensor":
        """Returns one-hot tensor of shape (B_variants * n_hosts, 4, SEQ_LEN)
        with each variant spliced into all 20 host REF windows."""
        import torch as _torch

        B = len(variant_seqs)
        n = len(self.contexts)
        # Pre-compute per-variant ALT blocks: one-hot, RC where needed per host.
        # Resulting layout: per (var, host) pair, splice the variant block (RC'd
        # if host is - strand) at var_start_in_window.
        alt = self._ref_oh_gpu.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n, 4, SEQ_LEN).clone()

        for v_idx, seq in enumerate(variant_seqs):
            seq_upper = seq.upper()
            if len(seq_upper) != VAR_LEN:
                raise ValueError(f"variant_seq {v_idx} has {len(seq_upper)} nt")
            # Build both fwd and rc one-hots once per variant (cheap, 36 nt).
            fwd = _torch.from_numpy(alt_block_oh(seq_upper, needs_revcomp=False)).to(self.model.device)
            rc  = _torch.from_numpy(alt_block_oh(seq_upper, needs_revcomp=True)).to(self.model.device)
            for h_idx, ctx in enumerate(self.contexts):
                row = v_idx * n + h_idx
                s = ctx.var_start_in_window
                alt[row, :, s : s + VAR_LEN] = rc if ctx.var_needs_revcomp else fwd
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
                f"got {set(library_ids)}"
            )

        n_variants = len(variant_seqs)
        n_hosts = len(self.contexts)
        scores = np.empty(n_variants, dtype=np.float64)
        ref_log_t = _torch.from_numpy(self._ref_log).to(self.model.device).to(_torch.float32)  # (n_hosts,)

        for batch_start in tqdm(
            range(0, n_variants, self.batch_size),
            desc=f"Shorkie Chen marginalized {self.library}",
        ):
            batch_end = min(batch_start + self.batch_size, n_variants)
            batch = variant_seqs[batch_start:batch_end]
            B = len(batch)

            alt_batch = self._splice_batch(batch)                # (B*n_hosts, 4, SEQ_LEN)
            alt_sums = self._predict_exon_sums(alt_batch)         # (B*n_hosts,)
            alt_sums = alt_sums.view(B, n_hosts)
            alt_log = _torch.log2(alt_sums + 1.0)
            logsed = (alt_log - ref_log_t.unsqueeze(0))           # (B, n_hosts)
            scores[batch_start:batch_end] = logsed.mean(dim=1).cpu().numpy()

        return scores
