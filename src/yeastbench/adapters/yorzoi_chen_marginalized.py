"""Yorzoi adapter for the Chen synonymous-mutation MPRA benchmark.

Marginalised version — see ``shorkie_chen_marginalized.py`` for the
motivation. Differs from the Shorkie analog in the model geometry and
in **strand-aware track aggregation**: a + strand host uses Yorzoi's
plus-strand tracks [0:81]; a − strand host uses [81:162].
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
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
    YORZOI_MINUS_TRACK_IDS,
    YORZOI_PLUS_TRACK_IDS,
)
from yeastbench.adapters.protocols import LocalCodingVariantPredictor
from yeastbench.models.yorzoi import Yorzoi

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


class YorzoiChenPredictor(LocalCodingVariantPredictor):
    def __init__(
        self,
        model: Yorzoi,
        fasta_path: str | Path,
        library: str,
        hosts_path: str | Path,
        data_dir: str | Path,
        batch_size: int = 8,
    ) -> None:
        import pysam
        import torch as _torch

        self.model = model
        self.library = library
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
        log.info("%s: built %d host contexts (Yorzoi geometry)",
                 library, len(self.contexts))

        n = len(self.contexts)
        ref_np = np.zeros((n, SEQ_LEN, 4), dtype=np.float32)
        for i, ctx in enumerate(self.contexts):
            ref_np[i] = one_hot_encode_channels_first(ctx.window_seq).T
        self._ref_oh_gpu = _torch.from_numpy(ref_np).to(self.model.device)

        # Per-host strand-matched track index slice.
        self._track_slices = [
            (0, 81) if c.host.strand == "+" else (81, 162)
            for c in self.contexts
        ]

        # Precompute REF exon-sum per host (cross-track-mean over strand-matched).
        self._ref_exon_sums = self._predict_exon_sums(self._ref_oh_gpu).cpu().numpy()  # (n_hosts,)
        self._ref_log = np.log2(self._ref_exon_sums + 1.0)
        log.info(
            "%s: Yorzoi REF exon-sums per host (min / max / median): %.2f / %.2f / %.2f",
            library,
            float(self._ref_exon_sums.min()), float(self._ref_exon_sums.max()),
            float(np.median(self._ref_exon_sums)),
        )

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        fasta_path: str | Path,
        library: str,
        hosts_path: str | Path,
        data_dir: str | Path,
        device: str = "cuda",
        batch_size: int = 8,
        use_rc: bool = True,
        autocast: bool = True,
    ) -> "YorzoiChenPredictor":
        return cls(
            Yorzoi.from_pretrained(
                hf_repo, device=device, use_rc=use_rc, autocast=autocast,
            ),
            fasta_path=fasta_path,
            library=library,
            hosts_path=hosts_path,
            data_dir=data_dir,
            batch_size=batch_size,
        )

    # ── Internals ────────────────────────────────────────────────────

    def _predict_exon_sums(self, batch_oh: "torch.Tensor") -> "torch.Tensor":
        """``batch_oh`` shape (B, SEQ_LEN, 4). B must be a multiple of n_hosts;
        row i maps to host ``i % n_hosts``. Returns (B,) with the
        strand-matched cross-track-mean CDS-bin sum per row."""
        import torch as _torch

        n = len(self.contexts)
        B = batch_oh.shape[0]
        if B % n != 0:
            raise ValueError(f"batch size {B} must be a multiple of n_hosts={n}")

        with _torch.no_grad():
            pred = self.model.forward_tracks_binned(batch_oh).float()  # (B, 162, OUT_BINS)

        out = _torch.zeros(B, device=pred.device, dtype=pred.dtype)
        for h, ctx in enumerate(self.contexts):
            ts, te = self._track_slices[h]
            rows = _torch.arange(h, B, n, device=pred.device)
            # Sum over CDS bins, mean across strand-matched tracks.
            cds = pred[rows][:, ts:te, ctx.cds_bin_lo:ctx.cds_bin_hi].sum(dim=2)  # (R, 81)
            out[rows] = cds.mean(dim=1)
        return out

    def _splice_batch(
        self, variant_seqs: Sequence[str],
    ) -> "torch.Tensor":
        """Returns (B_variants * n_hosts, SEQ_LEN, 4) one-hot tensor."""
        import torch as _torch

        B = len(variant_seqs)
        n = len(self.contexts)
        alt = self._ref_oh_gpu.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n, SEQ_LEN, 4).clone()

        for v_idx, seq in enumerate(variant_seqs):
            seq_upper = seq.upper()
            if len(seq_upper) != VAR_LEN:
                raise ValueError(f"variant_seq {v_idx} has {len(seq_upper)} nt")
            fwd = _torch.from_numpy(alt_block_oh(seq_upper, needs_revcomp=False).T).to(self.model.device)
            rc  = _torch.from_numpy(alt_block_oh(seq_upper, needs_revcomp=True).T).to(self.model.device)
            for h_idx, ctx in enumerate(self.contexts):
                row = v_idx * n + h_idx
                s = ctx.var_start_in_window
                alt[row, s : s + VAR_LEN, :] = rc if ctx.var_needs_revcomp else fwd
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
                f"got {set(library_ids)}"
            )

        n_variants = len(variant_seqs)
        n_hosts = len(self.contexts)
        scores = np.empty(n_variants, dtype=np.float64)
        ref_log_t = _torch.from_numpy(self._ref_log).to(self.model.device).to(_torch.float32)

        for batch_start in tqdm(
            range(0, n_variants, self.batch_size),
            desc=f"Yorzoi Chen marginalized {self.library}",
        ):
            batch_end = min(batch_start + self.batch_size, n_variants)
            batch = variant_seqs[batch_start:batch_end]
            B = len(batch)

            alt_batch = self._splice_batch(batch)
            alt_sums = self._predict_exon_sums(alt_batch)
            alt_sums = alt_sums.view(B, n_hosts)
            alt_log = _torch.log2(alt_sums + 1.0)
            logsed = (alt_log - ref_log_t.unsqueeze(0))
            scores[batch_start:batch_end] = logsed.mean(dim=1).cpu().numpy()

        return scores
