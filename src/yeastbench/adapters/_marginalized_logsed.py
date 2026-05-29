"""Shared engine for the marginalized-logSED adapter family.

Six adapters (Shorkie/Yorzoi × MPRA-marginalized / Shalem-terminator /
Chen-synonymous) share one algorithm: cache a REF window per context on
GPU, run a batched REF forward to a per-context baseline, then for each
candidate insert/variant clone the REF one-hots, splice the candidate in,
run a batched ALT forward, and score ``log2(alt+1) − log2(ref+1)``
aggregated across contexts.

Predictions are read on the **per-base, untransformed** scale: the
``ModelCoverage`` strategies call the wrapper's per-base forwards, which
(for Yorzoi) apply the Borzoi inverse transform *per element before*
RC-averaging and then unbin, so every downstream sum/mean here operates
on raw predicted counts. The exon readout selects exact **base
positions** (``gene_exon_base_positions``), not bin indices.

The two models reduce in *opposite order*:
Shorkie's ``forward_track_mean_perbase`` folds the cross-track mean into
the wrapper (returns ``(B, bins*BIN_WIDTH)``); Yorzoi's
``forward_tracks_perbase`` returns ``(B, 162, bins*BIN_WIDTH)`` and the
adapter base-sums first, then takes a strand-matched track mean. To keep
the engine track-axis-agnostic, the entire per-context reduction lives
inside a :class:`ModelCoverage` strategy — the mixin never sees a track
axis. See the model wrapper docstrings for why the orderings are kept
distinct.

``MarginalizedLogSED`` owns the orchestration; a task subclass supplies
the per-context accessors, the candidate encoder, and the aggregation.

The Shalem-terminator and MPRA-marginalized pairs (4 adapters) are built
on this engine. The Chen-synonymous pair is intentionally **not**: it
batches variants × hosts in a single forward (throughput), precomputes the
REF log in float64 numpy (≠ the engine's float32 ``torch.log2`` by ~1 ULP),
and uses a per-host reverse-complement flag plus a contiguous CDS-bin-slice
readout. Forcing it onto the engine would either shift its published
scores or require Chen-only hooks that defeat the de-duplication.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._genome import one_hot_encode_channels_first

if TYPE_CHECKING:
    import torch

# Yorzoi track layout: indices 0..80 are '+' strand, 81..161 are '-'.
_YORZOI_N_PLUS_TRACKS = 81
_YORZOI_N_TRACKS_TOTAL = 162


class ModelCoverage(Protocol):
    """Model-specific encode + forward + reduce-to-per-context-scalar.

    Encapsulates the one-hot memory layout, the splice axis, the forward
    call, and the full reduction order so the engine stays unaware of any
    track axis.
    """

    device: "torch.device"

    def one_hot(self, seq: str) -> np.ndarray:
        """Encode ``seq`` in the model's native one-hot layout."""
        ...

    def splice(
        self, alt: "torch.Tensor", row: int, start: int,
        block_oh: "torch.Tensor", length: int,
    ) -> None:
        """In-place splice ``block_oh`` into ``alt[row]`` at ``start``."""
        ...

    def forward(self, batch_oh: "torch.Tensor") -> "torch.Tensor":
        """Ensemble/RC-averaged forward over a one-hot batch."""
        ...

    def readout(
        self, out: "torch.Tensor", row: int,
        bins_t: "torch.Tensor", strand: str,
    ) -> "torch.Tensor":
        """Per-context pre-log scalar: reduce ``out[row]`` over the
        readout ``bins_t`` (and, for stranded models, the strand-matched
        track slice)."""
        ...


class ShorkieCoverage:
    """Shorkie coverage strategy: channels-first one-hots, the
    track-mean-folded **per-base** forward, and a plain exon-base sum (no
    track axis remains after the wrapper; the softplus head is already in
    raw counts, so the per-base forward is unbin-only)."""

    def __init__(self, model, track_subset: Sequence[int]) -> None:
        import torch as _torch

        self.model = model
        self.device = model.device
        self._track_idx_t = _torch.tensor(
            list(track_subset), device=model.device, dtype=_torch.long
        )

    def one_hot(self, seq: str) -> np.ndarray:
        return one_hot_encode_channels_first(seq)  # (4, L)

    def splice(self, alt, row, start, block_oh, length) -> None:
        alt[row, :, start : start + length] = block_oh

    def forward(self, batch_oh):
        return self.model.forward_track_mean_perbase(batch_oh, self._track_idx_t)

    def readout(self, out, row, bins_t, strand):
        # ``bins_t`` holds exon BASE positions; out is (B, bins*BIN_WIDTH).
        return out[row].index_select(0, bins_t).sum()


class YorzoiCoverage:
    """Yorzoi coverage strategy: channels-last one-hots, the full
    per-track **per-base** forward (Borzoi inverse applied per-pass before
    RC-averaging inside the wrapper, then unbinned to raw counts), then an
    exon-base sum followed by a strand-matched cross-track mean (matches
    the logSED_agg ordering of the original Yorzoi adapters, now on the
    raw-count scale)."""

    def __init__(self, model) -> None:
        self.model = model
        self.device = model.device

    def one_hot(self, seq: str) -> np.ndarray:
        return one_hot_encode_channels_first(seq).T  # (L, 4)

    def splice(self, alt, row, start, block_oh, length) -> None:
        alt[row, start : start + length, :] = block_oh

    def forward(self, batch_oh):
        # (B, 162, OUTPUT_BINS*BIN_WIDTH) raw per-base counts (already float32).
        return self.model.forward_tracks_perbase(batch_oh)

    def readout(self, out, row, bins_t, strand):
        # ``bins_t`` holds exon BASE positions; out is (162, bins*BIN_WIDTH).
        per_track = out[row].index_select(1, bins_t).sum(dim=1)  # (162,)
        if strand == "+":
            return per_track[0:_YORZOI_N_PLUS_TRACKS].mean()
        return per_track[_YORZOI_N_PLUS_TRACKS:_YORZOI_N_TRACKS_TOTAL].mean()


class MarginalizedLogSED:
    """Mixin owning the REF→ALT→logSED orchestration.

    A concrete adapter must:

    * set ``self._cov`` (a :class:`ModelCoverage`), ``self.batch_size``,
      ``self.n_sample``, ``self.seed`` and ``self._contexts`` in
      ``__init__``, then call :meth:`_init_baselines`;
    * implement the per-context accessors (:meth:`_ref_window_seq`,
      :meth:`_ctx_strand`, :meth:`_ctx_splice_start`, :meth:`_ctx_bins`),
      the candidate encoder (:meth:`_encode_candidate`), and the
      cross-context aggregation (:meth:`_aggregate`).
    """

    # ── hooks (implemented by the task subclass) ──────────────────────

    def _ref_window_seq(self, ctx) -> str:
        raise NotImplementedError

    def _ctx_strand(self, ctx) -> str:
        raise NotImplementedError

    def _ctx_splice_start(self, ctx) -> int:
        raise NotImplementedError

    def _ctx_bins(self, ctx) -> np.ndarray:
        raise NotImplementedError

    def _encode_candidate(self, candidate: str) -> tuple[str, str, int]:
        """Return ``(fwd_seq, rc_seq, block_len)`` — the genomic-forward
        block and its reverse complement, both spliced in by strand."""
        raise NotImplementedError

    def _aggregate(self, logsed_per_ctx: "torch.Tensor") -> float:
        raise NotImplementedError

    # ── orchestration (shared) ────────────────────────────────────────

    def _init_baselines(self, desc: str = "REF baseline") -> None:
        """Cache REF one-hots on GPU and precompute the per-context REF
        baseline scalar. Call once at the end of ``__init__``."""
        import torch as _torch

        ref_np = np.stack(
            [self._cov.one_hot(self._ref_window_seq(c)) for c in self._contexts]
        )
        self._ref_ohs_gpu = _torch.from_numpy(ref_np).to(self._cov.device)
        self._ref_values = self._reduce_batches(self._ref_ohs_gpu, desc)

    def _reduce_batches(self, x_all: "torch.Tensor", desc: str) -> "torch.Tensor":
        """Forward every cached window in batches and reduce each to its
        per-context scalar."""
        import torch as _torch

        n = len(self._contexts)
        out_vals = _torch.zeros(n, device=self._cov.device, dtype=_torch.float32)
        for bs in tqdm(range(0, n, self.batch_size), desc=desc):
            be = min(bs + self.batch_size, n)
            with _torch.no_grad():
                cov = self._cov.forward(x_all[bs:be])
            for j in range(be - bs):
                ctx = self._contexts[bs + j]
                bins_t = _torch.as_tensor(
                    self._ctx_bins(ctx), device=self._cov.device, dtype=_torch.long
                )
                out_vals[bs + j] = self._cov.readout(
                    cov, j, bins_t, self._ctx_strand(ctx)
                )
        return out_vals

    def _score_one(self, candidate: str) -> float:
        """logSED-aggregated score for one candidate insert/variant."""
        import torch as _torch

        fwd_seq, rc_seq, block_len = self._encode_candidate(candidate)
        fwd_oh = _torch.from_numpy(self._cov.one_hot(fwd_seq)).to(self._cov.device)
        rc_oh = _torch.from_numpy(self._cov.one_hot(rc_seq)).to(self._cov.device)

        n = len(self._contexts)
        alt_values = _torch.zeros(n, device=self._cov.device, dtype=_torch.float32)
        for bs in range(0, n, self.batch_size):
            be = min(bs + self.batch_size, n)
            alt = self._ref_ohs_gpu[bs:be].clone()
            for j in range(be - bs):
                ctx = self._contexts[bs + j]
                block = rc_oh if self._ctx_strand(ctx) == "-" else fwd_oh
                self._cov.splice(
                    alt, j, self._ctx_splice_start(ctx), block, block_len
                )
            with _torch.no_grad():
                cov = self._cov.forward(alt)
            for j in range(be - bs):
                ctx = self._contexts[bs + j]
                bins_t = _torch.as_tensor(
                    self._ctx_bins(ctx), device=self._cov.device, dtype=_torch.long
                )
                alt_values[bs + j] = self._cov.readout(
                    cov, j, bins_t, self._ctx_strand(ctx)
                )

        logsed_per_ctx = (
            _torch.log2(alt_values + 1.0) - _torch.log2(self._ref_values + 1.0)
        )
        return self._aggregate(logsed_per_ctx)

    def _predict(self, seqs: Sequence[str], desc: str) -> np.ndarray:
        """Shared predict loop with optional ``n_sample`` subsampling."""
        n = len(seqs)
        scores = np.full(n, np.nan, dtype=np.float64)

        if self.n_sample is not None and self.n_sample < n:
            rng = np.random.default_rng(self.seed)
            sample_idx = rng.choice(n, size=self.n_sample, replace=False)
        else:
            sample_idx = np.arange(n)

        for idx in tqdm(sample_idx, desc=desc):
            scores[idx] = self._score_one(seqs[idx])
        return scores
