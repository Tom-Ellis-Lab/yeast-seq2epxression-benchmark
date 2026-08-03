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

    def splice_batch(
        self, alt: "torch.Tensor", start: int,
        blocks: "torch.Tensor", length: int,
    ) -> None:
        """Vectorized splice: write ``blocks`` (one per row) into every row
        of ``alt`` at the same ``start`` (the position is per-context, shared
        across the candidate batch)."""
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

    def readout_batch(
        self, out: "torch.Tensor", bins_t: "torch.Tensor", strand: str,
    ) -> "torch.Tensor":
        """Batched :meth:`readout` over a candidate batch that all share one
        context's ``bins_t``/``strand``. Returns a ``(b,)`` tensor whose
        entries equal ``readout`` applied row by row."""
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

    def splice_batch(self, alt, start, blocks, length) -> None:
        # Vectorized splice: ``blocks`` is (b, 4, length); every row gets the
        # block at the same ``start`` (the position is per-context, not
        # per-candidate).
        alt[:, :, start : start + length] = blocks

    def forward(self, batch_oh):
        return self.model.forward_track_mean_perbase(batch_oh, self._track_idx_t)

    def readout(self, out, row, bins_t, strand):
        # ``bins_t`` holds exon BASE positions; out is (B, bins*BIN_WIDTH).
        return out[row].index_select(0, bins_t).sum()

    def readout_batch(self, out, bins_t, strand):
        # Batched ``readout``: out is (b, bins*BIN_WIDTH); all rows share one
        # context's exon ``bins_t``. Returns (b,) — the same per-row sum as
        # ``readout`` applied row by row.
        return out.index_select(1, bins_t).sum(dim=1)


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

    def splice_batch(self, alt, start, blocks, length) -> None:
        # Vectorized channels-last splice: ``blocks`` is (b, length, 4).
        alt[:, start : start + length, :] = blocks

    def forward(self, batch_oh):
        # (B, 162, OUTPUT_BINS*BIN_WIDTH) raw per-base counts (already float32).
        return self.model.forward_tracks_perbase(batch_oh)

    def readout(self, out, row, bins_t, strand):
        # ``bins_t`` holds exon BASE positions; out is (162, bins*BIN_WIDTH).
        per_track = out[row].index_select(1, bins_t).sum(dim=1)  # (162,)
        if strand == "+":
            return per_track[0:_YORZOI_N_PLUS_TRACKS].mean()
        return per_track[_YORZOI_N_PLUS_TRACKS:_YORZOI_N_TRACKS_TOTAL].mean()

    def readout_batch(self, out, bins_t, strand):
        # Batched ``readout``: out is (b, 162, bins*BIN_WIDTH). Returns (b,).
        per_track = out.index_select(2, bins_t).sum(dim=2)  # (b, 162)
        if strand == "+":
            return per_track[:, 0:_YORZOI_N_PLUS_TRACKS].mean(dim=1)
        return per_track[:, _YORZOI_N_PLUS_TRACKS:_YORZOI_N_TRACKS_TOTAL].mean(dim=1)


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
        baseline scalar plus the per-context readout/strand/splice metadata
        (materialized once here, reused across every candidate instead of
        being rebuilt per ``(candidate, context)``). Call once at the end of
        ``__init__``."""
        import torch as _torch

        ref_np = np.stack(
            [self._cov.one_hot(self._ref_window_seq(c)) for c in self._contexts]
        )
        self._ref_ohs_gpu = _torch.from_numpy(ref_np).to(self._cov.device)

        self._ctx_bins_t = [
            _torch.as_tensor(
                self._ctx_bins(c), device=self._cov.device, dtype=_torch.long
            )
            for c in self._contexts
        ]
        self._ctx_strands = [self._ctx_strand(c) for c in self._contexts]
        self._ctx_splice_starts = [
            int(self._ctx_splice_start(c)) for c in self._contexts
        ]

        self._ref_values = self._reduce_batches(self._ref_ohs_gpu, desc)

    def _reduce_batches(self, x_all: "torch.Tensor", desc: str) -> "torch.Tensor":
        """Forward every cached REF window in batches and reduce each to its
        per-context scalar. Each REF row carries its own exon readout, so
        (unlike the ALT path) the per-row reduction can't be batch-collapsed
        here."""
        import torch as _torch

        n = len(self._contexts)
        out_vals = _torch.zeros(n, device=self._cov.device, dtype=_torch.float32)
        for bs in tqdm(range(0, n, self.batch_size), desc=desc):
            be = min(bs + self.batch_size, n)
            with _torch.no_grad():
                cov = self._cov.forward(x_all[bs:be])
            for j in range(be - bs):
                out_vals[bs + j] = self._cov.readout(
                    cov, j, self._ctx_bins_t[bs + j], self._ctx_strands[bs + j]
                )
        return out_vals

    def _score_chunk(self, cands: Sequence[str]) -> list[float]:
        """logSED-aggregated scores for a *chunk* of candidate inserts.

        Batches over candidates rather than contexts: for each context the
        whole candidate chunk is spliced into that context's REF window in
        one vectorized op, then forwarded in ``batch_size`` rows — so the GPU
        sees a batch drawn from the 71k candidate axis instead of the ~22
        host-gene contexts. Per ``(candidate, context)`` the spliced input
        (and therefore the model output and readout) is identical to the
        per-candidate path; only the batch composition changes."""
        import torch as _torch

        C = len(cands)
        N = len(self._contexts)
        dev = self._cov.device

        encoded = [self._encode_candidate(c) for c in cands]
        block_lens = {bl for _, _, bl in encoded}
        assert len(block_lens) == 1, (
            f"_score_chunk needs a uniform block length per chunk, got {block_lens}"
        )
        blk = block_lens.pop()
        fwd_blocks = _torch.from_numpy(
            np.stack([self._cov.one_hot(f) for f, _, _ in encoded])
        ).to(dev)
        rc_blocks = _torch.from_numpy(
            np.stack([self._cov.one_hot(r) for _, r, _ in encoded])
        ).to(dev)

        alt_values = _torch.zeros(C, N, device=dev, dtype=_torch.float32)
        for j in range(N):
            strand = self._ctx_strands[j]
            blocks = rc_blocks if strand == "-" else fwd_blocks
            start = self._ctx_splice_starts[j]
            bins_t = self._ctx_bins_t[j]
            ref_j = self._ref_ohs_gpu[j]
            for bs in range(0, C, self.batch_size):
                be = min(bs + self.batch_size, C)
                alt = ref_j.unsqueeze(0).expand(be - bs, *ref_j.shape).clone()
                self._cov.splice_batch(alt, start, blocks[bs:be], blk)
                with _torch.no_grad():
                    cov = self._cov.forward(alt)
                alt_values[bs:be, j] = self._cov.readout_batch(cov, bins_t, strand)

        logsed = (
            _torch.log2(alt_values + 1.0)
            - _torch.log2(self._ref_values.unsqueeze(0) + 1.0)
        )
        return [self._aggregate(logsed[c]) for c in range(C)]

    def _predict(self, seqs: Sequence[str], desc: str) -> np.ndarray:
        """Shared predict loop with optional ``n_sample`` subsampling.

        Candidates are processed in chunks; within each chunk every context
        is scored with the candidate axis batched onto the GPU."""
        n = len(seqs)
        scores = np.full(n, np.nan, dtype=np.float64)

        if self.n_sample is not None and self.n_sample < n:
            rng = np.random.default_rng(self.seed)
            sample_idx = rng.choice(n, size=self.n_sample, replace=False)
        else:
            sample_idx = np.arange(n)

        cand_chunk = max(self.batch_size, 128)
        pbar = tqdm(total=len(sample_idx), desc=desc)
        for cs in range(0, len(sample_idx), cand_chunk):
            chunk_idx = sample_idx[cs : cs + cand_chunk]
            chunk_scores = self._score_chunk([seqs[int(i)] for i in chunk_idx])
            for k, gi in enumerate(chunk_idx):
                scores[int(gi)] = chunk_scores[k]
            pbar.update(len(chunk_idx))
        pbar.close()
        return scores
