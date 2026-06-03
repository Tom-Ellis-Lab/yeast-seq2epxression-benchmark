"""Shorkie adapter for the Meneu foreign-DNA tiled-coverage benchmark.

Identical forward path to the Brooks adapter — Shorkie's Poisson/softplus
head emits raw predicted counts directly and its RNA-seq tracks are
unstranded coverage bigwigs, so ``predict_coverage_batch`` already returns
the unstranded per-base coverage the Meneu benchmark wants. The only
difference from ``ShorkieBrooksPredictor`` is that this class is the one
the registry constructs for ``TiledCoverageTrackPredictor`` tasks (Meneu
tiles whole contigs and stitches them, rather than scoring isolated
SCRaMBLE constructs). The dispatch is by the *task's* ``adapter_protocol``,
so the adapter needs no extra attribute — keeping a distinct class just
gives the registry build fn something to construct.

Geometry, track subset (``SHORKIE_T0_RNA_SEQ_TRACK_IDS``) and
``varies_by_strain = False`` are all inherited unchanged from
``ShorkieBrooksPredictor``; the Meneu contig sequence is the same 16,384 bp
window length, so no per-base assertion needs to change.
"""
from __future__ import annotations

import logging

from yeastbench.adapters.shorkie_brooks import ShorkieBrooksPredictor

log = logging.getLogger(__name__)


class ShorkieMeneuPredictor(ShorkieBrooksPredictor):
    """Thin subclass of :class:`ShorkieBrooksPredictor` for Meneu.

    No behavioural change — Shorkie is unstranded and emits raw counts, so
    the inherited ``predict_coverage_batch`` already returns the unstranded
    per-base coverage over the central ``seq_len - 2*crop`` region. The
    class exists purely so the registry can build a Meneu-specific adapter
    (the task dispatches on ``TiledCoverageTrackPredictor``)."""
