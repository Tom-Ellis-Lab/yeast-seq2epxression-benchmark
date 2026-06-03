"""ExoShorkie adapter for the Meneu foreign-DNA tiled-coverage benchmark.

Meneu is ExoShorkie's most in-distribution task — zero-shot RNA-seq coverage of
foreign DNA tiled across whole contigs. The forward path is identical to the
Brooks adapter: ``ExoShorkieBrooksPredictor.predict_coverage_batch`` already
inverts the log-z normalization to count space and returns the per-base coverage
over the central ``seq_len - 2*crop`` region, RC-averaged across the 6 students.

Like ``ShorkieMeneuPredictor`` (a no-op subclass of ``ShorkieBrooksPredictor``),
this class exists purely so the registry can build a Meneu-specific adapter — the
Meneu task dispatches on ``TiledCoverageTrackPredictor`` rather than
``CoverageTrackPredictor``. Geometry (16,384 bp window) and ``varies_by_strain``
are inherited unchanged.

Note on the unstranded contract: Meneu compares against unstranded ``fwd + rev``
truth. ExoShorkie's RC averaging returns the per-strand *mean* (0.5·(fwd+rev))
rather than the sum, a constant 2× that cancels in Meneu's scale-invariant
metrics (per-window Pearson, fold-change ratios) — the same treatment as
Shorkie's inherited Brooks/Meneu path.
"""
from __future__ import annotations

import logging

from yeastbench.adapters.exoshorkie_brooks import ExoShorkieBrooksPredictor

log = logging.getLogger(__name__)


class ExoShorkieMeneuPredictor(ExoShorkieBrooksPredictor):
    """Thin subclass of :class:`ExoShorkieBrooksPredictor` for the Meneu task.

    No behavioural change — the inherited ``predict_coverage_batch`` already
    returns count-space per-base coverage over the central ``seq_len - 2*crop``
    region. The class exists so the registry can build a Meneu adapter (the task
    dispatches on ``TiledCoverageTrackPredictor``)."""
