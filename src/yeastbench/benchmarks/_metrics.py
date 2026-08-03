"""Shared metric result types used by multiple benchmarks.

Lives here (rather than inside any task-specific benchmark) so benchmarks
don't import from each other — removing one task benchmark must not
break another. See PR #1 review.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MPRAStratumResult:
    """Per-stratum regression result for any expression-prediction benchmark
    (overall set or a named subset): sample count + Pearson r + Spearman ρ.
    Used by Rafi (`benchmarks/mpra.py`) and Shalem (`benchmarks/shalem.py`)."""

    name: str
    n: int
    pearson_r: float
    spearman_rho: float
