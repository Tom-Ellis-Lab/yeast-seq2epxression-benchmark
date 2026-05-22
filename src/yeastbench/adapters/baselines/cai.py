"""CAI baseline for the Chen synonymous-mutation benchmark.

Reads the per-variant ``CAI`` column the Chen authors ship in supp tables
S7/S8/S9 (carried through by ``scripts/chen/build_distribution_tsvs.py``)
and returns it as the score, indexed by the (library_id, variable_seq)
pair. No reference-set choice, no codon-usage table, no genomic model.

See ``benchmarks/chen_synonymous.md`` for why we use Chen's column rather
than recomputing CAI ourselves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from yeastbench.adapters.protocols import LocalCodingVariantPredictor


class CAIBaselinePredictor(LocalCodingVariantPredictor):
    """Look up Chen's precomputed CAI by (library_id, variable_seq).

    A single instance can serve multiple libraries — register one CAI
    map per library via ``register_library``. The registry usually wires
    one instance per task, so in practice this is one library per
    adapter, but the multi-library path is here for completeness.
    """

    def __init__(self) -> None:
        self._tables: dict[str, dict[str, float]] = {}

    def register_library(self, library_id: str, data_path: Path) -> None:
        df = pd.read_csv(data_path, sep="\t")
        if "CAI" not in df.columns:
            raise ValueError(f"{data_path}: no CAI column")
        seqs = df["variable_seq"].astype(str).str.upper()
        self._tables[library_id] = dict(zip(seqs, df["CAI"].astype(float)))

    @classmethod
    def from_task(cls, task, **_ignored) -> "CAIBaselinePredictor":
        # The Chen task carries ``library`` and ``data_path``. We register
        # just that library — the benchmark calls predict only with
        # matching library_ids.
        instance = cls()
        instance.register_library(task.library, task.data_path)
        return instance

    def predict_local_variants(
        self,
        library_ids: Sequence[str],
        variant_seqs: Sequence[str],
    ) -> np.ndarray:
        out = np.empty(len(library_ids), dtype=float)
        for i, (lib, seq) in enumerate(zip(library_ids, variant_seqs)):
            if lib not in self._tables:
                raise KeyError(
                    f"CAI baseline has no table for library_id {lib!r}; "
                    f"known: {sorted(self._tables)}"
                )
            try:
                out[i] = self._tables[lib][seq.upper()]
            except KeyError:
                out[i] = float("nan")
        return out
