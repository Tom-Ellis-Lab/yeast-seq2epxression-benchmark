from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from yeastbench.adapters.protocols import Variant, VariantEffectScorer
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo


@dataclass(frozen=True)
class EQTLIterationResult:
    name: str
    scores: np.ndarray            # signed logSED_agg, shape (2N,)
    labels: np.ndarray            # 1/0 alternating pos/neg, shape (2N,)
    pairs: pd.DataFrame           # pair-level metadata (pair_id, distances, ...)
    auroc_signed: float
    auprc_signed: float
    auroc_abs: float              # |score| is the classification signal
    auprc_abs: float


@dataclass(frozen=True)
class EQTLResults:
    per_iter: list[EQTLIterationResult]

    def _agg(self, key: str) -> tuple[float, float]:
        xs = np.asarray([getattr(r, key) for r in self.per_iter], dtype=float)
        mean = float(xs.mean())
        sem = float(xs.std(ddof=1) / np.sqrt(xs.size)) if xs.size > 1 else 0.0
        return mean, sem

    @property
    def mean_auroc(self) -> float:
        return self._agg("auroc_abs")[0]

    @property
    def sem_auroc(self) -> float:
        return self._agg("auroc_abs")[1]

    @property
    def mean_auprc(self) -> float:
        return self._agg("auprc_abs")[0]

    @property
    def sem_auprc(self) -> float:
        return self._agg("auprc_abs")[1]


class EQTLClassificationBenchmark(Benchmark[VariantEffectScorer, EQTLResults]):
    adapter_protocol: ClassVar[type] = VariantEffectScorer

    def __init__(self, distribution_dir: Path, info: BenchmarkInfo) -> None:
        self.distribution_dir = Path(distribution_dir)
        self.info = info
        self.iteration_files = sorted(self.distribution_dir.glob("negset_*.tsv"))

    @property
    def fasta_path(self) -> Path:
        return self.distribution_dir / "reference" / "R64-1-1.fa"

    @property
    def gtf_path(self) -> Path:
        return self.distribution_dir / "reference" / "R64-1-1.115.gtf"

    def evaluate(self, adapter: VariantEffectScorer) -> EQTLResults:
        per_iter: list[EQTLIterationResult] = []
        for tsv in self.iteration_files:
            pairs = pd.read_csv(
                tsv, sep="\t", dtype={"pos_chrom": str, "neg_chrom": str}
            )
            variants: list[Variant] = []
            labels: list[int] = []
            meta_rows: list[dict] = []
            for row in pairs.itertuples():
                variants.append(
                    Variant(
                        row.pos_chrom, row.pos_pos, row.pos_ref, row.pos_alt, row.pos_gene
                    )
                )
                variants.append(
                    Variant(
                        row.neg_chrom, row.neg_pos, row.neg_ref, row.neg_alt, row.neg_gene
                    )
                )
                labels.extend([1, 0])
                meta_rows.append(
                    {
                        "pair_id": int(row.pair_id),
                        "pos_distance_to_tss": int(row.pos_distance_to_tss),
                        "neg_distance_to_tss": int(row.neg_distance_to_tss),
                    }
                )
            scores = np.asarray(adapter.score_variants(variants), dtype=float)
            labels_a = np.asarray(labels)
            per_iter.append(
                EQTLIterationResult(
                    name=tsv.stem,
                    scores=scores,
                    labels=labels_a,
                    pairs=pd.DataFrame(meta_rows),
                    auroc_signed=float(roc_auc_score(labels_a, scores)),
                    auprc_signed=float(average_precision_score(labels_a, scores)),
                    auroc_abs=float(roc_auc_score(labels_a, np.abs(scores))),
                    auprc_abs=float(average_precision_score(labels_a, np.abs(scores))),
                )
            )
        return EQTLResults(per_iter=per_iter)

    def plot(self, results: EQTLResults, out_dir: Path) -> None:
        raise NotImplementedError
