from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from yeastbench.adapters.protocols import Variant, VariantEffectScorer
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo


@dataclass(frozen=True)
class EQTLIterationResult:
    auroc: float
    auprc: float
    fpr: np.ndarray
    tpr: np.ndarray
    precision: np.ndarray
    recall: np.ndarray


@dataclass(frozen=True)
class EQTLResults:
    per_iter: list[EQTLIterationResult]

    @property
    def mean_auroc(self) -> float:
        return float(np.mean([r.auroc for r in self.per_iter]))

    @property
    def sem_auroc(self) -> float:
        return _sem([r.auroc for r in self.per_iter])

    @property
    def mean_auprc(self) -> float:
        return float(np.mean([r.auprc for r in self.per_iter]))

    @property
    def sem_auprc(self) -> float:
        return _sem([r.auprc for r in self.per_iter])


class EQTLClassificationBenchmark(Benchmark[VariantEffectScorer, EQTLResults]):
    adapter_protocol: ClassVar[type] = VariantEffectScorer

    def __init__(self, distribution_dir: Path, info: BenchmarkInfo) -> None:
        self.distribution_dir = distribution_dir
        self.info = info
        self.iteration_files = sorted(distribution_dir.glob("negset_*.tsv"))

    def evaluate(self, adapter: VariantEffectScorer) -> EQTLResults:
        per_iter: list[EQTLIterationResult] = []
        for tsv in self.iteration_files:
            pairs = pd.read_csv(
                tsv, sep="\t", dtype={"pos_chrom": str, "neg_chrom": str}
            )
            variants: list[Variant] = []
            labels: list[int] = []
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
            scores = np.asarray(adapter.score_variants(variants), dtype=float)
            per_iter.append(_compute_iteration_result(scores, np.asarray(labels)))
        return EQTLResults(per_iter=per_iter)

    def plot(self, results: EQTLResults, out_dir: Path) -> None:
        raise NotImplementedError


def _compute_iteration_result(
    scores: np.ndarray, labels: np.ndarray
) -> EQTLIterationResult:
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    return EQTLIterationResult(
        auroc=float(roc_auc_score(labels, scores)),
        auprc=float(average_precision_score(labels, scores)),
        fpr=fpr,
        tpr=tpr,
        precision=precision,
        recall=recall,
    )


def _sem(xs: list[float]) -> float:
    arr = np.asarray(xs, dtype=float)
    if arr.size < 2:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))
