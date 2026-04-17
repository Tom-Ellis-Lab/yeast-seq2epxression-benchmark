"""Registry of models and tasks. Adding a new model or task is one entry.

Each model factory has the signature ``(task, device, **model_config) → adapter``.
Each task factory has the signature ``(**task_config) → Benchmark``.

The runner builds the task first, then the adapter, passing the task and
global device into the adapter factory so it can pull task-specific
reference files (FASTA, GTF, …) without the YAML duplicating them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from yeastbench.adapters.protocols import (
    SequenceExpressionPredictor,
    VariantEffectScorer,
)
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo
from yeastbench.benchmarks.eqtl import EQTLClassificationBenchmark
from yeastbench.benchmarks.mpra import MPRAMarginalizedBenchmark, MPRARegressionBenchmark

ModelFactory = Callable[..., Any]
TaskFactory = Callable[..., Benchmark]


# ──────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────


def _build_shorkie(task: Benchmark, device: str, **cfg: Any) -> Any:
    if task.adapter_protocol is SequenceExpressionPredictor:
        # Pop marginalized-only kwargs so they don't leak to fixed-context.
        n_sample = cfg.pop("n_sample", None)
        seed = cfg.pop("seed", 42)

        if hasattr(task, "fasta_path") and hasattr(task, "gtf_path"):
            from yeastbench.adapters.shorkie_mpra_marginalized import (
                ShorkieMPRAMarginalizedPredictor,
            )

            return ShorkieMPRAMarginalizedPredictor.from_checkpoints(
                fasta_path=task.fasta_path,
                gtf_path=task.gtf_path,
                device=device,
                n_sample=n_sample,
                seed=seed,
                **cfg,
            )

        from yeastbench.adapters.shorkie_mpra import ShorkieMPRAPredictor

        return ShorkieMPRAPredictor.from_checkpoints(device=device, **cfg)

    from yeastbench.adapters.shorkie_eqtl import ShorkieVariantScorer

    return ShorkieVariantScorer.from_checkpoints(
        fasta_path=task.fasta_path,
        gtf_path=task.gtf_path,
        device=device,
        **cfg,
    )


def _build_yorzoi(task: Benchmark, device: str, **cfg: Any) -> Any:
    if task.adapter_protocol is SequenceExpressionPredictor:
        n_sample = cfg.pop("n_sample", None)
        seed = cfg.pop("seed", 42)

        if hasattr(task, "fasta_path") and hasattr(task, "gtf_path"):
            from yeastbench.adapters.yorzoi_mpra_marginalized import (
                YorzoiMPRAMarginalizedPredictor,
            )

            return YorzoiMPRAMarginalizedPredictor.from_pretrained(
                fasta_path=task.fasta_path,
                gtf_path=task.gtf_path,
                device=device,
                n_sample=n_sample,
                seed=seed,
                **cfg,
            )

        from yeastbench.adapters.yorzoi_mpra import YorzoiMPRAPredictor

        return YorzoiMPRAPredictor.from_pretrained(device=device, **cfg)

    from yeastbench.adapters.yorzoi_eqtl import YorzoiVariantScorer

    return YorzoiVariantScorer.from_pretrained(
        fasta_path=task.fasta_path,
        gtf_path=task.gtf_path,
        device=device,
        **cfg,
    )


MODELS: dict[str, ModelFactory] = {
    "shorkie": _build_shorkie,
    "yorzoi": _build_yorzoi,
}


# ──────────────────────────────────────────────────────────────
# Tasks
# ──────────────────────────────────────────────────────────────


def _build_caudal_eqtl(distribution_dir: str | Path) -> Benchmark:
    return EQTLClassificationBenchmark(
        distribution_dir=Path(distribution_dir),
        info=BenchmarkInfo(
            name="caudal_eqtl",
            version="v1",
            description="Caudal et al. yeast cis-eQTL classification",
            distribution_uri="gs://yeast-seq2expression/caudal_eqtl_v1/",
        ),
    )


def _build_rafi_mpra_promoter(data_dir: str | Path) -> Benchmark:
    return MPRARegressionBenchmark(
        data_dir=Path(data_dir),
        info=BenchmarkInfo(
            name="rafi_mpra_promoter",
            version="v1",
            description="Rafi / deBoer DREAM MPRA promoter expression (fixed-context)",
            distribution_uri="",
        ),
    )


def _build_rafi_mpra_marginalized(
    data_dir: str | Path,
    fasta_path: str | Path,
    gtf_path: str | Path,
) -> Benchmark:
    return MPRAMarginalizedBenchmark(
        data_dir=Path(data_dir),
        fasta_path=Path(fasta_path),
        gtf_path=Path(gtf_path),
        info=BenchmarkInfo(
            name="rafi_mpra_marginalized",
            version="v1",
            description="Rafi / deBoer MPRA promoter expression (marginalized / native-position)",
            distribution_uri="",
        ),
    )


TASKS: dict[str, TaskFactory] = {
    "caudal_eqtl": _build_caudal_eqtl,
    "rafi_mpra_promoter": _build_rafi_mpra_promoter,
    "rafi_mpra_marginalized": _build_rafi_mpra_marginalized,
}
