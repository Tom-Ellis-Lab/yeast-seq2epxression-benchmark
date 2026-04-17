"""Registry of models and tasks. Adding a new model or task is one entry.

Each task declares an ``adapter_protocol`` (the interface the model must
implement to run the benchmark). Each model has exactly one adapter class
per protocol; the registry looks the adapter up by protocol and builds
it with the task's kwargs plus anything in ``model_config``.

Task factory signature:  ``(**task_config) → Benchmark``
Model factory signature: ``(task, device, **model_config) → adapter``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from yeastbench.adapters.protocols import (
    MarginalizedSequenceExpressionPredictor,
    SequenceExpressionPredictor,
    VariantEffectScorer,
)
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo
from yeastbench.benchmarks.eqtl import EQTLClassificationBenchmark
from yeastbench.benchmarks.mpra import (
    MPRAMarginalizedBenchmark,
    MPRARegressionBenchmark,
)

ModelFactory = Callable[..., Any]
TaskFactory = Callable[..., Benchmark]


# ──────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────
#
# Each model exposes a dict ``{protocol_type: (build_fn, needs_refs)}``.
# ``build_fn`` takes ``(device, **cfg)`` plus optional fasta/gtf paths and
# returns a constructed adapter.  ``needs_refs`` controls whether the
# dispatcher passes ``fasta_path``/``gtf_path`` from the task.


def _shorkie_eqtl_adapter(device, fasta_path, gtf_path, **cfg):
    from yeastbench.adapters.shorkie_eqtl import ShorkieVariantScorer

    return ShorkieVariantScorer.from_checkpoints(
        fasta_path=fasta_path,
        gtf_path=gtf_path,
        device=device,
        **cfg,
    )


def _shorkie_mpra_fixed_adapter(device, **cfg):
    from yeastbench.adapters.shorkie_mpra import ShorkieMPRAPredictor

    return ShorkieMPRAPredictor.from_checkpoints(device=device, **cfg)


def _shorkie_mpra_marginalized_adapter(device, fasta_path, gtf_path, **cfg):
    from yeastbench.adapters.shorkie_mpra_marginalized import (
        ShorkieMPRAMarginalizedPredictor,
    )

    return ShorkieMPRAMarginalizedPredictor.from_checkpoints(
        fasta_path=fasta_path,
        gtf_path=gtf_path,
        device=device,
        **cfg,
    )


def _yorzoi_eqtl_adapter(device, fasta_path, gtf_path, **cfg):
    from yeastbench.adapters.yorzoi_eqtl import YorzoiVariantScorer

    return YorzoiVariantScorer.from_pretrained(
        fasta_path=fasta_path,
        gtf_path=gtf_path,
        device=device,
        **cfg,
    )


def _yorzoi_mpra_fixed_adapter(device, **cfg):
    from yeastbench.adapters.yorzoi_mpra import YorzoiMPRAPredictor

    return YorzoiMPRAPredictor.from_pretrained(device=device, **cfg)


def _yorzoi_mpra_marginalized_adapter(device, fasta_path, gtf_path, **cfg):
    from yeastbench.adapters.yorzoi_mpra_marginalized import (
        YorzoiMPRAMarginalizedPredictor,
    )

    return YorzoiMPRAMarginalizedPredictor.from_pretrained(
        fasta_path=fasta_path,
        gtf_path=gtf_path,
        device=device,
        **cfg,
    )


# protocol → (build_fn, needs_refs)
SHORKIE_ADAPTERS: dict[type, tuple[Callable, bool]] = {
    VariantEffectScorer: (_shorkie_eqtl_adapter, True),
    SequenceExpressionPredictor: (_shorkie_mpra_fixed_adapter, False),
    MarginalizedSequenceExpressionPredictor: (_shorkie_mpra_marginalized_adapter, True),
}

YORZOI_ADAPTERS: dict[type, tuple[Callable, bool]] = {
    VariantEffectScorer: (_yorzoi_eqtl_adapter, True),
    SequenceExpressionPredictor: (_yorzoi_mpra_fixed_adapter, False),
    MarginalizedSequenceExpressionPredictor: (_yorzoi_mpra_marginalized_adapter, True),
}


def _dispatch(
    adapters: dict[type, tuple[Callable, bool]],
    task: Benchmark,
    device: str,
    **cfg: Any,
) -> Any:
    protocol = task.adapter_protocol
    if protocol not in adapters:
        raise ValueError(
            f"No adapter registered for protocol {protocol.__name__}. "
            f"Known: {[p.__name__ for p in adapters]}"
        )
    build_fn, needs_refs = adapters[protocol]
    if needs_refs:
        return build_fn(
            device=device,
            fasta_path=task.fasta_path,
            gtf_path=task.gtf_path,
            **cfg,
        )
    return build_fn(device=device, **cfg)


def _build_shorkie(task: Benchmark, device: str, **cfg: Any) -> Any:
    return _dispatch(SHORKIE_ADAPTERS, task, device, **cfg)


def _build_yorzoi(task: Benchmark, device: str, **cfg: Any) -> Any:
    return _dispatch(YORZOI_ADAPTERS, task, device, **cfg)


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
