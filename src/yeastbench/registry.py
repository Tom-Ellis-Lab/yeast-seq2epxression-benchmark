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
    CassetteExpressionPredictor,
    CoverageTrackPredictor,
    FivePrimeUtrReporterExpressionPredictor,
    IGRInsertionExpressionPredictor,
    LocalCodingVariantPredictor,
    MarginalizedSequenceExpressionPredictor,
    TerminatorMarginalizedExpressionPredictor,
    TiledCoverageTrackPredictor,
    VariantEffectScorer,
)
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo
from yeastbench.benchmarks.chen import ChenSynonymousBenchmark
from yeastbench.benchmarks.eqtl import EQTLClassificationBenchmark
from yeastbench.benchmarks.hong_igr import HongIGRInsertionBenchmark
from yeastbench.benchmarks.mpra import MPRAMarginalizedBenchmark
from yeastbench.benchmarks.shalem import ShalemMPRAMarginalizedBenchmark
from yeastbench.benchmarks.rfpins import RFPInsertionBenchmark
from yeastbench.benchmarks.cuperus import CuperusUTRBenchmark

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


def _shorkie_shalem_adapter(device, fasta_path, gtf_path, **cfg):
    from yeastbench.adapters.shorkie_shalem import ShorkieShalemPredictor

    return ShorkieShalemPredictor.from_checkpoints(
        fasta_path=fasta_path,
        gtf_path=gtf_path,
        device=device,
        **cfg,
    )


def _yorzoi_shalem_adapter(device, fasta_path, gtf_path, **cfg):
    from yeastbench.adapters.yorzoi_shalem import YorzoiShalemPredictor

    return YorzoiShalemPredictor.from_pretrained(
        fasta_path=fasta_path,
        gtf_path=gtf_path,
        device=device,
        **cfg,
    )


def _shorkie_wu_adapter(device, fasta_path, gtf_path, **cfg):
    from yeastbench.adapters.shorkie_wu import ShorkieWuPredictor

    return ShorkieWuPredictor.from_checkpoints(
        fasta_path=fasta_path,
        gtf_path=gtf_path,
        device=device,
        **cfg,
    )


def _shorkie_chen_adapter(
    device, fasta_path, library, hosts_path, data_dir, **cfg,
):
    from yeastbench.adapters.shorkie_chen_marginalized import ShorkieChenPredictor

    return ShorkieChenPredictor.from_checkpoints(
        fasta_path=fasta_path,
        library=library,
        hosts_path=hosts_path,
        data_dir=data_dir,
        device=device,
        **cfg,
    )


def _yorzoi_chen_adapter(
    device, fasta_path, library, hosts_path, data_dir, **cfg,
):
    from yeastbench.adapters.yorzoi_chen_marginalized import YorzoiChenPredictor

    return YorzoiChenPredictor.from_pretrained(
        fasta_path=fasta_path,
        library=library,
        hosts_path=hosts_path,
        data_dir=data_dir,
        device=device,
        **cfg,
    )


def _yorzoi_wu_adapter(device, fasta_path, gtf_path, **cfg):
    from yeastbench.adapters.yorzoi_wu import YorzoiWuPredictor

    return YorzoiWuPredictor.from_pretrained(
        fasta_path=fasta_path,
        gtf_path=gtf_path,
        device=device,
        **cfg,
    )


def _shorkie_hong_adapter(device, fasta_path, **cfg):
    from yeastbench.adapters.shorkie_hong import ShorkieHongPredictor

    return ShorkieHongPredictor.from_checkpoints(
        fasta_path=fasta_path,
        device=device,
        **cfg,
    )


def _yorzoi_hong_adapter(device, fasta_path, **cfg):
    from yeastbench.adapters.yorzoi_hong import YorzoiHongPredictor

    return YorzoiHongPredictor.from_pretrained(
        fasta_path=fasta_path,
        device=device,
        **cfg,
    )


def _yorzoi_brooks_adapter(device, **cfg):
    from yeastbench.adapters.yorzoi_brooks import YorzoiBrooksPredictor

    return YorzoiBrooksPredictor.from_pretrained(device=device, **cfg)


def _shorkie_brooks_adapter(device, **cfg):
    from yeastbench.adapters.shorkie_brooks import ShorkieBrooksPredictor

    return ShorkieBrooksPredictor.from_checkpoints(device=device, **cfg)


def _shorkie_cuperus_adapter(device, fasta_path, **cfg):
    from yeastbench.adapters.shorkie_cuperus import ShorkieCuperusPredictor

    return ShorkieCuperusPredictor.from_checkpoints(
        fasta_path=fasta_path, device=device, **cfg,
    )


def _yorzoi_cuperus_adapter(device, fasta_path, **cfg):
    from yeastbench.adapters.yorzoi_cuperus import YorzoiCuperusPredictor

    return YorzoiCuperusPredictor.from_pretrained(
        fasta_path=fasta_path, device=device, **cfg,
    )


def _yorzoi_meneu_adapter(device, **cfg):
    from yeastbench.adapters.yorzoi_meneu import YorzoiMeneuPredictor

    return YorzoiMeneuPredictor.from_pretrained(device=device, **cfg)


def _shorkie_meneu_adapter(device, **cfg):
    from yeastbench.adapters.shorkie_meneu import ShorkieMeneuPredictor

    return ShorkieMeneuPredictor.from_checkpoints(device=device, **cfg)


# protocol → (build_fn, task_fields) — each name in task_fields is read
# from the constructed task object via getattr and forwarded to build_fn
# as a keyword argument.
REFS_FIELDS: tuple[str, ...] = ("fasta_path", "gtf_path")
FASTA_ONLY: tuple[str, ...] = ("fasta_path",)
CHEN_FIELDS: tuple[str, ...] = (
    "fasta_path", "library", "hosts_path", "data_dir",
)

SHORKIE_ADAPTERS: dict[type, tuple[Callable, tuple[str, ...]]] = {
    VariantEffectScorer: (_shorkie_eqtl_adapter, REFS_FIELDS),
    MarginalizedSequenceExpressionPredictor: (_shorkie_mpra_marginalized_adapter, REFS_FIELDS),
    TerminatorMarginalizedExpressionPredictor: (_shorkie_shalem_adapter, REFS_FIELDS),
    CassetteExpressionPredictor: (_shorkie_wu_adapter, REFS_FIELDS),
    IGRInsertionExpressionPredictor: (_shorkie_hong_adapter, FASTA_ONLY),
    FivePrimeUtrReporterExpressionPredictor: (_shorkie_cuperus_adapter, FASTA_ONLY),
    CoverageTrackPredictor: (_shorkie_brooks_adapter, ()),
    TiledCoverageTrackPredictor: (_shorkie_meneu_adapter, ()),
    LocalCodingVariantPredictor: (_shorkie_chen_adapter, CHEN_FIELDS),
}

YORZOI_ADAPTERS: dict[type, tuple[Callable, tuple[str, ...]]] = {
    VariantEffectScorer: (_yorzoi_eqtl_adapter, REFS_FIELDS),
    MarginalizedSequenceExpressionPredictor: (_yorzoi_mpra_marginalized_adapter, REFS_FIELDS),
    TerminatorMarginalizedExpressionPredictor: (_yorzoi_shalem_adapter, REFS_FIELDS),
    CassetteExpressionPredictor: (_yorzoi_wu_adapter, REFS_FIELDS),
    IGRInsertionExpressionPredictor: (_yorzoi_hong_adapter, FASTA_ONLY),
    FivePrimeUtrReporterExpressionPredictor: (_yorzoi_cuperus_adapter, FASTA_ONLY),
    CoverageTrackPredictor: (_yorzoi_brooks_adapter, ()),
    TiledCoverageTrackPredictor: (_yorzoi_meneu_adapter, ()),
    LocalCodingVariantPredictor: (_yorzoi_chen_adapter, CHEN_FIELDS),
}


def _dispatch(
    adapters: dict[type, tuple[Callable, tuple[str, ...]]],
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
    build_fn, task_fields = adapters[protocol]
    task_kwargs = {f: getattr(task, f) for f in task_fields}
    return build_fn(device=device, **task_kwargs, **cfg)


def _build_shorkie(task: Benchmark, device: str, **cfg: Any) -> Any:
    return _dispatch(SHORKIE_ADAPTERS, task, device, **cfg)


def _build_yorzoi(task: Benchmark, device: str, **cfg: Any) -> Any:
    return _dispatch(YORZOI_ADAPTERS, task, device, **cfg)


def _build_cai_baseline(task: Benchmark, device: str, **cfg: Any) -> Any:
    from yeastbench.adapters.baselines.cai import CAIBaselinePredictor

    if task.adapter_protocol is not LocalCodingVariantPredictor:
        raise ValueError(
            f"cai baseline only supports LocalCodingVariantPredictor tasks; "
            f"got {task.adapter_protocol.__name__}"
        )
    return CAIBaselinePredictor.from_task(task=task, **cfg)


def _build_codon_transformer_baseline(task: Benchmark, device: str, **cfg: Any) -> Any:
    from yeastbench.adapters.baselines.codon_transformer import (
        CodonTransformerBaselinePredictor,
    )

    if task.adapter_protocol is not LocalCodingVariantPredictor:
        raise ValueError(
            f"codon_transformer baseline only supports "
            f"LocalCodingVariantPredictor tasks; got "
            f"{task.adapter_protocol.__name__}"
        )
    return CodonTransformerBaselinePredictor.from_task(
        task=task, device=device, **cfg,
    )


MODELS: dict[str, ModelFactory] = {
    "shorkie": _build_shorkie,
    "yorzoi": _build_yorzoi,
    "cai": _build_cai_baseline,
    "codon_transformer": _build_codon_transformer_baseline,
}


# ──────────────────────────────────────────────────────────────
# Tasks
# ──────────────────────────────────────────────────────────────


def _build_caudal_eqtl(
    distribution_dir: str | Path,
    fasta_path: str | Path,
    gtf_path: str | Path,
) -> Benchmark:
    return EQTLClassificationBenchmark(
        distribution_dir=Path(distribution_dir),
        fasta_path=Path(fasta_path),
        gtf_path=Path(gtf_path),
        info=BenchmarkInfo(
            name="caudal_eqtl",
            version="v1",
            description="Caudal et al. yeast cis-eQTL classification",
            distribution_uri="gs://yeast-seq2expression/caudal_eqtl_v1/",
        ),
    )


def _build_kita_eqtl(
    distribution_dir: str | Path,
    fasta_path: str | Path,
    gtf_path: str | Path,
) -> Benchmark:
    return EQTLClassificationBenchmark(
        distribution_dir=Path(distribution_dir),
        fasta_path=Path(fasta_path),
        gtf_path=Path(gtf_path),
        info=BenchmarkInfo(
            name="kita_eqtl",
            version="v1",
            description="Kita et al. yeast cis-eQTL classification",
            distribution_uri="gs://yeast-seq2expression/kita_eqtl_v1/",
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


def _build_shalem_mpra_marginalized(
    data_path: str | Path,
    fasta_path: str | Path,
    gtf_path: str | Path,
) -> Benchmark:
    return ShalemMPRAMarginalizedBenchmark(
        data_path=Path(data_path),
        fasta_path=Path(fasta_path),
        gtf_path=Path(gtf_path),
        info=BenchmarkInfo(
            name="shalem_mpra_marginalized",
            version="v1",
            description="Shalem / Segal MPRA terminator expression (marginalized / native-position)",
            distribution_uri="",
        ),
    )


def _build_wu_rfpins(
    cassette_seq: str | Path,
    labels_path: str | Path,
    fasta_path: str | Path,
    gtf_path: str | Path,
) -> Benchmark:
    return RFPInsertionBenchmark(
        cassette_seq=Path(cassette_seq),
        labels_path=Path(labels_path),
        fasta_path=Path(fasta_path),
        gtf_path=Path(gtf_path),
        info=BenchmarkInfo(
            name="wu_rfpins",
            version="v1",
            description="Wu et al. genome-wide RFP-cassette position effects",
            distribution_uri="",
        ),
    )


def _build_hong_igr(
    labels_path: str | Path,
    cassette_seq: str | Path,
    fasta_path: str | Path,
) -> Benchmark:
    return HongIGRInsertionBenchmark(
        labels_path=Path(labels_path),
        cassette_seq=Path(cassette_seq),
        fasta_path=Path(fasta_path),
        info=BenchmarkInfo(
            name="hong_igr",
            version="v1",
            description="Hong et al. chromosomal-position effects on IGR-integrated mCherry",
            distribution_uri="",
        ),
    )


def _build_brooks_scramble(data_path: str | Path) -> Benchmark:
    from yeastbench.benchmarks.brooks import BrooksScrambleBenchmark

    return BrooksScrambleBenchmark(
        data_path=Path(data_path),
        info=BenchmarkInfo(
            name="brooks_scramble",
            version="v1",
            description="Brooks et al. SCRaMBLE structural-rearrangement effect",
            distribution_uri="",
        ),
    )


def _build_brooks_scramble_shorkie(data_path: str | Path) -> Benchmark:
    """Same benchmark class, but reads the 16,384 bp distribution
    rebuilt for Shorkie's receptive field. Until we have a unified
    max-window distribution (ROADMAP), separate task names are how
    we route the right TSV to the right model."""
    from yeastbench.benchmarks.brooks import BrooksScrambleBenchmark

    return BrooksScrambleBenchmark(
        data_path=Path(data_path),
        info=BenchmarkInfo(
            name="brooks_scramble_shorkie",
            version="v1-shorkie",
            description="Brooks et al. SCRaMBLE — 16,384 bp window (Shorkie)",
            distribution_uri="",
        ),
    )


def _build_meneu(
    data_path: str | Path =
        "data/tasks/meneu_foreign_dna/meneu_foreign_dna_v1.tsv",
) -> Benchmark:
    from yeastbench.benchmarks.meneu import MeneuForeignDNABenchmark

    return MeneuForeignDNABenchmark(
        data_path=Path(data_path),
        info=BenchmarkInfo(
            name="meneu_foreign_dna",
            version="v1",
            description="Meneu et al. foreign-DNA zero-shot coverage (4992 bp window)",
            distribution_uri="",
        ),
    )


def _build_meneu_shorkie(
    data_path: str | Path =
        "data/tasks/meneu_foreign_dna/meneu_foreign_dna_v1_w16384.tsv",
) -> Benchmark:
    """Same benchmark class, reads the 16,384 bp distribution for
    Shorkie's receptive field. Separate task names route the right TSV
    to the right model (mirrors the Brooks split)."""
    from yeastbench.benchmarks.meneu import MeneuForeignDNABenchmark

    return MeneuForeignDNABenchmark(
        data_path=Path(data_path),
        info=BenchmarkInfo(
            name="meneu_foreign_dna_shorkie",
            version="v1-shorkie",
            description="Meneu et al. foreign-DNA — 16,384 bp window (Shorkie)",
            distribution_uri="",
        ),
    )


def _build_chen(
    library: str,
    data_path: str | Path,
    fasta_path: str | Path,
    hosts_path: str | Path,
    data_dir: str | Path,
    replicate_ceiling_pearson: float,
    replicate_ceiling_spearman: float | None = None,
) -> Benchmark:
    return ChenSynonymousBenchmark(
        library=library,
        data_path=Path(data_path),
        fasta_path=Path(fasta_path),
        hosts_path=Path(hosts_path),
        data_dir=Path(data_dir),
        replicate_ceiling_pearson=float(replicate_ceiling_pearson),
        replicate_ceiling_spearman=(
            None if replicate_ceiling_spearman is None
            else float(replicate_ceiling_spearman)
        ),
        info=BenchmarkInfo(
            name=f"chen_{library}",
            version="v1",
            description=f"Chen et al. 2017 synonymous-mutation MPRA ({library})",
            distribution_uri="",
        ),
    )


def _build_cuperus_utr(
    random_path: str | Path,
    native_path: str | Path,
    fasta_path: str | Path,
) -> Benchmark:
    return CuperusUTRBenchmark(
        random_path=Path(random_path),
        native_path=Path(native_path),
        fasta_path=Path(fasta_path),
        info=BenchmarkInfo(
            name="cuperus_utr",
            version="v1",
            description="Cuperus et al. 2017 5'-UTR reporter expression",
            distribution_uri="",
        ),
    )


TASKS: dict[str, TaskFactory] = {
    "caudal_eqtl": _build_caudal_eqtl,
    "kita_eqtl": _build_kita_eqtl,
    "rafi_mpra_marginalized": _build_rafi_mpra_marginalized,
    "shalem_mpra_marginalized": _build_shalem_mpra_marginalized,
    "wu_rfpins": _build_wu_rfpins,
    "hong_igr": _build_hong_igr,
    "brooks_scramble": _build_brooks_scramble,
    "brooks_scramble_shorkie": _build_brooks_scramble_shorkie,
    "meneu_foreign_dna": _build_meneu,
    "meneu_foreign_dna_shorkie": _build_meneu_shorkie,
    "chen_gfp_r1": _build_chen,
    "chen_gfp_r2": _build_chen,
    "chen_tdh3": _build_chen,
    "cuperus_utr": _build_cuperus_utr,
}
