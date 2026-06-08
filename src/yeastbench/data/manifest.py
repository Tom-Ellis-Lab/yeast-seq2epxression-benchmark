"""Declarative artifact manifest — the companion to ``registry.py``.

Every blob the benchmark needs (task data, model weights, shared genome
references) is declared here as an :class:`Artifact`. Declarations are the
source of truth for *what* exists and *where* it can be fetched from; the
per-file checksums live in the generated lock (``manifest.lock.json``).

Adding a task or model to ``registry.py`` should come with an artifact here —
:func:`validate_against_registry` (exercised by the test suite) fails if a
registered task/model has no declared data.

v1 declares the *processed* tier (benchmark-ready files) and model weights.
The ``raw`` field is reserved for the v2 raw→build→reproduce path and is left
``None`` everywhere for now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


# ──────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────


class Kind(str, Enum):
    TASK_PROCESSED = "task-processed"
    MODEL_WEIGHTS = "model-weights"
    SHARED_REFERENCE = "shared-reference"


class BackendKind(str, Enum):
    HF = "hf"
    GCS = "gcs"
    HTTP = "http"
    # v2: FIGSHARE, ZENODO, SOURCE


class License(str, Enum):
    OWN = "own"                # our data — mirror freely
    CC_BY_4_0 = "cc-by-4.0"    # mirror freely with attribution
    PUBLIC = "public"          # publicly hosted by upstream; fetch from origin
    GEO_TERMS = "geo-terms"    # NCBI GEO — do not re-host
    UNKNOWN = "unknown"


# ──────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Mirror:
    """One place an artifact's files can be fetched from.

    The remote path of a file with local relative path ``rel`` is
    ``member_remote[rel]`` if present, else ``prefix + rel``. A
    ``member_remote`` value that starts with ``http`` is treated as an absolute
    URL (used for source mirrors with an irregular layout).
    """

    backend: BackendKind
    base: str = ""                     # HF repo id, or gs:// / https:// base (trailing slash)
    prefix: str = ""                   # remote dir prepended to each file's relpath
    repo_type: str = "dataset"         # HF only: "dataset" | "model"
    member_remote: Mapping[str, str] = field(default_factory=dict)

    def remote_for(self, rel: str) -> str:
        if rel in self.member_remote:
            return self.member_remote[rel]
        return f"{self.prefix}{rel}"


@dataclass(frozen=True)
class Artifact:
    id: str
    kind: Kind
    dest: str                          # dir relative to the data root
    mirrors: tuple[Mirror, ...]
    needed_by: tuple[str, ...] = ()     # registry task/model names
    requires: tuple[str, ...] = ()      # other artifact ids (e.g. shared refs)
    include: tuple[str, ...] = ("**/*",)
    # Keep the published distribution to runtime files: drop the raw-download
    # cache, build logs, and raw source formats (spreadsheets, GenBank) that the
    # benchmarks don't read at run time.
    exclude: tuple[str, ...] = (
        "**/_cache/**", "*.log", "*.xlsx", "*.xls", "*.gb", "*.gbk",
    )
    license: License = License.UNKNOWN
    redistributable: bool = False
    cache_only: bool = False           # HF-hub model loaded at runtime; warm cache, nothing under dest
    raw: object | None = None          # v2 RawSpec — not implemented in v1


# ──────────────────────────────────────────────────────────────
# Mirror helpers
# ──────────────────────────────────────────────────────────────

# Planned mirrors for our own processed task data. These repos/buckets are the
# publish targets; until `ybench data publish` (v2) runs, `get` from them will
# report "not yet published" rather than failing hard.
_HF_DATA_REPO = "tom-ellis-lab/yeast-seq2expression-data"
_GCS_DATA_BUCKET = "gs://yeast-seq2expression/"


def _task_mirrors(artifact_id: str) -> tuple[Mirror, ...]:
    """HF-first, GCS-secondary mirrors for one task's processed data."""
    return (
        Mirror(BackendKind.HF, base=_HF_DATA_REPO, prefix=f"{artifact_id}/"),
        Mirror(BackendKind.GCS, base=_GCS_DATA_BUCKET, prefix=f"{artifact_id}_v1/"),
    )


def _task(
    artifact_id: str,
    *,
    dest: str,
    needed_by: tuple[str, ...],
    requires: tuple[str, ...] = (),
    license: License = License.OWN,
    include: tuple[str, ...] = ("**/*",),
) -> Artifact:
    return Artifact(
        id=artifact_id,
        kind=Kind.TASK_PROCESSED,
        dest=dest,
        mirrors=_task_mirrors(artifact_id),
        needed_by=needed_by,
        requires=requires,
        include=include,
        license=license,
        redistributable=(license in (License.OWN, License.CC_BY_4_0)),
    )


# ──────────────────────────────────────────────────────────────
# Declarations
# ──────────────────────────────────────────────────────────────

_REFS = "refs"

ARTIFACTS: tuple[Artifact, ...] = (
    # Shared genome references — declared once, required by many tasks.
    Artifact(
        id=_REFS,
        kind=Kind.SHARED_REFERENCE,
        dest="data/tasks",
        include=(
            "R64-1-1.fa", "R64-1-1.fa.fai", "R64-1-1.115.gtf",
            "R64-5-1.fa", "R64-5-1.fa.fai",
        ),
        mirrors=_task_mirrors(_REFS),
        license=License.PUBLIC,  # SGD reference genome
        redistributable=True,
    ),

    # ── Task-processed data ────────────────────────────────────
    _task(
        "cuperus_mpra_5utr",
        dest="data/tasks/cuperus_mpra_5utr",
        needed_by=("cuperus_utr",),
        requires=(_REFS,),
        # Our processed TSVs (License.OWN, publishable). The underlying raw is
        # GEO GSE104252 (public); that raw belongs to the v2 raw tier.
    ),
    _task(
        "caudal_eqtl",
        dest="data/tasks/caudal_eqtl",
        needed_by=("caudal_eqtl",),
        requires=(_REFS,),
    ),
    _task(
        "kita_eqtl",
        dest="data/tasks/kita_eqtl",
        needed_by=("kita_eqtl",),
        requires=(_REFS,),
    ),
    _task(
        "rafi_mpra",
        dest="data/tasks/rafi_mpra",
        needed_by=("rafi_mpra_marginalized",),
        requires=(_REFS,),
    ),
    _task(
        "shalem_mpra_terminator",
        dest="data/tasks/shalem_mpra_terminator",
        needed_by=("shalem_mpra_marginalized",),
        requires=(_REFS,),
    ),
    _task(
        "wu_rfpins",
        dest="data/tasks/wu_rfpins",
        needed_by=("wu_rfpins",),
        requires=(_REFS,),
    ),
    _task(
        "hong",
        dest="data/tasks/hong",
        needed_by=("hong_igr",),
        requires=(_REFS,),
    ),
    _task(
        "chen_synonymous",
        dest="data/tasks/chen_synonymous",
        needed_by=("chen_gfp_r1", "chen_gfp_r2", "chen_tdh3"),
        requires=(_REFS,),
    ),
    _task(
        "brooks_scramble",
        dest="data/tasks/brooks_scramble",
        needed_by=("brooks_scramble", "brooks_scramble_shorkie"),
        license=License.OWN,  # Brooks lab Nanopore data, our processing
    ),
    _task(
        "meneu_foreign_dna",
        dest="data/tasks/meneu_foreign_dna",
        needed_by=("meneu_foreign_dna", "meneu_foreign_dna_shorkie"),
        license=License.CC_BY_4_0,  # ExoShorkie figshare
    ),

    # ── Model weights ──────────────────────────────────────────
    # Shorkie: authors' PUBLIC bucket — fetchable today, no re-hosting.
    # Local layout is f{i}.h5 + params.json; the bucket stores f{i}/model_best.h5
    # and params.json lives in the shorkie-pytorch repo.
    Artifact(
        id="shorkie",
        kind=Kind.MODEL_WEIGHTS,
        dest="data/models/shorkie",
        needed_by=("shorkie",),
        include=("checkpoints/f0.h5", "checkpoints/f1.h5", "checkpoints/f2.h5",
                 "checkpoints/f3.h5", "checkpoints/f4.h5", "checkpoints/f5.h5",
                 "checkpoints/f6.h5", "checkpoints/f7.h5", "params.json",
                 "targets.txt"),  # read at runtime by shorkie_hong
        license=License.PUBLIC,
        redistributable=False,  # upstream's to host; we point at their bucket
        mirrors=(
            Mirror(
                BackendKind.HTTP,
                base="https://storage.googleapis.com/seqnn-share/shorkie/",
                member_remote={
                    "checkpoints/f0.h5": "f0/model_best.h5",
                    "checkpoints/f1.h5": "f1/model_best.h5",
                    "checkpoints/f2.h5": "f2/model_best.h5",
                    "checkpoints/f3.h5": "f3/model_best.h5",
                    "checkpoints/f4.h5": "f4/model_best.h5",
                    "checkpoints/f5.h5": "f5/model_best.h5",
                    "checkpoints/f6.h5": "f6/model_best.h5",
                    "checkpoints/f7.h5": "f7/model_best.h5",
                    "params.json": (
                        "https://raw.githubusercontent.com/tdsone/"
                        "shorkie-pytorch/main/data/shorkie_params.json"
                    ),
                },
            ),
        ),
    ),
    # Yorzoi + CodonTransformer: HF-hub models the adapters load via
    # from_pretrained at runtime. `get` warms the HF cache; nothing lands
    # under data/, so there's nothing to lock/verify locally.
    Artifact(
        id="yorzoi",
        kind=Kind.MODEL_WEIGHTS,
        dest="",
        needed_by=("yorzoi",),
        cache_only=True,
        include=(),
        license=License.PUBLIC,
        mirrors=(Mirror(BackendKind.HF, base="tom-ellis-lab/yorzoi", repo_type="model"),),
    ),
    Artifact(
        id="codon_transformer",
        kind=Kind.MODEL_WEIGHTS,
        dest="",
        needed_by=("codon_transformer",),
        cache_only=True,
        include=(),
        license=License.PUBLIC,
        mirrors=(Mirror(BackendKind.HF, base="adibvafa/CodonTransformer", repo_type="model"),),
    ),
)


# Models that legitimately have no downloadable artifact (pure baselines that
# derive everything from task data already covered elsewhere).
MODELS_WITHOUT_ARTIFACTS: frozenset[str] = frozenset({"cai"})


# ──────────────────────────────────────────────────────────────
# Lookups & selection
# ──────────────────────────────────────────────────────────────

_BY_ID = {a.id: a for a in ARTIFACTS}


def artifact_by_id(artifact_id: str) -> Artifact:
    try:
        return _BY_ID[artifact_id]
    except KeyError:
        raise KeyError(
            f"No artifact '{artifact_id}'. Known: {sorted(_BY_ID)}"
        ) from None


def artifacts_for(
    tasks: list[str] | None = None,
    models: list[str] | None = None,
) -> list[Artifact]:
    """Resolve the artifacts needed by the given task/model names, closing
    over ``requires`` (shared references). With both ``None``, returns every
    artifact. Order is stable and deduplicated.

    Unknown names raise — a typo shouldn't silently download nothing.
    """
    if tasks is None and models is None:
        return list(ARTIFACTS)

    wanted = set(tasks or []) | set(models or [])
    selected: dict[str, Artifact] = {}

    def _add(art: Artifact) -> None:
        if art.id in selected:
            return
        selected[art.id] = art
        for dep_id in art.requires:
            _add(artifact_by_id(dep_id))

    matched_names: set[str] = set()
    for art in ARTIFACTS:
        hit = wanted.intersection(art.needed_by)
        if hit:
            matched_names |= hit
            _add(art)

    unmatched = wanted - matched_names - MODELS_WITHOUT_ARTIFACTS
    if unmatched:
        raise KeyError(
            f"No artifacts declared for: {sorted(unmatched)}. "
            f"Check the names against `ybench list`."
        )

    # Preserve ARTIFACTS declaration order.
    return [a for a in ARTIFACTS if a.id in selected]


def validate_against_registry() -> list[str]:
    """Return a list of problems: registered tasks/models with no artifact, and
    artifacts pointing at unknown task/model names. Empty list == all good.
    """
    from yeastbench.registry import MODELS, TASKS

    problems: list[str] = []

    covered_tasks = {n for a in ARTIFACTS for n in a.needed_by}
    for task_name in TASKS:
        if task_name not in covered_tasks:
            problems.append(f"task '{task_name}' has no declared artifact")

    covered_models = {n for a in ARTIFACTS for n in a.needed_by}
    for model_name in MODELS:
        if model_name in MODELS_WITHOUT_ARTIFACTS:
            continue
        if model_name not in covered_models:
            problems.append(f"model '{model_name}' has no declared artifact")

    known = set(TASKS) | set(MODELS)
    for art in ARTIFACTS:
        for name in art.needed_by:
            if name not in known:
                problems.append(
                    f"artifact '{art.id}' references unknown name '{name}'"
                )
    return problems
