"""Resolve, plan, fetch, and verify artifacts.

Glue between the manifest (what exists), the lock (per-file checksums), and the
backends (how to fetch). The CLI is a thin wrapper over these functions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from yeastbench.data.backends import BackendError, HfBackend, get_backend
from yeastbench.data.lock import LockedFile, locked_files, sha256_file
from yeastbench.data.manifest import (
    Artifact,
    BackendKind,
    Mirror,
    artifacts_for,
)

Log = Callable[[str], None]


def _noop(_: str) -> None:
    pass


# ──────────────────────────────────────────────────────────────
# Data root & selection
# ──────────────────────────────────────────────────────────────


def default_data_root() -> Path:
    """Repo root (nearest ancestor with pyproject.toml), else cwd. Artifact
    ``dest`` paths are resolved against this."""
    here = Path.cwd().resolve()
    for d in (here, *here.parents):
        if (d / "pyproject.toml").exists():
            return d
    return here


def selection_from_config(
    config_path: Path, model: str | None, task: str | None
) -> tuple[list[str], list[str]]:
    """(tasks, models) named by a run config, honoring --model/--task filters."""
    from yeastbench.config import load_config

    cfg = load_config(config_path).filtered(model, task)
    tasks = sorted({t for r in cfg.runs for t in r.tasks})
    models = sorted({r.model for r in cfg.runs})
    return tasks, models


# ──────────────────────────────────────────────────────────────
# Planning
# ──────────────────────────────────────────────────────────────


@dataclass
class FilePlan:
    relpath: str
    dest_file: Path
    sha256: str
    size: int
    state: str  # "ok" | "stale" | "missing"


@dataclass
class ArtifactPlan:
    artifact: Artifact
    files: list[FilePlan] = field(default_factory=list)
    mirror: Mirror | None = None
    mirror_available: bool = False
    note: str | None = None  # why nothing can be fetched, if applicable

    @property
    def cache_only(self) -> bool:
        return self.artifact.cache_only

    @property
    def to_fetch(self) -> list[FilePlan]:
        return [f for f in self.files if f.state != "ok"]


def _file_state(dest_file: Path, sha256: str) -> str:
    if not dest_file.exists():
        return "missing"
    return "ok" if sha256_file(dest_file) == sha256 else "stale"


def _choose_mirror(
    artifact: Artifact, from_kind: BackendKind | None
) -> tuple[Mirror | None, bool]:
    candidates = [
        m for m in artifact.mirrors if from_kind is None or m.backend == from_kind
    ]
    if not candidates:
        return None, False
    for m in candidates:
        if get_backend(m.backend).available():
            return m, True
    return candidates[0], False  # selected but its backend isn't usable here


def build_plans(
    artifacts: list[Artifact],
    lock: dict,
    from_kind: BackendKind | None,
    data_root: Path,
) -> list[ArtifactPlan]:
    plans: list[ArtifactPlan] = []
    for art in artifacts:
        mirror, available = _choose_mirror(art, from_kind)
        plan = ArtifactPlan(artifact=art, mirror=mirror, mirror_available=available)

        if art.cache_only:
            if mirror is None:
                plan.note = f"no mirror with backend {from_kind}"
            elif not available:
                plan.note = f"{mirror.backend.value} backend unavailable"
            plans.append(plan)
            continue

        locked: list[LockedFile] = locked_files(lock, art.id)
        if not locked:
            plan.note = "not in lock (run `ybench data lock` against a local copy)"
            plans.append(plan)
            continue

        for lf in locked:
            dest_file = data_root / art.dest / lf.relpath
            plan.files.append(
                FilePlan(
                    relpath=lf.relpath,
                    dest_file=dest_file,
                    sha256=lf.sha256,
                    size=lf.size,
                    state=_file_state(dest_file, lf.sha256),
                )
            )
        if plan.to_fetch and mirror is None:
            plan.note = f"no mirror with backend {from_kind}"
        elif plan.to_fetch and not available:
            plan.note = f"{mirror.backend.value} backend unavailable / not published"
        plans.append(plan)
    return plans


# ──────────────────────────────────────────────────────────────
# Get
# ──────────────────────────────────────────────────────────────


@dataclass
class GetSummary:
    fetched: int = 0
    skipped_ok: int = 0
    warmed: int = 0
    skipped: list[str] = field(default_factory=list)   # mirror not reachable yet
    failures: list[str] = field(default_factory=list)  # fetched but wrong / errored

    @property
    def ok(self) -> bool:
        return not self.failures


def run_get(plans: list[ArtifactPlan], log: Log = _noop) -> GetSummary:
    summary = GetSummary()
    for plan in plans:
        art = plan.artifact

        if plan.cache_only:
            if not plan.mirror_available or plan.mirror is None:
                summary.skipped.append(f"{art.id}: {plan.note}")
                log(f"  · {art.id}: {plan.note}")
                continue
            log(f"  warming HF cache: {plan.mirror.base}")
            try:
                HfBackend().warm(plan.mirror)
                summary.warmed += 1
            except BackendError as e:
                summary.failures.append(f"{art.id}: {e}")
                log(f"  ! {art.id}: {e}")
            continue

        to_fetch = plan.to_fetch
        summary.skipped_ok += len(plan.files) - len(to_fetch)
        if not to_fetch:
            log(f"  ✓ {art.id}: up to date ({len(plan.files)} files)")
            continue
        if plan.mirror is None or not plan.mirror_available:
            summary.skipped.append(f"{art.id}: {plan.note}")
            log(f"  · {art.id}: {plan.note} — skipping {len(to_fetch)} file(s)")
            continue

        backend = get_backend(plan.mirror.backend)
        for fp in to_fetch:
            remote = plan.mirror.remote_for(fp.relpath)
            tmp = fp.dest_file.with_name(fp.dest_file.name + ".part")
            log(f"  ↓ {art.id}/{fp.relpath}  ({_human(fp.size)})")
            try:
                backend.fetch(plan.mirror, remote, tmp)
                got = sha256_file(tmp)
                if got != fp.sha256:
                    tmp.unlink(missing_ok=True)
                    msg = f"{art.id}/{fp.relpath}: checksum mismatch"
                    summary.failures.append(msg)
                    log(f"  ! {msg}")
                    continue
                os.replace(tmp, fp.dest_file)
                summary.fetched += 1
            except BackendError as e:
                tmp.unlink(missing_ok=True)
                summary.failures.append(f"{art.id}/{fp.relpath}: {e}")
                log(f"  ! {art.id}/{fp.relpath}: {e}")
    return summary


# ──────────────────────────────────────────────────────────────
# Verify / status
# ──────────────────────────────────────────────────────────────


@dataclass
class VerifyRow:
    artifact_id: str
    relpath: str
    status: str  # "ok" | "stale" | "missing"


def run_verify(
    artifacts: list[Artifact], lock: dict, data_root: Path
) -> list[VerifyRow]:
    rows: list[VerifyRow] = []
    for art in artifacts:
        for lf in locked_files(lock, art.id):
            dest_file = data_root / art.dest / lf.relpath
            rows.append(
                VerifyRow(art.id, lf.relpath, _file_state(dest_file, lf.sha256))
            )
    return rows


@dataclass
class StatusRow:
    artifact_id: str
    kind: str
    present: int
    total: int
    bytes_present: int
    bytes_total: int
    cache_only: bool


def run_status(
    artifacts: list[Artifact], lock: dict, data_root: Path
) -> list[StatusRow]:
    rows: list[StatusRow] = []
    for art in artifacts:
        locked = locked_files(lock, art.id)
        present = bytes_present = 0
        bytes_total = sum(lf.size for lf in locked)
        for lf in locked:
            dest_file = data_root / art.dest / lf.relpath
            if dest_file.exists() and dest_file.stat().st_size == lf.size:
                present += 1
                bytes_present += lf.size
        rows.append(
            StatusRow(
                artifact_id=art.id,
                kind=art.kind.value,
                present=present,
                total=len(locked),
                bytes_present=bytes_present,
                bytes_total=bytes_total,
                cache_only=art.cache_only,
            )
        )
    return rows


def _human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.0f}{unit}" if unit == "B" else f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}TB"


__all__ = [
    "ArtifactPlan",
    "FilePlan",
    "GetSummary",
    "StatusRow",
    "VerifyRow",
    "artifacts_for",
    "build_plans",
    "default_data_root",
    "run_get",
    "run_status",
    "run_verify",
    "selection_from_config",
]
