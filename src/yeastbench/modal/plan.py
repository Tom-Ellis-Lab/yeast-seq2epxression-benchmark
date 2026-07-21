"""Pure helpers for the Modal backend — no ``modal`` import, so this module is
importable (and unit-testable) in the base install.

Keeps the config-reading / pair-enumeration logic (which needs no Modal client)
out of ``app.py`` (which defines the remote App, images and functions).
"""
from __future__ import annotations

import hashlib
import posixpath
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from yeastbench.config import load_config

# First path segment every ``out_dir`` must live under. The Modal results volume
# mounts at ``/repo/<RESULTS_ROOT>``, so anything written outside it is lost when
# the container exits. Single source of truth — ``app.py`` builds the mount from
# this rather than repeating the literal.
RESULTS_ROOT = "results"


@dataclass(frozen=True)
class RemotePlan:
    config_path: Path
    config_name: str  # basename, e.g. "default.yaml"
    config_bytes: bytes  # exact bytes → preserves config_hash remotely
    out_dir: str  # raw out_dir from the config, e.g. "results/default"
    pairs: list[tuple[str, str]]  # (model, task), honoring --model/--task filters

    @property
    def source_hash(self) -> str:
        """The 12-char config hash `ybench run` prints — derived from the bytes
        (same algorithm as config.load_config), so config_bytes stays the single
        source of truth rather than storing the hash alongside it."""
        return hashlib.sha256(self.config_bytes).hexdigest()[:12]

    def pair_dirs(self) -> list[str]:
        return [f"{m}__{t}" for m, t in self.pairs]


def build_plan(
    config_path: str | Path, model: str | None = None, task: str | None = None
) -> RemotePlan:
    """Validate the config locally and describe what the remote run will do.

    Raises the same way ``ybench run`` would (bad path, no matching runs), plus a
    guard that ``out_dir`` lives under ``results/`` — the results volume mounts at
    ``/repo/results``, so anything written elsewhere would be lost on container
    exit. Failing here means no container ever spins up for a doomed run.
    """
    path = Path(config_path)
    cfg = load_config(path).filtered(model, task)
    pairs = [(r.model, t) for r in cfg.runs for t in r.tasks]
    if not pairs:
        raise ValueError(
            f"No runs match filters (model={model!r}, task={task!r}) in {path}"
        )
    # Normalize to a POSIX-relative path BEFORE checking the first segment: a raw
    # `parts[:1]` test lets `results/../scratch` through, which would resolve
    # outside the mount on the container and be silently discarded on exit.
    # Normalizing also keeps the path POSIX for the Linux container regardless of
    # the client OS.
    raw = str(cfg.out_dir).replace("\\", "/")
    out_dir = posixpath.normpath(raw)
    if out_dir.startswith("/") or PurePosixPath(out_dir).parts[:1] != (RESULTS_ROOT,):
        raise ValueError(
            f"Modal backend requires out_dir under '{RESULTS_ROOT}/' "
            f"(got {str(cfg.out_dir)!r}, normalizes to {out_dir!r}); the results "
            f"volume mounts at /repo/{RESULTS_ROOT}, so output written elsewhere "
            "would be lost when the container exits."
        )
    return RemotePlan(
        config_path=path.resolve(),
        config_name=path.name,
        config_bytes=path.read_bytes(),
        out_dir=out_dir,
        pairs=pairs,
    )
