"""The checksum lock — authoritative per-file sha256 + size for every artifact.

The lock is generated from a local copy of the data (``ybench data lock``) and
committed (``manifest.lock.json``, shipped with the wheel). Downloads verify
fetched files against it; ``verify`` / ``status`` compare local files against
it.

Lock format::

    {
      "version": 1,
      "artifacts": {
        "<artifact id>": {
          "files": { "<relpath>": {"sha256": "...", "size": 123}, ... }
        }
      }
    }
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from yeastbench.data.manifest import Artifact

LOCK_VERSION = 1
LOCK_PATH = Path(__file__).with_name("manifest.lock.json")

_CHUNK = 1 << 20  # 1 MiB


# ──────────────────────────────────────────────────────────────
# Hashing & file discovery
# ──────────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _excluded(relpath: str, patterns: tuple[str, ...]) -> bool:
    parts = Path(relpath).parts
    for pat in patterns:
        token = pat.strip("*/")  # "**/_cache/**" -> "_cache"
        if token and token in parts:
            return True
        if fnmatch.fnmatch(relpath, pat):
            return True
    return False


def local_files(artifact: Artifact, data_root: Path) -> list[str]:
    """Relative paths (posix) of an artifact's local files, sorted, excludes
    applied. Empty for cache-only artifacts or a missing ``dest``."""
    if artifact.cache_only or not artifact.dest:
        return []
    base = data_root / artifact.dest
    if not base.exists():
        return []
    found: set[str] = set()
    for pattern in artifact.include:
        for path in base.glob(pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(base).as_posix()
            if _excluded(rel, artifact.exclude):
                continue
            found.add(rel)
    return sorted(found)


# ──────────────────────────────────────────────────────────────
# Build / read / write
# ──────────────────────────────────────────────────────────────


def build_lock(
    artifacts: list[Artifact], data_root: Path
) -> dict[str, dict]:
    """Compute the lock for ``artifacts`` from their local files. Artifacts
    with no local files are omitted."""
    out: dict[str, dict] = {}
    for art in artifacts:
        files = {}
        for rel in local_files(art, data_root):
            p = data_root / art.dest / rel
            files[rel] = {"sha256": sha256_file(p), "size": p.stat().st_size}
        if files:
            out[art.id] = {"files": files}
    return out


def merge_lock(existing: dict, fresh: dict[str, dict]) -> dict:
    """Overlay freshly computed artifact entries onto an existing lock,
    preserving entries for artifacts that weren't re-locked this run."""
    artifacts = dict(existing.get("artifacts", {}))
    artifacts.update(fresh)
    return {"version": LOCK_VERSION, "artifacts": artifacts}


def read_lock(path: Path = LOCK_PATH) -> dict:
    if not path.exists():
        return {"version": LOCK_VERSION, "artifacts": {}}
    return json.loads(path.read_text())


def write_lock(lock: dict, path: Path = LOCK_PATH) -> None:
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class LockedFile:
    relpath: str
    sha256: str
    size: int


def locked_files(lock: dict, artifact_id: str) -> list[LockedFile]:
    entry = lock.get("artifacts", {}).get(artifact_id)
    if not entry:
        return []
    return [
        LockedFile(rel, meta["sha256"], meta["size"])
        for rel, meta in sorted(entry["files"].items())
    ]


def artifact_ids_in_lock(lock: dict) -> set[str]:
    return set(lock.get("artifacts", {}))
