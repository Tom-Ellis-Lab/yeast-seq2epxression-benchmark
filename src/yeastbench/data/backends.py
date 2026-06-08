"""Download backends — one per :class:`BackendKind`.

Each backend writes a single remote file to a local path (the caller supplies a
temp ``.part`` path, verifies the checksum, then moves it into place — so
backends don't need to worry about atomicity or verification themselves).

- HTTP  — stdlib ``urllib``; public URLs, no auth. Resolves the Shorkie
          authors' public bucket today.
- HF    — ``huggingface_hub``; free, no-auth public mirror. Also warms the HF
          cache for runtime-loaded models.
- GCS   — shells out to ``gcloud storage cp`` (fallback ``gsutil cp``), reusing
          the user's existing gcloud auth.
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from yeastbench.data.manifest import BackendKind, Mirror


class BackendError(RuntimeError):
    """A fetch failed in a way the caller can report (missing file, no auth,
    mirror not published, …)."""


class Backend(Protocol):
    kind: BackendKind

    def available(self) -> bool:
        """Whether this backend can run here (tool installed / lib importable)."""

    def fetch(self, mirror: Mirror, remote: str, out_path: Path) -> None:
        """Download ``remote`` (resolved against ``mirror``) to ``out_path``."""


def _ensure_parent(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# HTTP
# ──────────────────────────────────────────────────────────────


class HttpBackend:
    kind = BackendKind.HTTP

    def available(self) -> bool:
        return True

    def _url(self, mirror: Mirror, remote: str) -> str:
        if remote.startswith("http://") or remote.startswith("https://"):
            return remote
        return f"{mirror.base}{remote}"

    def fetch(self, mirror: Mirror, remote: str, out_path: Path) -> None:
        url = self._url(mirror, remote)
        _ensure_parent(out_path)
        req = urllib.request.Request(url, headers={"User-Agent": "yeastbench-data"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(
                out_path, "wb"
            ) as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
        except urllib.error.HTTPError as e:
            raise BackendError(f"HTTP {e.code} fetching {url}") from e
        except urllib.error.URLError as e:
            raise BackendError(f"could not reach {url}: {e.reason}") from e


# ──────────────────────────────────────────────────────────────
# HuggingFace Hub
# ──────────────────────────────────────────────────────────────


class HfBackend:
    kind = BackendKind.HF

    def available(self) -> bool:
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            return False
        return True

    def fetch(self, mirror: Mirror, remote: str, out_path: Path) -> None:
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import (
                EntryNotFoundError,
                RepositoryNotFoundError,
            )
        except ImportError as e:
            raise BackendError(
                "huggingface_hub not installed; run `uv sync --extra data`"
            ) from e
        _ensure_parent(out_path)
        try:
            cached = hf_hub_download(
                repo_id=mirror.base,
                filename=remote,
                repo_type=mirror.repo_type,
            )
        except RepositoryNotFoundError as e:
            raise BackendError(
                f"HF repo '{mirror.base}' not found (not published yet?)"
            ) from e
        except EntryNotFoundError as e:
            raise BackendError(
                f"'{remote}' not found in HF repo '{mirror.base}'"
            ) from e
        shutil.copyfile(cached, out_path)

    def warm(self, mirror: Mirror) -> None:
        """Pull a whole HF repo into the local cache (runtime-loaded models)."""
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import RepositoryNotFoundError
        except ImportError as e:
            raise BackendError(
                "huggingface_hub not installed; run `uv sync --extra data`"
            ) from e
        try:
            snapshot_download(repo_id=mirror.base, repo_type=mirror.repo_type)
        except RepositoryNotFoundError as e:
            raise BackendError(f"HF repo '{mirror.base}' not found") from e


# ──────────────────────────────────────────────────────────────
# Google Cloud Storage (via gcloud / gsutil)
# ──────────────────────────────────────────────────────────────


class GcsBackend:
    kind = BackendKind.GCS

    def __init__(self) -> None:
        self._cli = shutil.which("gcloud") or shutil.which("gsutil")

    def available(self) -> bool:
        return self._cli is not None

    def fetch(self, mirror: Mirror, remote: str, out_path: Path) -> None:
        if self._cli is None:
            raise BackendError(
                "neither `gcloud` nor `gsutil` found on PATH; "
                "install the Google Cloud SDK or use `--from hf`"
            )
        src = f"{mirror.base}{remote}"
        _ensure_parent(out_path)
        if self._cli.endswith("gcloud"):
            cmd = [self._cli, "storage", "cp", src, str(out_path)]
        else:
            cmd = [self._cli, "cp", src, str(out_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise BackendError(
                f"`{' '.join(cmd)}` failed: {proc.stderr.strip() or proc.stdout.strip()}"
            )


# ──────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────

_BACKENDS: dict[BackendKind, Backend] = {
    BackendKind.HTTP: HttpBackend(),
    BackendKind.HF: HfBackend(),
    BackendKind.GCS: GcsBackend(),
}


def get_backend(kind: BackendKind) -> Backend:
    return _BACKENDS[kind]
