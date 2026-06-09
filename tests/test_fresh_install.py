"""Fresh-install acceptance test.

The release gate before cutting v1 (dev → main): a fresh checkout with **no
local data** must be able to pull everything and run the full benchmark matrix.

This is GPU-heavy (shorkie + yorzoi over every task) and needs the published
mirror reachable, so it does NOT run in normal CI. Enable it explicitly on a
GPU box once the mirror is published:

    YBENCH_FRESH_INSTALL=1 uv run pytest -m integration tests/test_fresh_install.py

The data-only half (`get` → `verify`) runs without a GPU; the model-run half is
additionally marked ``gpu``. Run on a fresh checkout (empty ``data/``) for a
true fresh-install test — ``get`` is idempotent, so on a box that already has
data it just re-verifies.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from yeastbench.config import load_config

REPO = Path(__file__).resolve().parents[1]
CONFIGS = ["configs/default.yaml", "configs/brooks.yaml", "configs/meneu.yaml"]

requires_fresh = pytest.mark.skipif(
    os.environ.get("YBENCH_FRESH_INSTALL") != "1",
    reason=(
        "fresh-install acceptance test — set YBENCH_FRESH_INSTALL=1 on a GPU "
        "box with the published mirror reachable to run it"
    ),
)


def _ybench(*args: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "yeastbench.cli", *args],
        cwd=REPO,
        check=True,
    )


@pytest.mark.integration
@requires_fresh
def test_pull_all_data_and_verify() -> None:
    """Pull every artifact from the mirror and checksum it against the lock."""
    _ybench("data", "get")
    _ybench("data", "verify")


@pytest.mark.integration
@pytest.mark.gpu
@requires_fresh
@pytest.mark.parametrize("config", CONFIGS)
def test_run_full_matrix(config: str) -> None:
    """Run every (model, task) pair in ``config`` and confirm each produced a
    summary. Assumes the data is already present (run after the pull test)."""
    _ybench("data", "get", "--config", config)
    _ybench("run", "--config", config)

    cfg = load_config(REPO / config)
    out_dir = REPO / cfg.out_dir  # cfg.out_dir is repo-relative
    missing = [
        f"{r.model}__{t}"
        for r in cfg.runs
        for t in r.tasks
        if not (out_dir / f"{r.model}__{t}" / "summary.json").exists()
    ]
    assert not missing, f"no summary.json for: {missing}"
