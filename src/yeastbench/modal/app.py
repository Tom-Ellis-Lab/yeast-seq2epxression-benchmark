"""Modal GPU compute backend — the engine behind ``ybench modal``.

Lift-and-shift: a CPU ``seed`` function fetches data/weights into a Volume, then
a GPU ``run_benchmark`` function runs the *unmodified* ``ybench`` CLI against
that Volume and returns the results tree. The eval/scoring code is never
reimplemented — the same bytes run remotely. See ``docs/modal_backend.md``.

Importing this module is cheap and offline: Modal Image/Volume/Function objects
are lazy specs; nothing builds or connects until the CLI enters ``app.run()``.

NOTE: the GPU image recipe (CUDA 13 + cu130 torch + the pinned flash-attn wheel)
is the one piece that can only be validated on a real Modal GPU — that's Phase 0
of docs/modal_backend.md (``scripts/modal/spike.py``). If it drifts, the
documented fallback is a GPU image without the ``yorzoi`` extra.
"""
from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import modal

from yeastbench.data.fetch import default_data_root

APP_NAME = "ybench"
GPU_DEFAULT = "A10"
CUDA_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
TORCH_CU130_INDEX = "https://download.pytorch.org/whl/cu130"
# Mirrors pyproject [tool.uv.sources].flash-attn; the constant is only a fallback
# for when pyproject.toml isn't on disk (e.g. an installed-wheel layout).
FLASH_ATTN_WHEEL_FALLBACK = (
    "https://github.com/adithyaxx/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3%2Bcu13torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)

_REPO_ROOT = default_data_root()
_IMAGE_IGNORE = ["data/**", "results/**", "**/__pycache__", ".venv/**"]
# .git is intentionally NOT ignored: `ybench run` records git_commit in
# run_metadata.json, so the checkout needs to be baked into the image.


def _flash_attn_wheel() -> str:
    pyproject = _REPO_ROOT / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text())
        return data["tool"]["uv"]["sources"]["flash-attn"]["url"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return FLASH_ATTN_WHEEL_FALLBACK


app = modal.App(APP_NAME)

# ── Volumes (two, per docs/modal_backend.md) ────────────────────────────────
# The HF cache rides inside the data volume (HF_HOME=/repo/data/.hf), so there's
# no separate cache volume: one "inputs" volume, one "results" volume.
data_vol = modal.Volume.from_name("ybench-data", create_if_missing=True)
results_vol = modal.Volume.from_name("ybench-results", create_if_missing=True)
VOLUMES = {"/repo/data": data_vol, "/repo/results": results_vol}
HF_ENV = {"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/repo/data/.hf"}

# ── Images ──────────────────────────────────────────────────────────────────
# Seed runs on a torch-free CPU container: just the data backend.
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .add_local_dir(str(_REPO_ROOT), "/repo", copy=True, ignore=_IMAGE_IGNORE)
    .uv_pip_install("/repo[data]", "hf_transfer")
    .env(HF_ENV)
)

# GPU run: CUDA 13 base so the cu13 flash-attn wheel can load, cp312, cu130 torch.
gpu_image = (
    modal.Image.from_registry(CUDA_BASE, add_python="3.12")
    .entrypoint([])  # clear the base image's entrypoint
    .apt_install("git")
    # So `git rev-parse` works on the baked /repo checkout (run by `ybench run`
    # to record git_commit in run_metadata.json) despite root/dubious-ownership.
    .run_commands("git config --global --add safe.directory /repo")
    # Pin torch to the cu130 build first so the wheel's ABI matches (the PyPI
    # default torch is cu12).
    .uv_pip_install("torch==2.11.*", index_url=TORCH_CU130_INDEX)
    .uv_pip_install(_flash_attn_wheel())
    .add_local_dir(str(_REPO_ROOT), "/repo", copy=True, ignore=_IMAGE_IGNORE)
    # Install the package + model deps WITHOUT the yorzoi extra's bare
    # `flash-attn` requirement (already satisfied by the wheel above) and pull
    # yorzoi explicitly. transformers + CodonTransformer are codon_transformer's
    # currently-undeclared runtime deps; the image papers over that gap.
    .uv_pip_install(
        "/repo[shorkie,dream_rnn,data]",
        "yorzoi==0.2.1",
        "hf_transfer",
        "transformers",
        "git+https://github.com/Adibvafa/CodonTransformer",
    )
    .env(HF_ENV)
)


def _write_config(config_bytes: bytes, config_name: str) -> str:
    """Write the user's config bytes into the container's configs/ dir under its
    original name. Preserves config_hash (same bytes) and the config basename;
    config_path in run_metadata.json is the container path (/repo/configs/<name>),
    which is environment-relative by design and has no effect on scores."""
    rel = f"configs/{config_name}"
    dest = Path("/repo") / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(config_bytes)
    return rel


@app.function(image=cpu_image, volumes=VOLUMES, timeout=2 * 3600)
def seed(
    config_bytes: bytes,
    config_name: str,
    model: str | None = None,
    task: str | None = None,
) -> None:
    """Fetch this config's data + weights into the volumes on a CPU container, so
    the ~1 GB download is never billed at GPU rates. Idempotent: the canonical
    ``ybench data get`` skips checksum-OK files, so re-seeding is near-free.
    ``model``/``task`` mirror the run's filters so a filtered run seeds only what
    it needs."""
    rel = _write_config(config_bytes, config_name)
    cmd = ["ybench", "data", "get", "--config", rel]
    if model:
        cmd += ["--model", model]
    if task:
        cmd += ["--task", task]
    subprocess.run(cmd, cwd="/repo", check=True)
    data_vol.commit()  # persists locked data/ AND the /repo/data/.hf cache


@app.function(image=gpu_image, gpu=GPU_DEFAULT, volumes=VOLUMES, timeout=4 * 3600)
def run_benchmark(
    config_bytes: bytes,
    config_name: str,
    out_dir: str = "results/default",
    model: str | None = None,
    task: str | None = None,
    device: str = "cuda",
) -> list[str]:
    """Run the unmodified ``ybench`` CLI on a GPU against the seeded volume,
    commit the results to the results volume, and return the list of pair dirs
    produced (a small manifest — the bulk results are downloaded by the caller
    via ``modal volume get``, which has no return-value size limit). ``model``/
    ``task`` are passed through so a filtered run runs only the selected pairs."""
    data_vol.reload()  # see what seed committed
    rel = _write_config(config_bytes, config_name)
    cmd = ["ybench", "run", "--config", rel, "--device", device]
    if model:
        cmd += ["--model", model]
    if task:
        cmd += ["--task", task]
    # The real run, including the auto-triggered cross-model compare. Data is
    # already on the volume (seed runs first); `ybench run`'s own preflight
    # reports clearly if something is missing.
    subprocess.run(cmd, cwd="/repo", check=True)
    results_vol.commit()
    produced = Path("/repo") / out_dir
    if not produced.exists():
        return []
    # Real pair dirs are "<model>__<task>"; skip the auto-compare/ output dir.
    return sorted(p.name for p in produced.iterdir() if p.is_dir() and "__" in p.name)
