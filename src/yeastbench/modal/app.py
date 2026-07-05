"""Modal GPU compute backend — the engine behind ``ybench modal``.

Lift-and-shift: a CPU ``seed`` function fetches data/weights into a Volume, then
a GPU ``run_benchmark`` function runs the *unmodified* ``ybench`` CLI against
that Volume and returns the results tree. The eval/scoring code is never
reimplemented — the same bytes run remotely. See ``docs/modal_backend.md``.

Importing this module is cheap and offline: Modal Image/Volume/Function objects
are lazy specs; nothing builds or connects until the CLI enters ``app.run()``.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import modal

from yeastbench.data.fetch import default_data_root

APP_NAME = "ybench"
GPU_DEFAULT = "A10"

_REPO_ROOT = default_data_root()
_IMAGE_IGNORE = ["data/**", "results/**", "**/__pycache__", ".venv/**"]
# .git is intentionally NOT ignored: `ybench run` records git_commit in
# run_metadata.json, so the checkout needs to be baked into the image.


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
    .uv_pip_install("/repo[data]", extra_options="-e")  # editable, mirrors local `uv sync`
    .uv_pip_install("hf_transfer")
    .env(HF_ENV)
)

# GPU run: a plain image + the model extras. yorzoi 0.2.1 no longer needs
# flash-attn, so torch comes in as an ordinary dependency of the extras (a
# standard CUDA-bundled wheel that runs on Modal's GPUs) — no CUDA base image,
# cu130 pin, or prebuilt wheel required.
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    # So `git rev-parse` works on the baked /repo checkout (run by `ybench run`
    # to record git_commit in run_metadata.json) despite root/dubious-ownership.
    .run_commands("git config --global --add safe.directory /repo")
    .add_local_dir(str(_REPO_ROOT), "/repo", copy=True, ignore=_IMAGE_IGNORE)
    # Install the `[all]`-equivalent extras (shorkie/dream_rnn/data) plus yorzoi.
    # EDITABLE (like the local `uv sync`), so the package lives at /repo/src and
    # `import yeastbench` resolves there: several adapters locate their frozen
    # default data files via `Path(__file__).resolve().parents[3]`, which only
    # equals the repo root in the src/ checkout layout. A non-editable install
    # (site-packages) would make parents[3] wrong — e.g. hong's cassette would
    # resolve to /data/tasks/... and 404. `include_source=False` on the functions
    # keeps this editable install the only copy of the package on the container's
    # path (no auto-mounted /root/yeastbench to shadow it).
    #
    # NOTE: `codon_transformer` is deliberately NOT installed. CodonTransformer
    # pins pandas<3 while the benchmark pins pandas>=3.0.2, so the two cannot
    # coexist in one environment — the same reason `codon_transformer` is absent
    # from the local `[all]` env. So this image mirrors local `[all]`, and
    # `codon_transformer` is unsupported on the Modal backend (see build_plan).
    .uv_pip_install("/repo[shorkie,dream_rnn,data]", extra_options="-e")
    .uv_pip_install("yorzoi==0.2.1", "hf_transfer")
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


@app.function(image=cpu_image, volumes=VOLUMES, timeout=2 * 3600, include_source=False)
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


@app.function(
    image=gpu_image, gpu=GPU_DEFAULT, volumes=VOLUMES, timeout=8 * 3600,
    include_source=False,
)
def run_benchmark(
    config_bytes: bytes,
    config_name: str,
    out_dir: str = "results/default",
    model: str | None = None,
    task: str | None = None,
) -> list[str]:
    """Run the unmodified ``ybench`` CLI on the GPU against the seeded volume,
    commit the results to the results volume, and return the list of pair dirs
    produced (a small manifest — the bulk results are downloaded by the caller
    via ``modal volume get``, which has no return-value size limit). ``model``/
    ``task`` are passed through so a filtered run runs only the selected pairs."""
    data_vol.reload()  # see what seed committed
    rel = _write_config(config_bytes, config_name)
    # Force torch device "cuda": a Modal GPU container exposes exactly one GPU as
    # cuda:0. We intentionally override the config's `device:` field (meant for
    # local multi-GPU boxes, e.g. "cuda:1", which would crash here). The Modal GPU
    # *type* (A10/L4/…) is a separate choice, set via the function's `gpu=`.
    cmd = ["ybench", "run", "--config", rel, "--device", "cuda"]
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
