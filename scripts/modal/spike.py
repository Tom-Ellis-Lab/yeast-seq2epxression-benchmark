"""Phase 0 spike for the Modal GPU backend (docs/modal_backend.md).

Run this once, BEFORE relying on the full integration, to prove the GPU image
actually works on a real Modal GPU:

    uv run modal run scripts/modal/spike.py

It builds the same image as the backend (``yeastbench.modal.app.gpu_image``) and,
on an A10, imports torch / flash_attn / yorzoi and prints the CUDA wiring. This
is the gate for the whole "run yorzoi remotely" promise — if it fails, rethink
the GPU image (documented fallback: drop the ``yorzoi`` extra and mark yorzoi
unsupported). Needs a Modal account: ``uv pip install modal && modal setup``.
"""
from __future__ import annotations

import modal

from yeastbench.modal.app import gpu_image

app = modal.App("ybench-spike")


@app.function(image=gpu_image, gpu="A10", timeout=600)
def check() -> None:
    import torch

    print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))

    import flash_attn

    print("flash_attn:", flash_attn.__version__)

    import yorzoi  # transitively imports flash_attn — the real ABI check

    print("yorzoi import OK:", getattr(yorzoi, "__version__", "ok"))


@app.local_entrypoint()
def main() -> None:
    check.remote()
