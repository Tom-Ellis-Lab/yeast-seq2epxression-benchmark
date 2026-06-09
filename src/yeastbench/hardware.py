"""Describe the compute device for a run banner.

No hard dependency on torch — the base install has none, so every path
degrades gracefully to "just print the device string" rather than raising.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    device: str
    name: str | None = None  # GPU model, if resolvable
    mem_free: int | None = None  # bytes
    mem_total: int | None = None  # bytes
    visible_devices: str | None = None  # CUDA_VISIBLE_DEVICES, if set
    note: str | None = None  # why we couldn't probe (torch missing, no CUDA, …)

    def lines(self) -> list[str]:
        """One or two banner lines: the device + optional CUDA_VISIBLE_DEVICES."""
        head = f"hardware:      {self.device}"
        if self.name:
            head += f"  {self.name}"
            if self.mem_total:
                head += f"  ({_gb(self.mem_free)}/{_gb(self.mem_total)} GB free)"
        elif self.note:
            head += f"  ({self.note})"
        out = [head]
        if self.visible_devices is not None:
            out.append(f"               CUDA_VISIBLE_DEVICES={self.visible_devices}")
        return out


def describe_device(device: str) -> DeviceInfo:
    """Resolve a device string ("cuda", "cuda:2", "cpu", …) to a banner-ready
    description. Probes torch for the GPU name + free/total VRAM when the device
    is CUDA and torch is importable; otherwise records a short note."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    info = DeviceInfo(device=device, visible_devices=visible)
    if not device.startswith("cuda"):
        return info

    try:
        import torch
    except ImportError:
        info.note = "torch not installed"
        return info

    if not torch.cuda.is_available():
        info.note = "CUDA not available"
        return info

    idx = 0
    if ":" in device:
        try:
            idx = int(device.split(":", 1)[1])
        except ValueError:
            idx = 0
    try:
        info.name = torch.cuda.get_device_name(idx)
        free, total = torch.cuda.mem_get_info(idx)
        info.mem_free, info.mem_total = free, total
    except Exception as e:  # bad index / driver hiccup — never kill the run
        info.note = f"cuda:{idx}: {e}"
    return info


def _gb(n: int | None) -> str:
    return "?" if n is None else f"{n / 1024**3:.1f}"
