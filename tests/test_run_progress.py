"""Run-banner hardware probe + progress/ETA formatting.

All pure-logic: the torch-dependent path is exercised via a simulated
ImportError, so these pass with or without torch / a GPU.
"""

from __future__ import annotations

import sys

from yeastbench.cli import _fmt_dur, _progress_line
from yeastbench.hardware import DeviceInfo, describe_device


# ── hardware probe ────────────────────────────────────────────


def test_cpu_device_has_no_gpu_fields():
    info = describe_device("cpu")
    assert info.device == "cpu"
    assert info.name is None and info.note is None
    assert info.lines()[0].startswith("hardware:")
    assert "cpu" in info.lines()[0]


def test_cuda_probe_never_raises():
    # Works whether or not torch/a GPU is present: must always yield a banner.
    info = describe_device("cuda")
    assert info.device == "cuda"
    assert info.lines()  # non-empty
    assert info.name is not None or info.note is not None


def test_missing_torch_is_a_note_not_a_crash(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` → ImportError
    info = describe_device("cuda:1")
    assert info.note == "torch not installed"
    assert info.name is None


def test_visible_devices_captured(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    info = describe_device("cpu")
    assert info.visible_devices == "2,3"
    assert any("CUDA_VISIBLE_DEVICES=2,3" in ln for ln in info.lines())


def test_lines_format_with_full_gpu_info():
    info = DeviceInfo(
        device="cuda:0",
        name="NVIDIA A100-80GB",
        mem_free=79 * 1024**3,
        mem_total=80 * 1024**3,
        visible_devices="2",
    )
    lines = info.lines()
    assert lines[0] == "hardware:      cuda:0  NVIDIA A100-80GB  (79.0/80.0 GB free)"
    assert lines[1].strip() == "CUDA_VISIBLE_DEVICES=2"


# ── duration + progress formatting ────────────────────────────


def test_fmt_dur():
    assert _fmt_dur(41) == "41s"
    assert _fmt_dur(118) == "1m58s"
    assert _fmt_dur(1024) == "17m04s"
    assert _fmt_dur(3700) == "1h01m"


def test_progress_line_shows_eta_until_last_pair():
    mid = _progress_line(3, 12, last_dt=84, durations=[41, 118, 84], elapsed=252)
    assert "done in 1m24s" in mid
    assert "mean " in mid and "ETA ~" in mid

    last = _progress_line(12, 12, last_dt=10, durations=[10] * 12, elapsed=1024)
    assert "done in 10s" in last
    assert "ETA" not in last  # no estimate once everything's done
