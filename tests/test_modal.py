"""Unit tests for the Modal backend's pure layer + CLI wiring.

The pure helpers (``yeastbench.modal.plan``) need no ``modal`` install and run in
CI. Anything that imports the Modal client is guarded with ``importorskip`` and
makes no remote calls — the actual remote dispatch is exercised manually behind
``modal setup`` (see docs/modal_backend.md), not here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from yeastbench.modal.plan import build_plan

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO / "configs" / "default.yaml"


# ── plan.build_plan ──────────────────────────────────────────────────────────


def test_build_plan_default_config():
    # filtered to a supported model (unfiltered default.yaml includes the
    # unsupported codon_transformer — see test_build_plan_rejects_codon_transformer)
    plan = build_plan(DEFAULT_CFG, model="shorkie")
    assert plan.config_name == "default.yaml"
    assert plan.config_bytes == DEFAULT_CFG.read_bytes()  # exact bytes → hash parity
    assert plan.out_dir == "results/default"
    assert plan.pairs  # non-empty
    assert all("__" in d for d in plan.pair_dirs())


def test_build_plan_filters_to_single_pair():
    plan = build_plan(DEFAULT_CFG, model="cai", task="chen_synonymous")
    assert plan.pairs == [("cai", "chen_synonymous")]
    assert plan.pair_dirs() == ["cai__chen_synonymous"]


def test_build_plan_no_match_raises():
    with pytest.raises(ValueError, match="No runs match"):
        build_plan(DEFAULT_CFG, model="does-not-exist")


def test_build_plan_rejects_codon_transformer():
    # default.yaml includes codon_transformer on chen_synonymous → must fail fast
    with pytest.raises(ValueError, match="can't run on the Modal backend"):
        build_plan(DEFAULT_CFG, model="codon_transformer")
    # and the supported models still plan fine
    assert build_plan(DEFAULT_CFG, model="shorkie").pairs


def test_build_plan_rejects_out_dir_outside_results(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "out_dir: elsewhere/run\ndevice: cpu\n"
        "runs:\n  - model: cai\n    tasks: [chen_synonymous]\n"
    )
    with pytest.raises(ValueError, match="out_dir under 'results/'"):
        build_plan(cfg)


def test_build_plan_accepts_results_subdir(tmp_path):
    cfg = tmp_path / "ok.yaml"
    cfg.write_text(
        "out_dir: results/custom\ndevice: cpu\n"
        "runs:\n  - model: cai\n    tasks: [chen_synonymous]\n"
    )
    assert build_plan(cfg).out_dir == "results/custom"


# ── CLI wiring + app spec (needs the modal client, but no remote calls) ──────


def test_modal_subapp_registered_on_main_cli():
    pytest.importorskip("modal")
    from typer.testing import CliRunner

    from yeastbench.cli import app

    result = CliRunner().invoke(app, ["modal", "--help"])
    assert result.exit_code == 0
    for cmd in ("run", "pull", "status", "data"):
        assert cmd in result.stdout


def test_modal_app_spec_is_two_volumes():
    pytest.importorskip("modal")
    from yeastbench.modal import app as modal_app

    assert modal_app.APP_NAME == "ybench"
    assert set(modal_app.VOLUMES) == {"/repo/data", "/repo/results"}
    assert modal_app.HF_ENV["HF_HOME"] == "/repo/data/.hf"
    # flash-attn wheel resolves from pyproject (single source of truth)
    assert "flash_attn" in modal_app._flash_attn_wheel()


def test_flash_attn_wheel_falls_back_without_pyproject(tmp_path, monkeypatch):
    pytest.importorskip("modal")
    from yeastbench.modal import app as modal_app

    monkeypatch.setattr(modal_app, "_REPO_ROOT", tmp_path)  # no pyproject.toml here
    assert modal_app._flash_attn_wheel() == modal_app.FLASH_ATTN_WHEEL_FALLBACK
