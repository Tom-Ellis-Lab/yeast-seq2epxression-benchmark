"""Tests for the cross-model comparison runner.

Covered:
- `_discover_results` groups `<model>__<task>/` dirs correctly, ignores
  `compare*` siblings, and honours `restrict_to_pairs`.
- The default `Benchmark.compare_plot` writes an SVG when given two
  fake summary dicts and returns ``None`` for a single model.
- `compare()` is scoped to the config's `(model, task)` pairs: it no-ops
  cleanly on a single-model fixture, writes outputs (per-task plot,
  summary.csv, summary.md) on a two-model fixture, and ignores stale
  on-disk results from pairs the config doesn't name.
"""
from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path

import pytest

from yeastbench.benchmarks.base import (
    _default_compare_plot,
)
from yeastbench.compare import (
    CompareSummary,
    _discover_results,
    compare,
)
from yeastbench.config import Config, RunSpec


# ── Fixtures ─────────────────────────────────────────────────────


def _write_summary(dirpath: Path, **metrics: float | int) -> None:
    """Create a `<model>__<task>/summary.json` fixture directory."""
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "summary.json").write_text(json.dumps(metrics, indent=2))


def _runs(*pairs: tuple[str, str]) -> list[RunSpec]:
    """Build `config.runs` from `(model, task)` pairs."""
    by_model: "OrderedDict[str, list[str]]" = OrderedDict()
    for model, task in pairs:
        by_model.setdefault(model, []).append(task)
    return [RunSpec(model=m, tasks=ts) for m, ts in by_model.items()]


def _config_at(
    out_dir: Path, source: Path, runs: list[RunSpec] | None = None,
) -> Config:
    """Minimal Config pointed at *out_dir*; bypasses YAML loading."""
    return Config(
        out_dir=out_dir,
        device="cpu",
        tasks_config={},
        runs=runs or [],
        source_path=source,
        source_hash="test-hash",
    )


# ── _discover_results ────────────────────────────────────────────


class TestDiscoverResults:
    def test_groups_by_task(self, tmp_path: Path):
        _write_summary(tmp_path / "yorzoi__task_a", r=0.5)
        _write_summary(tmp_path / "shorkie__task_a", r=0.3)
        _write_summary(tmp_path / "yorzoi__task_b", r=0.7)
        got = _discover_results(tmp_path)
        assert set(got) == {"task_a", "task_b"}
        assert set(got["task_a"]) == {"yorzoi", "shorkie"}
        assert set(got["task_b"]) == {"yorzoi"}

    def test_skips_dirs_without_summary(self, tmp_path: Path):
        (tmp_path / "yorzoi__task_a").mkdir()   # no summary.json
        _write_summary(tmp_path / "shorkie__task_a", r=0.3)
        got = _discover_results(tmp_path)
        assert got == {"task_a": {"shorkie": tmp_path / "shorkie__task_a"}}

    def test_skips_compare_subtree(self, tmp_path: Path):
        _write_summary(tmp_path / "yorzoi__task_a", r=0.5)
        _write_summary(tmp_path / "shorkie__task_a", r=0.3)
        # Old standalone-script output that should NOT be picked up
        # as a `compare__shared` task.
        _write_summary(tmp_path / "compare__shared", legacy=1.0)
        _write_summary(tmp_path / "compare", aggregate=1.0)
        got = _discover_results(tmp_path)
        assert set(got) == {"task_a"}

    def test_returns_empty_when_dir_missing(self, tmp_path: Path):
        assert _discover_results(tmp_path / "nope") == {}

    def test_restrict_to_pairs_filters(self, tmp_path: Path):
        _write_summary(tmp_path / "yorzoi__task_a", r=0.5)
        _write_summary(tmp_path / "shorkie__task_a", r=0.3)
        _write_summary(tmp_path / "yorzoi__task_b", r=0.7)
        got = _discover_results(
            tmp_path, restrict_to_pairs={("yorzoi", "task_a")},
        )
        assert got == {"task_a": {"yorzoi": tmp_path / "yorzoi__task_a"}}


# ── _default_compare_plot ────────────────────────────────────────


class TestDefaultComparePlot:
    def test_writes_svg_with_two_models(self, tmp_path: Path):
        _write_summary(tmp_path / "yorzoi__t", pearson_r=0.5, dir_acc=0.8)
        _write_summary(tmp_path / "shorkie__t", pearson_r=0.3, dir_acc=0.6)
        out = _default_compare_plot(
            {
                "yorzoi": tmp_path / "yorzoi__t",
                "shorkie": tmp_path / "shorkie__t",
            },
            tmp_path / "out",
        )
        assert out is not None
        assert out.exists()
        assert out.name == "plot.svg"

    def test_returns_none_with_single_model(self, tmp_path: Path):
        _write_summary(tmp_path / "yorzoi__t", pearson_r=0.5)
        out = _default_compare_plot(
            {"yorzoi": tmp_path / "yorzoi__t"}, tmp_path / "out",
        )
        assert out is None

    def test_returns_none_when_no_common_metrics(self, tmp_path: Path):
        _write_summary(tmp_path / "yorzoi__t", metric_a=0.5)
        _write_summary(tmp_path / "shorkie__t", metric_b=0.3)
        out = _default_compare_plot(
            {
                "yorzoi": tmp_path / "yorzoi__t",
                "shorkie": tmp_path / "shorkie__t",
            },
            tmp_path / "out",
        )
        # Intersection of metric keys is empty → nothing to plot
        assert out is None

    def test_skips_non_numeric_and_count_keys(self, tmp_path: Path):
        # n_*-prefixed integer keys are filtered out so they don't swamp
        # the y-axis; strings / nested dicts / NaN are filtered too.
        _write_summary(
            tmp_path / "yorzoi__t",
            n_scored=100, pearson_r=0.5,
            unrelated_string="foo",  # type: ignore[arg-type]
        )
        _write_summary(
            tmp_path / "shorkie__t",
            n_scored=80, pearson_r=0.3,
            unrelated_string="bar",  # type: ignore[arg-type]
        )
        out = _default_compare_plot(
            {
                "yorzoi": tmp_path / "yorzoi__t",
                "shorkie": tmp_path / "shorkie__t",
            },
            tmp_path / "out",
        )
        # pearson_r remains (numeric, not n_-prefixed) → plot is produced
        assert out is not None
        assert out.exists()


# ── compare() top-level ──────────────────────────────────────────


class TestCompareRunner:
    def test_silent_no_op_with_single_model(self, tmp_path: Path):
        _write_summary(tmp_path / "yorzoi__task_a", pearson_r=0.5)
        cfg = _config_at(
            tmp_path, tmp_path / "fake.yaml", runs=_runs(("yorzoi", "task_a")),
        )
        result = compare(cfg)
        assert isinstance(result, CompareSummary)
        assert result.empty is True
        assert result.tasks_compared == []
        assert result.tasks_skipped == ["task_a"]
        # No compare/ dir was created since nothing was comparable
        assert not (tmp_path / "compare").exists()

    def test_emits_outputs_with_two_models(self, tmp_path: Path):
        _write_summary(tmp_path / "yorzoi__task_a", pearson_r=0.5, dir_acc=0.8)
        _write_summary(tmp_path / "shorkie__task_a", pearson_r=0.3, dir_acc=0.6)
        cfg = _config_at(
            tmp_path, tmp_path / "fake.yaml",
            runs=_runs(("yorzoi", "task_a"), ("shorkie", "task_a")),
        )
        result = compare(cfg)
        assert result.tasks_compared == ["task_a"]
        assert result.tasks_skipped == []
        assert result.summary_csv is not None and result.summary_csv.exists()
        assert result.summary_md is not None and result.summary_md.exists()
        assert "task_a" in result.per_task_plots
        assert result.per_task_plots["task_a"].exists()

    def test_ignores_pairs_not_in_config(self, tmp_path: Path):
        # task_a is in the config (both models); stale_task is a leftover on
        # disk from another run and must NOT be compared or surface anywhere.
        _write_summary(tmp_path / "yorzoi__task_a", pearson_r=0.5)
        _write_summary(tmp_path / "shorkie__task_a", pearson_r=0.3)
        _write_summary(tmp_path / "yorzoi__stale_task", pearson_r=0.9)
        _write_summary(tmp_path / "shorkie__stale_task", pearson_r=0.8)
        cfg = _config_at(
            tmp_path, tmp_path / "fake.yaml",
            runs=_runs(("yorzoi", "task_a"), ("shorkie", "task_a")),
        )
        result = compare(cfg)
        assert result.tasks_compared == ["task_a"]
        assert "stale_task" not in result.per_task_plots
        assert "stale_task" not in result.tasks_skipped
        assert "stale_task" not in result.summary_md.read_text()

    def test_summary_csv_long_format(self, tmp_path: Path):
        _write_summary(tmp_path / "yorzoi__t", r=0.5, n_scored=100)
        _write_summary(tmp_path / "shorkie__t", r=0.3, n_scored=80)
        cfg = _config_at(
            tmp_path, tmp_path / "fake.yaml",
            runs=_runs(("yorzoi", "t"), ("shorkie", "t")),
        )
        result = compare(cfg)
        rows = list(csv.DictReader(result.summary_csv.open()))
        assert {r["task"] for r in rows} == {"t"}
        assert {r["model"] for r in rows} == {"yorzoi", "shorkie"}
        assert {r["metric"] for r in rows} == {"r", "n_scored"}
        # Look up Yorzoi r in the long-format dump
        y_r = next(r for r in rows if r["model"] == "yorzoi" and r["metric"] == "r")
        assert float(y_r["value"]) == pytest.approx(0.5)

    def test_summary_md_per_task_section(self, tmp_path: Path):
        _write_summary(tmp_path / "yorzoi__t", r=0.5)
        _write_summary(tmp_path / "shorkie__t", r=0.3)
        cfg = _config_at(
            tmp_path, tmp_path / "fake.yaml",
            runs=_runs(("yorzoi", "t"), ("shorkie", "t")),
        )
        result = compare(cfg)
        text = result.summary_md.read_text()
        assert "## t" in text
        assert "| metric |" in text
        # Both models in the header row, r row present
        assert "shorkie" in text and "yorzoi" in text
        assert "| r |" in text
