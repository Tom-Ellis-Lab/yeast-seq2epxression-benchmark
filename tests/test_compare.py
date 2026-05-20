"""Tests for the cross-model comparison runner.

Covered:
- `_discover_results` groups `<model>__<task>/` dirs correctly and
  ignores `compare*` siblings.
- The default `Benchmark.compare_plot` writes a PNG when given two
  fake summary dicts and returns ``None`` for a single model.
- `compare()` no-ops cleanly on a single-model fixture and writes
  outputs (per-task plot, summary.csv, summary.md, overview.png) on
  a two-model fixture.
- Group aliasing via ``Benchmark.compare_task_name`` pulls two
  registry tasks (e.g. ``brooks_scramble`` + ``brooks_scramble_shorkie``)
  into the same comparison group.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from yeastbench.benchmarks.base import (
    Benchmark,
    BenchmarkInfo,
    _default_compare_plot,
)
from yeastbench.compare import (
    CompareSummary,
    _discover_results,
    _group_by_compare_task,
    compare,
)
from yeastbench.config import Config


# ── Fixtures ─────────────────────────────────────────────────────


def _write_summary(dirpath: Path, **metrics: float | int) -> None:
    """Create a `<model>__<task>/summary.json` fixture directory."""
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "summary.json").write_text(json.dumps(metrics, indent=2))


def _config_at(out_dir: Path, source: Path) -> Config:
    """Minimal Config pointed at *out_dir*; bypasses YAML loading."""
    return Config(
        out_dir=out_dir,
        device="cpu",
        tasks_config={},
        runs=[],
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


# ── _default_compare_plot ────────────────────────────────────────


class TestDefaultComparePlot:
    def test_writes_png_with_two_models(self, tmp_path: Path):
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
        assert out.name == "plot.png"

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
        cfg = _config_at(tmp_path, tmp_path / "fake.yaml")
        result = compare(cfg)
        assert isinstance(result, CompareSummary)
        assert result.empty is True
        assert result.tasks_compared == []
        assert result.tasks_skipped == ["task_a"]
        assert result.overview_path is None
        # No compare/ dir was created since nothing was comparable
        assert not (tmp_path / "compare").exists()

    def test_emits_outputs_with_two_models(self, tmp_path: Path):
        _write_summary(tmp_path / "yorzoi__task_a", pearson_r=0.5, dir_acc=0.8)
        _write_summary(tmp_path / "shorkie__task_a", pearson_r=0.3, dir_acc=0.6)
        cfg = _config_at(tmp_path, tmp_path / "fake.yaml")
        result = compare(cfg)
        assert result.tasks_compared == ["task_a"]
        assert result.tasks_skipped == []
        assert result.summary_csv is not None and result.summary_csv.exists()
        assert result.summary_md is not None and result.summary_md.exists()
        assert result.overview_path is not None and result.overview_path.exists()
        assert "task_a" in result.per_task_plots
        assert result.per_task_plots["task_a"].exists()

    def test_summary_csv_long_format(self, tmp_path: Path):
        _write_summary(tmp_path / "yorzoi__t", r=0.5, n_scored=100)
        _write_summary(tmp_path / "shorkie__t", r=0.3, n_scored=80)
        cfg = _config_at(tmp_path, tmp_path / "fake.yaml")
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
        cfg = _config_at(tmp_path, tmp_path / "fake.yaml")
        result = compare(cfg)
        text = result.summary_md.read_text()
        assert "## t" in text
        assert "| metric |" in text
        # Both models in the header row, r row present
        assert "shorkie" in text and "yorzoi" in text
        assert "| r |" in text


# ── compare_task_name grouping ───────────────────────────────────


class TestGroupByCompareTask:
    def test_brooks_aliasing_collapses_two_registry_tasks(self, tmp_path: Path):
        # Real BrooksScrambleBenchmark.compare_task_name → "brooks_scramble"
        # for both `brooks_scramble` (4992) and `brooks_scramble_shorkie`
        # (16384). Validate via the TASKS factories.
        _write_summary(tmp_path / "yorzoi__brooks_scramble", r=0.22)
        _write_summary(
            tmp_path / "shorkie__brooks_scramble_shorkie", r=-0.01,
        )
        by_task = _discover_results(tmp_path)
        assert set(by_task) == {"brooks_scramble", "brooks_scramble_shorkie"}

        # Build a minimal TSV that satisfies the Brooks loader. We only
        # need the benchmark to instantiate so we can read its
        # `compare_task_name`; no `evaluate()` is run.
        import pandas as pd
        from yeastbench.benchmarks.brooks import WINDOW_LEN

        rng = np.random.default_rng(0)
        seq = "".join(rng.choice(list("ACGT"), size=WINDOW_LEN))
        cov = ",".join(map(str, rng.poisson(0.5, WINDOW_LEN).astype(int).tolist()))
        df = pd.DataFrame([{
            "sample_id": "S:G:0", "gene_id": "YIR000W", "strain": "S",
            "copy_idx": 0, "n_copies": 1, "strand": "+",
            "rearr_class": "context_change",
            "syn_contig": "S_1", "cds_start": 100, "cds_end": 1000,
            "window_len": WINDOW_LEN,
            "cds_start_in_window": 2000, "cds_end_in_window": 2700,
            "alt_seq": seq, "native_seq": seq,
            "true_cov_alt": cov, "true_cov_native": cov,
            "strain_reads": 50,
            "js94_reads_runs": "20,30,40",
            "size_factor_strain": 4.0, "norm_cov_strain": 30.0,
            "norm_cov_js94_mean": 10.0,
            "norm_cov_js94_runs": "10.000,10.000,10.000",
            "true_lfc": 1.5, "low_support": False,
        }])
        tsv = tmp_path / "fake_brooks.tsv"
        df.to_csv(tsv, sep="\t", index=False)
        tasks_config = {
            "brooks_scramble": {"data_path": tsv},
            "brooks_scramble_shorkie": {"data_path": tsv},
        }
        groups = _group_by_compare_task(by_task, tasks_config)
        # Both registry tasks fold into one group "brooks_scramble"
        assert set(groups) == {"brooks_scramble"}
        assert set(groups["brooks_scramble"]) == {
            "brooks_scramble", "brooks_scramble_shorkie",
        }

    def test_unknown_task_falls_through_to_registry_name(self, tmp_path: Path):
        _write_summary(tmp_path / "yorzoi__unknown_task", r=0.5)
        _write_summary(tmp_path / "shorkie__unknown_task", r=0.3)
        by_task = _discover_results(tmp_path)
        groups = _group_by_compare_task(by_task, {})
        assert set(groups) == {"unknown_task"}
