"""Tests for CLI output persistence format.

These tests verify that the summary.json and .npy files written by
_run_pair have the expected schema, using the real Yorzoi results as
a regression baseline where available.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

YORZOI_RESULTS = Path("results/default/yorzoi__caudal_eqtl")


@pytest.mark.skipif(
    not YORZOI_RESULTS.exists(),
    reason="Yorzoi results not present (GPU run required)",
)
class TestYorzoiResultsRegression:
    """Regression tests against existing Yorzoi results on disk."""

    def test_summary_json_schema(self):
        summary = json.loads((YORZOI_RESULTS / "summary.json").read_text())
        assert summary["model"] == "yorzoi"
        assert summary["task"] == "caudal_eqtl"
        assert summary["task_version"] == "v1"
        assert isinstance(summary["per_iteration"], list)
        assert len(summary["per_iteration"]) == 4
        assert "auroc_abs_mean" in summary
        assert "auroc_abs_sem" in summary
        assert "auprc_abs_mean" in summary
        assert "auprc_abs_sem" in summary

    def test_per_iteration_schema(self):
        summary = json.loads((YORZOI_RESULTS / "summary.json").read_text())
        for it in summary["per_iteration"]:
            assert "name" in it
            assert "n_pairs" in it
            assert "auroc_signed" in it
            assert "auprc_signed" in it
            assert "auroc_abs" in it
            assert "auprc_abs" in it
            assert "zero_frac" in it
            assert it["n_pairs"] == 1846

    def test_aggregate_metrics_match(self):
        summary = json.loads((YORZOI_RESULTS / "summary.json").read_text())
        aurocs = [it["auroc_abs"] for it in summary["per_iteration"]]
        expected_mean = float(np.mean(aurocs))
        expected_sem = float(np.std(aurocs, ddof=1) / np.sqrt(len(aurocs)))
        assert summary["auroc_abs_mean"] == pytest.approx(expected_mean, abs=1e-10)
        assert summary["auroc_abs_sem"] == pytest.approx(expected_sem, abs=1e-10)

    def test_npy_files_exist_and_match(self):
        for i in range(1, 5):
            scores = np.load(YORZOI_RESULTS / f"negset_{i}_scores.npy")
            labels = np.load(YORZOI_RESULTS / f"negset_{i}_labels.npy")
            assert scores.shape == labels.shape
            assert len(scores) == 2 * 1846  # 1846 pairs
            assert set(labels.tolist()) == {0, 1}

    def test_pairs_tsv_files(self):
        for i in range(1, 5):
            pairs = pd.read_csv(YORZOI_RESULTS / f"negset_{i}_pairs.tsv", sep="\t")
            assert "pair_id" in pairs.columns
            assert "pos_distance_to_tss" in pairs.columns
            assert "neg_distance_to_tss" in pairs.columns
            assert len(pairs) == 1846

    def test_labels_alternate_pos_neg(self):
        for i in range(1, 5):
            labels = np.load(YORZOI_RESULTS / f"negset_{i}_labels.npy")
            expected = np.tile([1, 0], len(labels) // 2)
            np.testing.assert_array_equal(labels, expected)

    def test_run_metadata_schema(self):
        meta = json.loads((YORZOI_RESULTS / "run_metadata.json").read_text())
        assert meta["model"] == "yorzoi"
        assert meta["task"] == "caudal_eqtl"
        assert "config_hash" in meta
        assert "git_commit" in meta
        assert "timestamp_utc" in meta
        assert "elapsed_eval_s" in meta

    def test_plots_exist(self):
        assert (YORZOI_RESULTS / "primary_roc_pr.png").exists()
        assert (YORZOI_RESULTS / "close_only_roc_pr.png").exists()
        assert (YORZOI_RESULTS / "distance_stratified.png").exists()
