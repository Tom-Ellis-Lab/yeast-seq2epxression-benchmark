"""Tests for YAML config loading and filtering."""
from __future__ import annotations

from pathlib import Path

import pytest

from yeastbench.config import Config, RunSpec, load_config


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    yaml_content = """\
out_dir: results/test
device: cpu

tasks_config:
  task_a:
    distribution_dir: data/a
  task_b:
    distribution_dir: data/b

runs:
  - model: model_x
    tasks: [task_a, task_b]
    model_config:
      batch_size: 4
  - model: model_y
    tasks: [task_a]
"""
    p = tmp_path / "test.yaml"
    p.write_text(yaml_content)
    return p


class TestLoadConfig:
    def test_basic_fields(self, sample_config):
        cfg = load_config(sample_config)
        assert cfg.out_dir == Path("results/test")
        assert cfg.device == "cpu"
        assert len(cfg.runs) == 2
        assert cfg.source_path == sample_config.resolve()
        assert len(cfg.source_hash) == 12

    def test_tasks_config(self, sample_config):
        cfg = load_config(sample_config)
        assert "task_a" in cfg.tasks_config
        assert cfg.tasks_config["task_a"]["distribution_dir"] == "data/a"

    def test_run_specs(self, sample_config):
        cfg = load_config(sample_config)
        assert cfg.runs[0].model == "model_x"
        assert cfg.runs[0].tasks == ["task_a", "task_b"]
        assert cfg.runs[0].model_config["batch_size"] == 4
        assert cfg.runs[1].model == "model_y"
        assert cfg.runs[1].tasks == ["task_a"]
        assert cfg.runs[1].model_config == {}

    def test_hash_changes_with_content(self, tmp_path):
        p1 = tmp_path / "a.yaml"
        p1.write_text("out_dir: a\nruns: []\n")
        p2 = tmp_path / "b.yaml"
        p2.write_text("out_dir: b\nruns: []\n")
        assert load_config(p1).source_hash != load_config(p2).source_hash


class TestConfigFiltering:
    def test_filter_by_model(self, sample_config):
        cfg = load_config(sample_config).filtered(model="model_x", task=None)
        assert len(cfg.runs) == 1
        assert cfg.runs[0].model == "model_x"
        assert cfg.runs[0].tasks == ["task_a", "task_b"]

    def test_filter_by_task(self, sample_config):
        cfg = load_config(sample_config).filtered(model=None, task="task_b")
        assert len(cfg.runs) == 1  # only model_x has task_b
        assert cfg.runs[0].model == "model_x"
        assert cfg.runs[0].tasks == ["task_b"]

    def test_filter_by_both(self, sample_config):
        cfg = load_config(sample_config).filtered(model="model_x", task="task_a")
        assert len(cfg.runs) == 1
        assert cfg.runs[0].tasks == ["task_a"]

    def test_filter_no_match(self, sample_config):
        cfg = load_config(sample_config).filtered(model="nonexistent", task=None)
        assert len(cfg.runs) == 0

    def test_no_filter(self, sample_config):
        cfg = load_config(sample_config).filtered(model=None, task=None)
        assert len(cfg.runs) == 2

    def test_filter_preserves_metadata(self, sample_config):
        original = load_config(sample_config)
        filtered = original.filtered(model="model_x", task=None)
        assert filtered.out_dir == original.out_dir
        assert filtered.device == original.device
        assert filtered.source_hash == original.source_hash
        assert filtered.tasks_config == original.tasks_config
