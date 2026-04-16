"""Tests for the model/task registry."""
from __future__ import annotations

from pathlib import Path

from yeastbench.adapters.protocols import VariantEffectScorer
from yeastbench.benchmarks.base import Benchmark
from yeastbench.benchmarks.eqtl import EQTLClassificationBenchmark
from yeastbench.registry import MODELS, TASKS


class TestRegistry:
    def test_caudal_eqtl_registered(self):
        assert "caudal_eqtl" in TASKS

    def test_shorkie_registered(self):
        assert "shorkie" in MODELS

    def test_yorzoi_registered(self):
        assert "yorzoi" in MODELS

    def test_caudal_factory_produces_benchmark(self, synthetic_distribution):
        task = TASKS["caudal_eqtl"](distribution_dir=synthetic_distribution)
        assert isinstance(task, Benchmark)
        assert isinstance(task, EQTLClassificationBenchmark)

    def test_caudal_adapter_protocol(self, synthetic_distribution):
        task = TASKS["caudal_eqtl"](distribution_dir=synthetic_distribution)
        assert task.adapter_protocol is VariantEffectScorer
