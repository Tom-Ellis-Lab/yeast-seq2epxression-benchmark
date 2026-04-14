from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from yeastbench.benchmarks.base import Benchmark


def run(benchmark: Benchmark, adapter: Any, out_dir: Path) -> Any:
    expected = benchmark.adapter_protocol
    if not isinstance(adapter, expected):
        raise TypeError(
            f"{type(adapter).__name__} does not implement "
            f"{expected.__name__}, required by {benchmark.info.name}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    results = benchmark.evaluate(adapter)
    benchmark.plot(results, out_dir)
    _write_results_json(results, out_dir / "results.json")
    return results


def _write_results_json(results: Any, path: Path) -> None:
    path.write_text(json.dumps(_to_json(results), indent=2))


def _to_json(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_json(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
