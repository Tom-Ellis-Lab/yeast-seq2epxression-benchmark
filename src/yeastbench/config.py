"""YAML run-spec loader.

A config file describes one or more ``(model, task)`` runs plus their
per-run settings. See ``configs/default.yaml`` for the canonical schema.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunSpec:
    model: str
    tasks: list[str]
    model_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    out_dir: Path
    device: str
    tasks_config: dict[str, dict[str, Any]]
    runs: list[RunSpec]
    source_path: Path
    source_hash: str

    def filtered(self, model: str | None, task: str | None) -> "Config":
        """Return a new Config whose runs are filtered by model and/or task name."""
        out_runs: list[RunSpec] = []
        for r in self.runs:
            if model and r.model != model:
                continue
            kept_tasks = [t for t in r.tasks if task is None or t == task]
            if not kept_tasks:
                continue
            out_runs.append(RunSpec(model=r.model, tasks=kept_tasks, model_config=r.model_config))
        return Config(
            out_dir=self.out_dir,
            device=self.device,
            tasks_config=self.tasks_config,
            runs=out_runs,
            source_path=self.source_path,
            source_hash=self.source_hash,
        )


def load_config(path: str | Path) -> Config:
    p = Path(path)
    raw_bytes = p.read_bytes()
    data = yaml.safe_load(raw_bytes)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {path}")

    runs_raw = data.get("runs") or []
    runs = [
        RunSpec(
            model=r["model"],
            tasks=list(r["tasks"]),
            model_config=dict(r.get("model_config") or {}),
        )
        for r in runs_raw
    ]

    return Config(
        out_dir=Path(data.get("out_dir", "results/default")),
        device=str(data.get("device", "cuda")),
        tasks_config={k: dict(v or {}) for k, v in (data.get("tasks_config") or {}).items()},
        runs=runs,
        source_path=p.resolve(),
        source_hash=hashlib.sha256(raw_bytes).hexdigest()[:12],
    )
