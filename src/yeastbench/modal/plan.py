"""Pure helpers for the Modal backend — no ``modal`` import, so this module is
importable (and unit-testable) in the base install.

Keeps the config-reading / pair-enumeration logic (which needs no Modal client)
out of ``app.py`` (which defines the remote App, images and functions).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from yeastbench.config import load_config

# Models the Modal image can't host. CodonTransformer pins pandas<3 while the
# benchmark pins pandas>=3, so they can't share one environment — which is also
# why `codon_transformer` is absent from the local `[all]` env. Run it in its
# own (pandas<3) environment, not on Modal.
UNSUPPORTED_ON_MODAL = frozenset({"codon_transformer"})


@dataclass(frozen=True)
class RemotePlan:
    config_path: Path
    config_name: str  # basename, e.g. "default.yaml"
    config_bytes: bytes  # exact bytes → preserves config_hash remotely
    out_dir: str  # raw out_dir from the config, e.g. "results/default"
    source_hash: str
    pairs: list[tuple[str, str]]  # (model, task), honoring --model/--task filters

    def pair_dirs(self) -> list[str]:
        return [f"{m}__{t}" for m, t in self.pairs]


def build_plan(
    config_path: str | Path, model: str | None = None, task: str | None = None
) -> RemotePlan:
    """Validate the config locally and describe what the remote run will do.

    Raises the same way ``ybench run`` would (bad path, no matching runs), plus a
    guard that ``out_dir`` lives under ``results/`` — the results volume mounts at
    ``/repo/results``, so anything written elsewhere would be lost on container
    exit. Failing here means no container ever spins up for a doomed run.
    """
    path = Path(config_path)
    cfg = load_config(path).filtered(model, task)
    pairs = [(r.model, t) for r in cfg.runs for t in r.tasks]
    if not pairs:
        raise ValueError(
            f"No runs match filters (model={model!r}, task={task!r}) in {path}"
        )
    bad = sorted({m for m, _ in pairs if m in UNSUPPORTED_ON_MODAL})
    if bad:
        raise ValueError(
            f"Model(s) {bad} can't run on the Modal backend: CodonTransformer "
            "requires pandas<3 but the benchmark requires pandas>=3, so they "
            "can't share one image (the same reason it's not in the local `all` "
            "env). Exclude it — e.g. `--model shorkie` (or `--model yorzoi` / "
            "`cai` / `dream_rnn`), or point at a config without it — and run "
            "codon_transformer in its own pandas<3 environment."
        )
    out_dir = str(cfg.out_dir)
    if Path(out_dir).parts[:1] != ("results",):
        raise ValueError(
            f"Modal backend requires out_dir under 'results/' (got {out_dir!r}); "
            "the results volume mounts at /repo/results, so output written "
            "elsewhere would be lost when the container exits."
        )
    return RemotePlan(
        config_path=path.resolve(),
        config_name=path.name,
        config_bytes=path.read_bytes(),
        out_dir=out_dir,
        source_hash=cfg.source_hash,
        pairs=pairs,
    )
