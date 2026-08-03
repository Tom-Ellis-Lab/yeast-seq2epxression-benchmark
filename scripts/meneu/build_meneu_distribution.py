#!/usr/bin/env python
"""Build the Meneu foreign-DNA coverage distribution.

Mirrors ``scripts/brooks/build_brooks_distribution.py`` but for whole megabase
contigs instead of gene-centred windows. The *only* component that touches the
network: it downloads the ExoShorkie figshare release (genome FASTAs + per-base
normalized RNA-seq coverage, as ``.npz`` numpy arrays keyed by contig) into a
cache, then writes one window-agnostic sidecar ``.npz`` per genome holding:

  * ``seq`` — the full contig DNA (uint8/ASCII),
  * ``fwd`` / ``rev`` — per-base normalized coverage.

The benchmark tiles each contig to the model's receptive field *at run time*
(``yeastbench.benchmarks.meneu.tile_contig``), so a single artifact serves every
model — there is no longer a per-window TSV. At eval time the benchmark depends
on these built files alone (no figshare).

Run:
    uv run python scripts/meneu/build_meneu_distribution.py

See ``docs/benchmarks/meneu_foreign_dna.md``.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────
OUT_DIR = Path("data/tasks/meneu_foreign_dna")
CACHE = OUT_DIR / "_cache"
FIGSHARE = "https://ndownloader.figshare.com/files"

# ExoShorkie figshare 10.6084/m9.figshare.31075375 (v2). Each genome ships a
# FASTA (single exogenous contig) plus fwd/rev per-base normalized-coverage
# .npz dicts keyed by contig (yeast chroms + the exogenous contig).
GENOMES: dict[str, dict[str, str]] = {
    "Mpneumo": {  # M. pneumoniae M129, ~818 kb, 40% GC — densely transcribed
        "strain": "RSG_Y960",
        "fasta": "61042573",
        "fwd": "61042570",
        "rev": "61042576",
    },
    "Mmmyco": {  # M. mycoides PG1, ~1.22 Mb, 24% GC — near-silent
        "strain": "RSG_Y712",
        "fasta": "61042561",
        "fwd": "61042564",
        "rev": "61042567",
    },
}

# ── Helpers (read_fasta mirrors the Brooks builder) ──────────────────────
def fetch(file_id: str, name: str) -> Path:
    """Download a figshare file once into the cache; return local path."""
    local = CACHE / name
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{FIGSHARE}/{file_id}", local)
    return local


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    cur = None
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                cur = line[1:].split()[0]
                seqs[cur] = []
            elif cur is not None:
                seqs[cur].append(line.strip())
    return {k: "".join(v).upper() for k, v in seqs.items()}


def load_coverage(contig: str, cfg: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-base fwd/rev normalized coverage for the exogenous contig."""
    fwd = np.load(fetch(cfg["fwd"], f"{contig}_fwd_norm.npz"))[contig].astype(np.float32)
    rev = np.load(fetch(cfg["rev"], f"{contig}_rev_norm.npz"))[contig].astype(np.float32)
    return fwd, rev


def build(genomes: list[str]) -> None:
    """Write one window-agnostic ``meneu_cov_<contig>.npz`` per genome holding
    the full contig sequence + per-base fwd/rev coverage. The benchmark tiles
    these at run time to the model's receptive field."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for contig in genomes:
        cfg = GENOMES[contig]
        seq = read_fasta(fetch(cfg["fasta"], f"{contig}.fa"))[contig]
        fwd, rev = load_coverage(contig, cfg)
        assert len(seq) == len(fwd) == len(rev), (len(seq), len(fwd), len(rev))
        cov_path = OUT_DIR / f"meneu_cov_{contig}.npz"
        np.savez_compressed(
            cov_path,
            seq=np.frombuffer(seq.encode("ascii"), dtype=np.uint8),
            fwd=fwd,
            rev=rev,
        )
        print(f"  wrote {cov_path}  (len {len(seq):,}, "
              f"unstranded mean {(fwd + rev).mean():.3g}, "
              f"frac0 {np.mean((fwd + rev) == 0):.3f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genomes", nargs="+", default=list(GENOMES),
                    choices=list(GENOMES), help="Which exogenous contigs to bake.")
    args = ap.parse_args()
    build(args.genomes)


if __name__ == "__main__":
    main()
