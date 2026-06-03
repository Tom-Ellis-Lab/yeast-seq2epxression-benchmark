#!/usr/bin/env python
"""Build the Meneu foreign-DNA coverage distribution.

Mirrors ``scripts/brooks/build_brooks_distribution.py`` but for a tiled
megabase contig instead of gene-centred windows. The *only* component that
touches the network: it downloads the ExoShorkie figshare release (genome
FASTAs + per-base normalized RNA-seq coverage, as ``.npz`` numpy arrays keyed
by contig) into a cache, then bakes:

  * one TSV per receptive-field window size (tile metadata + input sequence),
  * one coverage sidecar ``.npz`` per genome (per-base fwd/rev, window-agnostic).

At eval time the benchmark depends on these built files alone (no figshare).

Run:
    uv run python scripts/meneu/build_meneu_distribution.py --window 4992
    uv run python scripts/meneu/build_meneu_distribution.py --window 16384

See ``benchmarks/meneu_foreign_dna.md``.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

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

# Receptive fields: window -> crop_bp_each_side (Yorzoi 4992/996, Shorkie 16384/1024).
WINDOW_CROP: dict[int, int] = {4992: 996, 16384: 1024}
DEFAULT_WINDOW = 4992


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


def tile_rows(contig: str, strain: str, seq: str, window: int, crop: int) -> list[dict]:
    """Tile the contig so each window's central (predicted) region of length
    ``stride = window - 2*crop`` tiles [0, L) contiguously. Windows overlap
    neighbours by ``crop`` on each side; contig ends are N-padded."""
    L = len(seq)
    stride = window - 2 * crop
    n_tiles = (L + stride - 1) // stride
    rows = []
    for i in range(n_tiles):
        center_start = i * stride
        ws = center_start - crop  # window start in contig coords (may be < 0)
        # N-padded window slice [ws, ws+window)
        left_pad = max(0, -ws)
        right_pad = max(0, (ws + window) - L)
        core = seq[max(0, ws): min(L, ws + window)]
        win = "N" * left_pad + core + "N" * right_pad
        assert len(win) == window, (len(win), window, i)
        rows.append(
            dict(
                tile_id=f"{contig}_{i}",
                chrom=contig,
                strain=strain,
                window_start=ws,          # contig coord of window col 0 (may be negative)
                center_start=center_start,  # contig coord where the predicted region starts
                window_len=window,
                crop_bp_each_side=crop,
                seq=win,
            )
        )
    return rows


def build(window: int, genomes: list[str]) -> Path:
    crop = WINDOW_CROP[window]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for contig in genomes:
        cfg = GENOMES[contig]
        seq = read_fasta(fetch(cfg["fasta"], f"{contig}.fa"))[contig]
        # Coverage sidecar is window-agnostic — write once.
        cov_path = OUT_DIR / f"meneu_cov_{contig}.npz"
        if not cov_path.exists():
            fwd, rev = load_coverage(contig, cfg)
            assert len(seq) == len(fwd) == len(rev), (len(seq), len(fwd), len(rev))
            np.savez_compressed(cov_path, fwd=fwd, rev=rev)
            print(f"  wrote {cov_path}  (len {len(seq):,}, "
                  f"unstranded mean {(fwd + rev).mean():.3g}, frac0 {np.mean((fwd + rev) == 0):.3f})")
        all_rows.extend(tile_rows(contig, cfg["strain"], seq, window, crop))

    df = pd.DataFrame(all_rows)
    out = (OUT_DIR / "meneu_foreign_dna_v1.tsv" if window == DEFAULT_WINDOW
           else OUT_DIR / f"meneu_foreign_dna_v1_w{window}.tsv")
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}  ({len(df)} tiles across {len(genomes)} genomes, window {window})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    choices=sorted(WINDOW_CROP),
                    help=f"Receptive-field window (bp). {DEFAULT_WINDOW} Yorzoi, 16384 Shorkie.")
    ap.add_argument("--genomes", nargs="+", default=list(GENOMES),
                    choices=list(GENOMES), help="Which exogenous contigs to tile.")
    args = ap.parse_args()
    build(args.window, args.genomes)


if __name__ == "__main__":
    main()
