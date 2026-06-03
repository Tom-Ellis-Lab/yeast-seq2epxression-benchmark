"""Recompute the per-genome log-z normalization constants (mu, sigma) for ExoShorkie.

The distilled students predict in z-space; count space requires the inverse
`count = expm1(z*sigma + mu)` with the genome's (mu, sigma). The authors don't ship
these, but they're reproducible from the public figshare coverage NPZ
(DOI 10.6084/m9.figshare.31075375) using their exact recipe: pool the cropped+binned
forward coverage and the reversed reverse-strand coverage over the exogenous
chromosome key(s), log1p, take global mean/std (`compute_logz_stats_multi`).

Auto-detects the exogenous key(s) as every NPZ key that is not a S. cerevisiae
chromosome (chrI..chrXVI, Mito). For genomes shipped in parts (Human chr7 = 2 NPZ
pairs whose keys differ), all exogenous keys across the supplied NPZs are pooled —
matching the genome-global stat the teachers encode.

Downloads any missing NPZ from figshare by file id into a local cache, then writes
the 12 constants to data/models/exoshorkie/logz_stats.json.

Usage:
    uv run python scripts/exoshorkie/compute_logz_stats.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

CROP_BP = 1024
BIN_BP = 16
CACHE = Path("/home/tds122/exoshorkie-weights/coverage")
MENEU_CACHE = Path("/home/tds122/yeast-meneu/data/tasks/meneu_foreign_dna/_cache")
OUT = Path("data/models/exoshorkie/logz_stats.json")

YEAST_KEYS = {f"chr{r}" for r in
              ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
               "XI", "XII", "XIII", "XIV", "XV", "XVI"]} | {"Mito"}

# genome (our HF dir name) -> list of (fwd_figshare_id, rev_figshare_id, basename)
GENOMES: dict[str, list[tuple[int, int, str]]] = {
    "M_pneumoniae":     [(61042570, 61042576, "Mpneumo")],
    "M_mycoides":       [(61042564, 61042567, "Mmmyco")],
    "Data_storage_chr": [(61042516, 61042519, "dChr")],
    "HPRT1":            [(61042525, 61042528, "HPRT1")],
    "HPRT1R":           [(61042537, 61042534, "HPRT1R")],
    "Human_chr_7":      [(61042543, 61042549, "Human_part1"),
                         (61042558, 61042552, "Human_part2")],
}


def crop_and_bin_cov(cov: np.ndarray) -> np.ndarray:
    x = np.abs(np.asarray(cov, dtype=np.float64))[CROP_BP : cov.shape[0] - CROP_BP]
    L = (x.shape[0] // BIN_BP) * BIN_BP
    return x[:L].reshape(-1, BIN_BP).sum(axis=1)


def fetch(fid: int, name: str) -> Path:
    """Return a local path to NPZ file id `fid`; use Meneu cache or download."""
    cached = MENEU_CACHE / f"{name}_norm.npz"
    if cached.exists():
        return cached
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{name}_norm.npz"
    if not dest.exists():
        url = f"https://ndownloader.figshare.com/files/{fid}"
        print(f"  downloading {name} ({fid}) ...")
        subprocess.run(["curl", "-fsSL", "-o", str(dest), url], check=True)
    return dest


def exogenous_keys(npz) -> list[str]:
    return [k for k in npz.files if k not in YEAST_KEYS]


def main() -> None:
    stats: dict[str, dict] = {}
    for genome, parts in GENOMES.items():
        print(f"[{genome}]")
        pooled: list[np.ndarray] = []
        keys_used: list[str] = []
        for fwd_id, rev_id, base in parts:
            fwd_path = fetch(fwd_id, f"{base}_fwd")
            rev_path = fetch(rev_id, f"{base}_rev")
            fwd_npz, rev_npz = np.load(fwd_path), np.load(rev_path)
            for key in exogenous_keys(fwd_npz):
                fwd, rev = fwd_npz[key], rev_npz[key]
                pooled.append(crop_and_bin_cov(fwd))
                pooled.append(crop_and_bin_cov(rev[::-1]))
                keys_used.append(key)
                print(f"    key {key!r}: {fwd.shape[0]:,} bp")
        ys_log = np.log1p(np.concatenate(pooled))
        mu, sigma = float(ys_log.mean()), float(ys_log.std() or 1e-8)
        stats[genome] = {"mu": mu, "sigma": sigma, "keys": keys_used}
        print(f"    => mu={mu:.4f}  sigma={sigma:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stats, indent=2))
    print(f"\nwrote {OUT}")
    print(json.dumps({g: {"mu": round(s["mu"], 4), "sigma": round(s["sigma"], 4)}
                      for g, s in stats.items()}, indent=2))


if __name__ == "__main__":
    main()
