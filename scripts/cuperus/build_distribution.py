"""Build the Cuperus 5'-UTR distribution from the vendored GEO tables.

Reads ``archive/cuperus/`` (GEO GSE104252) and writes the committed-layout
distribution under ``data/tasks/cuperus_mpra_5utr/``:

  - ``random_utrs.tsv``  — 489,348 rows: ``UTR, growth_rate, t0, t1,
    depth_bucket (1-5), is_paper_top5``.
  - ``native_utrs.tsv``  — 11,856 rows: ``UTR_name, UTR, growth_rate, t0, t1``.
  - ``manifest.json``    — row counts, the t0 bucket edges, the paper-top5
    cohort size, and sha256 of every input and output file.

Depth buckets stratify the random library by input read depth ``t0`` (see
``docs/benchmarks/cuperus_mpra_5utr.md`` → Read-depth strata). ``is_paper_top5``
marks the exact 24,468-row top-5%-by-``t0`` cohort used for the CNN R2=0.62
comparison, reproduced via the Seeliglab Notebook_1 index split.

Usage:  uv run python scripts/cuperus/build_distribution.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# t0 thresholds; np.digitize(t0, EDGES) + 1 -> bucket in 1..5
#   1: t0<10 (noisy)  2: 10-29  3: 30-59  4: 60-100  5: t0>=101 (clean ~ paper top-5%)
BUCKET_EDGES = [10, 30, 60, 101]


def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "archive" / "cuperus").exists():
            return parent
    raise FileNotFoundError("could not locate repo root (archive/cuperus) from %s" % p)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _depth_bucket(t0: pd.Series) -> np.ndarray:
    return np.digitize(t0.to_numpy(), BUCKET_EDGES) + 1


def _paper_top5_mask(df: pd.DataFrame) -> np.ndarray:
    """Exact top-5%-by-t0 cohort, Seeliglab Notebook_1 index split.
    Ties at the boundary are broken by sort order, giving 24,468 rows."""
    sorted_inds = df.sort_values("t0", kind="stable").index
    top5 = sorted_inds[int(0.95 * len(df)):]
    return df.index.isin(top5)


def build(root: Path) -> None:
    src = root / "archive" / "cuperus"
    out = root / "data" / "tasks" / "cuperus_mpra_5utr"
    out.mkdir(parents=True, exist_ok=True)

    random_csv = src / "GSM2793752_Random_UTRs.csv"
    native_csv = src / "GSM2793754_Native_UTRs.csv"

    # --- random library ---
    rand = pd.read_csv(random_csv, index_col=0)
    assert (rand["UTR"].str.len() == 50).all(), "random UTRs must all be 50 bp"
    rand["depth_bucket"] = _depth_bucket(rand["t0"])
    rand["is_paper_top5"] = _paper_top5_mask(rand)
    rand_out = rand[["UTR", "growth_rate", "t0", "t1", "depth_bucket", "is_paper_top5"]]
    rand_path = out / "random_utrs.tsv"
    rand_out.to_csv(rand_path, sep="\t", index=False)

    bucket_counts = {int(b): int(n) for b, n in rand["depth_bucket"].value_counts().sort_index().items()}
    n_top5 = int(rand["is_paper_top5"].sum())

    # --- native library ---
    nat = pd.read_csv(native_csv, index_col=0)
    nat_out = nat[["UTR_name", "UTR", "growth_rate", "t0", "t1"]]
    nat_path = out / "native_utrs.tsv"
    nat_out.to_csv(nat_path, sep="\t", index=False)

    manifest = {
        "bucket_edges_t0": BUCKET_EDGES,
        "random": {
            "n": int(len(rand)),
            "depth_bucket_counts": bucket_counts,
            "paper_top5_n": n_top5,
        },
        "native": {"n": int(len(nat))},
        "inputs": {p.name: _sha256(p) for p in (random_csv, native_csv)},
        "outputs": {p.name: _sha256(p) for p in (rand_path, nat_path)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"random: {len(rand):,} rows -> {rand_path}")
    print(f"  depth buckets: {bucket_counts}")
    print(f"  paper top-5%: {n_top5:,} rows")
    print(f"native: {len(nat):,} rows -> {nat_path}")
    print(f"manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    build(_find_repo_root())
