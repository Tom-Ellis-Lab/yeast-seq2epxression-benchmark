"""Build a per-gene median TPM table from DEE2 S. cerevisiae data.

Samples N runs randomly (seeded) from the PASS set of DEE2's S. cerevisiae
accessions, fetches each run's STAR gene-count matrix via the per-run
CGI endpoint, computes TPM using the per-gene longest-isoform length from
GeneInfo.tsv, and writes a median-TPM table across runs.

Output TSV columns:
    gene_id, median_tpm, n_runs_nonzero

This is intentionally approximate — the gene selection doesn't have to be
perfect, it just needs a reasonable expression ranking to stratify host
genes into low/medium/high tertiles.
"""
from __future__ import annotations

import argparse
import bz2
import io
import logging
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

DEE2_CGI = "https://dee2.io/cgi-bin/request.sh"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "dee2"
DEFAULT_N_RUNS = 50
DEFAULT_SEED = 42


def _pass_accessions(accessions_bz2: Path) -> list[str]:
    with bz2.open(accessions_bz2, "rt") as fh:
        df = pd.read_csv(fh, sep="\t")
    return df.loc[df["QC_summary"] == "PASS", "SRR_accession"].tolist()


def _fetch_run(srr: str, retries: int = 3) -> bytes:
    url = DEE2_CGI
    for attempt in range(retries):
        try:
            r = requests.get(url, params={"org": "scerevisiae", "x": srr}, timeout=60)
            r.raise_for_status()
            if len(r.content) < 10_000:
                raise RuntimeError(f"suspicious payload size {len(r.content)}")
            return r.content
        except Exception as e:
            if attempt + 1 == retries:
                raise
            log.warning("retry %d/%d for %s: %s", attempt + 1, retries, srr, e)
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _parse_zip(payload: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (gene_counts, gene_info) parsed from a DEE2 zip payload."""
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        with z.open("GeneCountMatrix.tsv") as fh:
            counts = pd.read_csv(fh, sep="\t", index_col=0)
        with z.open("GeneInfo.tsv") as fh:
            info = pd.read_csv(fh, sep="\t", index_col=0)
    return counts, info


def _counts_to_tpm(counts: pd.Series, lengths: pd.Series) -> pd.Series:
    """TPM = (count / length) / sum_j(count_j / length_j) * 1e6."""
    common = counts.index.intersection(lengths.index)
    c = counts.loc[common].astype(float)
    l = lengths.loc[common].astype(float)
    rate = c / l
    denom = rate.sum()
    if denom <= 0:
        return pd.Series(0.0, index=common)
    return rate / denom * 1e6


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    accessions = args.data_dir / "scerevisiae_accessions.tsv.bz2"
    out_tsv = args.out or args.data_dir / "scerevisiae_gene_median_tpm.tsv"

    pass_runs = _pass_accessions(accessions)
    log.info("PASS runs available: %d", len(pass_runs))

    rng = random.Random(args.seed)
    sampled = sorted(rng.sample(pass_runs, args.n_runs))
    log.info("Sampled %d runs (seed=%d)", len(sampled), args.seed)

    gene_info: pd.DataFrame | None = None
    tpm_cols: dict[str, pd.Series] = {}
    skipped: list[str] = []
    for i, srr in enumerate(sampled, start=1):
        log.info("[%d/%d] fetching %s", i, len(sampled), srr)
        try:
            payload = _fetch_run(srr)
            counts, info = _parse_zip(payload)
            if counts.empty or srr not in counts.columns:
                raise RuntimeError("empty or missing column in GeneCountMatrix")
        except Exception as e:
            log.warning("skipping %s: %s", srr, e)
            skipped.append(srr)
            continue
        if gene_info is None:
            gene_info = info
        tpm_cols[srr] = _counts_to_tpm(counts[srr], info["longest_isoform"])
        time.sleep(0.2)  # courtesy
    if skipped:
        log.info("Skipped %d runs with bad payloads: %s", len(skipped), skipped)

    tpm_df = pd.DataFrame(tpm_cols)  # (genes × runs)
    log.info("TPM matrix: %s", tpm_df.shape)

    median_tpm = tpm_df.median(axis=1)
    n_nonzero = (tpm_df > 0).sum(axis=1)

    out = pd.DataFrame({
        "gene_id": median_tpm.index,
        "median_tpm": median_tpm.values,
        "n_runs_nonzero": n_nonzero.values,
    }).sort_values("median_tpm", ascending=False).reset_index(drop=True)

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_tsv, sep="\t", index=False)
    log.info("Wrote %d genes → %s", len(out), out_tsv)
    log.info(
        "Median TPM percentiles: p10=%.2f p50=%.2f p90=%.2f max=%.2f",
        out["median_tpm"].quantile(0.1),
        out["median_tpm"].quantile(0.5),
        out["median_tpm"].quantile(0.9),
        out["median_tpm"].max(),
    )


if __name__ == "__main__":
    main()
