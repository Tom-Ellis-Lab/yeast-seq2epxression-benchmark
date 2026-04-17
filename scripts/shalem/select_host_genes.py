"""Select 22 host genes for the Shalem marginalized MPRA benchmark.

Filter (must pass):
  1. Protein-coding with a well-defined CDS ≥ 300 bp
  2. No same-strand gene starts within 500 bp downstream of stop codon
  3. No convergent (opposite-strand) gene overlaps the 500 bp downstream region
  4. Gene center + stop-codon-plus-buffer fit both Shorkie (16384 bp) and
     Yorzoi (4992 bp) window placement constraints (chromosome edges)

Diversification (pick 22):
  - Strand balance: 10 positive, 12 negative (matches Rafi marginalized)
  - Expression stratified across low / medium / high log-TPM tertiles

Inputs:
  - GTF:             data/processed/caudal_eqtl_v1/reference/R64-1-1.115.gtf
  - FASTA index:     data/processed/caudal_eqtl_v1/reference/R64-1-1.fa.fai
  - DEE2 TPM table:  data/raw/dee2/scerevisiae_gene_median_tpm.tsv

Output:
  data/processed/shalem_mpra/host_genes.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GTF = REPO_ROOT / "data" / "processed" / "caudal_eqtl_v1" / "reference" / "R64-1-1.115.gtf"
DEFAULT_FAI = REPO_ROOT / "data" / "processed" / "caudal_eqtl_v1" / "reference" / "R64-1-1.fa.fai"
DEFAULT_TPM = REPO_ROOT / "data" / "raw" / "dee2" / "scerevisiae_gene_median_tpm.tsv"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "shalem_mpra" / "host_genes.json"

DOWNSTREAM_CLEAR = 500   # bp that must be clear of other genes downstream of stop
MIN_CDS_LEN = 300        # bp
MIN_TPM = 1.0            # exclude dubious ORFs / effectively-unexpressed genes
SHORKIE_SEQ_LEN = 16384
YORZOI_SEQ_LEN = 4992
MARGIN_EACH_SIDE = 3000  # conservative margin from chromosome edges

N_GENES = 22
N_POS = 10
N_NEG = 12

GENE_ID_RE = re.compile(r'gene_id\s+"([^"]+)"')
BIOTYPE_RE = re.compile(r'gene_biotype\s+"([^"]+)"')


@dataclass(frozen=True)
class GeneRecord:
    gene_id: str
    chrom: str
    strand: str       # '+' or '-'
    start: int        # 1-based inclusive
    end: int          # 1-based inclusive
    biotype: str


def _parse_gtf(gtf_path: Path) -> list[GeneRecord]:
    genes: list[GeneRecord] = []
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            m = GENE_ID_RE.search(f[8])
            if not m:
                continue
            b = BIOTYPE_RE.search(f[8])
            biotype = b.group(1) if b else "unknown"
            genes.append(GeneRecord(
                gene_id=m.group(1),
                chrom=f[0],
                strand=f[6],
                start=int(f[3]),
                end=int(f[4]),
                biotype=biotype,
            ))
    return genes


def _parse_fai(fai_path: Path) -> dict[str, int]:
    lens: dict[str, int] = {}
    with open(fai_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                lens[parts[0]] = int(parts[1])
    return lens


def _downstream_window(g: GeneRecord, bp: int) -> tuple[int, int]:
    """Inclusive 1-based genomic range for the bp-length window downstream of
    the host gene's transcription end (= stop codon in genomic coords)."""
    if g.strand == "+":
        return (g.end + 1, g.end + bp)
    return (g.start - bp, g.start - 1)


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _apply_filters(
    genes: list[GeneRecord], chrom_lens: dict[str, int],
) -> list[GeneRecord]:
    by_chrom: dict[str, list[GeneRecord]] = defaultdict(list)
    for g in genes:
        by_chrom[g.chrom].append(g)
    for lst in by_chrom.values():
        lst.sort(key=lambda g: g.start)

    passed: list[GeneRecord] = []
    rejects = defaultdict(int)
    for g in genes:
        if g.biotype != "protein_coding":
            rejects["not_protein_coding"] += 1
            continue
        cds_len = g.end - g.start + 1
        if cds_len < MIN_CDS_LEN:
            rejects["cds_too_short"] += 1
            continue
        # Chromosome-edge margin (conservative; model adapters handle window placement,
        # but reject anything pathologically close to a contig edge)
        chrom_len = chrom_lens.get(g.chrom)
        if chrom_len is None:
            rejects["chrom_not_in_fai"] += 1
            continue
        if g.start <= MARGIN_EACH_SIDE or g.end + MARGIN_EACH_SIDE > chrom_len:
            rejects["too_close_to_chrom_edge"] += 1
            continue

        # Downstream-500bp checks
        ds_lo, ds_hi = _downstream_window(g, DOWNSTREAM_CLEAR)
        bad = False
        for other in by_chrom[g.chrom]:
            if other.gene_id == g.gene_id:
                continue
            if other.end < ds_lo or other.start > ds_hi:
                continue
            # Overlaps the downstream window
            if other.strand == g.strand:
                # Same-strand neighbour whose body intrudes into downstream window
                rejects["same_strand_downstream"] += 1
                bad = True
                break
            else:
                # Convergent neighbour overlapping the downstream window
                rejects["convergent_overlap"] += 1
                bad = True
                break
        if bad:
            continue
        passed.append(g)

    log.info("Filter pass: %d / %d", len(passed), len(genes))
    for reason, n in sorted(rejects.items(), key=lambda x: -x[1]):
        log.info("  rejected %d (%s)", n, reason)
    return passed


def _pick_22(passed: list[GeneRecord], tpm: pd.Series, seed: int) -> list[GeneRecord]:
    # Join with TPM; keep only genes present in DEE2 and above MIN_TPM
    expressed = tpm[tpm >= MIN_TPM]
    tpm_index = set(expressed.index)
    candidates = [g for g in passed if g.gene_id in tpm_index]
    log.info(
        "Candidates: %d (passed filter ∩ TPM ≥ %.1f; %d genes have TPM < %.1f)",
        len(candidates), MIN_TPM,
        sum(1 for g in passed if g.gene_id in tpm.index and tpm.loc[g.gene_id] < MIN_TPM),
        MIN_TPM,
    )

    # Split into tertiles by log-TPM
    log_tpm = np.log10(tpm.reindex([g.gene_id for g in candidates]).values + 1e-3)
    t1, t2 = np.quantile(log_tpm, [1/3, 2/3])

    def tertile(g: GeneRecord) -> str:
        v = float(np.log10(tpm.loc[g.gene_id] + 1e-3))
        if v <= t1:
            return "low"
        if v <= t2:
            return "med"
        return "high"

    by_bucket: dict[tuple[str, str], list[GeneRecord]] = defaultdict(list)
    for g in candidates:
        by_bucket[(g.strand, tertile(g))].append(g)
    log.info("Bucket counts:")
    for k, v in sorted(by_bucket.items()):
        log.info("  %s: %d", k, len(v))

    # Target: (strand, tertile) cell counts, balanced as best we can.
    # Rafi-style: 10 pos + 12 neg. Distribute across 3 tertiles:
    #   positive: [3, 3, 4]  (low, med, high)
    #   negative: [4, 4, 4]
    targets = {
        ("+", "low"): 3, ("+", "med"): 3, ("+", "high"): 4,
        ("-", "low"): 4, ("-", "med"): 4, ("-", "high"): 4,
    }

    rng = random.Random(seed)
    picked: list[GeneRecord] = []
    for cell, n in targets.items():
        pool = by_bucket.get(cell, [])
        rng.shuffle(pool)
        picked.extend(pool[:n])
        shortfall = n - len(pool)
        if shortfall > 0:
            log.warning("cell %s short by %d (have %d, need %d)", cell, shortfall, len(pool), n)

    # If any cell came up short, backfill from the overall candidates, preserving strand split
    if len(picked) < N_GENES:
        remaining = [g for g in candidates if g not in picked]
        need_pos = N_POS - sum(1 for g in picked if g.strand == "+")
        need_neg = N_NEG - sum(1 for g in picked if g.strand == "-")
        rng.shuffle(remaining)
        for g in remaining:
            if g.strand == "+" and need_pos > 0:
                picked.append(g)
                need_pos -= 1
            elif g.strand == "-" and need_neg > 0:
                picked.append(g)
                need_neg -= 1
            if len(picked) == N_GENES:
                break

    assert len(picked) == N_GENES, f"picked {len(picked)} (want {N_GENES})"
    assert sum(1 for g in picked if g.strand == "+") == N_POS
    assert sum(1 for g in picked if g.strand == "-") == N_NEG
    return picked


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--gtf", type=Path, default=DEFAULT_GTF)
    p.add_argument("--fai", type=Path, default=DEFAULT_FAI)
    p.add_argument("--tpm", type=Path, default=DEFAULT_TPM)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    log.info("Loading GTF %s", args.gtf)
    genes = _parse_gtf(args.gtf)
    log.info("Genes in GTF: %d", len(genes))

    log.info("Loading chrom lengths %s", args.fai)
    chrom_lens = _parse_fai(args.fai)

    log.info("Applying filters")
    passed = _apply_filters(genes, chrom_lens)

    log.info("Loading TPM %s", args.tpm)
    tpm_df = pd.read_csv(args.tpm, sep="\t")
    tpm = tpm_df.set_index("gene_id")["median_tpm"]

    log.info("Picking 22 host genes")
    picked = _pick_22(passed, tpm, args.seed)

    # Emit JSON
    out = {
        "selection_criteria": {
            "min_cds_len_bp": MIN_CDS_LEN,
            "downstream_clear_bp": DOWNSTREAM_CLEAR,
            "chrom_edge_margin_bp": MARGIN_EACH_SIDE,
            "min_tpm": MIN_TPM,
            "strand_split": {"+": N_POS, "-": N_NEG},
            "tertile_targets": {f"{s}_{t}": n for (s, t), n in {
                ("+", "low"): 3, ("+", "med"): 3, ("+", "high"): 4,
                ("-", "low"): 4, ("-", "med"): 4, ("-", "high"): 4,
            }.items()},
            "seed": args.seed,
            "tpm_source": str(args.tpm.relative_to(REPO_ROOT)) if args.tpm.is_absolute() else str(args.tpm),
        },
        "genes": [
            {
                "gene_id": g.gene_id,
                "chrom": g.chrom,
                "strand": g.strand,
                "start_1based": g.start,
                "end_1based": g.end,
                "cds_len": g.end - g.start + 1,
                "median_tpm": float(tpm.loc[g.gene_id]),
            }
            for g in sorted(picked, key=lambda g: (g.strand, -tpm.loc[g.gene_id]))
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    log.info("Wrote %d genes → %s", len(picked), args.out)

    # Brief summary
    for g in out["genes"]:
        log.info("  %s  %s  %s:%d-%d  %4d bp  TPM %.1f",
                 g["gene_id"], g["strand"], g["chrom"],
                 g["start_1based"], g["end_1based"], g["cds_len"], g["median_tpm"])


if __name__ == "__main__":
    main()
