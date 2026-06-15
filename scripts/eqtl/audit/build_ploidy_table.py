#!/usr/bin/env python3
"""Build a clean per-isolate ploidy / zygosity table for the 1011 yeast panel.

Source: Peter et al. 2018 (Nature, doi:10.1038/s41586-018-0030-5) supplementary
workbook (the bundled .xls of Tables S1-S21). We use:

  * Table S1  - master isolate table: Standardized name, Ploidy, Aneuploidies,
                Zygosity, clade, ecology, etc.
  * Table S16 - per-chromosome copy number for every isolate (local ploidy at
                the variant's chromosome).
  * Table S8  - genome-wide heterozygosity (proportion of heterozygous SNPs).

The "Standardized name" column matches the sample codes in 1011Matrix.gvcf.gz
exactly (verified: 0 of 1011 gVCF samples missing from Table S1), so this table
joins straight onto per-strain genotypes.

Output: a TSV keyed by `strain` (== gVCF sample == standardized name), one row
per isolate, with parsed integer ploidy, per-chromosome copy number, and flags.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

CHROMS = [f"chromosome{i}" for i in range(1, 17)]


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(how="all")


def _parse_ploidy(x) -> float:
    """Table S1 Ploidy is 1..5 or 'Xn' (unknown) or NaN."""
    s = str(x).strip()
    if s in ("", "nan", "Xn", "None"):
        return float("nan")
    try:
        return float(int(float(s)))
    except ValueError:
        return float("nan")


def _parse_chrom_copies(x) -> float:
    """Table S16 per-chromosome cells: '2', '3', or segmental like
    '2[1,200000];3[200000,END]'. Take the leading integer as the dominant
    copy number.

    Table S16 ("Segmental duplications") only fills per-chromosome values for
    strains that carry segmental/aneuploid variation; euploid strains are left
    as 0. We treat 0/unparseable as "not explicitly reported" -> NaN, and let
    the caller fall back to the genome-wide ploidy (Table S1)."""
    s = str(x).strip()
    m = re.match(r"^\s*(\d+)", s)
    if not m:
        return float("nan")
    v = int(m.group(1))
    return float(v) if v > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[3]
    ap.add_argument(
        "--xls",
        type=Path,
        default=repo / "results/caudal_ploidy_audit/peter2018_supp_tables.xls",
        help="Peter 2018 supplementary workbook (bundled Tables S1-S21).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=repo / "results/caudal_ploidy_audit/isolate_ploidy.tsv",
    )
    args = ap.parse_args()

    xl = pd.ExcelFile(args.xls)

    # --- Table S1: master isolate table (header on row index 3) ---
    s1 = _norm_cols(xl.parse("Table S1", header=3))
    s1 = s1[s1["Standardized name"].notna()].copy()
    s1["strain"] = s1["Standardized name"].astype(str).str.strip()
    s1["ploidy"] = s1["Ploidy"].map(_parse_ploidy)
    s1["zygosity"] = s1["Zygosity"].astype(str).str.strip().str.lower()
    aneu_raw = s1["Aneuploidies"].astype(str).str.strip()
    s1["is_aneuploid"] = ~aneu_raw.str.lower().str.startswith("euploid") & (
        aneu_raw.str.lower() != "nan"
    )
    s1["aneuploidies"] = aneu_raw
    out = s1[
        [
            "strain",
            "Isolate name",
            "ploidy",
            "zygosity",
            "is_aneuploid",
            "aneuploidies",
            "Clades",
            "Ecological origins",
        ]
    ].rename(
        columns={
            "Isolate name": "isolate_name",
            "Clades": "clade",
            "Ecological origins": "ecological_origin",
        }
    )

    # --- Table S16: per-chromosome copy number (header on row index 2) ---
    s16 = _norm_cols(xl.parse("Table S16", header=2))
    s16 = s16[s16["Isolate"].notna()].copy()
    s16["strain"] = s16["Isolate"].astype(str).str.strip()
    for ch in CHROMS:
        if ch in s16.columns:
            s16[f"cn_{ch}"] = s16[ch].map(_parse_chrom_copies)
    cn_cols = [f"cn_{ch}" for ch in CHROMS if f"cn_{ch}" in s16.columns]
    out = out.merge(s16[["strain"] + cn_cols], on="strain", how="left")
    # Effective local copy number: explicit S16 value where reported (>0),
    # otherwise the genome-wide ploidy from Table S1.
    for col in cn_cols:
        out[col] = out[col].fillna(out["ploidy"])

    # --- Table S8: genome-wide heterozygosity (header on row index 2) ---
    s8 = _norm_cols(xl.parse("Table S8", header=2))
    s8 = s8[s8["Standardized_name"].notna()].copy()
    s8["strain"] = s8["Standardized_name"].astype(str).str.strip()
    het_col = "proportion of heterozygous SNPs"
    s8["het_snp_proportion"] = pd.to_numeric(s8[het_col], errors="coerce")
    out = out.merge(s8[["strain", "het_snp_proportion"]], on="strain", how="left")

    out = out.drop_duplicates("strain").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)

    print(f"wrote {len(out)} isolates -> {args.out}")
    print("\nploidy distribution:")
    print(out["ploidy"].value_counts(dropna=False).sort_index().to_string())
    print("\nzygosity distribution:")
    print(out["zygosity"].value_counts(dropna=False).to_string())
    print(f"\naneuploid strains: {int(out['is_aneuploid'].sum())}")
    print(f"per-chromosome CN columns: {len(cn_cols)}")
    print(f"het_snp_proportion non-null: {out['het_snp_proportion'].notna().sum()}")


if __name__ == "__main__":
    main()
