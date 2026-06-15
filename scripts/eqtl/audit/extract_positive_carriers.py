#!/usr/bin/env python3
"""Per-strain VAF + ploidy for every Caudal cis-eQTL positive variant.

For each of the 1,901 Caudal positive (variant, gene) pairs (1,656 unique
variant positions), look the variant up in the 1011-isolate panel gVCF and, for
every isolate that *carries the alternate allele*, record:

  * the genotype call (GT) and its locus-level zygosity (het / hom-alt),
  * the allele-balance VAF from the AD field = alt_depth / (ref_depth+alt_depth)
    -- this is the read-level alt fraction, which betrays true dosage even
    though GT is diploidised for the whole panel (e.g. a triploid carrying one
    alt copy reads ~0.33 yet is often called 0/1),
  * the isolate's genome-wide ploidy, zygosity class, and the copy number of the
    variant's own chromosome (Peter 2018 Tables S1/S16),
  * the isolate's genome-wide heterozygous-SNP proportion (Table S8).

Why this matters: a sequence-to-expression model (Shorkie/Yorzoi) scores a full
ref->alt swap on ONE haploid genome. It has no concept of heterozygosity or
copy-number dosage. Positives whose alt allele is carried mostly by
heterozygous or non-diploid isolates are therefore measured by the eQTL study
in a context the model fundamentally cannot represent -- so they may be poor
labels. This script quantifies that per positive.

Outputs (parquet) under --out-dir:
  * carriers_long.parquet   - one row per (variant, carrier isolate)
  * variant_summary.parquet - one row per unique variant
  * pairs.parquet           - one row per (variant, gene) positive pair
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

REPO = Path(__file__).resolve().parents[3]
DATA = Path("/home/tds122/yeast-seq2epxression-benchmark/data/tasks")


def _classify(gt: tuple, alt_idx: int) -> tuple[str, int]:
    """Return (locus_zygosity, n_alt_copies_called) from a diploid GT tuple."""
    called = [a for a in gt if a is not None]
    n_alt = sum(1 for a in called if a == alt_idx)
    if n_alt == 0:
        return "non_carrier", 0
    has_other = any(a != alt_idx for a in called)
    return ("het" if has_other else "hom_alt"), n_alt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--positives",
        type=Path,
        default=DATA / "caudal_eqtl/_intermediates_prep/selected_eQTL/intersected_data_CIS.tsv",
    )
    ap.add_argument("--gvcf", type=Path, default=DATA / "1011Matrix.gvcf.gz")
    ap.add_argument(
        "--ploidy",
        type=Path,
        default=REPO / "results/caudal_ploidy_audit/isolate_ploidy.tsv",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=REPO / "results/caudal_ploidy_audit"
    )
    args = ap.parse_args()

    pos = pd.read_csv(args.positives, sep="\t")
    # Variant key: chromosome + position + ref + alt (gene-independent).
    pos = pos.rename(
        columns={"Chr": "chrom", "ChrPos": "pos", "Reference": "ref", "Alternate": "alt"}
    )
    pos["variant_id"] = (
        pos["chrom"].astype(str)
        + ":"
        + pos["pos"].astype(str)
        + ":"
        + pos["ref"].astype(str)
        + ">"
        + pos["alt"].astype(str)
    )
    pos["dist_to_gene_start"] = (pos["pos"] - pos["Pheno_pos"]).abs()
    uniq = pos.drop_duplicates("variant_id").reset_index(drop=True)
    print(f"{len(pos)} positive pairs, {len(uniq)} unique variants")

    ploidy = pd.read_csv(args.ploidy, sep="\t").set_index("strain")
    cn_lookup = {ch: ploidy[f"cn_{ch}"] for ch in
                 [f"chromosome{i}" for i in range(1, 17)] if f"cn_{ch}" in ploidy.columns}

    vf = pysam.VariantFile(str(args.gvcf))
    samples = list(vf.header.samples)

    carrier_rows: list[dict] = []
    var_rows: list[dict] = []
    n_no_record = 0

    for i, v in enumerate(uniq.itertuples(index=False)):
        if i % 250 == 0:
            print(f"  ...{i}/{len(uniq)} variants", flush=True)
        chrom, p, ref, alt = v.chrom, int(v.pos), str(v.ref), str(v.alt)
        rec = None
        for r in vf.fetch(chrom, p - 1, p):
            if r.pos == p and r.ref == ref and alt in r.alts:
                rec = r
                break
        if rec is None:
            n_no_record += 1
            var_rows.append(
                dict(variant_id=v.variant_id, chrom=chrom, pos=p, ref=ref, alt=alt,
                     allele_found=False)
            )
            continue

        alt_idx = rec.alleles.index(alt)
        info_af = rec.info.get("AF")
        info_af = float(info_af[alt_idx - 1]) if info_af is not None else np.nan
        info_ac = rec.info.get("AC")
        info_ac = int(info_ac[alt_idx - 1]) if info_ac is not None else -1
        info_an = int(rec.info.get("AN") or 0)
        cn_series = cn_lookup.get(chrom)

        gt_alt_alleles = 0  # for genotype-derived AF cross-check
        gt_called_alleles = 0
        for s in samples:
            d = rec.samples[s]
            gt = d.get("GT")
            if gt is None:
                continue
            called = [a for a in gt if a is not None]
            gt_called_alleles += len(called)
            gt_alt_alleles += sum(1 for a in called if a == alt_idx)
            zyg, n_alt = _classify(gt, alt_idx)
            if n_alt == 0:
                continue  # not a carrier
            ad = d.get("AD")
            ad_ref = int(ad[0]) if ad is not None else -1
            ad_alt = int(ad[alt_idx]) if ad is not None and len(ad) > alt_idx else -1
            denom = ad_ref + ad_alt
            vaf = ad_alt / denom if denom > 0 else np.nan
            pinfo = ploidy.loc[s] if s in ploidy.index else None
            carrier_rows.append(dict(
                variant_id=v.variant_id, chrom=chrom, pos=p, ref=ref, alt=alt,
                strain=s, gt="/".join("." if a is None else str(a) for a in gt),
                locus_zygosity=zyg, n_alt_copies_called=n_alt,
                ad_ref=ad_ref, ad_alt=ad_alt, dp=int(d.get("DP") or 0), vaf=vaf,
                strain_ploidy=(float(pinfo["ploidy"]) if pinfo is not None else np.nan),
                strain_zygosity=(pinfo["zygosity"] if pinfo is not None else None),
                local_cn=(float(cn_series[s]) if cn_series is not None and s in cn_series.index else np.nan),
                is_aneuploid=(bool(pinfo["is_aneuploid"]) if pinfo is not None else None),
                het_snp_proportion=(float(pinfo["het_snp_proportion"]) if pinfo is not None else np.nan),
                clade=(pinfo["clade"] if pinfo is not None else None),
            ))

        var_rows.append(dict(
            variant_id=v.variant_id, chrom=chrom, pos=p, ref=ref, alt=alt,
            allele_found=True, info_af=info_af, info_ac=info_ac, info_an=info_an,
            gt_af=(gt_alt_alleles / gt_called_alleles if gt_called_alleles else np.nan),
            n_called_strains=int(sum(
                1 for s in samples if rec.samples[s].get("GT") is not None
                and any(a is not None for a in rec.samples[s].get("GT"))
            )),
        ))

    vdf = pd.DataFrame(var_rows)
    cdf = pd.DataFrame(carrier_rows)

    # ---- per-variant carrier aggregates ----
    def agg(g: pd.DataFrame) -> pd.Series:
        nd = g["strain_ploidy"].fillna(-1)
        return pd.Series(dict(
            n_carriers=len(g),
            n_het_carriers=int((g["locus_zygosity"] == "het").sum()),
            n_hom_alt_carriers=int((g["locus_zygosity"] == "hom_alt").sum()),
            n_carriers_1n=int((nd == 1).sum()),
            n_carriers_2n=int((nd == 2).sum()),
            n_carriers_3n_plus=int((nd >= 3).sum()),
            n_carriers_aneuploid=int((g["is_aneuploid"] == True).sum()),  # noqa: E712
            n_carriers_local_cn_gt2=int((g["local_cn"].fillna(0) > 2).sum()),
            mean_carrier_vaf=g["vaf"].mean(),
            median_carrier_vaf=g["vaf"].median(),
            mean_het_carrier_vaf=g.loc[g["locus_zygosity"] == "het", "vaf"].mean(),
        ))

    if len(cdf):
        a = cdf.groupby("variant_id").apply(agg, include_groups=False).reset_index()
        vdf = vdf.merge(a, on="variant_id", how="left")
        for c in ["n_carriers", "n_het_carriers", "n_hom_alt_carriers",
                  "n_carriers_1n", "n_carriers_2n", "n_carriers_3n_plus",
                  "n_carriers_aneuploid", "n_carriers_local_cn_gt2"]:
            vdf[c] = vdf[c].fillna(0).astype(int)
        vdf["frac_het_carriers"] = vdf["n_het_carriers"] / vdf["n_carriers"].replace(0, np.nan)
        vdf["frac_nondiploid_carriers"] = (
            (vdf["n_carriers_1n"] + vdf["n_carriers_3n_plus"]) / vdf["n_carriers"].replace(0, np.nan)
        )
        vdf["frac_polyploid_or_aneuploid_carriers"] = (
            (vdf["n_carriers_3n_plus"] + vdf["n_carriers_aneuploid"]).clip(upper=vdf["n_carriers"])
            / vdf["n_carriers"].replace(0, np.nan)
        )

    # attach the variant's own MAF / n_genes from the positives file
    meta = (
        pos.groupby("variant_id")
        .agg(maf=("maf", "first"),
             n_genes=("Pheno", "nunique"),
             min_dist_to_gene_start=("dist_to_gene_start", "min"))
        .reset_index()
    )
    vdf = vdf.merge(meta, on="variant_id", how="left")

    # ---- per-pair table (1,901 rows) ----
    pairs = pos[["variant_id", "chrom", "pos", "ref", "alt", "Pheno", "maf",
                 "dist_to_gene_start"]].rename(columns={"Pheno": "gene"})
    pairs = pairs.merge(
        vdf.drop(columns=["chrom", "pos", "ref", "alt", "maf"], errors="ignore"),
        on="variant_id", how="left",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cdf.to_parquet(args.out_dir / "carriers_long.parquet", index=False)
    vdf.to_parquet(args.out_dir / "variant_summary.parquet", index=False)
    pairs.to_parquet(args.out_dir / "pairs.parquet", index=False)

    # ---- console summary + internal consistency check ----
    found = vdf[vdf["allele_found"] == True]  # noqa: E712
    print(f"\nvariants with matching gVCF allele: {len(found)}/{len(vdf)} "
          f"(no record/allele mismatch: {n_no_record})")
    if len(found):
        d = (found["gt_af"] - found["info_af"]).abs()
        print(f"genotype-AF vs INFO-AF: max|delta|={d.max():.4f}, "
              f"mean|delta|={d.mean():.5f}  (consistency check)")
    print(f"carrier rows: {len(cdf)}")
    print(f"\nwrote -> {args.out_dir}/(carriers_long|variant_summary|pairs).parquet")


if __name__ == "__main__":
    main()
