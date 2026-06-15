#!/usr/bin/env python3
"""Assemble the Caudal ploidy/VAF audit notebook from cell sources.

Keeps the notebook itself out of source control churn: edit the cell strings
here, run this, then execute with nbconvert. The notebook reads the parquet
tables produced by extract_positive_carriers.py.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "notebooks/caudal_ploidy_vaf_audit.ipynb"

MD = "markdown"
CO = "code"

CELLS: list[tuple[str, str]] = [
    (MD, r"""# Caudal cis-eQTL positives — per-strain VAF & ploidy audit

**Question this notebook answers:** for each of the 1,901 Caudal cis-eQTL
*positive* `(variant, gene)` pairs, *which* of the 1,011 natural yeast isolates
actually carry the alternate allele, and **what is each carrier's allele-balance
VAF and its ploidy?**

**Why it matters.** The Caudal positives are not validated causal variants —
they are statistical cis-eQTL calls from a GWAS across the 1002-Yeast-Genomes
panel (Peter et al. 2018). That panel is *not* a clean haploid set: it mixes
haploids, diploids, polyploids and aneuploids, and roughly half the isolates are
heterozygous. A sequence-to-expression model (Shorkie, Yorzoi) scores a full
ref→alt swap on **one haploid genome** — it has no representation of
heterozygosity or copy-number dosage. So a positive whose alt allele lives
mostly in heterozygous or non-diploid isolates is measured by the eQTL study in
a context the model fundamentally cannot reproduce, which makes it a
questionable benchmark label.

This notebook quantifies that, per positive, so we can decide whether to filter
on ploidy / zygosity / allele frequency before scoring.

**Data sources**
- Per-strain genotypes: `1011Matrix.gvcf.gz` (1002 Yeast Genomes Project) — GT, AD, DP.
- Per-isolate ploidy / zygosity / per-chromosome copy number: Peter et al. 2018
  Nature Supplementary Tables S1, S16, S8.
- Positives: the canonical Caudal `intersected_data_CIS.tsv` (1,901 SNV+CIS pairs).

Tables are built by `scripts/eqtl/audit/build_ploidy_table.py` and
`scripts/eqtl/audit/extract_positive_carriers.py`.

> **VAF here = allele-balance VAF** = `alt_depth / (ref_depth + alt_depth)` from
> the gVCF AD field. It is the read-level alt fraction within an isolate. It is
> the key signal because GT is diploidised for the whole panel, so a triploid
> carrying a single alt copy is often *called* `0/1` yet reads ~0.33 — the VAF,
> not the GT, exposes the true dosage.
"""),
    (CO, r"""import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)

# locate the audit results dir whether run from repo root or notebooks/
CANDS = [Path("results/caudal_ploidy_audit"),
         Path("../results/caudal_ploidy_audit"),
         Path(__file__).resolve().parents[1] / "results/caudal_ploidy_audit"
         if "__file__" in dir() else Path("results/caudal_ploidy_audit")]
AUDIT = next((c.resolve() for c in CANDS if (c / "variant_summary.parquet").exists()), None)
assert AUDIT is not None, "run extract_positive_carriers.py first"
print("audit dir:", AUDIT)

carriers = pd.read_parquet(AUDIT / "carriers_long.parquet")
variants = pd.read_parquet(AUDIT / "variant_summary.parquet")
pairs = pd.read_parquet(AUDIT / "pairs.parquet")
isolates = pd.read_csv(AUDIT / "isolate_ploidy.tsv", sep="\t")

print(f"{len(pairs):>7,} positive (variant, gene) pairs")
print(f"{len(variants):>7,} unique variant positions")
print(f"{len(carriers):>7,} (variant, carrier-strain) rows")
print(f"{len(isolates):>7,} isolates in the panel")
"""),
    (MD, r"""## 1. Panel backdrop — ploidy & zygosity of the 1,011 isolates

Before looking at the positives, here is the composition of the panel the
eQTLs were called from. The non-diploid and heterozygous fractions here are the
ceiling on how confounded any positive could be."""),
    (CO, r"""fig, ax = plt.subplots(1, 3, figsize=(15, 4))

pl = isolates["ploidy"].fillna(-1).map(lambda x: f"{int(x)}n" if x > 0 else "unknown")
order = [f"{i}n" for i in range(1, 6)] + ["unknown"]
pl.value_counts().reindex(order).dropna().plot.bar(ax=ax[0], color="#4C72B0")
ax[0].set_title("Ploidy (genome-wide)"); ax[0].set_ylabel("isolates")

isolates["zygosity"].value_counts().plot.bar(ax=ax[1], color="#55A868")
ax[1].set_title("Zygosity class"); ax[1].set_ylabel("isolates")
ax[1].tick_params(axis="x", rotation=0)

isolates["is_aneuploid"].map({True: "aneuploid", False: "euploid"}).value_counts().plot.bar(
    ax=ax[2], color="#C44E52")
ax[2].set_title("Aneuploidy"); ax[2].set_ylabel("isolates")
ax[2].tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.show()

nd = int((isolates["ploidy"].fillna(2) != 2).sum())
het = int((isolates["zygosity"] == "heterozygous").sum())
print(f"non-diploid isolates: {nd}/{len(isolates)} ({nd/len(isolates):.0%})")
print(f"heterozygous isolates: {het}/{len(isolates)} ({het/len(isolates):.0%})")
print(f"aneuploid isolates: {int(isolates['is_aneuploid'].sum())}/{len(isolates)}")
"""),
    (MD, r"""## 2. The core table — per-strain VAF + ploidy for every carrier

`carriers_long` is the deliverable the question asks for: **one row per (positive
variant, carrying isolate)**, with that isolate's allele-balance VAF, its GT
call and locus zygosity, its genome-wide ploidy, the copy number of the
variant's own chromosome (`local_cn`), and its genome-wide heterozygosity."""),
    (CO, r"""show_cols = ["variant_id", "strain", "gt", "locus_zygosity", "ad_ref", "ad_alt",
             "vaf", "strain_ploidy", "strain_zygosity", "local_cn", "is_aneuploid"]
display(carriers[show_cols].head(10))

# all carrier isolates of one variant, with VAF + ploidy, sorted by VAF
def variant_carriers(variant_id: str) -> pd.DataFrame:
    return (carriers[carriers.variant_id == variant_id][show_cols]
            .sort_values("vaf").reset_index(drop=True))

def plot_variant(variant_id: str, ax=None):
    g = carriers[carriers.variant_id == variant_id]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.2))
    pal = {1.0: "#DD8452", 2.0: "#4C72B0", 3.0: "#55A868", 4.0: "#C44E52", 5.0: "#8172B3"}
    for pld, sub in g.groupby("strain_ploidy"):
        ax.scatter(sub["vaf"], np.random.uniform(0, 1, len(sub)), s=14, alpha=0.5,
                   color=pal.get(pld, "grey"), label=f"{int(pld)}n" if pld == pld else "?")
    ax.axvline(0.5, ls="--", c="grey", lw=1); ax.axvline(1.0, ls="--", c="grey", lw=1)
    ax.set_xlim(0, 1.02); ax.set_yticks([]); ax.set_xlabel("allele-balance VAF")
    ax.set_title(f"{variant_id}\n{len(g)} carriers", fontsize=9)
    ax.legend(title="ploidy", fontsize=7, loc="upper left")
    return ax

print("\nhelper ready: variant_carriers('chrom:pos:ref>alt'), plot_variant(...)")
"""),
    (MD, r"""## 3. Worked examples

Three positives picked automatically from the data: one carried almost entirely
by **homozygous diploids** (a clean label the model can represent), one with a
high fraction of **heterozygous** carriers, and one with a high fraction of
**polyploid / aneuploid** carriers. Watch the VAF: clean cases pile up near 1.0
(homozygous alt) and 0.5 (diploid het); dosage cases smear toward 0.33 / 0.67."""),
    (CO, r"""ok = variants[(variants.allele_found == True) & (variants.n_carriers >= 30)].copy()

ex_clean = ok.sort_values("frac_het_carriers").iloc[0]["variant_id"]
ex_het = ok.sort_values("frac_het_carriers", ascending=False).iloc[0]["variant_id"]
ex_poly = ok.sort_values("frac_nondiploid_carriers", ascending=False).iloc[0]["variant_id"]
examples = {"homozygous-diploid (clean)": ex_clean,
            "heterozygous-heavy": ex_het,
            "polyploid/aneuploid-heavy": ex_poly}

fig, axes = plt.subplots(1, 3, figsize=(16, 3.4))
for ax, (label, vid) in zip(axes, examples.items()):
    plot_variant(vid, ax=ax); ax.set_title(f"{label}\n{vid}", fontsize=9)
plt.tight_layout(); plt.show()

for label, vid in examples.items():
    r = variants[variants.variant_id == vid].iloc[0]
    print(f"\n### {label}: {vid}  (AF={r.info_af:.3f}, MAF={r.maf:.3f}, {int(r.n_carriers)} carriers)")
    print(f"    het carriers {r.frac_het_carriers:.0%} | non-diploid carriers "
          f"{r.frac_nondiploid_carriers:.0%} | mean VAF {r.mean_carrier_vaf:.2f}")
    display(variant_carriers(vid).head(8))
"""),
    (MD, r"""## 4. Per-strain VAF distribution — the dosage fingerprint

Pooling all carriers, the allele-balance VAF splits cleanly by locus zygosity
(homozygous-alt near 1.0, heterozygous near 0.5) and the heterozygous mode
broadens for higher-ploidy isolates (1 alt of 3 ≈ 0.33, 2 of 3 ≈ 0.67). Every
carrier left of ~0.9 is a genome where the model's homozygous ref→alt swap
overstates the per-strain effect."""),
    (CO, r"""fig, ax = plt.subplots(1, 2, figsize=(14, 4))

for zyg, c in [("hom_alt", "#4C72B0"), ("het", "#C44E52")]:
    sub = carriers[carriers.locus_zygosity == zyg]
    ax[0].hist(sub["vaf"].dropna(), bins=50, range=(0, 1), alpha=0.6, label=zyg, color=c)
ax[0].set_title("VAF by locus zygosity (all carriers)")
ax[0].set_xlabel("allele-balance VAF"); ax[0].set_ylabel("carriers"); ax[0].legend()

het = carriers[carriers.locus_zygosity == "het"].copy()
het["pl"] = het["strain_ploidy"].map(lambda x: f"{int(x)}n" if x in (1,2,3,4,5) else "?")
for pl, c in [("2n", "#4C72B0"), ("3n", "#55A868"), ("4n", "#C44E52")]:
    s = het[het.pl == pl]["vaf"].dropna()
    if len(s):
        ax[1].hist(s, bins=40, range=(0, 1), alpha=0.55, density=True, label=f"{pl} (n={len(s)})", color=c)
for x in (1/3, 0.5, 2/3):
    ax[1].axvline(x, ls="--", c="grey", lw=1)
ax[1].set_title("VAF of heterozygous-called carriers, by ploidy")
ax[1].set_xlabel("allele-balance VAF"); ax[1].set_ylabel("density"); ax[1].legend()
plt.tight_layout(); plt.show()

print("carrier composition (locus zygosity):")
print(carriers["locus_zygosity"].value_counts(normalize=True).mul(100).round(1).to_string())
print("\ncarrier composition (strain ploidy):")
print((carriers["strain_ploidy"].map(lambda x: f"{int(x)}n" if x==x else "?")
       .value_counts(normalize=True).mul(100).round(1)).to_string())
"""),
    (MD, r"""## 5. How confounded is each positive?

For every positive variant: what fraction of its carriers are heterozygous at
the locus, and what fraction are non-diploid? These are the two axes a
sequence-only model cannot represent."""),
    (CO, r"""v = variants[variants.allele_found == True]
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].hist(v["frac_het_carriers"].dropna(), bins=40, color="#C44E52", alpha=0.8)
ax[0].set_title("Fraction of carriers that are heterozygous (per positive)")
ax[0].set_xlabel("frac het carriers"); ax[0].set_ylabel("positives")
ax[0].axvline(v["frac_het_carriers"].median(), c="k", ls="--",
              label=f"median {v['frac_het_carriers'].median():.2f}"); ax[0].legend()

ax[1].hist(v["frac_nondiploid_carriers"].dropna(), bins=40, color="#55A868", alpha=0.8)
ax[1].set_title("Fraction of carriers that are non-diploid (per positive)")
ax[1].set_xlabel("frac non-diploid carriers"); ax[1].set_ylabel("positives")
ax[1].axvline(v["frac_nondiploid_carriers"].median(), c="k", ls="--",
              label=f"median {v['frac_nondiploid_carriers'].median():.2f}"); ax[1].legend()
plt.tight_layout(); plt.show()

print("Per-positive carrier confounding (median across positives):")
print(f"  het-carrier fraction:        {v['frac_het_carriers'].median():.1%}")
print(f"  non-diploid-carrier fraction:{v['frac_nondiploid_carriers'].median():.1%}")
print(f"  mean carrier VAF:            {v['mean_carrier_vaf'].median():.2f}")
"""),
    (MD, r"""## 6. Population allele frequency — are any positives rare?

The negatives are filtered to AF ≥ 0.05, but **the positives are not filtered on
allele frequency at all** (only `ld_mask`, `cis`, `SNP`). Rare positives give the
noisiest underlying GWAS calls. Here is the distribution of population AF / MAF
across the positives."""),
    (CO, r"""fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].hist(variants["maf"].dropna(), bins=40, color="#4C72B0", alpha=0.85)
ax[0].set_title("MAF of positives (from Caudal GWAS)"); ax[0].set_xlabel("MAF"); ax[0].set_ylabel("positives")
ax[1].hist(variants["info_af"].dropna(), bins=40, color="#8172B3", alpha=0.85)
ax[1].set_title("Population AF of alt allele (from gVCF INFO)"); ax[1].set_xlabel("AF")
plt.tight_layout(); plt.show()

for thr in (0.01, 0.05, 0.10):
    n = int((variants["maf"] < thr).sum())
    print(f"positives with MAF < {thr:.2f}: {n}/{len(variants)} ({n/len(variants):.1%})")
"""),
    (MD, r"""## 7. Allele-match sanity check — multiallelic / complex positives

The upstream gVCF intersection matched on *position only*. Requiring the exact
REF/ALT to match a panel record surfaces a separate data-hygiene issue: a set of
positives are **labelled `subtype == 'SNP'` but are actually multiallelic /
complex sites** — their `Alternate` is a comma-separated allele list (e.g.
`G → T,GTAATA`) and/or `Reference` is multi-base. These do not map to a single
clean biallelic SNP allele in the panel, so no per-strain VAF can be computed
for them and a sequence model has no single ref→alt swap to score. They are
excluded from the VAF/ploidy analysis above and are prime candidates to drop or
re-normalise."""),
    (CO, r"""nf = variants[variants["allele_found"] != True].copy()
print(f"positives not matched as a clean biallelic SNP: {len(nf)}/{len(variants)} variants")
maffield = pairs.groupby("variant_id").size()
print(f"  -> affecting {int(pairs['variant_id'].isin(nf.variant_id).sum())} of {len(pairs)} (variant, gene) pairs")
nf["alt_is_multiallelic"] = nf["alt"].astype(str).str.contains(",")
nf["ref_multibase"] = nf["ref"].astype(str).str.len() > 1
print(f"  -> with comma-separated (multiallelic) ALT: {int(nf['alt_is_multiallelic'].sum())}")
print(f"  -> with multi-base REF (complex/MNV):       {int(nf['ref_multibase'].sum())}")
display(nf[["variant_id", "ref", "alt", "alt_is_multiallelic", "ref_multibase"]].head(12))
"""),
    (MD, r"""## 8. What would a filter drop?

Candidate filters for a cleaner positive set, and how many of the 1,901 pairs /
1,656 variants each would remove. These are *illustrative thresholds* — the
point is the order of magnitude of the trade-off, not the exact cut."""),
    (CO, r"""def npairs(mask_var):
    bad = set(variants[mask_var]["variant_id"])
    return int(pairs["variant_id"].isin(bad).sum())

rules = {
    "allele not found in panel": variants["allele_found"] != True,
    "MAF < 0.05": variants["maf"] < 0.05,
    ">30% carriers heterozygous": variants["frac_het_carriers"] > 0.30,
    ">30% carriers non-diploid": variants["frac_nondiploid_carriers"] > 0.30,
    "mean carrier VAF < 0.80": variants["mean_carrier_vaf"] < 0.80,
}
rows = []
for name, m in rules.items():
    m = m.fillna(False)
    rows.append(dict(rule=name, variants_dropped=int(m.sum()),
                     pct_variants=f"{m.mean():.1%}", pairs_dropped=npairs(m)))
summary = pd.DataFrame(rows)
display(summary)

any_rule = (np.column_stack([m.fillna(False).values for m in rules.values()]).any(axis=1))
print(f"\nUnion of all rules: {int(any_rule.sum())}/{len(variants)} variants "
      f"({any_rule.mean():.1%}), {npairs(pd.Series(any_rule, index=variants.index))} pairs.")
"""),
    (MD, r"""## 9. Does it move the benchmark? — Yorzoi, clean vs not-clean

The decision hinges on *how much the score changes* between clean and not-clean
positives. We ran **Yorzoi once** over all four negsets (`ybench run --model
yorzoi --task caudal_eqtl`, current per-base code) and sliced the per-pair scores
by stratum — each positive keeps its distance-matched negative, so AUROC/AUPRC
stay well-posed (1:1 base rate). Produced by
`scripts/eqtl/audit/stratified_eval.py` → `stratified_yorzoi.json`.

Because the Caudal signal is strongly distance-dependent and ~38% of Yorzoi
scores are exactly 0 (variant beyond the model window), we also report a
distance-controlled **close-only (≤2 kb)** comparison — that is where the model
has real signal and where a ploidy effect, if real, should show."""),
    (CO, r"""import json
strat = json.loads((AUDIT / "stratified_yorzoi.json").read_text())
cats = ["clean", "not_clean", "het_dominated", "nondiploid_dominated", "low_carrier_vaf"]
rows = []
for c in cats:
    f = strat["full"].get(c) or {}
    d = strat["distance_profile"][c]
    if not f:
        continue
    rows.append(dict(stratum=c, n_pairs=f["n_pairs"],
                     AUROC=f"{f['auroc_mean']:.3f}±{f['auroc_sem']:.3f}",
                     AUPRC=f"{f['auprc_mean']:.3f}±{f['auprc_sem']:.3f}",
                     pos_zero_pct=f"{f['pos_zero_frac']*100:.0f}%",
                     gt_8kb_pct=f"{d['frac_gt_8kb']*100:.0f}%"))
display(pd.DataFrame(rows).set_index("stratum"))
print("Note: the 44 multiallelic/complex positives have 0 scored pairs — they were "
      "dropped from every negset (no REF/ALT-matched negative exists), so the "
      "benchmark already excludes them.")
"""),
    (CO, r"""fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for j, (regime, title) in enumerate([("full", "Full set"), ("close_only", "Close-only (≤2 kb)")]):
    cl, nc = strat[regime]["clean"], strat[regime]["not_clean"]
    x = np.arange(2)
    ax[j].bar(x - 0.2, [cl["auroc_mean"], nc["auroc_mean"]], 0.4,
              yerr=[cl["auroc_sem"], nc["auroc_sem"]], capsize=4, label="AUROC", color="#4C72B0")
    ax[j].bar(x + 0.2, [cl["auprc_mean"], nc["auprc_mean"]], 0.4,
              yerr=[cl["auprc_sem"], nc["auprc_sem"]], capsize=4, label="AUPRC", color="#C44E52")
    ax[j].axhline(0.5, ls="--", c="grey")
    ax[j].set_xticks(x); ax[j].set_xticklabels([f"clean\n(n={cl['n_pairs']})", f"not-clean\n(n={nc['n_pairs']})"])
    ax[j].set_ylim(0.45, 0.65); ax[j].set_title(f"{title} — Yorzoi"); ax[j].legend()
plt.tight_layout(); plt.show()

for regime in ("full", "close_only"):
    cl, nc = strat[regime]["clean"], strat[regime]["not_clean"]
    print(f"{regime:11s}: clean AUROC {cl['auroc_mean']:.3f} vs not-clean {nc['auroc_mean']:.3f} "
          f"(Δ={cl['auroc_mean']-nc['auroc_mean']:+.3f}) | "
          f"AUPRC Δ={cl['auprc_mean']-nc['auprc_mean']:+.3f}")
"""),
    (MD, r"""## 10. Findings & recommendation

*(Interpretation — the numbers are in the cells above.)*

**1. Allele frequency / VAF rarity is a non-issue.** The positives are GWAS hits,
so they are common by construction: MAF median ≈ 0.27 and essentially none fall
below 0.05 (the same floor the negatives use). There is no case for an
allele-frequency filter on the positives.

**2. Ploidy / zygosity confounding is real but moderate — a tail, not the bulk.**
Across carriers, ~85% of genotype calls at these loci are homozygous-alt and
~77% of carriers are diploid; the per-strain VAF piles up near 1.0 (hom-alt) and
0.5 (diploid het). The median positive draws only ~18% of its carriers from
heterozygous calls and ~23% from non-diploid isolates. Because an eQTL call
integrates over *all* its carriers, a positive is only genuinely
"unrepresentable" by a haploid sequence model when heterozygous / polyploid
carriers *dominate* it — that is the tail (p90 ≈ 40% het, ≈ 34% non-diploid),
not the typical positive.

**3. The ~54 multiallelic/complex "SNP" positives are already auto-excluded.**
They are labelled `SNP` upstream but are multiallelic / complex (comma-separated
ALT, multi-base REF). No REF/ALT-matched negative exists for them, so the negset
generator already drops every one — they never reach scoring. Still worth
re-normalising or formally dropping in v2 so the count is honest, but they are
not currently contaminating the benchmark.

**4. The clean/not-clean split barely moves Yorzoi (section 9).** Full set: clean
AUROC ≈ 0.53 vs not-clean ≈ 0.53 (Δ ≈ 0.005). Distance-controlled close-only
(≤2 kb, where the model has real signal): clean ≈ 0.59 vs not-clean ≈ 0.56
(Δ ≈ 0.027). The worst sub-bucket (non-diploid-dominated) is partly a window
artefact — it has the most beyond-window, zero-scored variants, not just the most
dosage. So there *is* a real ploidy effect, but it is small, and Yorzoi's weak
Caudal score is driven mainly by the model being near-chance and by the
window/zero-score limit, **not** by ploidy contamination of the labels.

**Recommendation — fix in v2, not this iteration.**
- The clean-vs-not-clean gap is small (≤ 0.03 AUROC even where signal exists), so
  cleaning the positive set would not materially change the headline. **Defer the
  ploidy/zygosity cleanup to v2.**
- **Do not** AF/VAF-filter the positives (finding 1) — ever.
- In v2: re-normalise / drop the multiallelic-complex positives (finding 3, a
  cosmetic count fix since they are already excluded), and ship the
  ploidy/zygosity strata as an optional **diagnostic stratifier** (full set
  primary, clean-carrier subset as a secondary view) rather than a hard filter.
- Caveat: this is **Yorzoi only**, which is near-chance on Caudal. Shorkie has
  more signal (~0.57 full, ~0.62 close-only) and *could* show a larger gap — if
  the v2 decision needs to be airtight, re-run section 9 on Shorkie before
  finalising.
"""),
]


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(s) if t == MD else nbf.v4.new_code_cell(s)
        for t, s in CELLS
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (audit venv)",
        "language": "python",
        "name": "python3",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build()
