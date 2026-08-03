# Caudal eQTL — positive-set ploidy / VAF audit

Roadmap item: *"Audit discovery methodology — positives called from the
1011-strain panel with varying ploidy + per-variant VAFs; decide whether to
filter on ploidy / VAF before scoring."*

The Caudal positives are statistical cis-eQTL calls from a GWAS across the
1002-Yeast-Genomes panel (Peter et al. 2018). That panel mixes haploids,
diploids, polyploids and aneuploids, and ~half the isolates are heterozygous. A
sequence-to-expression model scores a full ref→alt swap on **one haploid
genome** — it cannot represent heterozygosity or copy-number dosage. This audit
asks, per positive, how much of its alt-allele signal comes from isolates a
haploid model cannot represent.

## Pipeline

```
build_ploidy_table.py        Peter 2018 Tables S1/S16/S8  -> isolate_ploidy.tsv
                             (standardized name == gVCF sample; 0/1011 missing)
extract_positive_carriers.py 1011Matrix.gvcf.gz x positives -> per-strain VAF+ploidy
build_notebook.py            assembles caudal_ploidy_vaf_audit.ipynb
```

Run order (uses the repo's gVCF + the canonical `intersected_data_CIS.tsv`):

```bash
python scripts/eqtl/audit/build_ploidy_table.py
python scripts/eqtl/audit/extract_positive_carriers.py   # ~4 min; needs gVCF .tbi
python scripts/eqtl/audit/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/caudal_ploidy_vaf_audit.ipynb
```

Outputs land in `results/caudal_ploidy_audit/` (git-ignored):
`isolate_ploidy.tsv`, `carriers_long.parquet` (one row per variant×carrier,
636,909 rows), `variant_summary.parquet`, `pairs.parquet`, and the cached
Peter-2018 workbook. The notebook is in `notebooks/`.

VAF here = allele-balance VAF = `alt_depth/(ref_depth+alt_depth)` from the gVCF
AD field — the read-level alt fraction, which exposes true dosage even though GT
is diploidised for the whole panel.

## Findings (2026-06-15)

Validation: genotype-derived AF matches the gVCF INFO AF to max |Δ| = 0.0005.

1. **AF / VAF rarity is a non-issue.** Positives are GWAS hits — MAF median 0.27,
   only 1 of 1,656 below 0.05. No case for an allele-frequency filter on the
   positives.
2. **Ploidy / zygosity confounding is real but moderate.** Across carriers, 85%
   of locus genotype calls are homozygous-alt and 77% of carriers are diploid.
   The median positive draws 18% of carriers from heterozygous calls and 23%
   from non-diploid isolates; the confounded cases are the tail (p90 ≈ 40% het,
   34% non-diploid), e.g. chr16:25159 G>T has 100% heterozygous carriers (mean
   VAF 0.46) — measured entirely in het backgrounds a haploid model can't model.
3. **~54 positives (44 variants) are not clean biallelic SNPs, and are already
   auto-excluded.** Labelled `subtype == 'SNP'` upstream but multiallelic /
   complex (comma-separated ALT, multi-base REF). No REF/ALT-matched negative
   exists, so the negset generator drops every one — they never reach scoring.
   Worth re-normalising in v2 for an honest count, but not contaminating results.

4. **Stratified Yorzoi run — clean vs not-clean barely moves.** One
   `ybench` run, sliced by stratum (`stratified_eval.py` → `stratified_yorzoi.json`):
   - Full set: clean AUROC 0.531 vs not-clean 0.526 (Δ +0.005); AUPRC Δ +0.014.
   - Close-only ≤2 kb (distance-controlled, where the model has signal): clean
     0.588 vs not-clean 0.561 (Δ +0.027); AUPRC Δ +0.028.
   - Worst bucket (non-diploid-dominated) is partly a window artefact (most
     beyond-window / zero-scored variants). Real ploidy effect exists but small;
     Yorzoi's weak Caudal score is mostly model-near-chance + window limit, not
     label contamination.

## Recommendation — fix in v2, not this iteration

- The clean/not-clean gap is small (≤ 0.03 AUROC even where signal exists), so
  cleaning the positive set would not materially change the headline → **defer
  to v2**.
- Do **not** AF/VAF-filter the positives (finding 1) — ever.
- v2: re-normalise/drop the multiallelic-complex positives (cosmetic; already
  excluded) and ship ploidy/zygosity as an optional **diagnostic stratifier**
  (full set primary, clean-carrier subset secondary), not a hard filter.
- Caveat: Yorzoi only (near-chance on Caudal). Shorkie has more signal (~0.57 /
  ~0.62 close-only) and could show a larger gap — re-run `stratified_eval.py`
  against a Shorkie run if the v2 decision needs to be airtight.
