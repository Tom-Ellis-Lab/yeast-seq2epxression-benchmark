# Kita et al. — yeast cis-eQTL classification

> **Status:** draft. Second worked example, used alongside
> [Caudal](caudal_eqtl.md) to stress-test the benchmark entry template.

## At a glance

| | |
| --- | --- |
| **Task** | Binary classification: is this variant a cis-eQTL for its target gene, or a distance-matched non-eQTL control? Identical task definition to [Caudal](caudal_eqtl.md). |
| **Source** | Kita et al. (TODO: full citation). PNAS supplementary file [`pnas.1717421114.sd01.txt`](https://www.pnas.org/doi/suppl/10.1073/pnas.1717421114/suppl_file/pnas.1717421114.sd01.txt) (DOI: [10.1073/pnas.1717421114](https://doi.org/10.1073/pnas.1717421114)). 1,640 raw eQTLs called from 85 *S. cerevisiae* isolates. |
| **Reference assembly** | *S. cerevisiae* R64-1-1, Ensembl release 115 (shared with Caudal). |
| **Background population for negatives** | 1011 yeast isolates panel (`1011Matrix.gvcf`, shared with Caudal). The Kita eQTLs themselves are called from a separate 85-isolate panel; the 1011 panel is used only as the source of distance-matched non-eQTL controls. |
| **Positives** | 683 cis-eQTLs selected from the raw 1,640, restricted to four genomic contexts — **Promoter, UTR5, UTR3, ORF**. The 8 kb cis threshold (`|ChrPos − TSS| ≤ 8000`) is then applied for symmetry with Caudal. The selection of 683 follows the Shorkie paper's reproduction; the upstream Kita release contains more variants but only these four context categories are used in the canonical evaluation. |
| **Negatives** | Identical procedure to Caudal: REF/ALT-matched non-coding variants from the 1011 panel with AF ≥ 0.05, distance-to-TSS-matched to ±100 bp (fallback ±200 bp), four independent iterations. |
| **Primary metric** | AUROC and AUPRC, no class balancing, mean ± SEM across the four negative-set iterations. **Per-context stratified report (Promoter / UTR5 / UTR3 / ORF) is treated as primary**, not secondary, because the context shortcut (see [Known cheats #1](#known-cheats)) is the dominant failure mode. |
| **Leakage status** | ⚠️ **Sequence leakage applies to any model pretrained on R64-1-1** (Shorkie, Yorzoi) — same mechanism as Caudal. Kita is *claimed* to be "more independent" than Caudal as a held-out test set; the working hypothesis is that this stems from Kita's eQTLs being called on a different 85-isolate panel and therefore decorrelating from any RNA-seq label leakage in pretraining, but the mechanism has not been verified. See [Open questions](#open-questions--todo). |

## Why this benchmark exists

Kita is the **counterpart benchmark to Caudal**: same task, same scoring
contract, same evaluation pipeline, same negative-construction recipe — but
positives drawn from a separate eQTL study with a different isolate panel.
The pair gives the suite two cis-eQTL classification benchmarks that share
all of their evaluation machinery and differ only in the source of
positives. That makes them ideal for diagnosing whether a model's apparent
skill on Caudal generalizes or is dataset-specific.

The expected use is to **report both benchmarks alongside each other** for
any pretrained-on-yeast model. A model that scores highly on Caudal but
poorly on Kita is leaning on something Caudal-specific — most plausibly
sequence-level memorization that happens to correlate with Caudal's eQTL
calling methodology. A model that scores comparably on both is more likely
to be learning a generalizable variant-effect signal (or, less optimistically,
to be exploiting a shortcut that both benchmarks share — which is what the
[per-context stratified report](#evaluation-protocol) is designed to expose
for Kita specifically).

The headline difference in dataset construction is that **Kita explicitly
selects positives in four genomic contexts (Promoter, UTR5, UTR3, ORF)**,
while Caudal accepts any variant within 8 kb of a regulated gene's TSS.
This makes Kita simultaneously easier (the 683 are concentrated in
functionally meaningful regions and have larger expected effect sizes) and
**much more vulnerable to a context-based shortcut** than Caudal — which
is precisely what makes the per-context breakdown the load-bearing
diagnostic for this benchmark.

## Dataset construction

### Positive set
1. Download `pnas.1717421114.sd01.txt` from the PNAS supplementary materials.
   This is the canonical Kita et al. release, containing 1,640 eQTLs called
   on 85 *S. cerevisiae* isolates.
2. Restrict to the 683 variants annotated in one of the four genomic
   contexts: **Promoter, UTR5, UTR3, ORF**. This selection is inherited
   from the Shorkie paper's reproduction; it is not part of the Kita
   release itself, and a future version of the benchmark could choose a
   different selection.
3. Compute the absolute distance between `ChrPos` and the target gene's TSS
   using the Ensembl 115 GTF. Classify as cis if `|ChrPos − TSS| ≤ 8000`
   on the same chromosome — the same threshold as Caudal. (Within Kita's
   selected 683, essentially all variants pass this threshold by
   construction; the cis filter is included for symmetry with Caudal so
   that the same downstream pipeline can consume both benchmarks.)
4. Retrieve the reference and alternate alleles from `1011Matrix.gvcf` to
   ensure the variant is observed in the population panel that negatives
   will be drawn from.
5. Each retained row provides a `(chrom, pos, ref, alt, gene, context)`
   tuple. The `context` annotation (Promoter / UTR5 / UTR3 / ORF) is
   carried through into the cooked distribution as an additional column;
   see [Schema additions](#distribution).

### Negative set
**Identical procedure to Caudal**, generated by the same script
(`scripts/eqtl/0_data_generation/1_generate_negs.py --dataset kita`). The
only Kita-specific bit is the input column parsing — the script reads
`#Gene` (with the literal `#` prefix) instead of Caudal's `Pheno`. The
algorithm itself is unchanged:

1. Compute distance from each positive's `pos` to its target gene's TSS.
2. Restrict candidates to non-coding variants in the 1011 panel gVCF with
   AF ≥ 0.05 and identical `(ref, alt)`.
3. Pick one candidate whose distance to a randomly chosen same-chromosome
   gene's TSS matches the positive's distance to within ±100 bp (fallback
   ±200 bp). The randomly chosen gene becomes the negative's "target gene".
4. Reject candidates that are themselves positives or that have already
   been used in the current iteration.
5. Repeat for four independent iterations.

The same matching properties (✅ REF/ALT, ✅ distance, ✅ MAF; ⚠️ random
target gene assignment, ⚠️ coding/non-coding asymmetry) apply, but **the
coding/non-coding asymmetry is much sharper for Kita than for Caudal**:
Kita explicitly selects ORF variants as positives, while negatives are
required to be non-coding by construction. This is no longer a marginal
asymmetry — it is the dominant cheat surface for this benchmark, and is
addressed in [Known cheats #1](#known-cheats).

## Distribution

Same two-layer split as [Caudal — Distribution](caudal_eqtl.md#distribution):
raw upstream (sumstats + 1011 gVCF + GTF) lives under `data/raw/`, and the
cooked benchmark distribution is a flat-file release that adapters consume
directly.

### Cooked file layout

```
kita_eqtl_v1/
├── README.md
├── reference/                  # may be shared/symlinked with caudal_eqtl_v1
│   ├── R64-1-1.fa
│   ├── R64-1-1.fa.fai
│   └── R64-1-1.115.gtf
├── negset_1.tsv                # ~1,366 rows: ~683 positives + ~683 negatives
├── negset_2.tsv
├── negset_3.tsv
└── negset_4.tsv
```

### Schema

The full v1 schema (`variant_id`, `chrom`, `pos`, `ref`, `alt`, `gene`,
`gene_strand`, `is_positive`, `distance_to_tss`, optional `pair_id`) is
identical to [Caudal — Schema](caudal_eqtl.md#schema). The same
conventions apply: Roman-numeral chromosomes without prefix, 1-based
inclusive coordinates, sort by `(chrom, pos, is_positive desc)`.

**Kita-specific addition:** one extra column to capture the context
stratification.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `context` | str | ✅ for Kita | One of `Promoter`, `UTR5`, `UTR3`, `ORF` for positives. For negatives this is `NA` — negatives are sampled from non-coding regions and do not carry a Kita context label. The asymmetry is itself a feature of the benchmark, not a flaw of the schema; see [Known cheats #1](#known-cheats). |

The `context` column is Kita-specific but the rest of the file format is
otherwise identical to Caudal's, so any adapter that reads Caudal's TSV
schema can read Kita's by ignoring the extra column.

## Model contract

**Identical** to [Caudal — Model contract](caudal_eqtl.md#model-contract).
Same two adapter forms (Python callable or CLI/TSV), same per-variant
inputs, same scalar output, same adapter responsibilities (window choice,
strand handling, ref-allele verification, output track choice, ref-vs-alt
reduction).

A model that has been wired up to score Caudal can score Kita with **no
adapter changes** — only the input file path changes. This is the
load-bearing reason to keep Caudal and Kita in the same evaluation family.

The Shorkie reference scoring procedure (16,384 bp window, strand
averaging, 8-fold ensemble, sum of coverage over the gene's exon bins,
log2 fold change) is the same as documented in
[Caudal — Reference example](caudal_eqtl.md#reference-example-shorkies-scoring-function).
In fact `scripts/eqtl/2_variant_scoring/score_variants_shorkie.py` is
currently *written against Kita's column names* (`#Gene`, `position`,
`Reference`, `Alternate`); the script also computes two variants of the
score in parallel — `logSED_agg` (cross-track average coverage summed over
the gene's exon bins, then log fold-changed) and `logSED_mean_pertrack`
(per-track log fold change, then averaged across tracks). The published
Caudal/Kita evaluations use `logSED_agg`. Adapters should pick one and
document it; mixing the two across rows is a silent miscalibration.

## Evaluation protocol

For each of the four negative-set iterations `i ∈ {1..4}`:
1. Score every positive and every paired negative with the model.
2. Compute AUROC and AUPRC over the union of positives and negatives in
   iteration `i`. **Do not subsample to balance classes.**
3. Record per-iteration metrics.

**Primary report:** mean ± SEM across the four iterations, plus the
per-iteration ROC and PR curves interpolated to a common grid with ±1 SEM
bands. Same plotting harness as Caudal:
`scripts/eqtl/3_visualization/1_roc_pr_shorkie_fold.py`.

**Primary stratified report (Kita-specific):** AUROC and AUPRC stratified
by `context` (Promoter / UTR5 / UTR3 / ORF). This is treated as part of
the primary metric, not as a secondary view, because the context shortcut
is the dominant failure mode for this benchmark. A model whose AUROC is
high on ORF and collapses on Promoter (or vice versa) is almost certainly
leaning on the coding/non-coding asymmetry of the negative set rather than
on variant-effect signal. Reporting only the pooled AUROC hides this.

**Standard secondary report:** AUROC and AUPRC stratified by
distance-to-TSS bin, using Kita-specific bins
`[0, 500, 1200, 2000, 3000]` (vs. Caudal's
`[0, 1000, 2000, 3000, 4000, 5000]`). The tighter binning reflects that
Kita's selected variants are concentrated closer to their target genes.
See `scripts/eqtl/3_visualization/2_AUROC_AUPRC_by_dsitance.py:165-169`.

## Known cheats

The cheat list is **largely shared with Caudal**, but the ordering and the
relative weight of each cheat are different because Kita's positive
selection introduces a new load-bearing shortcut. Read this as the
*Kita-specific* version of [Caudal — Known cheats](caudal_eqtl.md#known-cheats);
the cheats unchanged from Caudal are summarized only briefly.

1. **Genomic-context / coding-vs-non-coding shortcut.** *(NEW for Kita;
   load-bearing.)* Kita explicitly selects positives in Promoter, UTR5,
   UTR3, and ORF contexts. Negatives, by construction, are required to be
   non-coding. This means a model that does nothing more than detect
   coding context can rank ORF positives above all negatives without
   learning anything about variant effects. The same applies to a lesser
   extent for UTR5/UTR3 (annotated regions vs. arbitrary non-coding) and
   even to Promoter (proximal regulatory regions vs. arbitrary
   non-coding). **This is the dominant cheat for Kita** and is the reason
   the per-context stratified evaluation is treated as primary.
   *Mitigations*: (a) report per-context AUROC/AUPRC and treat the minimum
   across contexts as the headline number; (b) in a future v2, generate
   context-matched negatives (ORF positives paired with ORF negatives,
   etc.) to neutralize the cheat at construction time.

2. **Sequence memorization (leakage).** Same mechanism as Caudal: any
   model pretrained on R64-1-1 has seen every positive's flanking context
   verbatim. The Kita-vs-Caudal claim that Kita is "more independent" is a
   working hypothesis — most plausibly that Kita's eQTLs were called on a
   different 85-isolate panel and the called effect sizes therefore
   decorrelate from any RNA-seq label leakage in the pretrained model's
   training set — but it has not been verified. It should be quantified
   before being reported. See [Open questions](#open-questions--todo).

3. **Gene-identity shortcut.** Same as Caudal: negatives are paired with a
   random same-chromosome gene at the matched TSS distance, so a model
   that ignores the variant and just learns "gene X is highly expression-
   variable across isolates" can win.

4. **Distance-to-TSS shortcut.** Same as Caudal. The Kita distance bins
   are tighter (`[0, 500, 1200, 2000, 3000]`) because the selected
   variants concentrate closer to TSSs; the residual effect of TSS
   proximity is therefore harder to escape via stratification.

5. **Reference-allele bias.** Same as Caudal — a confound rather than a
   cheat per se; a well-calibrated adapter should correct for it.

## Paired diagnostics

> **Status:** scoping. Kita is the natural diagnostic *for* Caudal (and
> vice versa) — see [Why this benchmark exists](#why-this-benchmark-exists).
> Within Kita itself, the per-context stratified report addresses
> [Known cheats #1](#known-cheats).

Initial candidates:
- **For (1) context shortcut:** report per-context AUROC/AUPRC; in v2 of
  the benchmark, optionally generate context-matched negatives and report
  the resulting AUROC as the cheat-removed number.
- **For (2) leakage:** Caudal vs Kita gap, plus a HashFrag train/test
  sequence overlap report shared with Caudal.
- **For (3)–(5):** identical to Caudal.

## Files

| Purpose | Path |
| --- | --- |
| Raw Kita sumstats | TBD — not yet committed to `data/raw/eQTL/`. ([source](https://www.pnas.org/doi/suppl/10.1073/pnas.1717421114/suppl_file/pnas.1717421114.sd01.txt)) |
| Background gVCF (1011 panel) | `data/raw/eQTL/1011Matrix.gvcf` (shared with Caudal) |
| Reference GTF | `data/raw/Saccharomyces_cerevisiae.R64-1-1.115.gtf` (shared with Caudal) |
| Cooked benchmark distribution | `kita_eqtl_v1/` (TBD: GCS bucket or HuggingFace Datasets) |
| Negative-set generation | `scripts/eqtl/0_data_generation/1_generate_negs.py --dataset kita` |
| Shorkie variant scoring | `scripts/eqtl/2_variant_scoring/score_variants_shorkie.py` (currently hardcoded against Kita's column names) |
| Evaluation / curves | `scripts/eqtl/3_visualization/1_roc_pr_shorkie_fold.py` (handles all eQTL benchmarks) |
| Distance-stratified eval | `scripts/eqtl/3_visualization/2_AUROC_AUPRC_by_dsitance.py` (Kita uses `[0, 500, 1200, 2000, 3000]` bins) |

## Open questions / TODO

- The Kita raw sumstats file (`pnas.1717421114.sd01.txt`) is not committed
  to `data/raw/eQTL/`. Decide whether to vendor it (it is small) or
  download-on-demand from PNAS.
- Add the Kita et al. citation in canonical form.
- **Verify the "Kita is more independent" claim.** Three things would
  help: (a) check whether Kita's 85-isolate RNA-seq is part of the
  pretraining set for Shorkie / Yorzoi; (b) compute HashFrag-style
  sequence overlap between Kita positives and pretraining windows;
  (c) compare AUROC degradation when sequence context is shuffled for
  Kita vs Caudal. Until at least one of these is done, the "more
  independent" framing is a hypothesis, not a property of the benchmark.
- Confirm the per-context labels in the upstream PNAS file. The Shorkie
  reproduction selects 683 variants by context, but the exact column name
  and category encoding from `pnas.1717421114.sd01.txt` is not yet
  documented in this entry; we should pin it before shipping `v1`.
- Confirm the negative-generation script's `#Gene` column expectation
  matches the actual upstream file (the literal `#` prefix is unusual and
  may have been introduced by a preprocessing step).
- Decide whether to ship a context-matched negative set in `kita_eqtl_v2`.
  This would neutralize the largest known cheat at construction time but
  changes the dataset shape, so it warrants a version bump and a decision
  about whether the cooked distribution hosts both v1 and v2 in parallel.
- The Shorkie variant-scoring script
  (`scripts/eqtl/2_variant_scoring/score_variants_shorkie.py`) is
  currently hardcoded against Kita's column conventions. Generalize it
  to consume the v1 cooked schema directly so the same script works for
  both Kita and Caudal without per-dataset edits.
- Once a third eQTL benchmark exists (likely Renganaath), extract the
  shared infrastructure sections (Distribution, Schema, Model contract,
  Evaluation protocol) into `benchmarks/_template_eqtl.md` and have
  Caudal/Kita reference the template instead of cross-referencing each
  other.
