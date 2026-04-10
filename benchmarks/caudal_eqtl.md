# Caudal et al. — yeast cis-eQTL classification

> **Status:** draft. This is the worked example used to develop the benchmark
> entry template — fields and wording are still in flux.

## At a glance

| | |
| --- | --- |
| **Task** | Binary classification: is this variant a cis-eQTL for its target gene, or a distance-matched non-eQTL control? |
| **Source** | Caudal et al. (TODO: full citation + DOI). GWAS summary statistics: `GWAS_combined_lgcCorr_ldPruned_noBonferroni_20221207.tab`, downloaded from the [1002 Yeast Genome Project](http://1002genomes.u-strasbg.fr/files/RNAseq). |
| **Reference assembly** | *S. cerevisiae* R64-1-1, Ensembl release 115 (`Saccharomyces_cerevisiae.R64-1-1.115.gtf`) |
| **Background population** | 1011 yeast isolates panel (`1011Matrix.gvcf`, [1002 Yeast Genome Project](http://1002genomes.u-strasbg.fr/files/)) |
| **Positives** | ~1,901 *local* (cis) eQTLs: variants where the SNP and the regulated gene's TSS lie on the same chromosome and within 8,000 bp of each other. LD-masked variants (`ld_mask == "masked"`) and any variant absent from the 1011 panel are dropped. |
| **Negatives** | Non-coding common variants (AF ≥ 0.05) from the same panel, REF/ALT-matched and distance-to-TSS-matched to ±100 bp (fallback ±200 bp), each iteration drawing a fresh independent set. Four negative sets are generated and reported on. |
| **Primary metric** | AUROC and AUPRC, evaluated **without class balancing** (natural prevalence within each negative set), reported as mean ± SEM across the four negative-set iterations. Stratification by distance-to-TSS bin is reported as a standard secondary view. |
| **Leakage status** | ⚠️ **Sequence leakage for any model pretrained on the R64-1-1 reference.** This includes Shorkie *and* Yorzoi: both have seen the exact flanking context of every positive variant during pretraining. The DREAM Challenge MPRA models (DREAM-CNN, DREAM-RNN, DREAM-Atten) are *not* affected — they are trained from scratch on MPRA data — and serve as the leakage-free reference point. See [Known cheats #1](#known-cheats). |

## Why this benchmark exists

Caudal et al. is the largest publicly available set of statistically called
yeast cis-eQTLs (~1,901 local hits across ~1,000 isolates), with effect sizes
and per-gene phenotype annotations. Used as a binary classification task it
asks: *given a variant and a candidate target gene, can a sequence model rank
true eQTLs above distance-matched controls drawn from the same population?*

The benchmark's primary use is **comparing yeast sequence-to-expression models
against each other and against MPRA-trained baselines**. The leakage caveat
above is real but does not invalidate the comparison — it changes how the
numbers should be read:

- **Pretrained-on-yeast models vs each other** (e.g. Yorzoi vs Shorkie): both
  models have seen the test sequences in pretraining, so neither has a
  held-out advantage. The comparison is honest in the sense that both are
  exposed to the same leakage, but it still measures something closer to "how
  well did each model retain locus-specific signal from pretraining" than
  "how well does each model generalize to unseen variants".
- **Pretrained-on-yeast models vs DREAM MPRA models**: DREAM models are
  trained from scratch on MPRA data with no exposure to R64-1-1. The gap
  between a pretrained model and a DREAM model is interpretable as the value
  of pretraining *plus* whatever leakage advantage it confers. Pairing this
  benchmark with one that has cleaner held-out splits (Kita et al.) is the
  way to disentangle the two.

We keep Caudal in the suite because (a) it is the largest available cis-eQTL
set for yeast, (b) all current models in scope can be evaluated on it without
modification, and (c) the leakage gap *itself* is informative when compared
across benchmarks.

## Dataset construction

### Positive set
1. Start from `GWAS_combined_lgcCorr_ldPruned_noBonferroni_20221207.tab`
   (provided by Caudal et al., downloaded from the 1002 Yeast Genome
   Project). The filename describes the upstream processing: logistic genomic
   control correction, LD pruning, no Bonferroni cut applied (filtering on
   `PValue` is left to the user).
2. Drop rows where `ld_mask == "masked"` (LD-pruned-out variants).
3. Drop rows with missing or non-positive `PValue`.
4. Classify each variant as **local (cis)** if the SNP and the regulated
   gene's TSS are on the same chromosome (`Chr == Pheno_chr`) and within
   8,000 bp of each other (`|ChrPos − Pheno_pos| ≤ 8000`). The benchmark uses
   the local set only — non-local (trans) variants are not part of this
   benchmark. This reproduces the Shorkie paper's `is_local` definition; see
   `scripts/eqtl/0a_gwas_preprocessing/1_snp_position.py`.
5. Intersect with the 1011 panel gVCF on `(Chr, ChrPos)` to retain only
   variants that are observed segregating in the population. Variants in the
   eQTL set that are absent from the gVCF are dropped.
6. Each retained row provides a `(chrom, pos, ref, alt, gene)` 5-tuple. The
   gene is the phenotype the eQTL is called against (`Pheno` column).

After step 5 the local set contains ~1,901 cis-eQTLs. Caudal further
distinguishes SNP and CNV `subtype` rows; the published Shorkie evaluation
uses both but the SNP/CNV split is reported separately in supplementary
figures, and we adopt the same convention.

### Negative set
Negatives are generated by `scripts/eqtl/0_data_generation/1_generate_negs.py`.
For each positive eQTL `(chrom, pos, ref, alt, gene)`:

1. Compute distance from `pos` to the TSS of `gene` (parsed from the GTF).
2. Restrict candidate negatives to non-coding variants in the 1011 panel
   gVCF with MAF > 0.05 and **identical (ref, alt)** alleles.
3. Pick one candidate negative variant whose position lies at the same
   distance from the TSS of *some randomly chosen gene on the same
   chromosome*, within ±100 bp tolerance (fallback ±200 bp). The randomly
   chosen gene becomes the negative variant's "target gene".
4. Reject candidates that match a known positive `(chrom, pos, ref, alt)` or
   that have already been used in the current negative-set iteration.
5. Repeat for `--iterations` independent negative sets (CLI default 4).

The output is a TSV per iteration with paired `(positive, negative)` rows
sharing distance-to-TSS, REF, and ALT.

**Properties of this matching scheme:**
- ✅ REF/ALT distribution is identical between positives and negatives.
- ✅ Distance-to-TSS distribution is matched within tolerance.
- ✅ Negatives are MAF > 5%, so the model cannot trivially distinguish
  positives from rare/private variants.
- ⚠️ The "target gene" assigned to a negative is a *random* gene on the same
  chromosome at the matched TSS distance — it is not necessarily a gene whose
  expression is plausibly affected by that variant. This is a deliberate
  design choice (it gives every negative a well-defined gene context for
  scoring), but see [Known cheats #4](#known-cheats).
- ⚠️ Negatives are restricted to non-coding regions; positives are not. If
  Caudal's positives include any coding variants, this asymmetry is itself a
  feature a model could learn.

## Distribution

The benchmark is split into two layers so that adding a new model does not
require running the upstream pipeline:

- **Raw upstream** — the original GWAS sumstats, the 1011 panel gVCF, and
  the Ensembl GTF. These live under `data/raw/` and are reproduced from the
  1002 Yeast Genome Project links above. The pipeline that turns them into
  benchmark-ready files lives in `scripts/eqtl/`.
- **Cooked benchmark distribution** — four flat TSV files, one per
  negative-set iteration, plus a bundled reference FASTA and GTF. Hosted at
  TBD (GCS bucket vs. HuggingFace Datasets).

**Adapters consume the cooked distribution.** The pipeline is provided for
provenance and reproducibility; it is not on the critical path for adding a
new model.

### Cooked file layout

```
caudal_eqtl_v1/
├── README.md          # version, generation date, source commit
├── reference/
│   ├── R64-1-1.fa     # bundled reference FASTA, indexed
│   ├── R64-1-1.fa.fai
│   └── R64-1-1.115.gtf
├── negset_1.tsv       # ~3,802 rows: ~1,901 positives + ~1,901 negatives
├── negset_2.tsv
├── negset_3.tsv
└── negset_4.tsv
```

Each `negset_{i}.tsv` is the complete classification problem for that
iteration. An adapter scores all four files; the harness computes
per-iteration AUROC/AUPRC and reports mean ± SEM across iterations.

### Schema

One row per variant. The same chromosomal position can appear multiple times
in a single file (once as a positive paired with its real target gene,
separately as a negative paired with a randomly chosen same-chromosome gene
at a matched distance), so the unit of identification is
`(variant, target_gene)`, not `(chrom, pos)` alone.

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `variant_id` | str | ✅ | `{chrom}:{pos}:{ref}>{alt}:{gene}`. Primary key within a negset file. Not unique across negsets, since the same negative variant may be re-paired with a different randomly chosen target gene in another iteration. |
| `chrom` | str | ✅ | Roman-numeral form (`I`, `II`, …, `XVI`), no prefix. Matches the GTF and reference FASTA naming. |
| `pos` | int | ✅ | 1-based, on R64-1-1. |
| `ref` | str | ✅ | Reference allele. |
| `alt` | str | ✅ | Alternate allele. |
| `gene` | str | ✅ | Ensembl gene ID of the target gene. For positives this is the eQTL's regulated gene; for negatives it is the randomly chosen same-chromosome gene at the matched TSS distance. |
| `gene_strand` | str | ✅ | `+` or `−`, the target gene's strand from the GTF. Saves the adapter from a GTF lookup. |
| `is_positive` | bool | ✅ | `True` for positive eQTLs, `False` for matched negatives. The benchmark label. |
| `distance_to_tss` | int | ✅ | Unsigned bp distance from `pos` to the target gene's TSS. Already produced by `1_generate_negs.py`; consumed by the stratified eval. |
| `pair_id` | int | ⭕ optional | Within a negset file: links each negative to its source positive (both rows share the same `pair_id`). Reserved for future paired diagnostics; no current eval script uses it. |

Conventions locked in for v1:
- **Chromosome naming**: Roman numerals, no prefix (`I`, `II`, …, `XVI`).
  This matches the R64-1-1 GTF and FASTA. The pipeline normalizes the
  Caudal CSV's integer chromosomes and the 1011 gVCF's `chromosome{N}`
  form to this canonical convention.
- **Coordinates**: 1-based, inclusive (matches the GTF and the source
  CSV's `ChrPos`).
- **Sort order**: rows sorted by `(chrom, pos, is_positive desc)`.
- **Booleans**: serialized as `True` / `False` in the TSV.

### Versioning

The cooked distribution is versioned (`caudal_eqtl_v1`, `_v2`, …). A
version bump is required whenever any of: the source upstream data, the
cis threshold, the negative-generation parameters, or the schema changes.
Adapters record which benchmark version they ran against in their reported
numbers.

## Model contract

A model is evaluated on this benchmark by exposing an **adapter** that scores
variants. The adapter contract is intentionally framework-agnostic — the
benchmark does not import the model, and the model does not need to be in
PyTorch.

**Two adapter forms are supported:**

1. **Python callable** — a function with signature
   `score_variant(chrom: str, pos: int, ref: str, alt: str, gene_id: str) -> float`
   that the benchmark imports and calls. Use this if your model is
   Python-importable (PyTorch, TensorFlow, JAX, ONNX, etc.).
2. **CLI / TSV form** — a command that reads a TSV of
   `(chrom, pos, ref, alt, gene_id)` rows from stdin (or a file path) and
   writes a TSV with an added `score` column to stdout (or a file path). Use
   this if your model lives behind a different language, a Docker container,
   a remote endpoint, or a build system the benchmark cannot import directly.

Both forms produce the same artifact downstream: a TSV of scored variants
that the evaluation step consumes.

**Per-variant inputs the adapter receives:**

| Field | Required | Notes |
| --- | --- | --- |
| `chrom`, `pos`, `ref`, `alt` | yes | 1-based, on the R64-1-1 assembly. |
| `gene_id` | yes | The annotated target gene; needed by adapters whose scoring depends on a gene-specific output window. The reference is the Ensembl release 115 GTF. |

**Resources the adapter is given access to once at startup:**
the R64-1-1 reference FASTA and the Ensembl 115 GTF.

**Output:** a single scalar variant effect score per `(variant, gene_id)`
pair. The benchmark uses the score as a classifier rank — sign convention is
adapter-defined, but the score (or `|score|`, for unsigned scoring functions)
must be monotonic in "how strongly the model thinks this variant affects this
gene's expression". The adapter must document its sign convention.

**What the adapter is responsible for:**
- Choosing the input window length and centering.
- Strand handling.
- Verifying the reference allele in the FASTA matches the `ref` field at
  `pos` and failing loudly if not.
- Choosing which output track(s) to read variant effect from (relevant for
  models that expose multiple expression-related tracks; see
  [open questions](#open-questions--todo)).
- Computing the ref-vs-alt comparison and reducing it to a scalar.
- Returning a finite scalar even when the variant is near a chromosome edge.

The benchmark **does not** require a specific scoring function. It only
requires the adapter to commit to one and document it.

### Reference example: Shorkie's scoring function

For grounding, the canonical Shorkie scoring procedure (used in the figures
of the Shorkie paper) is:

1. Extract a **16,384 bp** window centered on the variant from the R64-1-1
   reference.
2. Verify that the reference base at `pos` matches the `ref` field.
3. One-hot encode both the reference and the alternate sequences.
4. Predict expression coverage tracks for both, **averaging predictions over
   the forward and reverse-complement strands**. Predictions are
   ensemble-averaged across 8 trained folds (`f0c0`–`f7c0`).
5. Sum predicted coverage over all output bins overlapping the annotated
   exons of the target gene `g`, yielding `Cov_ref` and `Cov_alt`.
6. Score `= log2(Cov_alt + 1) − log2(Cov_ref + 1)` (the log2 fold change of
   summed gene-body coverage).

See `scripts/eqtl/2_variant_scoring/score_variants_shorkie.py`. Other models
do not need to mimic this; it is documented here as one concrete realization
of the contract.

## Evaluation protocol

For each of the four negative-set iterations `i ∈ {1..4}`:
1. Score every positive and every paired negative with the model.
2. Compute AUROC and AUPRC over the union of positives and negatives in
   iteration `i`. **Do not subsample to balance classes** — the natural
   prevalence is informative.
3. Record per-iteration metrics.

**Primary report:** mean ± SEM across the four iterations, plus the
per-iteration ROC and PR curves interpolated to a common grid and shown with
±1 SEM bands (see
`scripts/eqtl/3_visualization/1_roc_pr_shorkie_fold.py`).

**Standard secondary report:** AUROC and AUPRC stratified by distance-to-TSS
bin (see `scripts/eqtl/3_visualization/2_AUROC_AUPRC_by_dsitance.py`). This
is reported as Figure 7G–H of the Shorkie paper and we treat it as part of
the canonical evaluation, not an optional add-on.

## Known cheats

Ways a model can score well on this benchmark without actually learning
variant effects on expression. Listed roughly in decreasing order of how
plausible / load-bearing we think each one is.

1. **Sequence memorization (leakage).** Any model whose pretraining corpus
   contains the R64-1-1 reference has seen every positive variant's flanking
   sequence verbatim. This affects Shorkie *and* Yorzoi and any future
   yeast-pretrained model. Such a model can in principle memorize "this
   sequence neighborhood is associated with strong eQTL signal in the
   training species" without ever modeling the variant effect. The benchmark
   cannot distinguish a memorized hit from a learned variant effect on its
   own. Two things make it interpretable anyway: (a) the DREAM MPRA models
   are not pretrained on R64-1-1 and serve as a leakage-free reference; (b)
   the same models can be evaluated on Kita et al., which has a different
   leakage profile, and the gap between the two benchmarks is informative.
   For pretrained-on-yeast models specifically, expect to pair this benchmark
   with a HashFrag-style train/test sequence overlap report before publishing
   absolute numbers.

2. **Gene-identity shortcut.** Negatives are paired with a *random* same-chrom
   gene at the matched TSS distance, so a model that ignores the variant
   entirely and just learns "gene X is highly expression-variable across
   isolates" can rank positives above negatives whenever Caudal's positive
   genes are over-represented among variable genes. This is an asymmetry of
   the negative-construction scheme, not of the underlying biology.

3. **Distance-to-TSS shortcut.** Distance is matched to ±100 bp tolerance, but
   the tolerance is loose enough that a model that simply assigns higher
   scores to variants closer to the TSS can still gain a small advantage if
   the matching residual is non-zero on average. The expected magnitude is
   small but non-zero.

4. **Coding/non-coding asymmetry.** Negatives are required to be non-coding;
   positives are not filtered. If any Caudal positives fall in coding
   regions, a model that detects coding context can exploit this directly.
   (TODO: audit Caudal positives for coding overlap and either filter them
   out or accept the asymmetry explicitly.)

5. **Reference-allele bias.** Standard variant scoring (ref vs alt) compares
   two sequences, both of which are syntactically valid genomes. A model that
   has a prior favoring the reference allele (e.g. because pretraining
   sequences look more like ref than alt) will produce systematic
   ref→alt shifts that are mostly noise. This is not a "cheat" so much as a
   confound — a well-calibrated adapter should correct for it.

## Paired diagnostics

> **Status:** scoping. We expect each cheat above to be addressable by either
> a stratified analysis on Caudal itself or by a sibling benchmark with
> different construction. This section will be filled in once the suite has
> at least two eQTL benchmarks committed.

Initial candidates:
- **For (1) leakage:** report Caudal alongside Kita et al. as a held-out
  comparison; quantify train/test sequence overlap with HashFrag.
- **For (2) gene-identity shortcut:** ablate by scoring each variant against
  a *shuffled* gene-target assignment and reporting the residual AUROC drop.
- **For (3) distance shortcut:** report AUROC stratified by distance-to-TSS
  bin (script already exists at
  `scripts/eqtl/3_visualization/2_AUROC_AUPRC_by_dsitance.py`).
- **For (4) coding asymmetry:** rerun with positives intersected against the
  same coding mask as negatives.
- **For (5) reference bias:** report ref-vs-shuffled-alt as a null distribution.

## Files

| Purpose | Path |
| --- | --- |
| Raw GWAS sumstats (Caudal) | `data/raw/eQTL/GWAS_combined_lgcCorr_ldPruned_noBonferroni_20221207.csv` ([source](http://1002genomes.u-strasbg.fr/files/RNAseq)) |
| Background gVCF (1011 panel) | `data/raw/eQTL/1011Matrix.gvcf` ([source](http://1002genomes.u-strasbg.fr/files/)) |
| Reference GTF | `data/raw/Saccharomyces_cerevisiae.R64-1-1.115.gtf` |
| Processed positives (CIS) | `data/processed/eQTL/GWAS/GWAS_combined_lgcCorr_ldPruned_noBonferroni_20221207_cleaned_CIS.tab` |
| Negative sets + per-model scores | `data/processed/revision_experiments/eQTL/` |
| Cooked benchmark distribution | `caudal_eqtl_v1/` (TBD: GCS bucket or HuggingFace Datasets) — see [Distribution](#distribution) |
| Cis-classification step (`is_local`) | `scripts/eqtl/0a_gwas_preprocessing/1_snp_position.py` |
| Negative-set generation | `scripts/eqtl/0_data_generation/1_generate_negs.py` |
| Shorkie variant scoring | `scripts/eqtl/2_variant_scoring/score_variants_shorkie.py` |
| Evaluation / curves | `scripts/eqtl/3_visualization/1_roc_pr_shorkie_fold.py` |
| TSS-distance stratified eval | `scripts/eqtl/3_visualization/2_AUROC_AUPRC_by_dsitance.py` |

## Open questions / TODO

- Add the Caudal et al. citation and DOI.
- Decide where the cooked distribution will be hosted (GCS bucket vs.
  HuggingFace Datasets) and pin a release URL for `caudal_eqtl_v1`.
- For the CLI/TSV adapter form: confirm whether the adapter is expected to
  read the bundled `reference/` directory directly, or be passed FASTA and
  GTF paths at invocation time. The row schema is settled; only the
  invocation contract is still open.
- Yorzoi exposes multiple expression-related output tracks (RNA-seq +
  Nanopore + others depending on training); decide whether the Yorzoi
  adapter sums over a fixed RNA-seq subset (most directly comparable to
  Shorkie) or over a model-author-chosen track set, and document whichever
  it is. This is the first cross-family case where the model contract has
  to confront real heterogeneity.
- Audit Caudal positives for coding-region overlap. Negatives are restricted
  to non-coding by construction; if a non-trivial fraction of positives are
  coding, the asymmetry is itself a learnable cheat (Known cheats #4) and
  we should either filter positives or accept and document the asymmetry.
- HashFrag (or equivalent) train/test sequence overlap report for each
  pretrained-on-yeast model in scope, to quantify Known cheats #1.
