# Chen et al. — Synonymous-mutation MPRA (mRNA level, codon-resolution)

> **Status:** spec, ready to implement. Three libraries; same custom
> integration construct; one benchmark class parameterized by library
> name. Headline metric is per-library Pearson *r* on `log2(R/D)`.

## At a glance

| | |
| --- | --- |
| **Task** | Regression: predict scalar mRNA level (log2 read-count ratio `log2(R/D)`) for ~4,079 *S. cerevisiae* synonymous variants of three genes inserted at a shared chromosome II integration construct. Each variant changes only a 36 nt block (12 synonymous codons); the surrounding ~1.7 kb of construct sequence is identical across the variants of one library. |
| **Source** | Chen S, Li K, Cao W, *et al.* 2017. *Codon-Resolution Analysis Reveals a Direct and Context-Dependent Impact of Individual Synonymous Mutations on mRNA Level*. **Molecular Biology and Evolution** 34(11):2944–2958. DOI: [10.1093/molbev/msx229](https://doi.org/10.1093/molbev/msx229). Open Access. |
| **Assay** | Pooled barcode-free MPRA in BY4742-derived haploid yeast. The strain's chrII has GAL7's CDS replaced with `dTomato` and GAL1's CDS replaced with the variant cassette `PGAL1-{GFP\|TDH3}-TADH1-LEU2-TGAL1`. Galactose (2 %) co-induces the variant gene from `PGAL1` and the normalizer `dTomato` from `PGAL7`. Total RNA → cDNA → variable-region amplicon → Illumina HiSeq 2500 (R count); genomic DNA → variable-region amplicon → Illumina (D count). Per-variant mRNA level = `R/D`. |
| **Libraries** | Three, sharing the same integration construct. **GFP r1**: 1,124 variants in GFP codons 41–52 (`TTRACNTTRAARTTYATYTGYACNACNGGNAARTTR`). **GFP r2**: 2,432 variants in GFP codons 156–167 (`CARAARAAYGGNATYAARGTNAAYTTYAARATYAGR`). **TDH3**: 523 variants in TDH3 codons 57–68 (`GARGTNTCNCAYGAYGAYAARCAYATHATHGTNGAY`). |
| **Expression labels** | Per-library z-centred `log2(R/D)`, **shipped per replicate** (not merged): `log2mRNA_rep1` and `log2mRNA_rep2` (GFP r1, GFP r2) and a single `log2mRNA` column for TDH3 (which Chen supplies pre-averaged in supp Table S9). Replicates correlate at *r* = 0.83 (GFP r1), 0.73 (GFP r2), and 0.72 (TDH3) — the TDH3 number caps achievable Pearson on that library at ~0.72. |
| **Bonus labels** | **Protein level** (GFP r1, GFP r2): FACS-seq across 7 bins of `GFP/dTomato` ratio, per-variant weighted mean of bin medians. **mRNA degradation rate** (GFP r1 only): slope of −ln(mRNA_t / mRNA_0) vs *t* over 7 timepoints (0, 5, 10, 20, 40, 80, 160 min) after thiolutin addition; 1,076 of the 1,124 variants. |
| **Primary metric** | **Both** Pearson *r* and Spearman ρ of `(pred, log2mRNA)`, reported side-by-side — neither alone is the headline. Computed separately against each replicate column (`rep1`, `rep2`) on the two-replicate libraries (GFP r1 / GFP r2) and against the single column on TDH3. We deliberately do **not** merge replicates into one label. **TDH3-specific caveat:** CAI on TDH3 is heavy-tailed (the top ~10 % of CAI values pulls Pearson r from 0.39 → 0.67 by leverage), so Pearson and Spearman diverge dramatically only on that library. Chen 2017 itself reports Spearman; we report both so the comparison is unambiguous either way. |
| **Adapter protocol** | New: `LocalCodingVariantPredictor` — see below. Reuses the marginalized-MPRA logSED machinery internally; the protocol surface is "given a list of (library, 36 nt variable block) pairs, return scalars". |

## Why this benchmark exists

The Rafi promoter MPRA, the Shalem terminator MPRA, and the Wu RFP-cassette benchmark all probe *cis* elements either in the UTR or in the integration locus. **None of them probe the coding sequence itself** — yet the coding sequence is the largest block of training-distribution mismatch for genomic models (Shorkie / Yorzoi were trained on native CDSs; the heterologous GFP CDS used here is OOD for the GFP libraries but in-distribution for TDH3).

Chen 2017 gives us a controlled probe of that signal:

1. **Tight perturbation.** Only 36 nt change per variant. Models that pick up codon-usage / mRNA-stability signal will rank-order correctly; models that don't will produce uncorrelated noise.
2. **Three libraries, one construct.** A model's behaviour can be compared across (a) two regions of the same heterologous CDS (GFP r1 vs GFP r2 — should agree if the signal is real and not region-specific), and (b) heterologous vs endogenous CDS (GFP vs TDH3).
3. **Reproducibility ceiling published.** Replicate-replicate Pearson is reported per library, so we have a hard upper bound for what any predictor can achieve.
4. **Hard for our models.** The two regions of GFP cover 36 nt out of a ~720 nt CDS, embedded inside an entirely synthetic locus. Shorkie/Yorzoi may resolve nothing here — that's an informative negative result that the benchmark can publish.

## The construct (shared across all three libraries)

The Chen libraries do not live at any native locus. The strain BY4742 is modified to replace two chrII CDSs:

```
                            chrII (S. cerevisiae R64-1-1)
       ...GAL7 promoter ──┬─ dTomato ─┬── GAL1 promoter ─┬── {GFP | TDH3} variant CDS ──┬── ADH1 term. ── LEU2 marker ── GAL1 term. ── (downstream native chrII)
                          |           |                  |                              |
   YBR018C (GAL7) CDS replaced       (between the two)  YBR020W (GAL1) CDS replaced     ...the cassette extends ~3.0 kb past the original GAL1 stop codon
```

Both `PGAL1` and `PGAL7` are kept intact and natively GAL4-induced. `dTomato` provides cell-by-cell normalization for galactose induction; only the variant gene's mRNA / protein is the readout.

**Implication for genomic-model adapters:** the input window cannot be cut from the unmodified R64-1-1 FASTA. The benchmark ships a **synthetic chrII FASTA + GTF** with the integration spliced in (see *Files*), and the adapter loads that instead of `R64-1-1.fa`. The synthetic FASTA is identical to R64-1-1 outside the chrII GAL1–GAL7 interval (≈275,000–283,000 bp); inside, it carries the construct.

### Variable-region site for each library

After splicing, the construct has one well-defined locus per library:

| Library | Gene in construct | Codon range (0-based protein positions) | nt offset of variable block (0-based, from CDS start) | Block length |
| --- | --- | --- | ---: | ---: |
| GFP r1 | GFP   | 41–52   | 123 | 36 nt |
| GFP r2 | GFP   | 156–167 | 468 | 36 nt |
| TDH3   | TDH3  | 56–67   | 168 | 36 nt |

(0-based protein positions count from the start Methionine. Chen's prose says "GFP codons 41–52" but his peptide identity `LTLKFICTTGKL` puts the variable block at 0-based residues 41–52 — i.e., 3 × 41 = 123 nt past the start codon. We use the peptide identity, not the prose codon numbers, as the source of truth.)

The 36 nt is substituted in directly — no flanking insert / scaffold. Everything upstream and downstream is fixed.

## Adapter protocol

```python
@runtime_checkable
class LocalCodingVariantPredictor(Protocol):
    """Predict scalar expression for synonymous (or local) coding-region
    variants of a single construct gene. Given a library name + a 36-nt
    variable block per variant, return one scalar per variant in
    adapter-defined units (e.g. logSED relative to the most-common-codon
    reference); the benchmark only requires monotone correspondence to
    measured expression for Pearson scoring.

    Adapter contract:
    - Loads the construct FASTA + GTF at init (paths come from the task,
      same dispatcher path as Rafi marginalized + Shalem).
    - Knows the variable-region locus for each library_id it advertises
      (a small per-library dict committed alongside the adapter; lives
      next to `_marginalized_mpra.py` etc.).
    - Applies each `variant_seq` (36 nt) to the construct at the right
      offset, runs a logSED-style forward pass over the construct CDS
      bins, returns a scalar."""

    def predict_local_variants(
        self,
        library_ids: Sequence[str],   # one of {"chen_gfp_r1", "chen_gfp_r2", "chen_tdh3"}
        variant_seqs: Sequence[str],  # 36 nt each
    ) -> np.ndarray: ...
```

`library_ids` is sequence-aligned with `variant_seqs`; the same adapter can score all three libraries in a single call (the benchmark currently makes one call per library, but the protocol allows mixing). Adapters are free to internally batch per-library to amortize the construct context computation.

### Why not reuse `MarginalizedSequenceExpressionPredictor`?

Marginalized scoring inserts the **same** sequence at **many** host-gene contexts and means the logSED. Chen's task is the inverse: **many** sequences at **one** context per library. Forcing it through the marginalized protocol with N=1 host gene would erase the signal that distinguishes "one site, varying inserts" from "many sites, one insert" and would lock the protocol semantics to a special case.

### Why not reuse `VariantEffectScorer`?

The eQTL protocol assumes ref/alt single-nucleotide variants. Chen variants differ in up to 12 nt simultaneously; encoding them as multi-nt VariantEffectScorer inputs is technically possible but loses the "one well-defined locus per library" structure that adapters want to exploit for caching.

## Baseline models

Two non-genomic baselines ship with v1 to establish a floor and to ground the headline numbers from genomic models. Both implement `LocalCodingVariantPredictor`. Neither needs the construct FASTA at inference time — they operate purely on the variant gene's protein and the 36 nt variable block. The registry surface and the `ybench compare` runner are identical for these adapters and the Shorkie/Yorzoi ones.

### CAI (Codon Adaptation Index)

We use the **`CAI` column the Chen authors ship in supp tables S7/S8/S9**, not a re-implementation. Per-variant CAI is precomputed in the supp tables against the authors' reference codon usage; our pipeline carries the column through to `{gfp_r1,gfp_r2,tdh3}.tsv`. The CAI baseline adapter simply reads that column and returns it as the per-variant score.

- **Why not recompute:** using the authors' values matches their reported ρ ≈ 0.3 by construction, removes the "which reference codon usage table" decision, and removes a code path that could drift from the paper.
- **Adapter:** `src/yeastbench/adapters/baselines/cai.py`, ~20 lines. Reads `data/tasks/chen_synonymous/{library}.tsv`'s `CAI` column at init, dispatches by `library_id` at predict time. No model dependency at all.
- **Limitation by design:** CAI has no position- or context-dependence — the same codon scores the same regardless of where it sits — so it cannot capture the position-specific effects the paper reports in Fig. 5 / supp S8. That's the point: a lower-bound floor for any context-aware model.

### CodonTransformer

Fallahpour *et al.* 2025 (*Nat Commun* 16:3205, doi [10.1038/s41467-025-58588-7](https://doi.org/10.1038/s41467-025-58588-7); GitHub: <https://github.com/Adibvafa/CodonTransformer>). A BigBird masked-LM trained on >1M protein–DNA pairs across 164 organisms including *S. cerevisiae*. Published as a codon optimizer (protein → optimized DNA), but the underlying HuggingFace `model(...)` forward pass exposes per-position codon logits that we can use to score arbitrary synonymous variants — no upstream patch required, only bypassing the `predict_dna_sequence()` convenience wrapper.

**Tokenizer recap.** Vocab is ~90 tokens: specials, per-aa "unknown codon" tokens (`k_unk`, …), and one token per (amino acid, codon) pair (`k_aaa`, `n_aac`, …). Output at position *i* is a distribution over (aa, codon) pairs; conditioning on the protein collapses this to a choice among the synonymous codons of that position's amino acid.

**Scoring (v1, fully-marginal approximation).** Per library, one forward pass; per variant, twelve tensor lookups:

```python
# at adapter init, per library_id: translate the variant gene's CDS from the
# construct → protein string of length L (239 aa for GFP, 332 aa for TDH3);
# cache the masked-LM forward pass over the all-unk merged sequence
merged = get_merged_seq(protein=protein, dna="")               # codons → *_unk
inputs = tokenizer(merged, return_tensors="pt", ...)
with torch.no_grad():
    log_p = model(**inputs).logits[0, 1:-1, :].log_softmax(-1)  # [L, vocab]

# at predict_local_variants: per variant, sum log_p over the 12 variable codons
def score(variant_codons, var_pos, protein):
    tok = [TOKEN2INDEX[f"{protein[var_pos[j]].lower()}_{variant_codons[j].lower()}"]
           for j in range(12)]
    return sum(log_p[var_pos[j], tok[j]].item() for j in range(12))
```

Three forward passes total at benchmark time (one per library, since each library's protein is fixed). Codons outside the variable block are identical across variants of one library, so they cancel — summing over only the 12 variable positions is sufficient for ranking and matches what the benchmark scores.

**Position-independence caveat — documented, not hidden.** A single masked-LM forward pass over the all-`*_unk` input gives per-position **marginals**, not the joint `log P(DNA | protein)`. Summing marginals across the 12 positions implicitly treats codon choices as independent given the protein. The bidirectional BigBird attention still lets each position's marginal condition on the full protein, so this is fine for ranking — but it isn't a formal sequence likelihood. The more faithful pseudo-likelihood (12 forward passes per library with flanking codons set to wild-type and only one variable position masked at a time) is a drop-in v2 swap if the v1 number looks suspiciously like CAI's; v1 keeps the cheap marginal version.

**Dependency surface.** CodonTransformer becomes an optional extra under `[project.optional-dependencies] baselines = ["CodonTransformer", "torch", ...]`; not pulled in by default `uv sync`. Adapter lives at `src/yeastbench/adapters/baselines/codon_transformer.py`.

### Why both, not just one

CAI is the literature reference floor — the Chen paper itself reports CAI's correlation (ρ ≈ 0.3) in passing. CodonTransformer is the strongest "no-genomic-context, codon-only" learned model available off the shelf. Reporting both lets us answer two distinct questions in the same figure:

1. **Does a genomic model (Shorkie / Yorzoi) extract anything beyond marginal codon usage?** → compare to CAI.
2. **Does a genomic model extract anything beyond what a dedicated codon-context model already knows?** → compare to CodonTransformer.

A genomic model that loses to CAI is a strong negative result; one that beats CAI but loses to CodonTransformer reads as "learned codon usage but not codon context"; one that beats CodonTransformer too is doing something neither baseline can — likely mRNA-stability or RNA-structure signal that depends on the surrounding construct.

## Evaluation protocol

For each of the three libraries, independently:

1. Read the per-variant TSV (`data/tasks/chen_synonymous/{gfp_r1,gfp_r2,tdh3}.tsv`).
2. Hand `(library_id, variant_seq)` for every row to the adapter; receive scalar prediction array.
3. Drop rows where the relevant `log2mRNA*` column is NaN (defensive — should be 0 drops on the committed distribution).
4. **Per-library Pearson *r* and Spearman ρ** on `(pred, log2mRNA)`, computed **separately per replicate** for the two-replicate libraries: `pearson_rep1` + `spearman_rep1` and `pearson_rep2` + `spearman_rep2` for GFP r1 / GFP r2; `pearson` + `spearman` for TDH3. Both metrics are reported side-by-side; on most adapters they agree to within a few decimals, but on TDH3 with CAI (and any other heavy-tailed scalar predictor) Pearson can be much higher than Spearman because of leverage from a small high-CAI tail. We deliberately do not pre-average replicates into one label — reporting both metrics against the published replicate-replicate ceiling is more informative than collapsing.
5. Plot: per-library scatter with regression line + reproducibility-ceiling band (the replicate-replicate Pearson reported by Chen). On the two-replicate libraries, two scatter panels per library — one per replicate column.

Across libraries:

6. Per replicate (rep1, rep2): z-score predictions per library (mean 0, std 1) and the same for measured `log2mRNA`. Concatenate and compute one **aggregated Pearson + Spearman**. For TDH3 (single replicate), use its column for both aggregated numbers (it contributes identically to the rep1 and rep2 aggregates).
7. Plot: three-panel scatter, each annotated with its per-replicate Pearsons and the replicate-ceiling.

### Bonus assays (optional v1 sub-targets)

| Assay | Libraries | Compute | Sign |
| --- | --- | --- | --- |
| Protein level | GFP r1, GFP r2 | Pearson + Spearman on `(pred, log2_protein)` | Same scalar prediction; protein is downstream of mRNA so direction should match. |
| mRNA degradation rate | GFP r1 only | Pearson + Spearman on `(pred, degradation_rate)` | **Negative**: high predicted mRNA level → low degradation rate. |

These are reported alongside the headline mRNA Pearson but **not** mixed into the aggregated headline number — they are separate columns in `summary.json`.

### Compare-task grouping

v1 keeps each library as its own compare group (`chen_gfp_r1` / `chen_gfp_r2` / `chen_tdh3`). The cross-model compare runner currently assumes one model per group; collapsing the three libraries under one shared `compare_task_name = "chen_synonymous"` confuses it (every model appears in all three sub-tasks). A single 3-panel "Chen panel" with three sub-rows is a v2 enhancement — it needs a custom `compare_plot` override on the benchmark class, which v1 does not ship.

### What we're *not* doing in v1

- **Per-codon-site ICE values.** The paper's Fig. 5 / Fig. 10 / supp Fig. S8 decompose effects to individual codon sites. Reportable from our per-variant predictions post-hoc, but adds a non-trivial chunk of plotting and per-codon test design — defer to v2.
- **Interaction-index (II) reproduction.** Fig. 7's pairwise II / G-tests are interesting but tangential to predictor benchmarking.
- **mRNA secondary-structure features** (Fig. 8) — the model is expected to learn them implicitly, we don't test them separately.
- **Bootstrap confidence intervals** (v2).
- **CodonTransformer pseudo-likelihood (12 masked forward passes per library).** v1 ships the cheap fully-marginal scoring; the per-position-masked version is a v2 swap if it materially separates from CAI.

## Files

### Raw upstream

- `papers/Codon-Resolution Analysis Reveals a Direct and Context-Dependent Impact of Individual Synonymous Mutations on mRNA Level_Chen_2017.pdf` — the article itself.
- `archive/chen/msx229_Supptables.xlsx` — the published MBE supp workbook (open access, from <https://doi.org/10.1093/molbev/msx229>). Tables S7/S8/S9 hold the per-variant rows: `variable_seq` (36 nt), pre-computed scalar features (`CAI`, `tAI` (S7/S8 only), `MFE`, `GC3`), per-replicate normalized `log2(mRNA)` (S7/S8) or single column (S9), per-replicate `log2(protein)` (S7/S8), and `degradation_rate` (S7 only).
- GSA accession `PRJCA000227` — raw Illumina reads (only needed if we want to rebuild counts from raw; v1 trusts the per-variant counts from the supp tables).

### Processed distribution (one-time build)

| File | Content |
| --- | --- |
| `data/tasks/chen_synonymous/construct_chrII.fa` | Modified chromosome II with the GAL7→dTomato + GAL1→cassette integration spliced in. All other chromosomes are byte-identical to `R64-1-1.fa`. Adapter passes this in place of the canonical fasta for Chen tasks. |
| `data/tasks/chen_synonymous/construct.gtf` | GTF entries for `dTomato`, `GFP_variant`, and `TDH3_variant` at the new coordinates (start, end, strand, CDS). Strand is +. Used by adapters to locate the variable region and define the CDS-binning window. |
| `data/tasks/chen_synonymous/library_loci.json` | `{library_id: {gene_id, cds_start_in_construct, var_start, var_end, ref_codons}}` — the per-library locus metadata an adapter needs to splice the 36 nt block into the construct. Committed once at distribution build time. |
| `data/tasks/chen_synonymous/gfp_r1.tsv` | 1,124 rows. Cols: `variant_id, variable_seq (36nt), CAI, tAI, MFE, GC3, log2mRNA_rep1, log2mRNA_rep2, log2protein_rep1, log2protein_rep2, degradation_rate`. CAI/tAI/MFE/GC3 are carried through from Chen's supp Table S7 verbatim. |
| `data/tasks/chen_synonymous/gfp_r2.tsv` | 2,432 rows. Cols: `variant_id, variable_seq, CAI, tAI, MFE, GC3, log2mRNA_rep1, log2mRNA_rep2, log2protein_rep1, log2protein_rep2` (no degradation column). |
| `data/tasks/chen_synonymous/tdh3.tsv` | 523 rows. Cols: `variant_id, variable_seq, CAI, MFE, GC3, log2mRNA` (single normalized column as published by Chen in supp Table S9; no tAI/protein/degradation). |
| `data/tasks/chen_synonymous/replicate_ceilings.json` | `{library_id: pearson_r_between_replicates}` from the paper (GFP r1: 0.83, GFP r2: 0.73, TDH3: 0.72). Used to plot the reproducibility ceiling band. |

### Build script (one-off, lives in `scripts/chen/`)

- `scripts/chen/build_construct_reference.py` — fetches the dTomato + GFP + TDH3 + ADH1term + LEU2 + GAL1term sequences (commit one-off in the repo: dTomato and GFP are vendored from common plasmid sources, ADH1term and GAL1term are pulled from R64-1-1, LEU2 is YCL018W in R64-1-1), splices them into a modified chrII, writes `construct_chrII.fa` + `construct.gtf` + `library_loci.json`.
- `scripts/chen/build_distribution_tsvs.py` — parses MBE supp tables S7–S9, normalizes `log2(R/D)` per library, writes the three TSVs and `replicate_ceilings.json`.

Neither script is invoked by the benchmark at run time — they're one-off pipeline steps whose outputs are committed under `data/tasks/chen_synonymous/`.

### Sequence-format sanity checks (must pass at distribution build time)

- Each row's `variable_seq` is exactly 36 nt.
- Each `variable_seq` matches the degenerate pattern declared by the paper for its library (e.g. `TTRACN…AARTTR` for GFP r1, `CARAARA…AARATYAGR` for GFP r2, `GARGTNT…GTNGAY` for TDH3) when checked against IUPAC codes.
- Each `variable_seq` translates to the published 12-aa peptide for its library (`LTLKFICTTGKL` for GFP r1, `QKNGIKVNFKIR` for GFP r2, `EVSHDDKHIIVD` for TDH3).
- `log2mRNA_rep1` / `log2mRNA_rep2` (S7/S8) and `log2mRNA` (S9) are mean-centred within library to ~0 (paper's normalization convention, applied per replicate column).

## Registry surface

Three task entries in `TASKS`, all built by the same factory class with a `library` keyword. Same factory style as Brooks (one benchmark class, two registry entries):

```yaml
tasks_config:
  chen_gfp_r1:
    library: gfp_r1
    data_path: data/tasks/chen_synonymous/gfp_r1.tsv
    fasta_path: data/tasks/chen_synonymous/construct_chrII.fa
    gtf_path: data/tasks/chen_synonymous/construct.gtf
    library_loci_path: data/tasks/chen_synonymous/library_loci.json
    replicate_ceiling_pearson: 0.83
  chen_gfp_r2:
    library: gfp_r2
    ...
  chen_tdh3:
    library: tdh3
    ...
```

A single Shorkie adapter and a single Yorzoi adapter cover all three libraries via the `library_ids` argument (no per-library adapter classes).

## Open questions for implementation phase

1. **dTomato and GFP source sequences.** The paper cites a tag-free dTomato (Shaner 2004) and a generic *Aequorea victoria* GFP. Need to commit canonical FASTA strings (length-verified) in `scripts/chen/build_construct_reference.py`. Should the variant GFP in the construct use the GFP-S65T variant common in yeast or the wild-type *A. victoria* GFP? — pick whichever matches the published synonymous-codon catalog in S7/S8; verify at build time that all 12 aa peptides at codons 41–52 and 156–167 match the paper.
2. **Logits / logSED window inside the construct.** Shorkie has a 16,384 bp receptive field; Yorzoi has 4,992 bp. The construct integration site sits in a region of chrII with no known native expression there in BY4742 (GAL1/GAL7 are silent without galactose), but the *flanking* native chrII genes are real — the receptive window will spill onto them. That's expected, and the logSED is computed over the construct gene's CDS bins only.
3. **Score sign for degradation rate.** A higher *predicted* mRNA level should imply *lower* measured degradation rate, so the Pearson sign on `(pred, degradation_rate)` is negative. Mirror the Wu RFP-pins benchmark's sign-aware AUC computation: report the absolute correlation and document the expected sign.
4. **What if the supp tables are gated or hard to parse?** GSA accession PRJCA000227 has the raw reads, but rebuilding R/D counts from raw is a separate (~few-days) project. The supp tables S7–S9 are the canonical resource; if they're inaccessible we revisit.
