# Chen et al. — Synonymous-mutation MPRA (mRNA level, codon-resolution)

![image](/img/chen_codon_banner.png)

Image from Chen et al. (2017)
## At a glance

| | |
| --- | --- |
| **Task** | Regression: predict scalar mRNA level (log2 read-count ratio `log2(R/D)`) for ~4,079 *S. cerevisiae* synonymous variants of three genes inserted at a shared chromosome II integration construct. Each variant changes only a 36 nt block (12 synonymous codons); the surrounding ~1.7 kb of construct sequence is identical across the variants of one library. |
| **Source** | Chen S, Li K, Cao W, *et al.* 2017. *Codon-Resolution Analysis Reveals a Direct and Context-Dependent Impact of Individual Synonymous Mutations on mRNA Level*. **Molecular Biology and Evolution** 34(11):2944–2958. DOI: [10.1093/molbev/msx229](https://doi.org/10.1093/molbev/msx229). Open Access. |
| **Assay** | Pooled barcode-free MPRA in BY4742-derived haploid yeast. The strain's chrII has GAL7's CDS replaced with `dTomato` and GAL1's CDS replaced with the variant cassette `PGAL1-{GFP\|TDH3}-TADH1-LEU2-TGAL1`. Galactose (2 %) co-induces the variant gene from `PGAL1` and the normalizer `dTomato` from `PGAL7`. Total RNA → cDNA → variable-region amplicon → Illumina HiSeq 2500 (R count); genomic DNA → variable-region amplicon → Illumina (D count). Per-variant mRNA level = `R/D`. |
| **Libraries** | Three, sharing the same integration construct. **GFP r1**: 1,124 variants in GFP codons 41–52 (`TTRACNTTRAARTTYATYTGYACNACNGGNAARTTR`). **GFP r2**: 2,432 variants in GFP codons 156–167 (`CARAARAAYGGNATYAARGTNAAYTTYAARATYAGR`). **TDH3**: 523 variants in TDH3 codons 57–68 (`GARGTNTCNCAYGAYGAYAARCAYATHATHGTNGAY`). |
| **Expression labels** | Per-library z-centred `log2(R/D)`, **shipped per replicate** (not merged): `log2mRNA_rep1` and `log2mRNA_rep2` (GFP r1, GFP r2) and a single `log2mRNA` column for TDH3 (which Chen supplies pre-averaged in supp Table S9). Replicate-replicate ceilings: GFP r1 — Pearson 0.83 / Spearman 0.71; GFP r2 — Pearson 0.73 / Spearman 0.71; TDH3 — Pearson 0.72 / Spearman *unknown* (S9 ships only the merged column, so we can't recompute, and Chen 2017 only reports a Pearson). |
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

## The construct, in the original experiment vs in v1 scoring

Chen 2017's strain integrates the variant gene cassette at the chrII GAL1 locus:

```
                            chrII (S. cerevisiae R64-1-1)
       ...GAL7 promoter ──┬─ dTomato ─┬── GAL1 promoter ─┬── {GFP | TDH3} variant CDS ──┬── ADH1 term. ── LEU2 marker ── GAL1 term. ── (downstream native chrII)
```

Galactose is required to induce the variant gene from PGAL1; in glucose conditions PGAL1 is Mig1-repressed to near-zero transcription.

**Why v1 does *not* score at this locus.** Both Shorkie and Yorzoi were trained exclusively on glucose / standard-condition RNA-seq tracks (verified: zero galactose RNA-seq tracks in either model's target sheet). Scoring at the construct's actual chrII locus asks the model to predict variant effects at a promoter it knows is silent, which adds locus-specific calibration noise: see the investigation notebooks (`notebooks/chen_{shorkie,yorzoi}_investigation.ipynb`) — both models correctly predict native unmodified GAL1 in glucose as near-zero coverage but predict the same locus with the GFP CDS spliced in as 11-29× higher. The variant-effect signal that *does* survive the model's confusion at this locus is **CDS-intrinsic codon usage**: changing the 36 nt variable block modulates predicted coverage similarly across nearly all of the model's tracks, regardless of whether the host promoter is firing.

**What v1 scores instead — marginalisation over 20 active YPD hosts.** For each variant, splice the variant gene's CDS + TADH1 into 20 native R64-1-1 host gene loci (replacing each host's CDS, keeping the host's promoter and downstream context), score each (variant, host) pair, and average. The codon-effect signal is locus-independent, so it transfers cleanly; the locus-specific calibration noise averages out. The marginalised prediction does **not** correspond to "what Chen would have measured if the experiment were done in this gene's locus"; it corresponds to "the model's codon-effect signal averaged across active-in-YPD chromatin contexts" — which is the quantity the model can actually compute.

### Host-gene panel (20 hosts)

Curated for YPD-log-phase activity (so logSED has signal-to-noise) and span weak / medium / high promoter strengths (so per-locus prediction noise integrates out):

| Tier (DEE2 median TPM) | Hosts |
| --- | --- |
| Low (10-50)            | VPS52, RGI1, GPM3, PKC1 |
| Medium (50-250)        | ALG9, HXT1, RPE1, TUB1, SEC61 |
| Medium-high (250-1000) | DPM1, HXT3, CTS1, RPL11A, IPP1 |
| High (1000+)           | ACT1, RPL25, PGK1, ENO2, FBA1, TDH3 |

Functional mix: glycolysis (5), ribosomal / translation (2), transport (2 HXT), cytoskeleton (2), ER/Golgi/biosynthesis (4), cell-cycle / division (1), signalling (2), PPP (1), housekeeping (4). 14 of 16 chromosomes represented. Stress-induced / condition-specific genes (HSP family, MSN4, GAL family) deliberately excluded — they look "expressed" in pooled DEE2 averages only because the DEE2 mix includes stress samples; in pure YPD log phase they're near-silent. Pinned in `data/tasks/chen_synonymous/marginalized_hosts.json`.

### Per-library variant-block position in the cassette

| Library | Variant gene CDS | 0-based protein positions of the variable block | nt offset of variable block from CDS start | Block length |
| --- | --- | --- | ---: | ---: |
| GFP r1 | wild-type avGFP CDS (L29345-corrected), 717 nt | 41–52   | 123 | 36 nt |
| GFP r2 | same GFP CDS as above (one cassette per gene)   | 156–167 | 468 | 36 nt |
| TDH3   | native R64-1-1 TDH3 CDS (YGR192C), 996 nt          | 56–67   | 168 | 36 nt |

(0-based protein positions count from the start Methionine. Chen's prose says "GFP codons 41–52" but his peptide identity `LTLKFICTTGKL` puts the variable block at 0-based residues 41–52 — i.e., 3 × 41 = 123 nt past the start codon. We use the peptide identity, not the prose codon numbers, as the source of truth.)

The 12-codon variable block is substituted in directly — no flanking insert / scaffold. For each (variant, host) pair, only this 36 nt region differs between REF and ALT; everything upstream (host promoter, 5' UTR) and downstream (TADH1, host 3' context) is fixed. For − strand hosts the cassette is reverse-complemented before splicing into the genomic + strand at the host's CDS span.

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
        library_ids: Sequence[str],   # one of {"gfp_r1", "gfp_r2", "tdh3"}
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

One task, `chen_synonymous`, evaluates all three libraries and reports results **stratified per library** plus a single aggregate. For each library, independently:

1. Read the per-variant TSV (`data/tasks/chen_synonymous/{gfp_r1,gfp_r2,tdh3}.tsv`).
2. Hand `(library_id, variant_seq)` for every row to the adapter; receive scalar prediction array. The benchmark makes one call per library, so each library's scores are identical to scoring it on its own.
3. Drop rows where the relevant `log2mRNA*` column is NaN (defensive — should be 0 drops on the committed distribution).
4. **Per-library Pearson *r* and Spearman ρ** on `(pred, log2mRNA)`, computed **separately per replicate** for the two-replicate libraries. Every metric is **library-prefixed** in `summary.json`: `gfp_r1_pearson_rep1` / `gfp_r1_spearman_rep1` / `gfp_r1_pearson_rep2` / … for the GFP libraries, `tdh3_pearson` / `tdh3_spearman` for TDH3, plus per-library `*_ceiling_pearson` (and `*_ceiling_spearman` where published) and `*_n_*` counts. Both metrics are reported side-by-side; on TDH3 with CAI (and any heavy-tailed scalar predictor) Pearson can be much higher than Spearman because of leverage from a small high-CAI tail. We deliberately do not pre-average replicates into one label — reporting both against the published replicate-replicate ceiling is more informative than collapsing.

Across libraries:

5. **Aggregate**: `pearson_mean` and `spearman_mean` = the nan-aware mean over the **five replicate columns** (GFP r1 rep1+rep2, GFP r2 rep1+rep2, TDH3's single column). The five individual numbers are always reported too; the mean is just a single headline figure (a missing replicate is skipped rather than poisoning it).
6. Plot: one `scatter.png` with five panels (one per library/replicate column), each a measured-vs-predicted scatter with regression line + the replicate-replicate ceiling band.

### Bonus assays (optional v1 sub-targets)

| Assay | Libraries | Compute | Sign |
| --- | --- | --- | --- |
| Protein level | GFP r1, GFP r2 | Pearson + Spearman on `(pred, log2_protein)` | Same scalar prediction; protein is downstream of mRNA so direction should match. |
| mRNA degradation rate | GFP r1 only | Pearson + Spearman on `(pred, degradation_rate)` | **Negative**: high predicted mRNA level → low degradation rate. |

These are reported alongside the headline mRNA Pearson but **not** mixed into the aggregated headline number — they are separate columns in `summary.json`.

### Cross-model comparison

`chen_synonymous` is a single registry task, so it forms one compare group. The default grouped-bar compare plot puts every model side-by-side across the curated headline metrics — the five per-replicate Pearsons, the five Spearmans, and the two aggregates (see `headline_metric_labels`). The full per-library table (including counts and ceilings) lands in `compare/summary.csv` and `compare/summary.md`.

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
| `data/tasks/chen_synonymous/marginalized_hosts.json` | List of 20 curated YPD-active host genes (`gene_id`, `gene_name`, `chrom`, `strand`, `cds_start`, `cds_end`, `tier`, `dee2_tpm`). The adapter splices the variant gene's CDS + TADH1 into each host's CDS span at run time; no construct FASTA is built ahead of time. |
| `data/tasks/chen_synonymous/gfp_r1.tsv` | 1,124 rows. Cols: `variant_id, variable_seq (36nt), CAI, tAI, MFE, GC3, log2mRNA_rep1, log2mRNA_rep2, log2protein_rep1, log2protein_rep2, degradation_rate`. CAI/tAI/MFE/GC3 are carried through from Chen's supp Table S7 verbatim. |
| `data/tasks/chen_synonymous/gfp_r2.tsv` | 2,432 rows. Cols: `variant_id, variable_seq, CAI, tAI, MFE, GC3, log2mRNA_rep1, log2mRNA_rep2, log2protein_rep1, log2protein_rep2` (no degradation column). |
| `data/tasks/chen_synonymous/tdh3.tsv` | 523 rows. Cols: `variant_id, variable_seq, CAI, MFE, GC3, log2mRNA` (single normalized column as published by Chen in supp Table S9; no tAI/protein/degradation). |
| `data/tasks/chen_synonymous/replicate_ceilings.json` | Per-library replicate-replicate Pearson and Spearman ceilings (Pearson from Chen 2017, Spearman empirical from rep1/rep2 columns; TDH3 ships a merged column so Spearman is `null`). Used to plot the reproducibility ceiling band. |

The model adapter reads `R64-1-1.fa` and `marginalized_hosts.json` at init, builds the cassette internally (variant gene CDS + TADH1), and splices into each host's CDS span on the fly. There is no pre-built construct FASTA.

### Build script (one-off, lives in `scripts/chen/`)

- `scripts/chen/build_distribution_tsvs.py` — parses MBE supp tables S7–S9, normalizes `log2(R/D)` per library, writes the three TSVs and `replicate_ceilings.json` (with Pearson and Spearman ceilings).

The hosts JSON (`marginalized_hosts.json`) was hand-curated against DEE2 median TPM — see the `## The construct, in the original experiment vs in v1 scoring` section for the selection criteria. Not produced by an auto-run script.

### Sequence-format sanity checks (must pass at distribution build time)

- Each row's `variable_seq` is exactly 36 nt.
- Each `variable_seq` matches the degenerate pattern declared by the paper for its library (e.g. `TTRACN…AARTTR` for GFP r1, `CARAARA…AARATYAGR` for GFP r2, `GARGTNT…GTNGAY` for TDH3) when checked against IUPAC codes.
- Each `variable_seq` translates to the published 12-aa peptide for its library (`LTLKFICTTGKL` for GFP r1, `QKNGIKVNFKIR` for GFP r2, `EVSHDDKHIIVD` for TDH3).
- `log2mRNA_rep1` / `log2mRNA_rep2` (S7/S8) and `log2mRNA` (S9) are mean-centred within library to ~0 (paper's normalization convention, applied per replicate column).

## Registry surface

One task entry, `chen_synonymous`, whose config carries a `libraries:` list (one entry per library, with its own `data_path` + replicate ceilings) plus the shared `fasta_path` / `hosts_path` / `data_dir`:

```yaml
tasks_config:
  chen_synonymous:
    fasta_path: data/tasks/R64-1-1.fa
    hosts_path: data/tasks/chen_synonymous/marginalized_hosts.json
    data_dir: data/tasks/chen_synonymous
    libraries:
      - library: gfp_r1
        data_path: data/tasks/chen_synonymous/gfp_r1.tsv
        replicate_ceiling_pearson: 0.83
        replicate_ceiling_spearman: 0.71
      - library: gfp_r2
        data_path: data/tasks/chen_synonymous/gfp_r2.tsv
        replicate_ceiling_pearson: 0.73
        replicate_ceiling_spearman: 0.71
      - library: tdh3
        data_path: data/tasks/chen_synonymous/tdh3.tsv
        replicate_ceiling_pearson: 0.72   # no Spearman ceiling (single merged column)
```

`evaluate` scores each library with its own `predict_local_variants` call. A single Shorkie adapter and a single Yorzoi adapter cover all three libraries via the `library_ids` argument (no per-library adapter classes): each builds its per-library host contexts + REF caches on demand and caches them, so a library's scores match a standalone single-library run.

## Open questions for implementation phase

1. **GFP source sequence — resolved (issue #8).** The GFP is **wild-type *A. victoria* GFP**, not the S65T variant and not a codon-optimised synthesis. Evidence: Chen's construction primers (supp Table S1) encode the wild-type residues at the positions that distinguish variants (E172, Q157), and the protein matches PDB 1EMA everywhere except the engineered chromophore (S65T/Q80R). The coding DNA is `GFP_CDS_WT` in `src/yeastbench/adapters/_chen_gfp_reference.py`: GenBank **L29345.1** corrected to the canonical avGFP protein, with residue 172 set to `GAA` to match Chen's primer. Chen never published the full construct DNA, so codons outside the Table S1 flanks are L29345-native (the flanks we *do* have match this sequence base-for-base). The variable regions are still overwritten per-variant from the TSVs. **Do not** revert to a preferred-codon back-translation — that scored models on the wrong DNA.
2. **Logits / logSED window inside the construct.** Shorkie has a 16,384 bp receptive field; Yorzoi has 4,992 bp. The construct integration site sits in a region of chrII with no known native expression there in BY4742 (GAL1/GAL7 are silent without galactose), but the *flanking* native chrII genes are real — the receptive window will spill onto them. That's expected, and the logSED is computed over the construct gene's CDS bins only.
3. **Score sign for degradation rate.** A higher *predicted* mRNA level should imply *lower* measured degradation rate, so the Pearson sign on `(pred, degradation_rate)` is negative. Mirror the Wu RFP-pins benchmark's sign-aware AUC computation: report the absolute correlation and document the expected sign.
4. **What if the supp tables are gated or hard to parse?** GSA accession PRJCA000227 has the raw reads, but rebuilding R/D counts from raw is a separate (~few-days) project. The supp tables S7–S9 are the canonical resource; if they're inaccessible we revisit.
