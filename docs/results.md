# Benchmark results

Cross-model comparisons, one section per benchmark. Each table shows the
benchmark's **primary metric(s)** for every model that has run; the leading
value is **bold** and named in the per-section *Winner* line.

> [!NOTE]
> These numbers are from the latest available local run, and a couple of
> tables are still incomplete (see below). A full rerun of the canonical
> configs (`configs/default.yaml`, `configs/brooks.yaml`, `configs/meneu.yaml`)
> is pending and will refresh every table here. This document mainly shows the
> shape the results will take.

Conventions: higher is better unless a metric is marked ↓ (lower is better) or
the benchmark is a *negative result* (both models at chance / ≈ 0). Full
per-metric tables for each comparison live next to the runs in
`results/<group>/compare/summary.md`.

## Summary

| Benchmark | Models | Primary metric | Winner |
| --- | --- | --- | --- |
| [Caudal eQTL](#caudal-eqtl) | Shorkie, Yorzoi | \|score\| AUROC / AUPRC | **Shorkie** |
| [Kita eQTL](#kita-eqtl) | Yorzoi only | AUROC / AUPRC | _pending — needs a 2nd model_ |
| [Rafi / deBoer MPRA (promoter)](#rafi--deboer-mpra-promoter) | Shorkie, Yorzoi (+DREAM-RNN pending) | Pearson r / Spearman ρ | **Shorkie** |
| [Shalem MPRA (terminator)](#shalem-mpra-terminator) | Shorkie, Yorzoi | Pearson r / Spearman ρ | **Yorzoi** |
| [Chen synonymous MPRA](#chen-synonymous-mpra) | CAI, CodonTransformer, Shorkie, Yorzoi | Pearson r / Spearman ρ | **Mixed** (Shorkie on GFP, CAI on TDH3) |
| [Wu RFP insertions](#wu-rfp-insertions) | Shorkie, Yorzoi | Pearson r / Spearman ρ | **Neither** (both ≈ 0) |
| [Hong IGR insertions](#hong-igr-insertions) | Shorkie, Yorzoi | Spearman ρ (IntProp) | **Neither** (both ≈ 0) |
| [Brooks SCRaMBLE](#brooks-scramble) | Shorkie, Yorzoi | LFC direction balanced acc. | **Yorzoi** |
| [Cuperus 5′-UTR](#cuperus-5-utr) | Shorkie, Yorzoi | Spearman ρ (+ partial-corr) | **Shorkie** |
| [Meneu foreign DNA](#meneu-foreign-dna) | Shorkie, Yorzoi | per-window shape Pearson / JS↓ | **Yorzoi** |

---

## Caudal eQTL

Rank true single-nucleotide *cis*-eQTLs above distance-matched controls. Mean ±
SEM over 4 negative sets. ([spec](benchmarks/caudal_eqtl.md))

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| \|score\| AUROC | **0.5673 ± 0.0015** | 0.5296 ± 0.0007 |
| \|score\| AUPRC | **0.5793 ± 0.0022** | 0.5364 ± 0.0015 |

**Winner: Shorkie** — leads on both AUROC and AUPRC.

## Kita eQTL

Same task as Caudal on an independent positive panel (Promoter/UTR5/UTR3/ORF,
≤ 8 kb of TSS). ([spec](benchmarks/kita_eqtl.md))

> _No comparison yet._ Only Yorzoi has run on this task; a comparison needs at
> least two models. Placeholder — will be filled by the rerun.

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| \|score\| AUROC | _pending_ | _ran_ |
| \|score\| AUPRC | _pending_ | _ran_ |

## Rafi / deBoer MPRA (promoter)

~71k random 80 bp promoters, scored by marginalizing the insert's logSED effect
across native host-gene contexts. ([spec](benchmarks/rafi_mpra_promoter.md))

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| Pearson r | **0.7595** | 0.6062 |
| Spearman ρ | **0.7747** | 0.6254 |

**Winner: Shorkie** — leads on both. The DREAM-RNN supervised in-distribution
baseline is spec'd but not yet run; it will be added as a third column.

## Shalem MPRA (terminator)

Designed 3′-end / terminator variants, marginalized logSED scoring.
([spec](benchmarks/shalem_mpra_terminator.md))

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| Pearson r | 0.6430 | **0.7075** |
| Spearman ρ | 0.6511 | **0.7066** |

**Winner: Yorzoi** — leads on both.

## Chen synonymous MPRA

Synonymous-codon effect on mRNA abundance, three libraries. Includes the CAI and
CodonTransformer sequence-feature baselines alongside the two S2F models.
([spec](benchmarks/chen_synonymous.md))

**gfp_r1** (two replicates)

| metric | CAI | CodonTransformer | Shorkie | Yorzoi |
| --- | --- | --- | --- | --- |
| Pearson r (rep1) | 0.3141 | 0.2091 | **0.3863** | 0.3067 |
| Pearson r (rep2) | **0.2655** | 0.1498 | 0.2169 | 0.2481 |
| Spearman ρ (rep1) | 0.3275 | 0.2971 | **0.4749** | 0.3759 |
| Spearman ρ (rep2) | 0.3055 | 0.2731 | 0.2979 | **0.3424** |

**gfp_r2** (two replicates)

| metric | CAI | CodonTransformer | Shorkie | Yorzoi |
| --- | --- | --- | --- | --- |
| Pearson r (rep1) | 0.3219 | 0.5560 | **0.6008** | 0.5695 |
| Pearson r (rep2) | 0.2469 | **0.5932** | 0.5455 | 0.5458 |
| Spearman ρ (rep1) | 0.3158 | 0.5735 | **0.6220** | 0.5891 |
| Spearman ρ (rep2) | 0.2385 | **0.6059** | 0.5564 | 0.5566 |

**tdh3** (single column)

| metric | CAI | CodonTransformer | Shorkie | Yorzoi |
| --- | --- | --- | --- | --- |
| Pearson r | **0.6752** | 0.3609 | 0.5270 | 0.6061 |
| Spearman ρ | **0.3887** | 0.1876 | 0.3024 | 0.2305 |

**Winner: Mixed.** Shorkie leads the two GFP libraries (CodonTransformer ties it
on gfp_r2 rep2); the CAI baseline leads TDH3 outright.

## Wu RFP insertions

Fixed RFP cassette moved across ORF-deletion loci — a pure genomic *position*
effect. ([spec](benchmarks/wu_rfpins.md))

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| Pearson r | -0.0325 | 0.0159 |
| Spearman ρ | -0.0252 | 0.0212 |

**Winner: Neither** — both models sit at ≈ 0 (informative negative; position
effect not captured). Yorzoi is marginally less negative.

## Hong IGR insertions

Fixed reporter across intergenic loci — the same position-effect probe as Wu, at
preserved intergenic regions. ([spec](benchmarks/hong_igr.md))

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| IntProp Spearman ρ | -0.1475 | 0.0631 |
| IntTrain Spearman ρ | -0.2687 | -0.0728 |

**Winner: Neither** — both ≈ 0 / negative (informative negative).

## Brooks SCRaMBLE

SCRaMBLE genome rearrangement; per-construct log-fold-change of CDS coverage vs.
an unscrambled parental control, against a leave-one-out reproducibility ceiling.
([spec](benchmarks/brooks_scramble.md))

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| LFC direction balanced acc. | 0.5173 | **0.6348** |
| LFC Spearman ρ | -0.0024 | **0.3128** |
| LFC Pearson r | -0.0064 | **0.2205** |
| Shape Pearson (mean) | 0.4242 | **0.8346** |
| Shape JS (mean) ↓ | 0.3534 | **0.0674** |
| LOO ceiling (dir. bal. acc.) | 0.6938 | 0.8058 |

**Winner: Yorzoi** — leads every metric; Shorkie's direction accuracy is at
chance (0.5). Note: each model is scored over the subset its receptive field
supports (Shorkie n=528, Yorzoi n=327 scored), so a shared-cohort intersection
is reported separately in `results/brooks/compare__shared/`.

> These values come from a pre-refactor run that labelled the metrics
> `tier1_*`/`tier2_*`; they are shown here under the current `lfc_*`/`shape_*`
> names and will be regenerated verbatim by the rerun.

## Cuperus 5′-UTR

Random 50 bp 5′-UTRs (read-depth stratified) plus native yeast 5′-UTR fragments,
in the HIS3 reporter. ([spec](benchmarks/cuperus_mpra_5utr.md))

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| random Spearman ρ | **0.2518** | 0.1166 |
| random clean ρ (top read-depth bucket) | **0.2803** | 0.1142 |
| random partial-corr ρ (beyond Kozak) | **0.2701** | 0.0924 |
| native Spearman ρ | **0.2012** | 0.1639 |
| native partial-corr ρ (beyond Kozak) | **0.2611** | 0.0005 |

**Winner: Shorkie** — leads every cut, and retains signal in the partial
correlation (after regressing out hand-crafted Kozak features) where Yorzoi
drops to ≈ 0.

## Meneu foreign DNA

Whole bacterial chromosomes integrated into yeast; tile each contig and predict
RNA-seq coverage zero-shot (far-OOD). Scored per contig over 5 kb windows.
([spec](benchmarks/meneu_foreign_dna.md))

**_M. mycoides_** (Mmmyco)

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| shape Pearson | 0.2638 | **0.3843** |
| shape JS ↓ | 0.2844 | **0.2382** |
| magnitude fold-change error (mean) | 0.7027 | 0.8066 |

**_M. pneumoniae_** (Mpneumo)

| metric | Shorkie | Yorzoi |
| --- | --- | --- |
| shape Pearson | 0.0912 | **0.1103** |
| shape JS ↓ | 0.0982 | **0.0826** |
| magnitude fold-change error (mean) | -0.0702 | 0.0464 |

**Winner: Yorzoi** — higher within-window shape correlation and lower JS
divergence on both chromosomes.
