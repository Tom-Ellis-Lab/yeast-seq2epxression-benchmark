# Benchmarks

One file per benchmark. Each spec is the contract: the task, the data, the
construct, the metric, and the implementation status. This README is just the
index — read it to see what's here and pick the spec you need.

Cross-cutting commentary on *why the numbers look the way they do* (model
failure modes, noise ceilings) lives in [`model_failures.md`](model_failures.md),
not in the individual specs. The forward-looking task list lives in
[`../../ROADMAP.md`](../../ROADMAP.md).

## At a glance

| Benchmark | Probes | Task | Primary metric | Status |
| --- | --- | --- | --- | --- |
| [Caudal eQTL](caudal_eqtl.md) | *cis*-regulatory variants | binary classification | AUROC / AUPRC (mean ± SEM over 4 negative sets) | implemented |
| [Kita eQTL](kita_eqtl.md) | *cis*-regulatory variants (independent panel) | binary classification | AUROC / AUPRC | implemented |
| [Rafi / deBoer MPRA (promoter)](rafi_mpra_promoter.md) | promoter grammar | regression (marginalized logSED) | Pearson / Spearman | implemented (zero-shot); DREAM-RNN supervised baseline spec'd |
| [Shalem MPRA (terminator)](shalem_mpra_terminator.md) | 3′-end / termination grammar | regression (marginalized logSED) | Pearson *r* (Spearman alongside) | implemented |
| [Chen synonymous MPRA](chen_synonymous.md) | coding-sequence / codon-usage effect on mRNA | regression, 3 libraries | Pearson *r* + Spearman ρ on `log2(R/D)` | implemented |
| [Wu RFP insertions](wu_rfpins.md) | genomic *position* effect (ORF-deletion locus) | regression | Pearson *r* + Spearman ρ (+ tail AUROC) | implemented, GPU runs done |
| [Hong IGR insertions](hong_igr.md) | genomic *position* effect (intergenic) | regression | Spearman ρ on IntProp | implemented, GPU runs done |
| [Brooks SCRaMBLE](brooks_scramble.md) | genome-architecture rearrangement | coverage-track LFC | direction balanced accuracy, then Spearman / Pearson | implemented, GPU runs done |
| [Cuperus 5′-UTR](cuperus_mpra_5utr.md) | 5′-UTR / translational grammar | regression (HIS3 reporter) | Spearman/Pearson + partial-corr over Kozak features | implemented, GPU run done |
| [Meneu foreign DNA](meneu_foreign_dna.md) | foreign / OOD sequence (whole bacterial chromosomes in yeast) | zero-shot coverage-track prediction | per-window Pearson + JS divergence (+ fold-change error) | implemented, GPU runs done |

"Implemented" means the benchmark class + both model adapters are registered
([`src/yeastbench/registry.py`](../../src/yeastbench/registry.py)) and covered by
the test suite. The spec header on each file carries the finer-grained status
and any headline numbers.

## What each one is for

**eQTL classification — [Caudal](caudal_eqtl.md), [Kita](kita_eqtl.md).** Given
a `(variant, gene)` pair, rank true *cis*-eQTLs above distance-matched controls
drawn from the 1011-isolate panel. Same task, same scoring contract, same
negative-set recipe; they differ only in the source of positives (Caudal: ≤25 kb
of gene body; Kita: Promoter/UTR5/UTR3/ORF, ≤8 kb of TSS). Report them together
to see whether a model's skill generalizes or is dataset-specific.

**cis-element MPRA — Rafi (promoter), [Shalem](shalem_mpra_terminator.md)
(terminator), [Chen](chen_synonymous.md) (synonymous CDS).** Each probes a
different regulatory layer with a designed library, scored by marginalizing the
insert's logSED effect across native host-gene contexts (in-distribution for the
model, cancels scaffold-specific miscalibration). Rafi hits the promoter, Shalem
the 3′-end/termination layer, Chen the coding sequence itself — the largest block
of training-distribution mismatch for genomic models.

**Position effect — [Wu](wu_rfpins.md), [Hong](hong_igr.md).** The inverse of
every other benchmark: the cassette and its promoter are held constant, only the
genomic insertion site moves. Asks whether models capture *position* effects or
only local promoter grammar. Wu integrates at a deleted-ORF locus (~3.5 kb
cassette); Hong at a preserved intergenic region (~1.6 kb), closer to the
pathway-engineering question. Both are clean adversarial probes — and so far both
models score ≈ 0, an informative negative result.

**Structural rearrangement — [Brooks SCRaMBLE](brooks_scramble.md).** The
genome-scale companion to the position-effect benchmarks: SCRaMBLE perturbs
genome architecture while leaving each gene's CDS and promoter intact, decoupling
it from its native downstream context. Sequence-in / coverage-out, scored as a
log-fold-change of CDS coverage against an unscrambled parental control, read
against a leave-one-out reproducibility ceiling.

**5′-UTR grammar — [Cuperus](cuperus_mpra_5utr.md).** Random 50 bp 5′-UTRs (the full
~489k library, stratified by read depth) and 11,856 native yeast 5′-UTR fragments,
both in the `CYC1`pr–`HIS3`–`CYC1`term reporter and selected by growth. Probes
translation-initiation / uORF / structure grammar. Scored zero-shot in the literal
reporter context (no marginalization — the construct is native yeast sequence), with
a second metric isolating the model's mRNA-channel signal beyond hand-crafted Kozak
features. The assay reads protein-level growth, so an mRNA-abundance model only sees
the RNA-visible fraction (uORF→NMD, structure→stability).

**Foreign-DNA coverage — [Meneu](meneu_foreign_dna.md).** Whole bacterial
chromosomes (*M. pneumoniae*, *M. mycoides*) integrated into yeast and RNA-seq'd.
Tile the foreign chromosome and predict its RNA-seq coverage zero-shot — the
far-OOD, leakage-by-construction end of the suite. Scored per chromosome over 5 kb
windows: within-region shape (median Pearson + JS divergence) and across-region
magnitude (fold-change error), read against the ExoShorkie transfer-learning
numbers rather than the paper's (non-RNA-seq) CNN.

## Planned / v2

Future benchmarks are tracked in [`../../ROADMAP.md`](../../ROADMAP.md), not here, until
they have a spec file. Notable ones deferred past v1:

- **Species LM (Karollus et al.)** — sequence language-model evaluation, moved to v2.
- **Native-genome track prediction** — cross-model RNA-seq track R² on held-out
  yeast regions.
- **Condition coherence** — does the model respect promoter-driven OFF states?
  (the first benchmark probing *absolute* calibration rather than variant ranking).
