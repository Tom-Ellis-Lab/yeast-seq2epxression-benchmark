# Model failure modes

Observations about how the benchmarked models misbehave or hit
intrinsic limits on each task — the kind of context that explains
why headline numbers look the way they do. Each section is owned by
one benchmark spec. The spec is the contract; this file is the
commentary. This file is **not** a TODO or issue list — those live
in GitHub Issues.

## Hong et al. IGR insertion

### The published 0.847 SPCC ceiling is cross-promoter selection-biased

Hong reports **SPCC = 0.847** between RT-qPCR mCherry mRNA and
mCherry fluorescence across **30 (promoter × IGR) combinations** (Fig.
2H/I, with the underlying per-cell values visible as heatmaps in Fig.
S9A / S10A). The benchmark records this number in `summary.json` as
`mrna_fluo_ceiling_spcc_published` because it is the only published
mRNA↔fluorescence calibration Hong provides.

It is **not** the right ceiling for the 150-IGR fixed-TDH3p task we
benchmark, for two reasons:

1. **Cross-promoter dominance.** The 30 points span 6 promoters
   (GAL1p ≈ 2× TDH3p; TDH3p ≈ 2.6× CYC1p) × ≈ 5 IGRs each. The
   cross-promoter dynamic range is ~16× — far exceeding the
   within-promoter (IGR-driven) dynamic range of ~1.5× per row. Most
   of the rank correlation in the 30-point pool comes from "stronger
   promoters → more mRNA AND more protein," which is roughly
   tautological. Our benchmark fixes promoter = TDH3p; only the
   within-promoter variation is in scope.
2. **No published within-TDH3p calibration.** Hong's paper does not
   report a per-set (IntTrain-only / IntProp-only) qPCR-vs-fluo
   correlation. Fig. S9A/S10A's heatmaps display per-cell fold
   changes, not correlation coefficients. The notebook
   `notebooks/hong_predictions_deep_dive.ipynb` Section 6 hand-reads
   the heatmaps to compute within-promoter SPCC; for TDH3p the
   values rank identically between the two heatmaps, but the
   readout is quantized to ~0.1 log₂ steps so the within-promoter
   number is an upper bound, not a measurement.

So 0.847 should be quoted with the caveat that it pools across a
range we explicitly hold fixed.

### Yorzoi noise-limited ceiling on IntProp ≈ +0.32

The Yorzoi paper reports a per-locus magnitude error of σ_log₂ ≈ 0.9
(≈ −56% to +74% of true magnitude). The Hong IntProp label's log₂
standard deviation across 52 loci is only ≈ 0.31 — signal-to-noise
ratio < 0.4.

Simulation (perfect-on-average prediction + σ_log₂ = 0.9 Gaussian
noise, 5000 trials, n = 52) gives **achievable IntProp Spearman ρ ≈
0.32, 90% CI [+0.11, +0.52]** — reproduced in Section 7 of the
deep-dive notebook. **No model with Yorzoi-quoted magnitude noise
can hit 0.847 on this task**, independent of architecture.

This is a **Yorzoi-specific** ceiling: it is set by Yorzoi's
published noise σ_log₂ = 0.9, not by Hong's biology. A model with a
smaller σ_log₂ has a higher achievable ρ:
- **YeIP** (trained directly on the 98 IntTrain loci, tabular
  AutoGluon on 10 hand-engineered features) reports σ_log₂ ≈ 0.4
  and SPCC = 0.556 on IntProp. The smaller noise lets it sit close
  to the honest within-promoter ceiling.
- **Shorkie / Yorzoi zero-shot** sit below the Yorzoi-derived ~+0.32
  band on Primary (RNA-seq × cassette CDS). Shorkie's
  IntTrain-fitted IntProp ρ at +0.185 closes most of that gap by
  swapping the readout to predicted H3 nucleosome density on the
  immediate native flank.

The point of the benchmark is to make the gap to this ceiling
visible without the reader having to chase the noise number through
the Yorzoi paper.

### Chromatin signal at the flank vs RNA-seq at the cassette — track incoherence

Shorkie's two paths to the same Hong IGR locus give opposite-sign
correlations with measured cassette fluorescence:

- **Primary** (RNA-seq T0 × cassette CDS): IntProp ρ = **−0.148**.
- **IntTrain-fitted** (H3 nucleosome density × flank both 1 kb,
  sign-flipped): IntProp ρ = **+0.185**.

A coherent model would propagate "low-nucleosome-density flank →
transcribable" into "more predicted RNA-seq at the cassette." It
doesn't. Two contributors:

1. **OOD cassette sequence.** TDH3p-mCherry-ADH1t is foreign;
   Shorkie has never seen it. RNA-seq prediction over the cassette
   is noisy and apparently slightly anti-correlated.
2. **Different readout locations.** Chromatin is read at the native
   flank (in-distribution); RNA-seq is read at the cassette (OOD).
   They're answering different questions.

Section 8 of `notebooks/hong_predictions_deep_dive.ipynb` shows a
second flavour. Predicted **active marks** at the 3 kb flank
correlate *negatively* with measured fluorescence — H3K27ac ρ = −0.36,
H3K9ac ρ = −0.30 — where the biological prior says active marks
should be positive. The most charitable mechanism: summed active-mark
coverage is bounded above by total nucleosome density (active marks
live on histones), so denser-chromatin sites get more total
active-mark signal in absolute counts even though the
marks-per-nucleosome ratio would have the expected sign.

*Implication.* Track predictions are not internally coherent at the
"more permissive chromatin → more transcription" level. Benchmarks
that read the model's transcript prediction at the cassette as a
proxy for chromatin understanding will systematically underrate the
model. IntTrain-fitted IntProp ρ works around this by letting the
benchmark pick whichever readout actually correlates.

## Chen et al. synonymous-mutation MPRA

### The GAL1 story: a model "predicting" a repressed promoter

The Chen 2017 synonymous-MPRA places GFP under PGAL1, run in 2%
galactose. Neither model has galactose-condition tracks (Yorzoi:
zero galactose entries in `yorzoi/track_annotation.json`; Shorkie:
the T0 RNA-seq subset is all glucose rich-media TF-knockout time
courses). PGAL1 is glucose-repressed in every condition either
model saw during training, yet both produce non-trivial Pearson r
on the Chen GFP_r2 codon variants — Yorzoi r ≈ 0.5–0.6, Shorkie
r ≈ 0.6.

`notebooks/chen_{yorzoi,shorkie}_investigation.ipynb` queries each
model at the unmodified native GAL1 locus and compares against
native TDH3 (constitutive control). Cross-track mean CDS-sum,
GAL1 / TDH3 ratio:

- **Shorkie: 0.005** — native GAL1 ≈ 0.5% of TDH3. Biologically
  correct (off in glucose).
- **Yorzoi: 0.702** — native GAL1 ≈ 70% of TDH3. Biologically
  wrong: strongly repressed promoter predicted nearly as active as
  the strongest constitutive promoter in yeast.

*Implication for codon-level benchmarks.* The variant-discriminating
signal that gives Yorzoi r ≈ 0.5 on Chen GFP_r2 is **not** "real
PGAL1 transcription in galactose" — the model has no notion of
galactose. The signal lives downstream of the cassette codon swap
itself (likely a coding-density / GC / RNA-stability heuristic
learned from native CDS). Treat Pearson r on such benchmarks as
orthogonal to "does the model understand condition-specific
transcription."
