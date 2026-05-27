# Benchmark findings

Investigations, caveats, and contextual notes that don't belong in
the formal benchmark specs but inform how to read the headline
numbers. Each section is owned by one benchmark spec and may be
extended over time. The spec is the contract; this file is the
commentary.

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
