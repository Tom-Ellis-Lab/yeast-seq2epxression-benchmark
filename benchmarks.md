## Benchmark Plan

## 1) Native-genome track prediction (RNA-seq only)
- Goal: directly compare performance at predicting tracks on native genome.
- Metrics:
  - Pearson correlation
  - KL divergence (shape-only)
  - Log error (magnitude-only)
- Caveat: not a fully clean direct comparison, since the models were trained on different experiments.

## 2) eQTL datasets (Shorkie)
- Task: binary classification (`eQTL` vs `non-eQTL` variant).
- Datasets:
  - **Caudal et al.**: significant train/test leakage for Shorkie.
  - **Kita et al.**: more independent test set.
- Reproducibility note: exact evaluation procedure is hard to reconstruct from code + paper; author has not responded yet.

## 3) MPRA generalization (no finetuning)
- Question: can models predict MPRA outcomes without finetuning?

### 3.1) Rafi et al. (Promoter)
- Fixed original context (from Yorzoi): direct comparison possible.
- Marginalized over selected genes/positions (from Shorkie): introduces some effective "finetuning" via position selection.
- Overall: still a good evaluation.
- TODO: stratify by promoter type (truly random, motif tiling, etc.).

### 3.2) Shalem et al. (Terminator)
- Clean comparison possible.
- Fixed original context (from Yorzoi).

### 3.3) Brooks SCRaMBLE rearrangement effect — IMPLEMENTED
- See `benchmarks/brooks_scramble.md` (local spec) and PR #2.
- Sequence-in / coverage-out: alt construct vs gene-centred parental
  window on JS96_1 (= JS94 parental synIXR).
- 56 SCRaMBLE strains; 698 samples at 4992 bp / 1055 samples at 16384 bp;
  per-replicate LFCs from the 3 JS94 deep-WT runs; LOO reproducibility
  ceiling; calibration metrics (within-range hit rate, mean |z|).
- Both Yorzoi and Shorkie ship via `CoverageTrackPredictor`. Yorzoi
  uses per-strain matched-track routing; Shorkie averages the T0
  RNA-seq track subset and broadcasts native across JS94 replicates.
- Cross-model headline numbers come from the **shared sample set**
  (`scripts/brooks/compare_models.py`); per-model full-set numbers
  are reported as secondary.
- Substantive findings: Yorzoi recovers 28% of the noise-ceiling
  Pearson r (training-set leakage caveat); Shorkie's r ≈ 0 on
  SCRaMBLE rearrangements. Yorzoi over-predicts upregulation when
  surrounding context is heavily replaced.

## 4) ExoShorkie evaluations
- Additional artificial chromosomes.

## 5) Species LM
- Keren et al.

## 6) 5′ UTR regulatory grammar (Cuperus et al., 2017)
- Paper: *Deep learning of the regulatory grammar of yeast 5′ untranslated regions from 500,000 random sequences* ([Genome Research PDF](https://pmc.ncbi.nlm.nih.gov/articles/PMC5741052/pdf/2015.pdf)).
- Task framing: predict protein expression from 50-nt yeast 5′ UTR sequence.
- Dataset scale: 489,348 detected random 5′ UTR variants (library size ~500k).
- Suggested benchmark mode:
  - **In-distribution**: held-out random 5′ UTRs.
  - **Cross-context/generalization**: native S. cerevisiae 5′ UTR fragments.
- Notes:
  - Original model uses CNN and reports strong performance on both random and native UTRs.
  - Useful for sequence-to-expression evaluation focused on translational/regulatory grammar.