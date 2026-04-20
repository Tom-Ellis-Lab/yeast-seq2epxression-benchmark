# Roadmap

## Infrastructure

- [x] Protocol-based adapter dispatch (`VariantEffectScorer`,
  `SequenceExpressionPredictor`, `MarginalizedSequenceExpressionPredictor`,
  `TerminatorMarginalizedExpressionPredictor`)
- [x] Benchmark ABC with `evaluate` / `plot` / `save_results` / `load_results`
  / `summary_dict` / `headline` contract
- [x] Registry (`SHORKIE_ADAPTERS` / `YORZOI_ADAPTERS` keyed by protocol,
  task factories in `TASKS`)
- [x] YAML run-spec CLI (`ybench run|replot|list` + `configs/default.yaml`)
- [x] 106-test pytest suite (eQTL, both MPRA variants, Shalem, registry,
  config, CLI persistence)
- [ ] Measured-RNA-seq oracle baseline benchmark (upper bound via
  Shorkie's expression scoring formula on real data)
- [ ] Refactor MPRA / Shalem benchmark classes to share an abstract
  `RegressionBenchmark` base (data-source-agnostic)

## eQTL

### Caudal et al. *cis*-eQTL classification

- [x] Benchmark class + per-iteration `negset_*.tsv` evaluation, |score|
  AUROC/AUPRC, close-only subset, distance-to-TSS stratified plots
- [x] Shorkie adapter (`ShorkieVariantScorer`, 8-fold ensemble, 1014
  RNA-seq tracks)
- [x] Yorzoi adapter (`YorzoiVariantScorer`, plus-strand tracks, strand-
  swap RC averaging)
- [ ] Check and fix divergence between Shorkie paper's Caudal numbers and
  our benchmark results (see `memory/project_shorkie_paper_eqtl_pipeline.md`)
- [ ] Per-(gene-strand, distance-bin) effect-size calibration plot

### Kita et al. eQTL

- [ ] Data preparation (positive / negative sets, matching Caudal-style
  pipeline)
- [ ] Benchmark class (reusable `EQTLClassificationBenchmark` if layout
  matches Caudal, else new)
- [ ] Shorkie + Yorzoi adapters

## MPRA

### Rafi / deBoer et al. (promoter, DREAM)

- [x] Fixed-context benchmark (`MPRARegressionBenchmark` + 8-stratum
  split: high/low/yeast/random/challenging + SNVs / motif_perturbation /
  motif_tiling pair strata)
- [x] Shorkie fixed-context adapter (r = 0.739 on all 71,103 seqs)
- [x] Yorzoi fixed-context adapter (r = 0.458 on all 71,103)
- [x] Marginalized / native-position benchmark
  (`MPRAMarginalizedBenchmark`, 22 host genes at 180 bp upstream of TSS,
  logSED_agg over host-gene exon bins)
- [x] Shorkie marginalized adapter (T0 RNA-seq tracks, 8-fold ensemble;
  full 71,103-seq run in progress)
- [x] Yorzoi marginalized adapter (strand-matched tracks; r = 0.608 on
  all 71,103 seqs; motif strata r ≈ 0.73–0.74)
- [ ] Pair-difference Pearson on full runs for SNV / motif pair strata
  (already saved raw scores; just needs replot)
- [ ] Bootstrap confidence intervals (10 000× 10 %-subsample per stratum)
- [ ] Compare marginalized vs fixed-context per stratum in a unified
  report

### Shalem / Segal et al. (terminator)

- [x] Spec finalized (`benchmarks/shalem_mpra_terminator.md`):
  marginalized-only, 150 bp insert + 300 bp CYC1 no-term filler,
  450 bp replacement at stop codon + 1
- [x] cycl-512 mutant UTR reconstruction from Guo 1995 (R64-1-1 + 40 bp
  deletion between two TATTTA motifs)
- [x] DEE2 per-gene median TPM table (49 PASS runs,
  `data/raw/dee2/scerevisiae_gene_median_tpm.tsv`)
- [x] Host-gene selection script (`scripts/shalem/select_host_genes.py`,
  filter + TPM-tertile stratification, 22 genes committed at
  `data/processed/shalem_mpra/host_genes.json`)
- [x] `ShalemMPRAMarginalizedBenchmark` benchmark class
- [x] `_shalem_scaffold.py` shared infrastructure (CYC1 filler
  construction, host-gene loading, insertion-context computation)
- [x] Shorkie adapter (`ShorkieShalemPredictor`, T0 tracks; full run in
  progress)
- [x] Yorzoi adapter (`YorzoiShalemPredictor`, strand-matched tracks;
  **full-run r = 0.708**, ρ = 0.707 on 14,172 non-NA rows)
- [ ] Per-stratum reporting: parse `Description` column's ~100
  `SetName` values into ~9 coarse strata (RBP random, scanning mut-
  pos / neg / quantile, native 3′ UTRs, motif moves, GC variants, etc.)
- [ ] Bootstrap CIs

### Cuperus et al. (5′ UTR)

- [ ] Source data acquisition + spec
- [ ] Marginalized benchmark (analogous to Rafi, upstream of TSS)
- [ ] Shorkie + Yorzoi adapters

## Native-genome track prediction

- [ ] Cross-model RNA-seq track-prediction benchmark on held-out yeast
  regions (paper-style R² per track, tissue-style grouping if relevant)
- [ ] Adapters implementing `TrackPredictor.predict_tracks(regions)`

## Documentation

- [x] Spec per benchmark (`benchmarks/*.md`)
- [x] Architecture doc (`benchmarks/architecture.md`)
- [x] README extension guide ("Adding a new benchmark / model")
- [ ] Results summary page pulling numbers from `results/default/*/summary.json`
- [ ] Per-stratum result tables auto-generated from saved scores

## Reproducibility

- [x] Run metadata (config hash, git commit, timestamp) per output dir
- [x] Raw scores / labels persisted so re-plotting doesn't require re-scoring
- [ ] CI / automated smoke-test run on synthetic data (no GPU)
- [ ] Lock data distribution versions in a manifest (SHA256 of each raw
  input file)
