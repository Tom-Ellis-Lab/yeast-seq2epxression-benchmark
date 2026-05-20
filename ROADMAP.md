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
- [ ] Audit Caudal eQTL discovery methodology — how were positives
  called from the 1011-strain panel? Strains have varying ploidy
  (haploid / diploid / polyploid) and per-variant VAFs that don't
  cleanly map to "the strain has allele X." Decide whether this
  contaminates our positive set and, if so, whether to filter on
  ploidy / VAF before scoring.
- [ ] Per-(gene-strand, distance-bin) effect-size calibration plot

### Kita et al. eQTL

- [x] Spec drafted (`benchmarks/kita_eqtl.md`)
- [x] Generalized eQTL data-prep scripts that handle Kita
  (`scripts/eqtl/0_data_generation/*` with `--dataset kita`)
- [x] Distribution committed at `data/tasks/kita_eqtl/`
  (negset_{1..4}.tsv, 619 rows each), built from Kuanhao's emailed
  negsets via `scripts/eqtl/build_kita_v1_distribution.py`
- [x] Wired into `TASKS` and `configs/default.yaml` (reuses
  `EQTLClassificationBenchmark`)
- [ ] Shorkie + Yorzoi runs on Kita (adapters already wired; needs an
  end-to-end run)

## MPRA

### Rafi / deBoer et al. (promoter, DREAM)

- [x] Fixed-context benchmark (`MPRARegressionBenchmark` + 8-stratum
  split: high/low/yeast/random/challenging + SNVs / motif_perturbation /
  motif_tiling pair strata)
- [x] Shorkie fixed-context adapter (r = 0.739 on all 71,103 seqs)
- [x] Yorzoi fixed-context adapter (r = 0.458 on all 71,103)
- [ ] DREAM-RNN supervised baseline adapter (`SequenceExpressionPredictor`,
  fixed-context only — DREAM-RNN's input is a fixed 80-bp promoter
  predicting a scalar, so it does not fit `MarginalizedSequenceExpressionPredictor`).
  DREAM-RNN is the optimal Prix Fixe model from the BHI core layer block
  in Rafi / de Boer et al.; weights and architecture from
  [random-promoter-dream-challenge-2022](https://github.com/de-Boer-Lab/random-promoter-dream-challenge-2022).
  Counts as a **supervised upper bound**, not zero-shot — DREAM-RNN was
  trained on this MPRA, so its score should be reported separately from
  the Shorkie/Yorzoi zero-shot numbers (clearly labelled as in-distribution).
  The Prix Fixe BHI architecture code is already locally available at
  `shorkie-paper/from_kuanhao/eQTL/data/eQTL_MPRA_models_eval/prixfixe/bhi/`
  (vendored via Kuanhao's email); pretrained weights need to come from
  the de-Boer-Lab repo or be retrained.
- [x] Marginalized / native-position benchmark
  (`MPRAMarginalizedBenchmark`, 22 host genes at 180 bp upstream of TSS,
  logSED_agg over host-gene exon bins)
- [x] Shorkie marginalized adapter (T0 RNA-seq tracks, 8-fold ensemble;
  full 71,103-seq run, r = 0.760)
- [x] Yorzoi marginalized adapter (strand-matched tracks; r = 0.608 on
  all 71,103 seqs; motif strata r ≈ 0.73–0.74)
- [x] Pair-difference Pearson on full runs for SNV / motif pair strata
  (`per_pair_stratum` with `diff_pearson_r` saved in both Shorkie and
  Yorzoi summaries)
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
  `data/tasks/dee2_gene_median_tpm.tsv`)
- [x] Host-gene selection script (`scripts/shalem/select_host_genes.py`,
  filter + TPM-tertile stratification, 22 genes committed at
  `data/tasks/shalem_mpra_terminator/host_genes.json`)
- [x] `ShalemMPRAMarginalizedBenchmark` benchmark class
- [x] `_shalem_scaffold.py` shared infrastructure (CYC1 filler
  construction, host-gene loading, insertion-context computation)
- [x] Shorkie adapter (`ShorkieShalemPredictor`, T0 tracks; full-run
  r = 0.643, ρ = 0.651 on 14,172 non-NA rows)
- [x] Yorzoi adapter (`YorzoiShalemPredictor`, strand-matched tracks;
  **full-run r = 0.708**, ρ = 0.707 on 14,172 non-NA rows)
- [ ] Per-stratum reporting: parse `Description` column's ~100
  `SetName` values into ~9 coarse strata (RBP random, scanning mut-
  pos / neg / quantile, native 3′ UTRs, motif moves, GC variants, etc.)
- [ ] Bootstrap CIs

### Wu et al. (RFP Genome wide position effects)
- [x] Spec complete (`benchmarks/wu_rfpins.md`); evaluation design
  settled, blocked on cassette-sequence verification
- [x] Labels (`table_s2_fluorescence_1044_loci.csv`, 1044 loci)
- [x] Cassette content verified (`scripts/wu/verify_cassette.py`: RFP =
  mCherry, native parts, Giaever-matched universal sites) + constant
  payload frozen to `expression_cassette.fasta`
  (`scripts/wu/build_cassette_fasta.py`)
- [ ] Per-ORF UPTAG/DNTAG barcodes from SGD deletion table →
  `barcodes.tsv` (non-blocking; inert for the readout)
- [x] Implement benchmark (`RFPInsertionBenchmark` + `_wu_scaffold.py`;
  protocol-name / registry-key / stub gaps all fixed;
  `tests/test_rfpins.py`, 20 tests; full suite green; end-to-end
  validated on real 1044-locus data with a mock adapter)
- [x] Implement Yorzoi & Shorkie adapters (`ShorkieWuPredictor`,
  `YorzoiWuPredictor`; mCherry-start-codon-centred window, absolute
  mCherry-CDS readout, RC + fold averaging)
- [x] Windowing: mCherry **stop codon at the downstream crop edge**
  (max upstream context), strand-aware.
- [x] Metrics: dropped the 5-class κ_w; added two binary tail tasks
  (extreme-low < 5, extreme-high ≥ 8) with AUROC + AUPRC and ROC/PR
  plots per task.
- [x] GPU run (RTX A6000), final windowing+metrics (n = 1043):
  **Shorkie** r = −0.035, ρ = −0.028 | xlow AUROC 0.507 AUPRC 0.083 |
  xhigh AUROC 0.467 AUPRC 0.089. **Yorzoi** r = +0.014, ρ = +0.021 |
  xlow AUROC 0.491 AUPRC 0.063 | xhigh AUROC 0.526 AUPRC 0.101.
  Every tail AUROC ≈ 0.5, AUPRC ≈ base rate — neither model detects
  either extreme above chance. Headline negative result confirmed:
  genome-wide position effects on a constant cassette are invisible to
  both models (max-upstream-context windowing did not change this).
- [x] Diagnostic: `scripts/wu/dump_yorzoi_tracks.py` — per-bin Yorzoi
  track dump (track + strand-matched mean) for all 1043 windows,
  annotated with cassette sub-features and native gene bodies; ROC/PR
  + detailed plots for measured extremes.

### Cuperus et al. (5′ UTR)

- [x] Spec drafted (`benchmarks/cuperus_mpra_5utr.md`, draft v0;
  open questions remain)
- [ ] Source data acquisition (top-5 % ~24,474-sequence test set + native
  11,962-sequence secondary eval)
- [ ] Marginalized benchmark (analogous to Rafi, upstream of TSS;
  new `FivePrimeUtrMarginalizedExpressionPredictor` protocol)
- [ ] Shorkie + Yorzoi adapters

## Structural rearrangements

### Brooks et al. SCRaMBLE chromosome 9

Tests whether a model can predict the effect of structural rearrangements
(translocations, duplications, inversions, 3′ UTR swaps) on gene
expression by comparing CDS-coverage in a SCRaMBLE-rearranged strain
against the unscrambled control. Reproduces the *Capturing the effects
of genetic neighborhoods on transcription* evaluation in the Yorzoi
paper (Schneider et al. 2025, Figure 4).

- [ ] Spec (`benchmarks/brooks_scramble_chr9.md`):
  - **Source:** Brooks et al. 2022, *Transcriptional neighborhoods regulate
    transcript isoform lengths and expression levels*, Science 375(6584)
    (DOI: [10.1126/science.abg0162](https://doi.org/10.1126/science.abg0162)).
    Supplementary `science.abg0162_table_s3.txt` lists all novel
    junctions called from whole-genome sequencing of the SCRaMBLE
    strains.
  - **Strains:** unscrambled control **JS94** (carries loxPsym sites
    on chr9 right arm but no induced recombination); five SCRaMBLE
    strains **JS606, JS707, JS711, JS731, JS732**.
  - **Samples:** 18 unique CDSs identified as rearranged into a novel
    context across the 5 strains → **41 (gene × strain) samples** in
    the Yorzoi paper. Lock the exact gene/strain list as part of the
    spec so the eval is reproducible.
  - **Coverage source:** long-read direct RNA-seq (Nanopore) from
    Brooks et al., per-strain. Strand-of-gene only.
  - **Per-sample label:** `Δtrue = (CDS coverage sum, SCRAMBLE strain) /
    (CDS coverage sum, JS94)`. Coverage summed over the gene's CDS
    interval, on the gene's strand.
  - **Per-sample prediction:** `Δpred = (predicted CDS coverage sum on
    the rearranged construct) / (predicted CDS coverage sum on the
    native unscrambled construct)`. The native construct centres the
    gene's CDS in the model's input window; the rearranged construct
    is built from the strain's WGS-called junction structure around
    the gene. The model's output is summed over the bins overlapping
    the CDS in each prediction.
  - **Metrics:** Pearson *r* and Spearman ρ of `(Δpred, Δtrue)` across
    all 41 samples, plus **balanced accuracy of direction**
    (sign of `Δpred − 1` vs `Δtrue − 1`). Yorzoi paper reports r = 0.33,
    ρ = 0.32, balanced accuracy = 0.62 — use these as the reference
    baseline.
- [x] Spec **design locked** (`benchmarks/brooks_scramble.md`, 2026-05-19):
  data characterised from `gs://brooks-nanopore`; per-copy sampling
  (dissolves dosage); native-genome median-of-ratios normalisation;
  objective locked sample rule; JS94×3 reproducibility ceiling; Tier-1
  LFC (dir-acc → ρ → r) + Tier-2 Yorzoi-only coverage shape
  (Pearson + Jensen–Shannon headline, KL secondary); gene-centred
  alt vs JS94-native constructs.
- [x] Source data located: `gs://brooks-nanopore/` (per-strain
  genomes/GFF/Nanopore BEDs). **No junction-walking needed** — each
  strain's `JS<n>_1` contig already encodes the rearranged synIXR;
  construct = window in that contig vs `JS94_1`.
- [x] Yorzoi leakage: **assumed not trained on any Brooks SCRaMBLEd
  sequences** (user, 2026-05-19) → treated zero-shot, no held-out
  filtering in v1; revisit only if the training manifest contradicts it.
- [x] **Single self-contained distribution built**
  (`scripts/brooks/build_brooks_distribution.py` →
  `data/tasks/brooks_scramble/brooks_scramble_v1.tsv`, 37 samples):
  JS96 parental sequence (no JS94 FASTA in bucket), JS94 deep-WT runs
  only (rrp6Δ/xrn1Δ + 2 failed runs excluded), total-native-reads size
  factor, per-copy sampling, deletions excluded, ceiling derivable from
  the file (`norm_cov_js94_runs`). Run-time deps: that one file only.
  TODO: scale `--strains all`; investigate JS707→0.
- [ ] New adapter protocol `CoverageTrackPredictor.predict_coverage(
  construct_seq, strand) -> np.ndarray` (per-bin window coverage); the
  benchmark derives Tier-1 CDS LFC + Tier-2 shape. Sequence-in /
  coverage-out, not variant-effect.
- [ ] `BrooksScrambleBenchmark` class — reads only
  `brooks_scramble_v1.tsv`; Tier-1 LFC (dir-acc → ρ → r vs the JS94
  reproducibility ceiling derived from `norm_cov_js94_runs`) + Tier-2
  Yorzoi coverage shape (Pearson + Jensen–Shannon); plots.
- [ ] Yorzoi `CoverageTrackPredictor` adapter (RC/strand machinery
  reused). Leakage assumed clean per user (2026-05-19).
- [ ] Shorkie Tier-1 substitute — deferred (not Nanopore-trained).

## Native-genome track prediction

- [ ] Cross-model RNA-seq track-prediction benchmark on held-out yeast
  regions (paper-style R² per track, tissue-style grouping if relevant)
- [ ] Adapters implementing `TrackPredictor.predict_tracks(regions)`

## Code-structure refactor (post-Brooks)

Discuss after the Brooks benchmark lands. Duplication is building up
across adapters and `src/yeastbench/adapters/` is getting crowded.

Concrete examples:
- `_forward_full_tracks` and `from_pretrained` in `yorzoi_shalem.py`,
  `yorzoi_wu.py`, and `yorzoi_brooks.py` are near-identical — RC averaging
  with the plus↔minus swap is implemented three times. Same story for
  the shorkie checkpoint loaders / forward path across `shorkie_*` files.
- The adapter folder is approaching ~15 files, with most pairs only
  differing in how the task-specific window/insertion is built and which
  tracks are read out.

Direction (revisit before designing):
- The original architecture intent was **models + tasks/benchmarks +
  thin adapters connecting them**, not a fat adapter per (model × task).
- A `Shorkie` / `Yorzoi` model class would own `from_pretrained` / RC
  averaging / strand-track machinery once; per-task adapters become a
  short specialisation that builds the input and picks the readout.
- Goal: reduce LOC, make adding a new benchmark or model O(one small
  file), keep the registry's protocol-driven dispatch.

Not blocking any benchmark; queued for explicit design discussion.

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
