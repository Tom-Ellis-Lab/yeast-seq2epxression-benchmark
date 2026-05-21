# Roadmap

## Infrastructure

- [x] Protocol-based adapter dispatch (`VariantEffectScorer`,
  `MarginalizedSequenceExpressionPredictor`,
  `TerminatorMarginalizedExpressionPredictor`,
  `CassetteExpressionPredictor`, `CoverageTrackPredictor`)
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
- [ ] **Automated cross-model comparison runner (`ybench compare` or
  equivalent).** Walks `results/<config>/<model>__<task>/summary.json`
  + raw arrays, intersects on a common per-task sample axis where
  applicable (mirroring `scripts/brooks/compare_models.py`), and
  emits a unified per-task comparison directory under
  `results/<config>/compare__shared/<task>/` with:
    - a side-by-side metrics table (per-task summary CSV/JSON
      with one row per model, primary metric column highlighted),
    - per-task standardised plot (scatter / dir-acc bar / ROC, etc.
      depending on protocol — defined by the benchmark class's new
      `compare_plot(results_by_model, out_dir)` hook), and
    - a top-level `compare__all/index.md` that aggregates the per-task
      summaries into a single results page (the existing Documentation
      "Results summary page" item collapses into this).
  Today only Brooks has this (`scripts/brooks/compare_models.py`).
  Generalising means defining a `Benchmark.compare(...)` classmethod
  + a per-protocol shared-cohort intersection helper.

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

- [x] Deleted the fixed-context variant (2026-05-21). The fixed-context
  `MPRARegressionBenchmark` + its Shorkie / Yorzoi adapters + the
  `rafi_mpra_promoter` task entry + the `SequenceExpressionPredictor`
  protocol are all gone. The marginalized / native-position variant is
  the honest evaluation for Shorkie / Yorzoi (sequence-in-yeast-context
  is what they were trained for); the fixed-context plasmid construct
  was an extra hop that didn't add benchmark signal we weren't already
  getting from the marginalized version. Doesn't kill the DREAM-RNN
  follow-up below: it'll pull 80-bp inserts straight from the
  marginalized benchmark's input sequences and persist scalars in the
  same `summary.json` shape — no fixed-context benchmark required.
- [ ] DREAM-RNN supervised reference baseline. DREAM-RNN is the
  optimal Prix Fixe model from the BHI core layer block in
  Rafi / de Boer et al.; weights + architecture from
  [random-promoter-dream-challenge-2022](https://github.com/de-Boer-Lab/random-promoter-dream-challenge-2022).
  Takes the raw 80-bp promoter and predicts a scalar — does NOT
  marginalize over host-gene contexts. Counts as a **supervised
  upper bound**, not zero-shot: DREAM-RNN was trained on this MPRA,
  so its score is reported separately from the Shorkie/Yorzoi
  marginalized numbers (clearly labelled as in-distribution).
  Implementation: minimal scoring adapter that pulls each row's
  raw 80-bp insert from the marginalized benchmark's input
  sequences, runs DREAM-RNN on the bare 80 bp, and persists scalars
  in the same `summary.json` layout so the compare runner picks it
  up alongside the marginalized predictions (different scoring
  paths, comparable headline metrics). The Prix Fixe BHI
  architecture code is already locally available at
  `shorkie-paper/from_kuanhao/eQTL/data/eQTL_MPRA_models_eval/prixfixe/bhi/`
  (vendored via Kuanhao's email); pretrained weights need to come
  from the de-Boer-Lab repo or be retrained.
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
- [ ] **ExoShorkie adapter** (`MarginalizedSequenceExpressionPredictor`).
  ExoShorkie is a transfer-learning extension of Shorkie on
  exogenous-RNA-seq-in-yeast (Mandl & Orenstein 2026,
  [DOI](https://doi.org/10.64898/2026.01.25.701486)). Random 80 bp
  promoters in the YFP-plasmid construct are the canonical
  exogenous-DNA-in-yeast setup — this is the task ExoShorkie was
  trained for, so expect it to outperform Shorkie/Yorzoi here.
  Architecture is Shorkie-compatible: add a `ExoShorkie` wrapper
  subclassing or paralleling `Shorkie` in `models/`, then a thin
  marginalized per-task adapter forwarding to it (per the Part-1
  refactor). Code + weights in `shorkie-paper/` (vendored locally).

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
- [ ] **ExoShorkie adapter**
  (`TerminatorMarginalizedExpressionPredictor`). 150 bp exogenous
  insert + 300 bp filler downstream of a native host gene's stop
  codon is structurally close to ExoShorkie's training distribution
  (foreign DNA inserted into native yeast context). Same wrapper as
  Rafi above + a per-task adapter. Weights + code from
  `shorkie-paper/`.

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
- [ ] **ExoShorkie adapter** (`CassetteExpressionPredictor`). The Wu
  task is literally "a constant foreign cassette (mCherry + URA3 +
  LEU2 etc.) inserted at varying yeast loci" — the canonical
  exogenous-DNA-in-yeast measurement ExoShorkie was trained on.
  Re-uses the `_wu_scaffold.py` insertion machinery; only the model
  wrapper changes. Weights + code from `shorkie-paper/`.

### Cuperus et al. (5′ UTR)

- [x] Spec drafted (`benchmarks/cuperus_mpra_5utr.md`, draft v0;
  open questions remain)
- [ ] Source data acquisition (top-5 % ~24,474-sequence test set + native
  11,962-sequence secondary eval)
- [ ] Marginalized benchmark (analogous to Rafi, upstream of TSS;
  new `FivePrimeUtrMarginalizedExpressionPredictor` protocol)
- [ ] Shorkie + Yorzoi adapters
- [ ] **ExoShorkie adapter** (same new protocol). Random 50 bp 5′
  UTRs in the CYC1-YFP plasmid construct are the same exogenous-in-
  yeast flavour as Rafi and Wu — should benefit from ExoShorkie's
  training distribution. Reuses the wrapper from Rafi.

## Structural rearrangements

### Brooks et al. SCRaMBLE chromosome 9

Tests whether a model can predict the effect of structural rearrangements
(translocations, duplications, inversions, 3′ UTR swaps) on gene
expression by comparing CDS-coverage in a SCRaMBLE-rearranged strain
against the unscrambled control. Reproduces and extends the *Capturing
the effects of genetic neighborhoods on transcription* evaluation in the
Yorzoi paper (Schneider et al. 2025, Figure 4 — 5 strains, 41 samples,
r=0.33 / ρ=0.32 / dir-acc=0.62 — kept as the reference baseline).

Source: Brooks et al. 2022, *Transcriptional neighborhoods regulate
transcript isoform lengths and expression levels*, Science 375(6584)
([DOI](https://doi.org/10.1126/science.abg0162)). Aligned Nanopore
direct-RNA BEDs + per-strain genomes + GFFs at `gs://brooks-nanopore/`.

**Done.**

- [x] Spec design locked (`benchmarks/brooks_scramble.md`, 2026-05-19):
  data characterised from `gs://brooks-nanopore`; per-copy sampling
  dissolves dosage; native-genome total-reads size factor (median-of-
  ratios deferred); objective locked sample rule (alt window differs
  from native within the receptive field; intact CDS; deletions
  excluded); gene-centred alt-vs-native constructs.
- [x] Source data located: `gs://brooks-nanopore/`. **No junction-
  walking needed** — each strain's `JS<n>_1` contig already encodes
  the rearranged synIXR; construct = gene-centred window in that contig
  vs the parental `JS96_1` (JS94 has no FASTA in the bucket but its
  GFF and reads share `JS94_1` coords, which are byte-identical to
  `JS96_1`).
- [x] Yorzoi leakage: **assumed not trained on any Brooks SCRaMBLEd
  sequences** (user, 2026-05-19) → treated zero-shot. Revisit only if
  the training manifest contradicts it.
- [x] Single self-contained distribution built
  (`scripts/brooks/build_brooks_distribution.py` →
  `data/tasks/brooks_scramble/brooks_scramble_v1.tsv`). **Full-strain
  rebuild (2026-05-20)**: **698 samples across 56 SCRaMBLE strains**
  (3 strains skipped — JS618/JS621 too shallow, JS720 missing GFF;
  JS94 control + JS96 parental excluded). JS96 parental sequence; JS94
  deep-WT runs only (rrp6Δ/xrn1Δ + two failed shallow runs excluded by
  `MIN_RUN_READS = 50_000`); total-native-reads size factor; per-copy
  sampling; deletions excluded; deduplicated byte-identical copies.
  Schema: per-replicate raw JS94 counts in `js94_reads_runs` (comma
  list of 3); strain-side raw counts in `strain_reads`; `low_support`
  is strain-side only (`strain_reads < 10`). Per-replicate normalised
  JS94 coverages in `norm_cov_js94_runs`. Run-time deps: that one file
  only.
- [x] Two strains produce zero samples (**JS707**, **JS710**) despite
  having heavily rearranged synIXR contigs. Most likely cause: gene-ID
  remapping in those strains' GFFs leaves no `gene_id` overlap with
  JS94's parental gene list, so the per-gene matching loop never fires.
  Acceptable loss (2 of 56 strains); follow-up to confirm if we want
  to recover them.
- [x] New adapter protocol `CoverageTrackPredictor.predict_coverage(
  construct_seq, strand) -> np.ndarray` returning **per-base raw
  predicted counts** over `seq_len − 2 × crop_bp_each_side`. Adapters
  invert any training-time transform and unbin internally.
- [x] Yorzoi adapter `YorzoiBrooksPredictor` (`yorzoi_brooks.py`): RC
  averaging with strand swap; inverse Borzoi `x^0.75 + sqrt` transform;
  per-base unbin (BIN_WIDTH = 10) — predictions are in raw count units
  matching the per-base Nanopore truth in the TSV.
- [x] **Shorkie adapter `ShorkieBrooksPredictor` (2026-05-20)** —
  8-fold ensemble; T0 RNA-seq tracks (384 unstranded); no transform
  inversion (Poisson/softplus head outputs raw counts directly);
  unbins 16 bp → per-base. `varies_by_strain = False` (no Brooks-
  specific output tracks → native predicted once and broadcast across
  the JS94 replicate axis). `batch_size = 4`. Headline (shared cohort
  with Yorzoi, n_scored = 327): Pearson r −0.020, Spearman ρ −0.038,
  dir-acc 0.553 — essentially blind to SCRaMBLE LFCs, in contrast
  to Yorzoi's leakage-driven r 0.222 / dir-acc 0.635.
- [x] **Batched protocol + adapters (2026-05-20)**. Protocol takes
  `predict_coverage_batch(seqs, strands, strains) → (B, out_len)`;
  adapters expose `batch_size`. Yorzoi: 5× speedup (28 s → 5.7 s).
  Shorkie: ~20× speedup (was running 35+ min unbatched; 128 s
  batched). `tqdm` progress streams per chunk with `PYTHONUNBUFFERED=1`.
- [x] `BrooksScrambleBenchmark` class — reads only `brooks_scramble_v1.tsv`.
  **Per-replicate LFC design** (2026-05-20): for each sample, compute
  0–3 true LFCs (one per JS94 deep run with raw_reads ≥ MIN_READS_PER_RUN).
  Two-tier headline:
    * scored (n_reps ≥ 1 AND not low_support): Pearson r, Spearman ρ,
      direction balanced accuracy of pred_LFC vs mean true LFC.
    * calibration (n_reps ≥ 2): within-range hit rate (does pred land
      in [min, max] of the supporting JS94 LFCs?) and mean |z| in
      envelope widths.
  Tier-2 (shape): per-base Pearson r + Jensen–Shannon divergence over
  the scored cohort. Scatter plot shows the JS94 replicate envelope as
  per-sample horizontal error bars.
- [x] **Cross-model comparison on the shared sample set (2026-05-20).**
  Different model receptive fields produce different distribution
  files (Yorzoi → 698 samples, Shorkie → 1055 samples) — the
  Shorkie-only extras (357 wider-window-distinct rearrangement
  contexts) bias side-by-side headlines. Convention: **headline
  metrics are computed on the intersection of the two models'
  sample sets** (the shared cohort), reported as the primary number.
  Full-set per-model numbers are reported as secondary so the gap is
  documented. `scripts/brooks/compare_models.py` reads each model's
  result dir, intersects on `sample_id`, recomputes per-replicate
  metrics + LOO ceiling on the shared cohort, and writes
  `results/brooks/compare__shared/summary.json` + a Tier-1 chart
  (`shared_tier1.png`). Yorzoi and Shorkie share 698 samples
  (= full Yorzoi set); on that subset Shorkie's dir-acc rises
  0.517 → 0.553 (the 357 wider-only extras were a bit harder), but
  Pearson r stays ≈ 0 and ρ stays ≈ 0 — qualitative finding
  preserved.
- [x] **Inter-run JS94 reproducibility diagnostic (2026-05-20).** On
  per-native-gene sense reads (log1p), pairwise Pearson r between the
  three JS94 deep runs lands at **0.67–0.86**, well below the > 0.95
  conventional good-replicate range. Direct-RNA at the observed depth
  (~50–130 reads/gene mean) gives sparse per-gene counts; this is
  plausibly the true noise floor for this protocol, not a single bad
  run. Implication: the upper bound on Pearson r achievable on
  Brooks Tier-1 is probably ~0.7–0.85 in absolute terms — important
  context for any headline number.
- [x] **Per-strain matched-track routing** (2026-05-20). Yorzoi adapter
  gains `track_mode = "all" | "nanopore_all" | "matched"` (default
  matched). Matched routes per-call: alt construct uses the supplied
  strain's Nanopore track(s); native uses JS94's 3 deep-WT tracks.
  Switching `"all" → "matched"` shifts Tier-1 dir-acc 0.609 → 0.642,
  Spearman ρ 0.288 → 0.348; Pearson r barely moves. Tier-2 is
  saturated by training leakage (Yorzoi was trained on these tracks).
- [x] **Per-replicate LFC framing + LOO noise ceiling** (2026-05-20).
  The Brooks benchmark now computes 0–3 *predicted* LFCs per sample
  (one per JS94 deep-WT replicate) symmetric with the 0–3 truth LFCs.
  Per-replicate metrics; headline = mean across replicates. LOO
  ceiling: for each k, compute Pearson(true_lfc_runs[:, k],
  mean(true_lfc_runs[:, j ≠ k])). Conservative (denominator-side noise
  only; strain side is single-rep). Same construction for dir-acc.
  Three JS94 virtual-strain aliases pinned in `_yorzoi_constants.py`:
  `JS94_r0 → [73]`, `JS94_r1 → [75]`, `JS94_r2 → [77]`.
- [x] **First Yorzoi headline with the ceiling (2026-05-20)**:
    * Tier-1 (n_scored = 327):
        - dir-acc 0.635 (ceiling **0.806**) → 79% of recoverable
        - Pearson r 0.222 (ceiling **0.805**) → 28% of recoverable
        - Spearman ρ 0.315
    * Per-replicate spread (r): [0.299, 0.213, 0.154]; ceilings
      [0.789, 0.954, 0.671]. Replicate 1 (20180628) is by far the
      most reliable JS94 run but Yorzoi predicts replicate 0
      (20180214) best — suggests bias toward run 0's noise pattern.
    * Calibration (n = 198): within-range 0.146, mean |z| 11.31.
    * Tier-2 (n_scored = 327): r̄ 0.834, JS̄ 0.067.
- [x] **Asymmetric over-prediction finding (2026-05-20)** —
  recorded from the per-sample interval plot
  (`tier1_per_sample.png`). Quantified over 576 (sample, replicate)
  pred-truth pairs in the scored cohort:
    * Truth is **65% downregulation** (median LFC −0.53, range
      [−4.1, +2.2]). SCRaMBLE rearrangements predominantly disrupt
      gene neighborhoods → downregulation is the baseline outcome.
    * Yorzoi predicts **57% upregulation** (median LFC +0.25, range
      [−7.6, +14.8]). Sign-correct on true positives 77%; on true
      negatives 54%.
    * Magnitude distribution is bimodal: downward predictions are
      bounded to roughly [−1, 0], while upward predictions are either
      ~0 or saturate in [+5, +12]. The middle band [+1, +5] is nearly
      empty.
    * Mechanism (hypothesis from earlier hamming-distance analysis):
      Yorzoi treats any heavily-replaced surrounding context as
      "fresh, transcriptionally active neighborhood" → strong
      upregulation. The native synIXR context is below the
      expression-density baseline of most yeast neighborhoods in
      training, so the asymmetry is structural to how the model
      learned context-vs-expression.
  Implication: scalar LFC predictions out of Yorzoi for novel
  structural rearrangements are not directly usable as effect-size
  estimates without a calibration correction. Rank order (Spearman ρ)
  and direction are more trustworthy than the absolute magnitudes.

**Open.**
- [ ] **Unified max-window distribution (post-Shorkie).** v1 ships two
  separate TSVs (`brooks_scramble_v1.tsv` @ 4,992 bp for Yorzoi,
  `brooks_scramble_v1_w16384.tsv` @ 16,384 bp for Shorkie) and routes
  via separate task names (`brooks_scramble` /
  `brooks_scramble_shorkie`). The clean end-state is one TSV at the
  maximum window any model needs, with the benchmark cropping the
  central `adapter.seq_len` bp per call (and slicing the stored
  per-base truth correspondingly). Avoids one TSV per receptive field
  as new models land. Defer until a third model needs a third window.
- [ ] **JS707 / JS710 recovery (low priority).** Both strains have
  heavily rearranged synIXR contigs (307 kb and 363 kb vs 98.7 kb
  parental) but produce zero samples because of suspected gene-ID
  remapping in their GFFs. Recoverable if needed by matching on
  coordinate-based homology rather than strict gene-ID equality.
- [ ] **Context-stratified headline (low priority).** Yorzoi's
  over-prediction of LFC magnitude is strongly clustered by how much
  of the 4992 bp window is preserved between alt and native. Reporting
  metrics on (low-hamming, high-hamming) subsets separately would make
  the model behaviour explicit. Quick to wire in once `hamming(alt,
  native)` is a stored column.
- [ ] **ExoShorkie adapter** (`CoverageTrackPredictor`). SCRaMBLE
  rearrangements aren't strictly exogenous DNA, but the rearranged
  synIXR contigs put native genes into novel sequence contexts that
  ExoShorkie's training distribution (foreign sequence in yeast)
  arguably covers better than vanilla Shorkie's all-native training.
  Worth comparing as a third model on the same per-replicate /
  shared-cohort framework PR #2 set up. Reuses the Shorkie wrapper
  pattern; new thin adapter only.

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

### Correctness sweep — always evaluate on the untransformed, unbinned scale

Yorzoi was trained with the Borzoi piecewise transform
``y = transform(bin_4bp(x))`` where ``transform(x) = min(x^0.75, 384 +
sqrt(x^0.75 − 384))`` (`yorzoi/yorzoi/utils.py`). Every metric we
compute on a transformed quantity is wrong in principle:
``sum(T(x_i)) ≠ T(sum(x_i))`` and the squash compresses log-ratios at
high counts. Rank-based metrics (Spearman / AUROC / balanced acc) are
roughly preserved; Pearson r and any scalar magnitude prediction are
distorted. Shorkie was trained with Poisson loss + softplus head, so its
outputs are already in raw-count units — no inverse-transform needed,
but its 16 bp output bins introduce CDS-boundary rounding that
unbinning to per-base eliminates.

**Audit (2026-05-20).** Only `yorzoi_brooks` operates on
untransformed, per-base predictions. Every other Yorzoi adapter sums
transformed binned values directly:

| Adapter | Aggregation | Transform-affected? | Bin-boundary-rounding? |
| --- | --- | :-: | :-: |
| `yorzoi_eqtl` | `log2(alt_sum+1) − log2(ref_sum+1)` over exon bins | yes | yes |
| `yorzoi_mpra` | scalar `cov[:, yfp_bins].sum()` (DREAM YFP) | yes | yes |
| `yorzoi_mpra_marginalized` | per-host logSED → mean | yes | yes |
| `yorzoi_shalem` | per-host logSED → mean | yes | yes |
| `yorzoi_wu` | strand-matched 81-track mean × CDS-bin sum | yes | yes |
| `yorzoi_brooks` | per-base, untransformed (fixed in PR #2) | — | — |
| `shorkie_*` (all five) | log2 / scalar sums over 16 bp bins | **no transform** | yes |

**Planned fix — in one place, as part of the refactor above.** The
`Yorzoi` model class owns `_borzoi_inv_transform` and the per-base
unbin; the `Shorkie` model class owns the per-base unbin. Adapters
become thin and consume per-base raw-count predictions exclusively —
exactly the protocol contract already established by
`CoverageTrackPredictor` for Brooks. Then every existing benchmark
re-runs naturally on the corrected scale; expect:

- Rank metrics (Spearman / AUROC / balanced acc): largely unchanged.
- Pearson r: slight shifts (typically improvement; matches Brooks's
  Tier-1 r and Tier-2 r̄/JS̄ behaviour after the fix).
- Scalar magnitude predictions (DREAM YFP, Wu mCherry): can shift
  meaningfully.

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
- [ ] `ybench data` CLI to fetch benchmark artifacts from a hosted store
  - `--all` or `--tasks t1,t2,...` (task names match `data/tasks/` subdirs:
    `caudal_eqtl`, `kita_eqtl`, `rafi_mpra`, `shalem_mpra_terminator`,
    `wu_rfpins`, `brooks_scramble`)
  - Resolves shared assets (e.g. `R64-1-1.fa`/GTF, `1011Matrix.gvcf.gz`)
    once across tasks rather than per-task
  - Verifies each file against the SHA256 manifest above; refuses partial
    downloads; idempotent (skip files already present and matching)
  - Open questions: ship prepared task artifacts vs raw inputs + local
    regeneration; hosting (Zenodo / HuggingFace datasets / GCS); whether
    to use `pooch` rather than hand-rolling fetch+verify

## v2 release

*Empty by design — anything that doesn't make the v1 cut goes here so
the v1 scope stays bounded. Move items in once v1 is locked.*
