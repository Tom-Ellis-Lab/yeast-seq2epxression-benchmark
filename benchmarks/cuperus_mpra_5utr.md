# Cuperus et al. — 5′-UTR MPRA expression (HIS3 reporter)

> **Status:** implemented; **v1 GPU run done (2026-06-02)** — see *Results*.
> Headline: Shorkie clean-bucket Spearman **0.28**, Yorzoi **0.11**; both
> metrics behaved as designed (per-bucket ρ climbs monotonically with depth for
> Shorkie; metric 2 + the depth split show Shorkie carries genuine mRNA signal
> while Yorzoi is near-noise on the clean data). Scoring is in the **natural
> HIS3 reporter context** — no marginalization; a divergence check over diverse
> backgrounds confirmed this is immaterial for the rank metrics (see *Why no
> marginalization*).

## At a glance

| | |
| --- | --- |
| **Task** | Regression: predict the growth-based expression of a 50 bp random 5′-UTR by scoring the model's predicted `HIS3` mRNA coverage in the actual Cuperus reporter construct (`CYC1` promoter – 50 bp UTR – `HIS3` – `CYC1` terminator). One forward pass per UTR; no host-gene marginalization. |
| **Source** | Cuperus *et al.* 2017, *Deep learning of the regulatory grammar of yeast 5′ UTRs from 500,000 random sequences*, Genome Research 27:2015–2024. DOI: [10.1101/gr.224964.117](https://doi.org/10.1101/gr.224964.117). PMC: [PMC5741052](https://pmc.ncbi.nlm.nih.gov/articles/PMC5741052/). Code/data: [Seeliglab/2017---Deep-learning-yeast-UTRs](https://github.com/Seeliglab/2017---Deep-learning-yeast-UTRs). PDF in `papers/`. |
| **Assay** | Massively parallel growth-based selection. ~500 k 50 bp random 5′-UTRs cloned into a single-copy `p415-CYC1` CEN plasmid, replacing the native 56-bp `CYC1` 5′-UTR immediately upstream of the `HIS3` ATG (`CYC1` promoter + `CYC1` terminator). Pool grown in SD-His + 1.5 mM 3-AT (His3 inhibitor) for ~6.2 doublings; pre/post-selection plasmid DNA deep-sequenced. Growth ∝ His3 protein ∝ 5′-UTR translational efficiency + mRNA level. |
| **Expression label** | Scalar per sequence: **`growth_rate`** (the committed column), a depth-normalized, +1-pseudocounted **natural-log** enrichment — see *Data → Label*. Higher = better 5′-UTR (more His3 → faster growth). Not normalized to any reference sequence; absolute values differ between experiments. |
| **Eval set** | Two libraries, both primary, scored and reported separately. **(1) Random** — all 489,348 sequences, scored in full (zero-shot, no train/test split); metric 1 reported **overall** and **stratified into 5 input-read-depth (`t0`) buckets** (clean bucket `t0 ≥ 101`, ~5 % ≈ the paper's top-5 %). **(2) Native** — 11,856 real yeast 5′-UTR fragments in the same reporter (its own selection experiment). See *Read-depth strata* and *Native library*. |
| **Primary metric** | **Metric 1:** Spearman ρ (headline) + Pearson *r* between the model's predicted `HIS3` coverage and `growth_rate`, reported overall and per depth bucket. **Metric 2:** the model's partial correlation / incremental R² over translation-only (Kozak) features. See *Metrics*. |
| **Adapter protocol** | `FivePrimeUtrReporterExpressionPredictor` (new): given a list of 50 bp UTRs, return one scalar per UTR = predicted `HIS3` expression in the reporter construct. Distinct from the marginalized-logSED protocols (Rafi/Shalem) — Cuperus scores a single fixed reporter, not a marginalization over host genes. |

## Why this benchmark exists

The Cuperus library is the only large yeast 5′-UTR MPRA at this scale. It probes a regulatory layer — translation initiation (Kozak), uORFs, 5′-UTR secondary structure, NMD-mediated mRNA decay — that the promoter (Rafi) and terminator (Shalem) benchmarks miss. Run zero-shot, it tests whether models trained on native-genome RNA-seq capture 5′-UTR-mediated regulation.

### Caveat: RNA-seq vs translation (the ceiling)

The label reflects **protein level** (His3 activity → growth). 5′-UTR changes act largely on **translation** (initiation efficiency, ribosome scanning). Shorkie and Yorzoi predict **mRNA coverage**, so they can only see the part of the 5′-UTR effect that flows through to mRNA abundance:

- **uORFs → NMD** → mRNA drops (visible).
- **Secondary structure → mRNA stability** → mRNA shifts (partly visible).
- **TSS / 5′-end effects** → abundance shifts (visible).

Pure translational-efficiency differences (e.g. a better Kozak that raises protein without changing mRNA) are **invisible** to an mRNA model. So the achievable correlation is bounded by the RNA-visible fraction of enrichment, and this benchmark is expected to correlate more weakly than Rafi/Shalem. A near-zero result is a legitimate finding ("RNA-seq-trained models don't see translation"); a moderate positive would reflect the NMD + stability channels making it through. Metric 2 is built to separate "model sees the mRNA channel" from "model is just re-deriving the Kozak context a hand feature already has."

### Why no marginalization

Rafi/Shalem/Chen marginalize a designed insert across many native host genes for two reasons: (a) to put an *unnatural* insert (a heterologous GFP CDS, a designed terminator oligo) into in-distribution genomic context, and (b) to cancel the model's absolute miscalibration by averaging logSED over contexts. Cuperus needs neither. The reporter is built from native yeast sequence the model saw in training (`CYC1` promoter, `HIS3` CDS, `CYC1` terminator), and the measured quantity is tied to `HIS3` specifically — marginalizing over 22 random hosts would measure a *different* quantity ("the insert's average effect across contexts") than what Cuperus assayed. And with the 50 bp UTR as the only thing varying, a fixed-reference logSED is just a constant offset from the raw log-coverage, so it's rank-identical — there is nothing to gain from a per-context REF. So v1 scores the literal construct, one forward pass per UTR. This is also ~22× cheaper (24,468 forwards, not ~538 k).

**Empirical confirmation (divergence check, 2026-06-03).** We re-scored the full clean bucket + native library against five diverse genomic backgrounds (within- and cross-chromosome, both strands, gene-dense and gene-sparse) plus a synthetic random-flank control, and compared single-`HIS3` to the background-marginalized score. The background shifts the readout *magnitude* near-uniformly across UTRs (per-UTR CV ≈ 0.10) but preserves rankings: Spearman(g_HIS3, g_marg) = 0.996, clean-bucket Spearman Δ = −0.0068, metric-2 partial Δ = −0.006 — all well inside any reasonable tolerance, and the shipped numbers are unchanged because production already scores single-`HIS3`. The effect is small-but-real and statistically resolvable (paired bootstrap CIs exclude zero; it is mildly *negative* for Shorkie and, on the native library, *positive* and sign-flipping for Yorzoi — a genuine flank×context interaction), so "no effect" would overstate it; the honest statement is **immaterial for the rank-based metrics this benchmark reports.** The marginalization machinery was therefore removed rather than shipped.

## The construct

Rebuild the literal Cuperus reporter as a single sequence and score it directly (it's a plasmid, not a genomic locus, so we don't place it at a chromosomal coordinate):

```
─── transcription direction (+ strand) ───

[ CYC1 promoter, ~298 nt ]   (TATA + HAP1/MIG1 UAS + TSS; from p415-CYC1)
[ 50 bp UTR insert ]         ← the variable region; replaces the native 56-bp CYC1 5′UTR
[ HIS3 CDS, 663 nt ]         (YOR202W; ATG immediately 3′ of the insert)
[ CYC1 terminator, ~250 nt ]
```

- Only the 50 bp insert varies. Everything else is constant across all test sequences.
- Total ~1.26 kb — fits comfortably in both windows (Shorkie 16,384 bp, Yorzoi 4992 bp); center the window on the `HIS3` CDS.
- Build the construct on the + strand; both adapters average forward + RC.

## What the model scores

For each test UTR `i`:
1. Build the construct above with insert `i`.
2. One forward pass (RC-averaged); read out the model's predicted RNA-seq coverage summed over the `HIS3` CDS (per-base, untransformed raw counts — the repo's per-base convention). Call this scalar `g_i`.
3. (Optional, interpretability only) report `logSED` vs the wild-type `CYC1` 5′-UTR construct — a constant offset, rank-identical to `log g_i`, so it does not change either metric.

`g_i` is the model's prediction; `E_i = growth_rate_i` is the measured label.

## Data

Vendored at `archive/cuperus/` (also on the Seeliglab GitHub at `Data/Random_UTRs.csv.gz`; byte counts differ from GEO but content is identical). Summary notebook: `notebooks/cuperus_data_summary.ipynb`.

| File | Rows | Contents |
| --- | --- | --- |
| `GSM2793752_Random_UTRs.csv.gz` | 489,348 | random library: `UTR` (50 bp), `growth_rate`, `t0`, `t1` |
| `GSM2793754_Native_UTRs.csv.gz` | 11,856 | native fragments: `UTR_name` (`GENE:frag_len:offset`), `UTR` (variable ≤ 50 bp), `growth_rate`, `t0`, `t1` |
| `GSM2793756_Random_Evolved_Seqs.csv.gz` | 2,734 | in-silico evolution (100 starts × 41 rounds); only Round0 measured |
| `GSM2793756_Native_Evolved_Seqs.csv.gz` | 1,874 | in-silico evolution from native seeds; only Round0 measured |

### Label

The label column `growth_rate` is given by the following closed form (verified against the committed data, max residual 5e-12):

```
growth_rate = ln( ((t1+1) / Σ(t1+1)) / ((t0+1) / Σ(t0+1)) )
```

i.e. the **natural-log** enrichment of **+1-pseudocounted, depth-normalized** frequencies. `t0` = pre-selection (input) read depth, `t1` = post-selection. The paper's figures label it "Enrichment (log2)", but the committed column is in nats with a pseudocount — three differences from a naive `log2(t1/t0)`: the +1 pseudocount (regularizes, makes `t1=0` rows finite, shrinks low-count UTRs toward 0), per-timepoint depth normalization (a per-construct constant), and ln vs log2. None of this changes rankings (Spearman is transform-invariant; the depth-norm is a constant Pearson absorbs), so use the `growth_rate` column as-is.

### Read-depth strata

We're zero-shot, so there is no train/test split — score **all 489,348** random sequences and report metric 1 both overall and stratified by **input read depth `t0`** (pre-selection depth). `t0` is the right stratifying axis because it is fixed before selection, so it is independent of the UTR's effect; bucketing by post-selection depth or by the label itself would bias the strata toward high/low expression.

Why stratify: the label is a ratio of two read counts, so its sampling noise scales with depth. A per-construct first-order proxy for the label's noise SD (counts modeled as Poisson, propagated through the label by the delta method) is

```
noise_SD ≈ √( 1/(t0+1) + 1/(t1+1) )   # nats; compare to the ~1.0 spread of growth_rate
```

— a *relative* reliability indicator for ranking strata, not a calibrated noise figure (real counts are overdispersed, so this is a floor). Measurement noise attenuates correlation (`r ≤ √(1 − Var_noise/Var_label)`), so each bucket has its own achievable ceiling, lowest in the noisy bucket.

Buckets (edges on `t0`; the partition is exact — every sequence lands in exactly one — and the per-sequence bucket label is pinned in the manifest):

| # | `t0` | n | % | median `noise_SD` | |
|---|---|---:|---:|---:|---|
| 1 | `< 10` | 37,992 | 7.8 % | 0.82 | noisy |
| 2 | `10–29` | 120,288 | 24.6 % | 0.38 | low |
| 3 | `30–59` | 192,541 | 39.3 % | 0.26 | medium |
| 4 | `60–100` | 113,859 | 23.3 % | 0.20 | high |
| 5 | `≥ 101` | 24,668 | 5.0 % | 0.16 | clean |

Bucket 5 (`t0 ≥ 101`) ≈ the paper's top-5 % test set. The paper's exact split takes 24,468 rows (it breaks the 1,128 ties at `t0 = 101` by sort order, as the Seeliglab `Notebook_1` does):

```python
sorted_inds = df.sort_values("t0").index
top5 = sorted_inds[int(0.95 * len(df)):]   # N = 489,348 → 24,468 rows
```

Use that exact 24,468-row cohort for the CNN R²=0.62 comparison (see *Ceiling anchor*); the threshold form of bucket 5 (24,668 rows) is what the stratified report uses.

### Native library (second primary eval)

The 11,856 native fragments are scored in the **same `HIS3` reporter**, the variable slot holding each native 5′-UTR fragment in place of the 50 bp random insert. What shapes the eval:

- **Variable length ≤ 50 bp** (tiled from real yeast 5′-UTRs with 25 bp overlap; 81 % are 50 bp, min 2 bp). The construct slot and the metric-2 Kozak window (−5…−1 before the ATG) must handle this; for the 0.3 % of fragments < 5 bp the window spills into the fixed construct sequence upstream of the slot — immaterial at that frequency, but note it.
- **Its own experiment.** `growth_rate` uses the same closed form (verified, residual 5e-12) but with the native library's own depth normalizers, so its absolute scale differs from random (native range ≈ [−2.6, 2.4], sd ≈ 0.48 vs random ≈ 1.1). **Don't pool the two** — report each library's metrics separately.
- **Deeply sequenced.** Median `t0` = 131 (vs 42 for random), so most native measurements are clean and the 5-bucket scheme doesn't fit. Report metric 1 **overall** (headline) plus a robustness check on the `t0 ≥ 10` subset, which drops the noisy ~14 % tail (including the ~10 % of rows with `t0 = 0`, finite only via the pseudocount).
- **Generalization test, not leakage.** These are real genomic UTR sequences the models saw during training — but scored here in the synthetic `HIS3` reporter, not at their native loci, so it's a generalization test with a mild memorization caveat, not direct label leakage.

Metrics 1 and 2 and the feature set `f` are identical to the random library; the native ceiling anchor is the Cuperus CNN's **R²=0.60** on native UTRs (Fig 3B).

### Evolved sequences (out of scope)

In-silico optimization trajectories with only Round0 measured — model predictions, not the paper's 573 experimentally-measured constructs (those are not in the vendored tables). Not usable as a measured eval beyond Round0.

## Metrics

Let `E` = `growth_rate`, `g` = the model's predicted `HIS3` coverage, `f` = the translation-only features below. Both metrics are computed **per library** (random and native) and reported separately — the two labels come from different selection experiments and aren't comparable in absolute scale.

### Metric 1 — direct, zero-shot

**Spearman ρ (headline)** and **Pearson *r* (secondary)** between `g` and `E`, reported **overall (all 489,348)** and **per depth bucket** (see *Read-depth strata*). Spearman leads: it is robust to the pseudocount shrinkage and the nonlinear enrichment→protein→mRNA chain. For Pearson, correlate `log g` with `E` (both on a log scale). Expect the correlation to climb monotonically from bucket 1 (noise-limited) to bucket 5 (clean ≈ paper top-5 %); its ceiling is the RNA-visible fraction (see *Caveat*).

### Metric 2 — the model's signal beyond hand-crafted translation

Quantifies whether `g` carries mRNA-channel signal *beyond* what a pure translation-initiation feature already explains. Fit `E ~ f + g` and report:

- **Headline: partial correlation** `corr(E_r, g_r)`, where `E_r` and `g_r` are the residuals of `E` and `g` after regressing each on `f`.
- **Alongside: incremental R²** = `R²(E ~ f + g) − R²(E ~ f)`.

Rules:
- **Scope.** K-fold CV **within the clean bucket** (`t0 ≥ 101`, ≈ 24.5k rows) — fit `f` and evaluate the partial correlation on the same low-noise, in-distribution data. That bucket keeps ~80 % of the full `growth_rate` variance (sd 1.02 vs 1.14) at the lowest label noise, and ≈ 24.5k rows is ample for the 15-column `f` (~1.6k rows/parameter). Fitting `f` on the noisier low-depth bulk is deliberately avoided: the pseudocount shrinks `growth_rate` there, attenuating the Kozak coefficients, so they would under-remove Kozak on the clean set and hand the model spurious credit for it. Full-library CV is a secondary diagnostic only.
- **Cross-validated.** Fit the `f`-regressions on train folds, evaluate the residual correlation out-of-fold. Use the **same folds and the same `f`** across every model compared, so the only thing that varies is `g`.
- **Interpretation caveat.** Metric 2 *understates* a model that already captures the same mechanism `f` encodes (a model that has learned Kozak gets no credit for it here) — by design, since we want credit only for the mRNA channel `f` cannot reach. The screen below shows `f` is small, so this understatement is small.

#### The translation-only features `f`

`f` must be **exclusively translational and not mRNA-mediated**, or it leaks the channel we want to attribute to the model. From the paper's feature analysis (Fig 1; *Effects of 5′ UTR features*):

- **Include — Kozak / start-codon context (this is all of `f`).** One-hot encoding of the 5 nt immediately 5′ of the `HIS3` ATG (the last 5 nt of the insert, positions −5 … −1), with −3 dominant (A at −3 favorable). The feature screen (`notebooks/cuperus_translation_features.ipynb`, clean bucket) confirms it carries real, translation-grounded signal — Spearman ρ ≈ 0.14 (random) / 0.32 (native), A-at-−3 worth +0.31 nats — and that it is **orthogonal to the mRNA channel**: incremental R² is 0.04 (Kozak) vs 0.40 (uORF block) vs 0.43 (both), so Kozak adds ~its full standalone share on top of uORF and double-counts nothing.
- **Exclude — uORFs / upstream AUGs.** By far the strongest feature (ρ ≈ −0.63, −1.27 nats), but it acts through **NMD-driven mRNA decay** — exactly the channel an RNA-seq model can legitimately capture. Putting it in `f` would steal the model's credit. (It is instead the basis for the optional uORF validity-gate below.)
- **Exclude — secondary structure (MFE).** Weak (ρ ≈ 0.23, R² ≈ 0.06, matching the paper's 0.078) and partly mRNA-stability-mediated; construct-dependent. Not cleanly translation-only.
- **Exclude — in-frame uAUG extension.** Its sign flips between the random (+) and native (−) libraries — confounded (with uORF-absence) rather than a stable translation feature. Dropped.

**What this means for metric 2.** Because `f` (Kozak) removes only ~4 % of the variance and is orthogonal to the dominant uORF channel (~40 %), the partial correlation sits close to the raw metric-1 correlation. So metric 2 is not a large reweighting — it is a **validity check**: it confirms a model's metric-1 score is not merely re-derived Kozak context (which `f` already holds). Equivalently, metric 1 is already a fairly clean read of the mRNA channel, because the only clean translation-only feature is small. If a later run wants `f` to carry more, the principled additions stay translation-only and non-mRNA (e.g. a ribosome-load or CNN-derived translation component), never uORF/structure.

### Ceiling anchor

Report the Cuperus CNN as a fixed reference (sequence→enrichment, captures both translation and mRNA channels). Its accuracy is itself depth-dependent: **R²=0.62** on the top-5 %-by-depth set (≈ bucket 5) vs **R²=0.47** on a randomly chosen, depth-mixed 5 %. So read each random-library bucket against a depth-appropriate ceiling (≈0.62 for bucket 5; the overall number sits nearer the depth-mixed 0.47), not a single anchor. For the native library the anchor is the CNN's **R²=0.60** on native UTRs (Fig 3B).

### Optional secondaries (not in the v1 headline)

Carry as diagnostics; include only if metric 2 needs backup:
- **uORF sign test** — is `g` lower for uORF-containing inserts than for matched non-uORF inserts? (Known NMD direction; no fitting.) A direct probe of the mRNA channel.
- **Top/bottom-decile AUROC** — rank-only, robust to the noisy label tail.

## Results (v1 — 2026-06-02 GPU run)

Both models scored zero-shot over the full library (489,348 random + 11,856 native) on one RTX A6000. Spearman is the headline (scale-free); the clean bucket is `t0 ≥ 101` (≈ the paper's top-5 %).

| Spearman ρ | Shorkie | Yorzoi |
| --- | ---: | ---: |
| random — overall | 0.252 | 0.117 |
| random — clean bucket (`t0 ≥ 101`) | **0.280** | 0.114 |
| random — metric 2 partial (beyond Kozak) | 0.270 | 0.092 |
| random — metric 2 incremental R² | 0.072 | 0.009 |
| native — overall | 0.201 | 0.164 |
| native — clean-bucket raw (`t0 ≥ 101`) | 0.373 | −0.010 |
| native — metric 2 partial (clean) | **0.259** | **0.005** |
| native — metric 2 incremental R² | 0.058 | 0.002 |

Depth stratification — random-library Spearman ρ by `t0` bucket:

| bucket | `t0` | n | Shorkie | Yorzoi |
| --- | --- | ---: | ---: | ---: |
| 1 | < 10 | 37,992 | 0.152 | 0.090 |
| 2 | 10–29 | 120,288 | 0.240 | 0.120 |
| 3 | 30–59 | 192,541 | 0.262 | 0.121 |
| 4 | 60–100 | 113,859 | 0.276 | 0.115 |
| 5 | ≥ 101 | 24,668 | **0.280** | 0.114 |

1. **Shorkie ≫ Yorzoi** — ~2.5× the correlation on the clean random bucket (0.280 vs 0.114). A per-track-group check ruled out a Yorzoi readout artifact: all 81 plus-strand track groups (extra-chromosome strains, Illumina, JS Nanopore) give ρ ≈ 0.09–0.12, so the gap is genuine, not track selection.
2. **Depth stratification validated.** Shorkie's per-bucket ρ climbs monotonically with read depth (0.152 → 0.280) exactly as predicted — the low-`t0` bucket is measurement-noise-limited, the clean bucket is the cleanest read, and it reproduces an independent 2k sanity sample (0.279) to three decimals. Yorzoi's is flat (~0.11–0.12): its signal sits near the noise floor everywhere.
3. **Metric 2 + the depth split separate genuine signal from artifact.** On the clean random bucket both models' correlation survives residualizing out Kozak (Shorkie partial 0.270 vs raw 0.280; Yorzoi 0.092 vs 0.114) — neither wins via Kozak; Shorkie just has ~3× more real mRNA-channel signal. Native is starker: Shorkie's clean-bucket native correlation is strong and survives Kozak (raw ρ 0.37 → partial 0.26), while **Yorzoi has essentially no native signal on the clean bucket** (raw ρ ≈ 0, partial 0.005) — its all-depth native 0.164 *reverses* with read depth (0.164 → 0.108 at `t0 ≥ 10` → ≈ 0 at `t0 ≥ 101`), marking it a low-depth confound rather than expression signal. Net: Shorkie carries genuine 5′-UTR→mRNA signal on both libraries; Yorzoi does not on native.

**Sign:** positive for both, as expected (higher predicted `HIS3` coverage → higher `growth_rate`).

**Ceiling context.** Shorkie's clean-bucket ρ = 0.280 → r² ≈ 0.078, against the Cuperus CNN's R² = 0.62 on the same top-5 % split — so an mRNA-coverage model recovers ~12 % of the sequence-achievable variance, i.e. the RNA-visible (NMD / stability) slice of a protein-level assay, as the benchmark's framing predicts.

Artifacts: `results/default/{shorkie,yorzoi}__cuperus_utr/` (per model) and `results/default/compare/` (cross-model). Mechanism breakdown — uORF effect, worked examples, the Shorkie/Yorzoi native contrast — in `notebooks/cuperus_predictions.ipynb` (generator: `scripts/cuperus/build_predictions_notebook.py`).

## Sign convention

Higher `growth_rate` = better 5′-UTR = more His3 protein. A 5′-UTR that raises `HIS3` mRNA (e.g. by avoiding a uORF → less NMD) → higher `g`. Expected correlation: **positive**, attenuated by the translation-only fraction the model can't see. **Confirmed positive in the v1 run** (Shorkie / Yorzoi both > 0; see *Results*).

## Files (target layout)

### Raw upstream
- `archive/cuperus/GSM2793752_Random_UTRs.csv.gz` — random library (vendored).

### Processed distribution (`data/tasks/cuperus_mpra_5utr/`)
- `random_utrs.tsv` — all 489,348 random sequences with `growth_rate`, `t0`, `t1`, and a precomputed depth-bucket label (1–5); the bucket edges and the exact paper top-5 % index pinned by SHA256 in a manifest.
- `native_utrs.tsv` — the 11,856 native fragments with `UTR_name`, `UTR` (variable length), `growth_rate`, `t0`, `t1`.
- `construct.json` (or `.fa`) — the reconstructed `CYC1`pr / `HIS3` CDS / `CYC1` terminator flanks and the 50 bp slot coordinates.

### Track subsets / RC averaging
- Shorkie: T0 RNA-seq tracks (same subset as Rafi/Shalem marginalized).
- Yorzoi: plus-strand tracks (construct built on the + strand); swap on the RC pass.
- Both adapters average forward + RC.

## Open questions / TODO

- **`f` validation (after the first GPU run).** Check whether metric 2 separates from metric 1; if `f` is too weak, consider adding an in-frame-uAUG term or a CNN-derived translation component (keeping it translation-only).
- **Native sub-50 bp fragments.** The scaffold assembles `promoter + fragment + HIS3`, so the slot length follows the fragment; revisit whether very short native fragments should carry their flanking native UTR context (minor — 0.3 % are < 5 bp).

*Resolved:* the construct sequences are pinned + verified (`scripts/cuperus/build_construct.py` reconstructs them from the genome; junctions match the paper's cloning overhangs), and the scaffold reuses `_cassette_scaffold.py`. The single-`HIS3`-vs-marginalized divergence check ran 2026-06-03 and confirmed marginalization is immaterial for the rank metrics (see *Why no marginalization*); the `backgrounds=` machinery was removed.
