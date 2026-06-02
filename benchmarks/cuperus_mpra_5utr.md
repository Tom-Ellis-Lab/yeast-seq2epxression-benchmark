# Cuperus et al. — 5′-UTR MPRA expression (HIS3 reporter)

> **Status:** spec, ready to implement. Data is vendored (`archive/cuperus/`);
> the two metrics are defined below. v1 scores in the **natural HIS3 reporter
> context** — no marginalization (see *Why no marginalization*).

## At a glance

| | |
| --- | --- |
| **Task** | Regression: predict the growth-based expression of a 50 bp random 5′-UTR by scoring the model's predicted `HIS3` mRNA coverage in the actual Cuperus reporter construct (`CYC1` promoter – 50 bp UTR – `HIS3` – `CYC1` terminator). One forward pass per UTR; no host-gene marginalization. |
| **Source** | Cuperus *et al.* 2017, *Deep learning of the regulatory grammar of yeast 5′ UTRs from 500,000 random sequences*, Genome Research 27:2015–2024. DOI: [10.1101/gr.224964.117](https://doi.org/10.1101/gr.224964.117). PMC: [PMC5741052](https://pmc.ncbi.nlm.nih.gov/articles/PMC5741052/). Code/data: [Seeliglab/2017---Deep-learning-yeast-UTRs](https://github.com/Seeliglab/2017---Deep-learning-yeast-UTRs). PDF in `papers/`. |
| **Assay** | Massively parallel growth-based selection. ~500 k 50 bp random 5′-UTRs cloned into a single-copy `p415-CYC1` CEN plasmid, replacing the native 56-bp `CYC1` 5′-UTR immediately upstream of the `HIS3` ATG (`CYC1` promoter + `CYC1` terminator). Pool grown in SD-His + 1.5 mM 3-AT (His3 inhibitor) for ~6.2 doublings; pre/post-selection plasmid DNA deep-sequenced. Growth ∝ His3 protein ∝ 5′-UTR translational efficiency + mRNA level. |
| **Expression label** | Scalar per sequence: **`growth_rate`** (the committed column), a depth-normalized, +1-pseudocounted **natural-log** enrichment — see *Data → Label*. Higher = better 5′-UTR (more His3 → faster growth). Not normalized to any reference sequence; absolute values differ between experiments. |
| **Test set size** | **24,468** random sequences — the top 5 % of the 489,348-variant library by input read depth (`t0`), reproduced deterministically (see *Data → Test split*). Native stratum (11,856 yeast-genome fragments) is an optional secondary. |
| **Primary metric** | **Metric 1:** Spearman ρ (headline) + Pearson *r* between the model's predicted `HIS3` coverage and `growth_rate`. **Metric 2:** the model's partial correlation / incremental R² over translation-only (Kozak) features. See *Metrics*. |
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

The committed column is **`growth_rate`**, not "log₂ enrichment". Exact closed form (verified against the data, max residual 5e-12):

```
growth_rate = ln( ((t1+1) / Σ(t1+1)) / ((t0+1) / Σ(t0+1)) )
```

i.e. the **natural-log** enrichment of **+1-pseudocounted, depth-normalized** frequencies. `t0` = pre-selection (input) read depth, `t1` = post-selection. The paper's figures label it "Enrichment (log2)", but the committed column is in nats with a pseudocount — three differences from a naive `log2(t1/t0)`: the +1 pseudocount (regularizes, makes `t1=0` rows finite, shrinks low-count UTRs toward 0), per-timepoint depth normalization (a per-construct constant), and ln vs log2. None of this changes rankings (Spearman is transform-invariant; the depth-norm is a constant Pearson absorbs) — but use the column as-is and name it correctly.

### Test split

Top 5 % by input read depth (`t0`), reproduced exactly as the Seeliglab `Notebook_1` does:

```python
sorted_inds = df.sort_values("t0").index
test = sorted_inds[int(0.95 * len(df)):]   # N = 489,348 → 24,468 rows
```

~1,128 rows sit at the `t0 = 101` boundary and are sort-order-dependent, so **pin an explicit index** in the distribution manifest rather than recomputing on the fly. The paper's CNN R²=0.62 is reported on this split.

### Native and evolved strata

- **Native (11,856 rows):** optional secondary. Fragments are **variable length ≤ 50 bp** (tiled, with overlaps), so the construct's variable slot is not a fixed 50 bp here — the adapter must handle variable insert lengths. These are native yeast sequences the models saw in training (contamination flag).
- **Evolved:** in-silico optimization trajectories with only Round0 measured — model predictions, **not** the paper's 573 experimentally-measured constructs (those are not in the vendored tables). Not usable as a measured eval beyond Round0. Out of scope.

## Metrics

Let `E` = `growth_rate`, `g` = the model's predicted `HIS3` coverage, `f` = the translation-only features below.

### Metric 1 — direct, zero-shot

**Spearman ρ (headline)** and **Pearson *r* (secondary)** between `g` and `E` over the 24,468 test sequences. Spearman leads: it is robust to the pseudocount shrinkage and the nonlinear enrichment→protein→mRNA chain. For Pearson, correlate `log g` with `E` (both on a log scale). This is the simple, fully comparable number; its ceiling is the RNA-visible fraction (see *Caveat*).

### Metric 2 — the model's signal beyond hand-crafted translation

Quantifies whether `g` carries mRNA-channel signal *beyond* what a pure translation-initiation feature already explains. Fit `E ~ f + g` and report:

- **Headline: partial correlation** `corr(E_r, g_r)`, where `E_r` and `g_r` are the residuals of `E` and `g` after regressing each on `f`.
- **Alongside: incremental R²** = `R²(E ~ f + g) − R²(E ~ f)`.

Rules:
- **Cross-validated.** Fit the `f`-regressions on train folds, evaluate the residual correlation out-of-fold. Use the **same folds and the same `f`** across every model compared, so the only thing that varies is `g`.
- **Interpretation caveat.** Metric 2 *understates* a model that already captures the same mechanism `f` encodes (a model that has learned Kozak gets no credit for it here). That is by design — we want credit only for the mRNA channel `f` cannot reach.

#### The translation-only features `f`

`f` must be **exclusively translational and not mRNA-mediated**, or it leaks the channel we want to attribute to the model. From the paper's feature analysis (Fig 1; *Effects of 5′ UTR features*):

- **Include — Kozak / start-codon context.** One-hot encoding of the 5 nt immediately 5′ of the `HIS3` ATG (= the last 5 nt of the insert, positions −5 … −1), with −3 carrying most of the signal (A at −3 is most favorable). This is pure initiation efficiency; it does not act through mRNA abundance.
- **Exclude — uORFs / upstream AUGs.** The strongest single feature, but it acts partly through **NMD-driven mRNA decay** — exactly the channel an RNA-seq model can legitimately capture. Putting it in `f` would steal the model's credit. (It is instead the basis for the optional uORF validity-gate below.)
- **Exclude — secondary structure (MFE).** Affects translation but also mRNA stability, is construct-dependent, and is weak (paper R²=0.078). Not cleanly translation-only.

This is the **try-then-validate** starting point you flagged: Kozak alone is a weak predictor, so metric 2 may collapse toward metric 1 (if `f` removes little variance) — in which case `f` is too weak or the model already encodes Kozak. We decide empirically, after the first run, whether metric 2 earns its place and whether `f` needs enriching (e.g. an in-frame-uAUG term, or a CNN-derived translation component).

### Ceiling anchor

Report the Cuperus CNN's **R²=0.62** on this exact top-5 % split as a fixed reference (sequence→enrichment, captures both translation and mRNA channels), so metric 1 reads as a fraction of achievable signal.

### Optional secondaries (not in the v1 headline)

Carry as diagnostics; include only if metric 2 needs backup:
- **uORF sign test** — is `g` lower for uORF-containing inserts than for matched non-uORF inserts? (Known NMD direction; no fitting.) A direct probe of the mRNA channel.
- **Top/bottom-decile AUROC** — rank-only, robust to the noisy label tail.

## Sign convention (verify empirically)

Higher `growth_rate` = better 5′-UTR = more His3 protein. A 5′-UTR that raises `HIS3` mRNA (e.g. by avoiding a uORF → less NMD) → higher `g`. Expected correlation: **positive**, but attenuated by the translation-only fraction the model can't see. Confirm the sign on the first run — the RNA-vs-translation indirection makes it less certain than for Rafi/Shalem.

## Files (target layout)

### Raw upstream
- `archive/cuperus/GSM2793752_Random_UTRs.csv.gz` — random library (vendored).

### Processed distribution (`data/tasks/cuperus_mpra_5utr/`)
- `test_set.tsv` — the 24,468 top-`t0` sequences with `growth_rate`; the test index pinned by SHA256 in a manifest.
- `construct.json` (or `.fa`) — the reconstructed `CYC1`pr / `HIS3` CDS / `CYC1` terminator flanks and the 50 bp slot coordinates.

### Track subsets / RC averaging
- Shorkie: T0 RNA-seq tracks (same subset as Rafi/Shalem marginalized).
- Yorzoi: plus-strand tracks (construct built on the + strand); swap on the RC pass.
- Both adapters average forward + RC.

## Open questions / TODO

- **Construct sequence sourcing.** Pin the exact `CYC1` promoter (298 nt) and `CYC1` terminator from `p415-CYC1` (Mumberg et al. 1995 / pRS415 map), and the `HIS3` CDS (YOR202W). The 50 bp insert replaces the native 56-bp `CYC1` 5′-UTR immediately upstream of the `HIS3` ATG. Confirm against the Seeliglab construct.
- **`f` validation.** After the first run, check whether metric 2 separates from metric 1; if `f` is too weak, consider adding an in-frame-uAUG term or a CNN-derived translation component (keeping it translation-only).
- **Native stratum.** Decide whether to include the 11,856 variable-length native fragments as a secondary run (needs variable-insert handling + a contamination flag).
- **Scaffold reuse.** `_cuperus_scaffold.py` (build a fixed construct with one variable slot) is close to `_cassette_scaffold.py`; unify if the abstractions line up.
