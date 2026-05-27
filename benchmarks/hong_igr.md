# Hong et al. — Chromosomal-position effects on IGR-integrated mCherry

> **Status:** **implemented & tested** (`tests/test_hong_igr.py`,
> 32 tests; full suite green). `HongIGRInsertionBenchmark` +
> `_hong_scaffold.py` + `_cassette_scaffold.py` +
> `ShorkieHongPredictor` / `YorzoiHongPredictor` wired into the
> registry/config (`hong_igr`). Distribution committed at
> `data/tasks/hong/hong_igr_v1.tsv` (98 IntTrain + 52 IntProp = 150
> rows, all gRNA-verified against R64-5-1; 2 IntTrain fall back to
> Table S4 `int_site`). Cassette FASTA + xlsx source committed at
> `data/tasks/hong/`. **GPU runs done** — headline numbers in the
> *Results* section below.

Implements the IGR-integration assay from Hong, Cai, Wang, Dong,
Zhang & Lian 2026, *Exploring Chromosomal Position Effects for
Predictable Tuning of Metabolic Pathways in Yeast*, bioRxiv
[10.64898/2026.04.06.716637](https://doi.org/10.64898/2026.04.06.716637).
Supplementary data vendored at `data/tasks/hong/YeIP_supp_table_R2_20260411.xlsx`;
YeIP source at https://github.com/daftpunksss/YeIP.

## At a glance

| | |
| --- | --- |
| **Task** | Regression: predict mean mCherry fluorescence (normalized to a reference site) of a **constant** `TDH3p-mCherry-ADH1t` reporter cassette CRISPR-Cas9 integrated into one of 150 distinct intergenic regions (IGRs) in *S. cerevisiae*. **Position-effect** task: cassette/promoter/CDS/terminator are fixed; only the genomic insertion site varies. |
| **Source** | Hong et al. 2026, bioRxiv ([DOI](https://doi.org/10.64898/2026.04.06.716637)). PDF + supp data in `archive/hong/` + `data/tasks/hong/`. |
| **Assay** | The `TDH3p-mCherry-ADH1t` expression cassette is integrated into a chosen IGR of *S. cerevisiae* BY4741 (`MATa his3Δ1 leu2Δ0 met15Δ0 ura3Δ0`) using CRISPR-Cas9. Donor DNA is PCR-amplified from the cassette template with primers carrying 40 bp homology arms matching the chromosome up- and downstream of the gRNA-defined cut site. Steady-state mCherry mean fluorescence is read by flow cytometry (BD LSRFortessa, 561 nm ex / 610/20 nm em, 30 000 events per sample), normalized to the reference site `IntTrain92` (fluorescence ≔ 1.0). |
| **Sibling of Wu (vs differences)** | Wu integrates into a *deleted ORF span* (CDS replacement at the YKO *kanMX* locus); Hong integrates into a *preserved intergenic region* between two intact genes. Wu's cassette ≈ 3.5 kb consumes ~70 % of Yorzoi's receptive field; Hong's ≈ 1.6 kb consumes ~30 %. The biology Hong tests is *position-effect on a foreign cassette in preserved native context* — closer to the rational pathway-engineering question. |
| **Reference assembly** | *S. cerevisiae* R64-5-1 (SGD release 2024-05-29). Matches what the YeIP code repo bundles; Hong's paper Methods say R64-4-1 but the running code uses R64-5-1, and they're sequence-identical for our IGR set. FASTA at `data/tasks/R64-5-1.fa`. |
| **Expression label** | `fluorescence_norm_intrain92`: scalar per locus, mCherry mean fluorescence ÷ `IntTrain92` fluorescence. Dynamic range across IntProp 90/10 percentile ≈ **1.73×** (~0.7–1.2); IntTrain wider, ~2.3× 90/10. Distribution roughly unimodal centred near 1.0 for IntProp, near 0.75 for IntTrain. |
| **Test set size** | **150 unique measured loci (98 IntTrain + 52 IntProp), 0 drops.** No training set on our side — the model is zero-shot. *YeIP* was trained on the 98 IntTrain and validated on the 52 IntProp; we mirror that split for tiered reporting. |
| **Primary metric** | **Spearman ρ** on IntProp (n = 52), using RNA-seq × mCherry-CDS readout. Fixed per model, apples-to-apples across models. Tests the principled causal chain: mRNA → fluorescence. |
| **Diagnostic B (secondary)** | **Spearman ρ** on IntProp using the **best (track group × readout region) combination selected on IntTrain** (signed scores; biological signs applied). A soft form of supervised feature engineering. Diagnoses each model's *upper bound* given its track inventory. Adapters opt in by implementing `predict_diagnostic_readouts(loci)`. |
| **Adapter protocol** | `IGRInsertionExpressionPredictor` — distinct from Wu's `CassetteExpressionPredictor` since the cassettes, locus shapes, and window-anchor conventions differ. Optional extension: `predict_diagnostic_readouts(loci) → dict[name, scores]` for Diagnostic B. |

## Headline ceiling: 0.847 is misleading; the real ceiling is ~0.3

Hong reports **SPCC = 0.847** between RT-qPCR mCherry mRNA and
fluorescence intensity across 30 (promoter × IGR) combinations
(Fig. 2I). **That number is structurally inflated for our task and
should not be treated as an achievable ceiling.** Reasons:

1. **The 30 points span 6 promoters × ≈ 6 IGRs**, where the
   cross-promoter dynamic range (TDH3p strongest, CYC1p ~ 2.6× weaker;
   GAL1p induced ~ 2× stronger than TDH3p) far exceeds the
   within-promoter (IGR-driven) dynamic range (~ 1.5× per promoter
   row). Most of the rank correlation in the 30-point set is carried
   by cross-promoter ordering ("strong promoters → more mRNA AND more
   protein"), which is roughly tautological.
2. **Hong does not report the within-TDH3p (fixed-promoter, varying-
   IGR) rank correlation** — the directly relevant number for our
   fixed-TDH3p, 150-IGR benchmark. From the heatmap (Fig. S9A), the
   within-promoter rank correlation across 6 IGRs is severely
   underpowered (n = 6 per row) and likely much lower than 0.847.
3. **Noise-limited ceiling.** Yorzoi's published per-locus magnitude
   error is σ_log₂ ≈ 0.9 (≈ −56 % to +74 % of true magnitude). The
   IntProp label's log₂ standard deviation across 52 loci is only
   ≈ 0.31 — signal-to-noise ratio < 0.4. Simulation (perfect-on-
   average + σ_log₂ = 0.9 Gaussian noise, 5000 trials) gives
   **achievable IntProp ρ ≈ 0.32, 90% CI [+0.11, +0.52]**. Repeated
   in section 7 of the deep-dive notebook (`notebooks/
   hong_predictions_deep_dive.ipynb`). **No model with Yorzoi-quoted
   magnitude noise can hit 0.847.**

What this means for the benchmark:
- **The 0.556 ceiling YeIP reports on IntProp is likely close to the
  honest within-promoter ceiling**, achievable because YeIP uses a
  tabular AutoGluon model on hand-engineered chromatin features (and
  was trained directly on these IGRs). YeIP's effective σ_log₂ ≈ 0.4
  vs Yorzoi's ≈ 0.9.
- **For zero-shot Shorkie / Yorzoi the relevant ceiling is ~0.3**,
  set by the model's per-locus magnitude noise + the narrow IntProp
  dynamic range. The benchmark's job is to make the gap to this
  ceiling visible.

The benchmark `summary.json` records the published 0.847 as
`mrna_fluo_ceiling_spcc_published` for transparency, but headline
results should be read against the ~0.3 noise-limited ceiling and
YeIP's 0.556, not against 0.847.

## Results

| Model | Primary IntProp ρ | Primary IntTrain ρ | Diagnostic B readout | Diag B IntProp ρ | Diag B IntTrain ρ | Candidates considered |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| **Shorkie** | −0.148 | −0.269 | **H3 (nucleosome density) × flank both 1 kb** | **+0.185** | +0.345 | 36 |
| **Yorzoi**  | +0.057 | −0.070 | SCRaMBLE strains × flank L 1 kb | −0.030 | +0.107 | 24 |
| *Reference: YeIP (supervised, published)* | — | — | tabular features on 10 hand-engineered features | **+0.556** | — | — |
| *Reference: noise-limited ceiling (σ_log₂=0.9)* | — | — | — | **≈ +0.32** | — | — |

(See `results/default/{shorkie,yorzoi}__hong_igr/summary.json` for
the full per-tier breakdown.)

### What the numbers say

- **Shorkie's Diagnostic B is a clean, biologically interpretable
  positive result.** The picked combo — predicted H3 nucleosome
  density at the immediate native flank, sign-flipped — gives
  IntProp ρ = +0.185, a ~0.33 absolute improvement over its Primary.
  Direction agrees with YeIP's "nucleosome density lower in
  high-expression IGRs". The combo sits within the noise-limited
  ceiling band (≈ +0.32 ± noise), so this is effectively the most
  the model can extract.
- **Yorzoi's Diagnostic B is *worse* than its Primary** (−0.030 vs
  +0.057). Selection on IntTrain picks a combo that doesn't
  generalize, because Yorzoi's RNA-seq-only track inventory has no
  chromatin-density tracks. The asymmetry is itself the finding:
  models with richer track inventories pick up more of the position-
  effect signal.
- **Apples-to-apples (Primary), Yorzoi > Shorkie** (+0.057 vs
  −0.108). On the same readout strategy, Yorzoi is mildly better.
  Both are far below the YeIP supervised baseline.
- **Both are within the noise-limited band** (|ρ| < 0.3 on
  IntProp). No deep failure mode here — zero-shot genomic models
  just don't beat their characterised magnitude noise on a target
  this dynamic-range-compressed.

## Dataset construction

### Sources

Tables from `data/tasks/hong/YeIP_supp_table_R2_20260411.xlsx`:

| Sheet | What it gives us | What we ignore |
| --- | --- | --- |
| **Table S1** | 98 IntTrain locus names + gRNA sequences (used to verify each cut site against R64-5-1). 2 IntTrain rows have no listed gRNA. | — |
| **Table S2** | 52 IntProp locus names + gRNA sequences + **fluorescence labels**. Plus the IntTrain92 reference row. | — |
| **Table S3** | Expression cassette sequences. Row 1 = `TDH3p-mCherry-ADH1t`, 1595 bp — the cassette we use. | The other 8 promoter variants. |
| **Table S4** | 98 IntTrain loci with `chr`, `int_site`, **`fluo_intensity`** (label). Only the label is read; coords are re-derived from gRNA matching where possible. | The 10 YeIP feature columns. |
| **Table S5** | Genome-wide YeIP predictions for 589 IGRs. **The `chr`/`int_site` columns are inconsistent with Table S2's gRNA assignments for 17 of 52 IntProp loci.** We use Table S2's gRNA as ground truth and ignore S5 entirely. | All of it. |

### gRNA-derived coordinate resolution

Every locus's `integration_coord` is the experimentally-realized
Cas9 cut site, derived from Table S1/S2's gRNA by searching R64-5-1
and applying the SpCas9 cut-position rule (between protospacer
0-indexed positions 16 and 17):

| Set | gRNA verified | Falls back to Table S4 `int_site` |
| --- | ---: | ---: |
| **IntTrain (98)** | 96 | 2 (no gRNA listed in Table S1) |
| **IntProp (52)** | 52 | 0 |

**Total: 150 loci, 0 drops.**

### Output

`data/tasks/hong/hong_igr_v1.tsv`, **150 rows**:

| Column | Type | Notes |
| --- | --- | --- |
| `locus_id` | str | `IntTrain{N}` or `IntProp{N}` |
| `set` | str | `IntTrain` (98) / `IntProp` (52) |
| `chrom` | str | Roman without `chr` prefix (`I`..`XVI`) |
| `integration_coord` | int | 1-based; experimentally-realized cut site |
| `fluorescence_norm_intrain92` | float | label |
| `gRNA` | str (nullable) | gRNA from Table S1 / S2; NaN for the 2 IntTrain fallback rows |

## The construct

Cassette: `TDH3p-mCherry-ADH1t`, **1595 bp total**, at
`data/tasks/hong/expression_cassette.fasta`. Verified sub-feature
layout:

| Feature | 0-based offset | Length |
| --- | ---: | ---: |
| **TDH3 promoter** | 0–686 | 686 |
| **mCherry CDS** (readout) | 686–1397 | 711 incl. TAA stop |
| **ADH1 terminator** | 1397–1595 | 198 |

### Cassette orientation: always `+`

Hong's donor DNA is PCR-amplified from the cassette template with
primers carrying 40 bp tails matching the chromosome. The forward
primer's tail is upstream-of-cut + strand; the reverse primer's
tail is downstream-of-cut + strand (RC'd on the reverse primer).
After HDR, the chromosome's + strand reads
`[upstream + strand] [cassette + strand] [downstream + strand]`.
Cassette is therefore always integrated with its written 5'→3'
direction matching the chromosome's + strand. mCherry transcription
is always on +. For Yorzoi this means we always use the +-strand
track subset (tracks 0–80) — no per-locus strand routing.

## Model contract

`HongIGRInsertionBenchmark.evaluate(adapter)` accepts any
`IGRInsertionExpressionPredictor`:

```python
class IGRInsertionExpressionPredictor(Protocol):
    def predict_expressions(self, loci) -> np.ndarray:
        """Primary scoring (RNA-seq × mCherry-CDS readout)."""
        ...

    # Optional Diagnostic B extension (hasattr-based duck typing)
    def predict_diagnostic_readouts(self, loci) -> dict[str, np.ndarray]:
        """Multiple candidate readouts with biological signs applied.
        Returns dict {readout_name: signed_scores_per_locus} including
        a 'primary' key. All other keys are candidates for IntTrain
        selection."""
        ...
```

`HongLocus` (frozen dataclass): `locus_id, set, chrom,
integration_coord`. Window construction (cassette centered, window
seq_len model-dependent) lives in `_hong_scaffold.py`, which
delegates to the shared `_cassette_scaffold.py`.

### Window-anchor strategy: centered cassette

Cassette midpoint at window midpoint. Yorzoi's 4992 bp window
leaves ~1.7 kb of native flank each side; Shorkie's 16,384 bp
leaves ~7.4 kb each side. This differs from Wu's
"readout-at-downstream-edge" anchor and is more appropriate for
intergenic insertion where signal can come from either flank.

### Diagnostic B selection space

Per the spec design discussion, biological signs are pre-declared
(not data-fit). Active marks and RNA-seq tracks get sign +1
(more = more expression); nucleosome density gets sign −1.

**Shorkie selection space** (36 candidates):

| Track group | Sign | Source |
| --- | ---: | --- |
| RNA-seq T0 | +1 | `SHORKIE_T0_RNA_SEQ_TRACK_IDS` |
| H3K27ac (active) | +1 | Chip-MNase H3K27AC_S{0,1} |
| H3K4me3 (active promoter) | +1 | Chip-MNase H3K4ME3_S{0,1} |
| H3K9ac (active) | +1 | Chip-MNase H3K9AC_S{0,1} |
| H3K36me3 (gene body) | +1 | Chip-MNase H3K36ME3_S{0,1} |
| H3 (nucleosome density) | **−1** | Chip-MNase H3_S{0,1,2} |

**Yorzoi selection space** (24 candidates, RNA-seq only):

| Track group | Sign | Source |
| --- | ---: | --- |
| All + tracks (baseline) | +1 | tracks 0..80 of `track_annotation.json['+']` |
| JS94 (WT yeast) | +1 | `JS94_*` tracks |
| SCRaMBLE strains | +1 | `JS\d+_*` minus JS94 |
| Brooks Nanopore yeast | +1 | All `JS\d+_*` tracks |

**Readout regions** (same set for both models; computed in window
coordinates around the centered cassette):

- cassette CDS (= the mCherry CDS bins, = Primary readout location)
- cassette full (TDH3p + mCherry + ADH1t)
- flank L 1 kb (1 kb immediately 5′ of cassette)
- flank R 1 kb (1 kb immediately 3′ of cassette)
- flank both 1 kb (union)
- flank both 3 kb (3 kb each side, union)

The Primary readout for Shorkie is `RNA-seq T0 × cassette CDS`;
for Yorzoi, `All + tracks (baseline) × cassette CDS`. Both are
included in the diagnostic candidate set, so selection can in
principle pick them again — in which case Diagnostic B equals
Primary (no improvement, which is itself an informative outcome).

## Evaluation protocol

1. Adapter returns scores. If it implements `predict_diagnostic_readouts`,
   the benchmark uses that dict; otherwise calls `predict_expressions`.
2. **Per-tier Primary metrics** for each of `{IntTrain, IntProp, pooled}`:
   - Spearman ρ (primary), Pearson r, top-k enrichment at k ∈ {3, 5, 7, 10}.
3. **Diagnostic B selection** (if readouts available):
   - For each non-`'primary'` candidate: compute signed ρ on IntTrain.
   - Pick the argmax. If the pick doesn't beat the Primary's IntTrain ρ,
     report Diagnostic B = Primary (no improvement).
   - Compute the picked readout's signed ρ on IntProp + pooled.
4. **Headline**: per-model line containing both Primary IntProp ρ and
   Diagnostic B IntProp ρ (+ the readout name picked).
5. **Plots**: per-tier Primary scatter, top-k enrichment, per-tier
   Diagnostic B scatter (if available).

### What we're *not* doing in v1

- **Cold-spot tier** (Supp Fig. S14): per-locus values unavailable
  in supp tables.
- **YeIP supervised reference baseline** (roadmap item).
- **Kong et al. 2022 external validation set** (roadmap item).
- **Promoter × IGR and carbon-source sub-tasks** (roadmap items).
- **5-fold CV selection-stability** as a scored output. The deep-
  dive notebook does the CV check; if Diagnostic B promotion to
  the headline becomes load-bearing, we can add a `diag_b_cv_stable`
  flag to `summary.json` in v2.
- **Bootstrap CIs** (standard v2 deferral).

## Files

### Inputs (committed to the repo)
- `data/tasks/hong/YeIP_supp_table_R2_20260411.xlsx` — supplementary
  tables S1–S6, source of every per-locus value.
- `data/tasks/R64-5-1.fa` (gitignored; ~12 MB; downloaded from
  SGD's R64-5-1 archive, headers normalized to bare Roman).

### Reference materials (not consumed by code)
- `archive/hong/YeIP_supp_R2_20260407.pdf` — supplementary text +
  figures. Human-reference only.

### Build script
- `scripts/hong/build_hong_distribution.py` — reads the xlsx, derives
  each locus's experimental cut site from its gRNA → R64-5-1 match,
  writes the TSV + cassette FASTA to `data/tasks/hong/`. Asserts
  all 52 IntProp gRNAs uniquely resolve in R64-5-1.

### Processed distribution
- `data/tasks/hong/hong_igr_v1.tsv` — 150 rows.
- `data/tasks/hong/expression_cassette.fasta` — frozen
  `TDH3p-mCherry-ADH1t` cassette, 1595 bp.

### Code
- `src/yeastbench/adapters/_cassette_scaffold.py` — shared
  insertion-context machinery (used by both Wu and Hong).
- `src/yeastbench/adapters/_wu_scaffold.py` — thin Wu layer.
- `src/yeastbench/adapters/_hong_scaffold.py` — thin Hong layer +
  diagnostic readout-region helpers.
- `src/yeastbench/adapters/{shorkie,yorzoi}_hong.py` — adapters with
  Primary + Diagnostic B.
- `src/yeastbench/benchmarks/hong_igr.py` — benchmark class.
- `src/yeastbench/adapters/protocols.py` — `IGRInsertionExpressionPredictor`.
- `tests/test_hong_igr.py` — 32 tests covering scaffold, benchmark,
  Diagnostic B selection, save/load roundtrip, back-compat.

### Diagnostic notebook (gitignored)
- `notebooks/hong_predictions_deep_dive.ipynb` — annotated per-locus
  sample predictions, Section 7 noise-limited ceiling simulation,
  Section 8 multi-track readout sweep, Section 10 IntTrain →
  IntProp selection + 5-fold CV, Section 12 upstream-only H3
  length sweep.

## Open questions / future work

- **Diagnostic B CV stability is borderline.** 5-fold CV on
  IntTrain in the notebook shows Shorkie picks 2 different combos
  across folds (both H3 nucleosome at flank, different region
  widths). Selection is "stable in track type, unstable in exact
  region." For v2 we could (a) report `diag_b_cv_stable: bool` in
  summary.json with a warning, or (b) pick a more conservative
  selection rule (e.g., majority pick across folds).
- **The Shorkie chromatin advantage might not generalize across
  models.** Yorzoi's lack of histone-mark tracks means it can't
  pick up Hong's chromatin signal. Future models with richer
  yeast track inventories should automatically benefit; future
  models with only RNA-seq won't. The benchmark surfaces this
  asymmetry; we shouldn't paper over it.
- **The upstream-only H3 length sweep** (notebook Section 12)
  suggests the best upstream window is 750 bp on IntProp (ρ ≈
  +0.28) but 3 kb on IntTrain (ρ ≈ +0.19). Not promoted into
  Diagnostic B's selection space — the "flank L 1 kb" and "flank
  both 1 kb" regions roughly capture the same signal — but worth
  exploring further if the chromatin readout becomes load-bearing.
