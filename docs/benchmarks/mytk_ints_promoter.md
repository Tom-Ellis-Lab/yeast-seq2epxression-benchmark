# MYTK — Promoter × integration-site position effects

> **Status:** benchmark class, protocol, and tests implemented (mock-adapter
> tested). **Pending:** payload/CDS sequences, the model adapters, and a config
> entry — see [Open items](#open-items).

## At a glance

| | |
| --- | --- |
| **Task** | Regression: predict mScarlet reporter expression (mean fluorescence, fold-over-background) of an integrated payload across **11 integration sites × 3 promoters = 33 constructs**. Both the integration *site* and the *promoter* vary; the rest of the payload (mScarlet + terminator + marker/vector) is constant. |
| **Source** | A Multiplex MoClo Toolkit (MYTK) for Extensive and Flexible Engineering of *Saccharomyces cerevisiae*, ACS Synth. Biol. 2024 ([DOI 10.1021/acssynbio.3c00423](https://doi.org/10.1021/acssynbio.3c00423)). SI table `sb3c00423_si_003.xlsx`. |
| **Assay** | mScarlet reporter payloads (three promoters: `pTDH3` strong, `pRPL18B` medium, `pREV1` weak) integrated by the MYTK markerless integration vectors at 10 newly-defined genomic loci (`Int.1`…`Int.10`) plus the `ura3` control locus. Steady-state mScarlet fluorescence read by flow cytometry, reported as fold-over-background. 3 biological replicates per (locus, promoter). |
| **Expression label** | `mean`: scalar per (locus, promoter), mean fluorescence fold-over-background (no log). Per-promoter base levels differ ~100× (pTDH3 ≈ 1300, pRPL18B ≈ 185, pREV1 ≈ 8); the within-promoter spread across the 11 sites is the small (~±15–35 %) position-effect signal. |
| **Test set size** | **33 constructs (11 loci × 3 promoters), 0 drops.** Zero-shot — no training split on our side. |
| **Primary metric** | **Per-promoter Spearman ρ across the 11 integration sites** (3 values + their mean). Holding the promoter fixed, does the model rank integration sites correctly? This is the position-effect signal; n = 11 each. |
| **Secondary metric** | **Pooled Spearman ρ across all 33.** Dominated by the >100× cross-promoter strength difference, so any non-broken model scores high; reported for context, not as the headline. |
| **Adapter protocol** | `PromoterIntegrationExpressionPredictor` — distinct from Hong's `IGRInsertionExpressionPredictor` because the promoter is part of the varied construct (Hong's promoter is fixed; only the site varies). |

## Why this benchmark exists

Most of our position-effect data (Hong, Wu) holds the promoter fixed and varies
only the genomic site. MYTK adds an orthogonal axis: the *same* set of
integration sites is measured with three promoters spanning ~100× in strength.
That lets us ask a sharper question — does a sequence-to-expression model
predict **position effects** (how a given promoter's output changes across
integration sites), separately from the much easier **promoter-strength
ranking**.

## Why per-promoter is primary and per-site is not reported

The 33 points form an 11 × 3 matrix, but the two axes are **not** statistically
symmetric:

- **Across sites, per promoter (n = 11):** the position-effect signal. Small
  (~±15–35 % swings), genuinely hard, and n = 11 is enough for a meaningful
  Spearman. → **primary** (3 ρ's + their mean).
- **Across promoters, per site (n = 3):** would measure only whether the model
  ranks pTDH3 > pRPL18B > pREV1. Two problems make it near-useless as a metric:
  (1) n = 3 is statistically degenerate — Spearman can only land on a tiny
  discrete set; (2) the >100× promoter separation makes ρ ≈ +1 trivially for any
  non-broken model, so it carries almost no power to discriminate models.
  → **not reported.**
- **Pooled (n = 33):** captures the cross-promoter strength ranking more
  robustly than 11 noisy n = 3 correlations, and is honest about being
  promoter-dominated. → **secondary / context.**

Treating the two axes symmetrically would be misleading: the per-site column
would sit at ~1.0 and visually swamp the per-promoter column (the hard part).
Promoter × site *interaction* effects are a legitimate question but need a
two-way analysis on the 33 points, not per-site Spearman; deferred.

## The construct

For each (locus, promoter) the model is fed native genomic sequence around the
integration site, with the **full integrated payload** spliced in at a single
point and the native flanks left intact (**Hong-style point insertion**, not a
deletion/replacement):

```
─── native genome ───[ full integrated payload (promoter … mScarlet … terminator … marker) ]─── native genome ───
                      ▲
                      integration_coord (midpoint of the 5′/3′ span)
```

- **Insertion point:** the midpoint of each locus's `integration_coord_5_prime`
  / `integration_coord_3_prime` span. The 5′/3′ coordinates are homology-arm
  targets; the payload is modelled as inserted at their midpoint with native
  sequence otherwise unchanged.
- **Full payload, per promoter:** unlike Hong (which inserts only the
  `promoter-mScarlet-terminator` transcription unit), here the model sees the
  *entire* integrated DNA — including any selection marker / vector backbone
  that lands at the site. The three payloads differ only by the promoter region;
  everything else is constant.
- **Readout region — the mScarlet CDS:** predicted coverage is summed over the
  mScarlet CDS only (not the whole payload — marker/vector sequence would dilute
  the signal). The CDS is located inside each payload by exact substring match
  against the `mScarlet_cds` reference record (see [Files](#sequences-to-add)),
  so no manual offsets and correct for each promoter regardless of payload
  length.

> **Receptive-field caveat.** A full payload (with marker/vector) can be several
> kb. Yorzoi's window is ~5 kb and Shorkie's ~16 kb, so a large payload plus
> native flanks may fill or exceed Yorzoi's window in particular — the adapter
> must place the window so the mScarlet CDS stays inside the output crop, and a
> payload that overruns the window is a real limitation to record per model.

## Evaluation protocol

For each of the 33 constructs, in order:
1. Build the input window: native context around the locus midpoint with the
   full per-promoter payload inserted at the midpoint.
2. Forward-pass; REF baseline (native, no payload) cached per locus.
3. Read out predicted expression over the mScarlet-CDS bins (CDS located by
   substring match within the payload).
4. Per-promoter Spearman ρ + Pearson r across the 11 sites; pooled across 33.

### What we're *not* doing in v1
- **Per-site ρ** (n = 3) — see rationale above.
- **Promoter × site interaction** modelling.
- **Bootstrap CIs / replicate-noise ceiling** — `sd` is carried in the table for
  context but not yet used as a noise floor.

## Files

### Inputs

- `data/tasks/mytk_ints_promoter/mytk_ints_promoters_expression.csv` — raw SI
  expression export (wide, spreadsheet layout).
- `data/tasks/mytk_ints_promoter/mytk_int_coordinates.csv` — raw per-locus
  integration coordinates (`int_locus_id, chromosome, integration_coord_5_prime,
  integration_coord_3_prime, difference`), 1-based.

### Processed distribution (built)

- `data/tasks/mytk_ints_promoter/mytk_ints_promoter.tsv` — the benchmark table,
  one row per (locus, promoter). Built by
  `scripts/mytk/build_expression_table.py`:

  | Column | Meaning |
  | --- | --- |
  | `locus_id` | `ura3` / `Int.1` … `Int.10` |
  | `promoter` | `pTDH3` / `pRPL18B` / `pREV1` |
  | `chrom` | Roman contig name (`I`…`XVI`), matching the R64 FASTA |
  | `integration_coord` | 1-based midpoint of the 5′/3′ span |
  | `rep1`,`rep2`,`rep3` | the 3 replicate fluorescence values |
  | `mean` | label = mean fluorescence (fold-over-background) |
  | `sd` | sample SD (÷N−1), matching the published table; context only |

  The build maps the raw `chromosome` integers to Roman via the same
  `ARABIC_TO_ROMAN` convention as `yeastbench.adapters._genome`.

### Sequences (to add)

A single multi-record FASTA, `data/tasks/mytk_ints_promoter/cassette_payloads.fasta`,
assembled in Benchling — **4 records**:

```
>pTDH3
…           full integrated payload for the pTDH3 construct, 5′→3′
>pRPL18B
…           full integrated payload for the pRPL18B construct
>pREV1
…           full integrated payload for the pREV1 construct
>mScarlet_cds
ATG…TAA     the mScarlet CDS reference (readout region)
```

Format rules:
1. The three **payload record names must exactly match** the `promoter` values
   in the TSV (`pTDH3`, `pRPL18B`, `pREV1`) — the adapter resolves id → payload
   by name.
2. Each payload is the **complete integrated DNA**, 5′→3′, in the orientation it
   is integrated (the cassette is self-contained; the adapter handles RC
   averaging).
3. The `mScarlet_cds` record must occur **exactly once** as a substring of each
   payload — the adapter locates the readout region by exact match (the adapter
   asserts a unique hit).
4. **Uppercase `ACGT`.**

## Open items

- **Sequences:** the `cassette_payloads.fasta` above (3 full payloads +
  `mScarlet_cds`).
- **Adapters:** `shorkie_mytk.py` / `yorzoi_mytk.py` implementing
  `PromoterIntegrationExpressionPredictor` (likely on the shared
  marginalized/insertion scaffold, as this is Hong-shaped), then registered in
  `SHORKIE_ADAPTERS` / `YORZOI_ADAPTERS`. Needs the sequences + a GPU.
- **Reference assembly:** confirm which R64 release the integration coordinates
  are in (R64-1-1 vs R64-5-1) and point `fasta_path` at the matching FASTA — the
  coordinates must agree with the FASTA the adapter fetches from.
- **Config:** a `configs/*.yaml` entry wiring `data_path` + `fasta_path` for the
  `mytk_ints_promoter` task once an adapter exists.
- **Data manifest:** once finalised, declare the processed artifact + FASTA in
  `src/yeastbench/data/manifest.py` + lock so `ybench data` can fetch them.
