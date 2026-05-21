# Cuperus et al. — 5′-UTR MPRA expression (marginalized)

> **Status:** draft v0 — initial spec. Several TODOs remain; see *Open
> questions* at the bottom.

## At a glance

| | |
| --- | --- |
| **Task** | Regression: predict scalar protein-expression effect of a 50 bp random 5′-UTR sequence by marginalizing each sequence's predicted logSED across a fixed set of 22 native host genes. |
| **Source** | Cuperus *et al.* 2017, *Deep learning of the regulatory grammar of yeast 5′ UTRs from 500,000 random sequences*, Genome Research 27:2015–2024. DOI: [10.1101/gr.224964.117](https://doi.org/10.1101/gr.224964.117). PMC: [PMC5741052](https://pmc.ncbi.nlm.nih.gov/articles/PMC5741052/). Code: [Seeliglab/2017---Deep-learning-yeast-UTRs](https://github.com/Seeliglab/2017---Deep-learning-yeast-UTRs). |
| **Assay** | Massively parallel growth-based enrichment. ~500 k 50 bp random 5′-UTRs cloned upstream of a `HIS3` ORF under a `CYC1` promoter + `CYC1` terminator. Yeast pool grown in SD-His + 1.5 mM 3-AT (His3 inhibitor) for ~6.2 doublings; pre/post-selection plasmid DNA deep-sequenced; per-sequence **log₂ enrichment** estimated. Enrichment scales with His3 protein level, which scales with 5′-UTR translational efficiency and mRNA stability. |
| **Expression label** | Scalar per sequence: **log₂ enrichment** post- vs pre-selection. Higher = better 5′-UTR (more His3 protein → faster growth → over-representation). Not bounded; enrichment scores differ between experiments and are not normalized to any reference sequence. |
| **Test set size** | **~24,474** random sequences — the top 5 % of the 489,348-variant library by input read depth (as defined in the paper's CNN evaluation). A second "native" set of 11,962 yeast-genome-derived 50 bp UTR fragments is available as a secondary eval. |
| **Primary metric** | Overall Pearson *r* on `(pred_logSED, log2_enrichment)` across the test set. Spearman ρ reported alongside. |
| **Adapter protocol** | `FivePrimeUtrMarginalizedExpressionPredictor` — new protocol, parallel to `MarginalizedSequenceExpressionPredictor` (promoter / Rafi) and `TerminatorMarginalizedExpressionPredictor` (Shalem). Distinct because the insertion site and scaffold semantics differ: the 5′-UTR MPRA inserts immediately upstream of the host ATG, with no filler. |

## Why this benchmark exists

The Cuperus library is the only large yeast 5′-UTR MPRA at this scale. It probes a regulatory layer — translation initiation, uORFs, Kozak context, 5′-UTR secondary structure, NMD-mediated mRNA decay — that is underrepresented in the promoter (Rafi) and terminator (Shalem) benchmarks. Running it zero-shot against sequence-to-expression models tests whether models trained on native-genome RNA-seq capture 5′-UTR-mediated regulation.

The marginalized approach is chosen for the same reasons as Rafi / Shalem marginalized:

1. In-distribution input (native genomic context + local perturbation, not a plasmid scaffold with `HIS3` where the model was trained on `YHR018C`).
2. Isolates the insert's effect via logSED, cancelling model miscalibration and scaffold noise.
3. Matches the relative nature of the enrichment label (log-ratio, not absolute).

### Caveat: RNA-seq vs translation

The Cuperus label reflects **protein level** (His3 activity → growth). 5′-UTR changes primarily affect **translation** (ribosome loading, uORFs, secondary structure). Shorkie and Yorzoi predict RNA-seq coverage, which is mRNA abundance — only indirectly sensitive to translation. mRNA-level effects we can expect to measure:

- **uORFs triggering NMD** → mRNA abundance drops.
- **Secondary structure affecting mRNA stability** → mRNA abundance shifts.
- **5′-UTR length extending past TSS signals** → transcription-start-site choice or abundance shifts.

Pure translational-efficiency differences without NMD/stability downstream effects will not show up in RNA-seq predictions. The benchmark is therefore expected to correlate more weakly than Rafi / Shalem; the headline number measures the fraction of 5′-UTR regulation that is *transcriptionally* visible.

## Construct (per host gene)

For each of the 22 selected yeast host genes, at each test sequence evaluation, the model is fed a window of native genomic sequence centred on the host gene's CDS, with the **50 bp immediately upstream of the ATG replaced** by the test 5′-UTR:

```
─── transcription direction ───

[native genomic upstream, up to gene_start − 51]
[50 bp CUPERUS INSERT]                  ← replaces native bp (gene_start−50) … (gene_start−1)
[native host CDS starting at ATG = gene_start]
[native genomic downstream]

─── rest of window native ───
```

- Only the 50 bp insert varies per test sequence.
- **No additional filler.** The downstream is the host's own native CDS + 3′-UTR (what we want). The upstream is the host's own native promoter (also what we want). The 50 bp replacement is smaller than most native 5′-UTRs (yeast 5′-UTRs are typically 50–200 bp) so some of the native UTR may remain between the replacement and the host's TSS — this is acceptable; the insert sits immediately 5′-of-ATG where Kozak context and NMD-detection mechanisms are most sensitive.
- For **− strand host genes** the 50 bp insert is reverse-complemented and placed at `[gene_end + 1, gene_end + 51)` in genomic coords.

## Host gene selection

22 host genes (10 positive-strand, 12 negative-strand) — analogous to Shalem, but filter is flipped to check **upstream** clearance.

### Filter (must pass)
1. `gene_biotype == "protein_coding"`.
2. CDS length ≥ 300 bp.
3. No same-strand gene **ends** within 500 bp upstream of the host's start codon (so the 50 bp replacement doesn't clobber a neighbour's terminator / 3′-UTR).
4. No convergent (opposite-strand) gene overlaps the 500 bp upstream region.
5. Gene fits within Shorkie (16 384 bp) and Yorzoi (4992 bp) window placement with chromosome-edge margin.
6. DEE2 median TPM ≥ 1.0 (reuse of `data/tasks/dee2_gene_median_tpm.tsv` from Shalem).

### Diversification
- Strand balance: 10 +, 12 − (matches Rafi / Shalem).
- TPM-tertile stratification: `{+low 3, +med 3, +high 4, −low 4, −med 4, −high 4}`.

### Selection artifact
`scripts/cuperus/select_host_genes.py` → `data/tasks/cuperus_mpra_5utr/host_genes.json`. Shares the DEE2 TPM table and tertile logic with Shalem's selection script; only the "which side of the gene to check for clearance" differs.

## Evaluation protocol

For each of the ~24,474 test sequences:
1. For each of 22 host genes:
   a. Build the input window: native context around host CDS, with the 50 bp replacement at `gene_start − 50` (+ strand) / `gene_end + 1` − strand RC.
   b. Forward-pass the model. `REF` baselines (native context, no replacement) are pre-cached per host gene.
   c. Compute logSED over the host gene's exon bins using the `logSED_agg` convention.
2. Mean logSED across 22 host genes → scalar per test sequence.
3. **Overall Pearson *r*** on `(pred, log2_enrichment)`. **Spearman ρ** alongside.
4. Plot: scatter of pred vs measured with regression line + r/ρ annotated.

### Not in scope for v1
- **Native-UTR stratum** (11,962 yeast-genome 5′-UTRs) — add as a secondary run with a contamination flag (models saw these during training).
- **Evolved stratum** (573 iteratively-optimized sequences) — nice sanity check but small.
- **Per-Kozak stratified Pearson** (the paper characterizes this extensively). Defer.
- **Bootstrap CIs.**

### Track subsets
- Shorkie: 384 T0 RNA-seq tracks (same as Rafi / Shalem marginalized).
- Yorzoi: 81 plus-strand tracks for + strand host genes, 81 minus-strand for − strand (same strand-matched scheme as Rafi / Shalem marginalized).

### RC averaging
Both adapters average forward + RC. Yorzoi swaps strand tracks on the RC pass.

## Files (target layout)

### Raw upstream
- `data/tasks/cuperus_mpra_5utr/random_utrs_enrichment.tsv` — per-sequence `(sequence, log2_enrichment)`, pulled from the Seeliglab GitHub repo or reconstructed from GEO GSE104252. **Not yet vendored — see open question below.**
- `data/tasks/dee2_gene_median_tpm.tsv` — shared with Shalem.

### Processed distribution
- `data/tasks/cuperus_mpra_5utr/host_genes.json` — 22 host genes + selection metadata.
- `data/tasks/cuperus_mpra_5utr/test_set.tsv` — the ~24,474 top-read-depth sequences used by the paper's CNN evaluation; pinned by SHA256 in a manifest.

### Sequence format

Each entry is exactly 50 bp of random ACGT. No primer flanks, no barcode — the sequence IS the variable region. Adapters insert the 50 bp as-is (positive-strand host genes) or reverse-complemented (negative-strand host genes).

### Sign convention (to verify)

Higher `log2_enrichment` = better 5′-UTR = more His3 protein. Positive logSED from the adapter = model predicts the insert boosts host-gene mRNA abundance. Expected correlation direction: positive. **Needs empirical verification after first run — the transcription-vs-translation indirection makes this less certain than for promoter / terminator benchmarks.**

## Open questions / TODO

- **Label table acquisition.** The paper does not provide a single master TSV with all 489,348 sequences and their enrichment scores. GEO GSE104252 has raw reads; the Seeliglab GitHub repo has the CNN training code and presumably the processed labels. Need to either:
  1. Locate the processed `(sequence, enrichment)` table in the Seeliglab repo (most likely under a `data/` or `processed/` directory).
  2. Re-process from GEO (align reads → count per-sequence → compute enrichment). Large effort.
  
  Action: inspect the Seeliglab repo. Block implementation until we have the labels.
- **Test-set definition.** The paper states "top 5 % by input read depth" without giving the exact 24,474-row index. Need to either reproduce that filtering (requires the read-count column to be available) or pick an equivalent subset and document the choice.
- **Native-UTR stratum.** 11,962 yeast native 5′-UTR fragments are a promising *secondary* eval (out-of-training-distribution for synthetic libraries, but native yeast sequences that the models have seen). Decide whether to include in v1 or defer.
- **Host-gene selection script.** Adapt `scripts/shalem/select_host_genes.py` → `scripts/cuperus/select_host_genes.py`. The only difference is the upstream-clearance check and the JSON output path. Consider refactoring to a shared selection helper with a `side ∈ {"upstream", "downstream"}` parameter.
- **Shared infrastructure.** `_cuperus_scaffold.py` will be near-trivial (no filler, small 50 bp replacement). Consider unifying with the Shalem `_shalem_scaffold.py` if the host-gene / insertion abstractions naturally share.
- **Replacement length vs native-UTR overlap.** The 50 bp replacement sits right at the ATG. Some host genes may have natural regulatory elements just upstream of the replacement (within their 5′-UTR). v1 accepts this as a limitation — the benchmark is not a pure isolate of the 50 bp effect, it's the 50 bp effect *in the context of the host's surrounding regulatory landscape*. Documented as a known caveat.
- **Expected correlation strength.** Given the RNA-seq-vs-translation indirection (see *Caveat* above), we should not be surprised by a lower Pearson than Rafi / Shalem marginalized. A negative result (r ≈ 0) is a legitimate finding — "RNA-seq-trained models don't capture translation-level regulation" — but a moderate positive (say 0.3–0.5) would be consistent with NMD + structural-stability effects making it through to mRNA abundance.
