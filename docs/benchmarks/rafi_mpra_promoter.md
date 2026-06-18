# Rafi / deBoer — random-promoter MPRA expression

![image](/img/mpra_banner.svg)

> **Status:** zero-shot marginalized eval **implemented** (Shorkie, Yorzoi).
> DREAM-RNN supervised baseline — **implemented + verified** (`dream_rnn` model;
> `tests/test_dream_rnn.py`). The published `0_1_1_0` checkpoint loads
> `strict=True` into the port, and a full end-to-end run reproduces a strong
> in-distribution reference (overall Pearson *r* ≈ 0.97; per-stratum *r*: broad
> strata 0.96–0.98, range-restricted high/low 0.72–0.74, native 0.89; SNV
> pair-difference Δ*r* ≈ 0.86). **Data-op follow-up:** publish
> `model_best.pth` + `plasmid.json` to the HF/GCS mirrors so `ybench data get
> dream_rnn` works on a fresh checkout (lock entry already committed).

## At a glance

| | |
| --- | --- |
| **Task** | Regression: predict scalar expression for 71,103 random-promoter sequences across eight DREAM test-set strata. |
| **Source** | Rafi, Nogina, Penzar, *et al.* 2024, *A community effort to optimize sequence-based deep learning models of gene regulation*, Nat Biotechnol 43:1373–1383. DOI: [10.1038/s41587-024-02414-w](https://doi.org/10.1038/s41587-024-02414-w). The Random Promoter DREAM Challenge 2022. |
| **Assay** | GPRA. 80 bp random inserts cloned upstream of YFP in the `yeast_DualReporter` vector (AddGene 127546); FACS-sorted into 18 bins on `log2(RFP/YFP)`; per-sequence expression = MAUDE-fit bin mean. |
| **Expression label** | Scalar per sequence (`el` column, MAUDE expression). Stored at `data/tasks/rafi_mpra/filtered_test_data_with_MAUDE_expression.txt` (71,103 rows, tab-separated `seq`, `el`). |
| **Sequence format** | All 71,103 rows are exactly **110 bp** = 17 bp constant left adapter (`TGCATTTTTTTCACATC`) + 80 bp variable insert + 13 bp constant tail (`GGTTACGGCTGTT`). Verified: 100 % of rows match this layout. |
| **Strata** | 8 subsets via `data/tasks/rafi_mpra/test_subset_ids/*.csv`: `high_exp`, `low_exp`, `yeast_exp`, `random_exp`, `challenging` (singleton) + `SNVs`, `motif_perturbation`, `motif_tiling` (pairs). |
| **Primary metric** | Per-stratum Pearson *r* + Spearman ρ on `(pred, el)`; for the three pair strata, also pair-difference Pearson/Spearman on `(pred_alt − pred_ref, el_alt − el_ref)`. |
| **Adapter protocol** | `SequenceExpressionScorer.predict_expression_scores(seqs) -> np.ndarray` (renamed from `MarginalizedSequenceExpressionPredictor` — see *Protocol rename*). |

## Two evaluation modes on one benchmark

The benchmark feeds every entrant the **same** 110 bp sequences and asks for one scalar per sequence; `MPRAMarginalizedBenchmark.evaluate` then correlates those scalars against the measured expression per stratum (`src/yeastbench/benchmarks/mpra.py:138`). It does not prescribe *how* the scalar is produced — that is the model's business. Two models answer the same ranking question through different machinery:

1. **Zero-shot, marginalized (Shorkie, Yorzoi)** — *implemented.* The insert is spliced into 22 native host-gene loci 180 bp upstream of the TSS; the scalar is the mean logSED (log fold-change in predicted host-gene expression) across loci (`adapters/_marginalized_mpra.py`). This is the honest zero-shot probe for genomic foundation models — they predict *effects in native context*, never reporter readouts.

2. **Supervised, in-distribution (DREAM-RNN)** — *this spec.* A model trained directly on this assay predicts the reporter expression of the insert itself, in its own plasmid context. No host genes, no marginalization.

Both emit a per-sequence scalar, both flow through the identical per-stratum metric. Because the metrics are Pearson/Spearman (scale- and shift-invariant), "mean logSED" and "predicted reporter expression" are compared on equal footing: each just has to *rank* sequences by expression. The measured label **is** reporter expression, which the supervised model predicts directly, so it sets the in-distribution reference a zero-shot model is reaching for.

## The supervised baseline: DREAM-RNN

### What it is — one model, not an ensemble

DREAM-RNN is a **single neural network** with a single set of weights. The DREAM paper built it with the *Prix Fixe* framework: it deconstructed the top-three challenge submissions into interchangeable blocks and searched all combinations; DREAM-RNN is the best combination built around the BHI team's recurrent core. The "composite" is in the *architecture's parts* (blocks sourced from different teams' designs, stacked into one `nn.Module`), not in combining models at inference. The paper endorses it as the best-*generalizing* of the optimized models; on the yeast task itself it scores ~0.815 weighted Pearson, within noise of the best (DREAM-CNN 0.822).

We use **only** DREAM-RNN — not DREAM-CNN, DREAM-Attn, or any original team model.

### Architecture (per the reference implementation)

Assembled as `PrixFixeNet(first, core, final)`, following the de-Boer-Lab `DREAMNets_BuildModel_Train_and_Predict.ipynb` and the vendored eval (`shorkie-paper/from_kuanhao/eQTL/data/eQTL_MPRA_models_eval/2_predict_seq_pos.py:90–123`):

| Block | Upstream class | What it is |
| --- | --- | --- |
| first | `BHIFirstLayersBlock` | parallel multi-kernel conv (kernels 9, 15; out 320; dropout 0.2) |
| core | `BHICoreBlock` | **bidirectional LSTM** (hidden 320/dir → 640) + multi-kernel conv (out 320; dropout1 0.2, dropout2 0.5) |
| final | `AutosomeFinalLayersBlock` | 1×1 conv → 18 bins → global avg-pool → softmax → expected-value scalar |

> The upstream class names record which *team* designed each block; they are not separate models. When porting, rename to neutral single-model names (`DreamRnn`, block classes without the team prefixes) so nothing implies more than one model. The paper's Methods prose calls DREAM-RNN's first block "same as DREAM-CNN" (Autosome's two-kernel conv); the reference code uses `BHIFirstLayersBlock` — both are the same kernel-9/15 design, and we follow the code since that matches the distributed weights.

### Input / output

- **Input length 150 bp, 6 channels.** The 80 bp insert is placed back in the plasmid context the model was trained on and 5′-padded to 150 bp; encoded as 4 one-hot base channels (`N → 0.25`) + a reverse-flag channel (0 forward / 1 RC) + a zero singleton channel.
- **Reflanking (= "crop to the variable sequence").** The wrapper strips the 17 bp left adapter, prepends 150 bp of upstream plasmid from `plasmid.json` (the `N×80` slot at index 3648; the 17 bp adapter sits immediately upstream at 3631), and keeps the last 150 bp → `[57 bp plasmid incl. adapter] + [80 bp insert] + [13 bp tail]`. This discards all genomic/host context — the supervised model never sees it.
- **Output: one scalar** per sequence (softmax over 18 bins → expected bin value). Forward + reverse-complement are run through the same single network and averaged (test-time augmentation, not an ensemble).

### Weights

Zenodo record **10633252** (DOI [10.5281/zenodo.10633252](https://doi.org/10.5281/zenodo.10633252)), `prixfixe_model_weights.tar.gz` (2.3 GB). DREAM-RNN is **`0_1_1_0/model_best.pth`** — a plain torch `state_dict`. No retraining needed.

> **Combo-index scheme (verified, not from notebook comments).** The dir name is `<dataprocessor>_<first>_<core>_<final>` with team codes `0`=Autosome, `1`=BHI, `2`=UnlockDNA. So `0_1_1_0` = Autosome data-processor + BHI first + **BHI (Bi-LSTM) core** + Autosome final = DREAM-RNN; `0_1_0_0` = BHI first + Autosome (CNN) core = DREAM-CNN; `0_0_2_0` = Autosome first + UnlockDNA (attention) core = DREAM-Attn. This was **confirmed by a strict `load_state_dict`**: `0_1_1_0` has `core.lstm.*` and `final.mapper.0` in_channels 320; `0_1_0_0` has an Autosome `core.seqextractor.*` and final in_channels 64. The de-Boer notebook comments label `0_1_0_0` as RNN — that is wrong; the vendored eval script's mapping is right.

If we ever retrain instead: PyTorch, ~80 epochs, batch 1024, single 16 GB GPU, training data on the same Zenodo record (6,739,258 sequences).

### Adapter

A new `dream_rnn` model whose adapter implements the protocol by running the network directly — no host-gene loop:

```python
class DreamRnnRafiPredictor(SequenceExpressionScorer):
    def predict_expression_scores(self, seqs):
        return np.array([self._predictor.predict(s) for s in seqs])
```

`self._predictor` wraps the one DREAM-RNN net and does the reflank → encode → forward+RC → scalar pipeline above. Cheap: ~142 k forward passes of a ~4 M-param net (71,103 × 2), seconds-to-minutes on GPU, vs. the foundation models' 22-loci marginalization.

## Protocol rename

`MarginalizedSequenceExpressionPredictor.predict_marginalized_expressions` is a misnomer for a model that does no marginalizing. Rename to a neutral name that is honest for both modes (a per-sequence scalar that should track expression, however computed):

- protocol `MarginalizedSequenceExpressionPredictor` → **`SequenceExpressionScorer`**
- method `predict_marginalized_expressions` → **`predict_expression_scores`**

Pure rename, behavior-preserving. Touches `adapters/protocols.py`, the marginalized base (`adapters/_marginalized_mpra.py`), the Rafi and Shalem adapters + benchmarks (both use this protocol), and `registry.py`. Avoid the name `SequenceExpressionPredictor` — that protocol was deleted on 2026-05-21 and reusing it would confuse.

## Reporting: fed its native substrate

There is no model-class/tier concept in the repo (`registry.py` keys are flat strings; `compare.py` groups by task only). To keep the supervised baseline visually distinct from the zero-shot models on the shared per-stratum plots:

1. **Style `dream_rnn` distinctly** in `benchmarks/mpra.py:compare_plot` (e.g. a hatched / outlined bar or a pinned reference color), so it doesn't read as one more zero-shot model.
2. **Caption requirement (mandatory):** every figure that places DREAM-RNN beside Shorkie/Yorzoi must state that *each model is fed its native substrate* — foundation models scored by marginalized logSED at native loci, DREAM-RNN scored on the reporter insert in its own plasmid context. Without this, the figure implies identical inputs, which is false.

## Files

### To add
- `data/tasks/rafi_mpra/plasmid.json` — 8,294 bp dual-reporter plasmid string, `N×80` insert slot at index 3648. Source: de-Boer-Lab/random-promoter-dream-challenge-2022 `data/plasmid.json` (`main`). Wire through the data manifest (`manifest.lock.json`), not a relative-cwd `open()` like the upstream code.
- DREAM-RNN weights `0_1_1_0/model_best.pth` from Zenodo 10633252 — stage like other model weights (not under `data/tasks/`).
- `src/yeastbench/models/dream_rnn/` — ported net (`PrixFixeNet` + the three blocks, neutrally renamed) + the predictor/preprocessing (`n2id`, `revcomp`, reflank, 6-channel encode, fwd+RC average).
- `src/yeastbench/adapters/dream_rnn_rafi.py` — `DreamRnnRafiPredictor`.

### To change
- `adapters/protocols.py`, `_marginalized_mpra.py`, Rafi + Shalem adapters/benchmarks, `registry.py` — protocol rename.
- `registry.py` — register `dream_rnn`; `configs/default.yaml` — enable `dream_rnn` on `rafi_mpra_marginalized`.
- `benchmarks/mpra.py:compare_plot` — distinct style + caption note.

### Already present
- `data/tasks/rafi_mpra/filtered_test_data_with_MAUDE_expression.txt` (71,103 rows), `test_subset_ids/*.csv`, `public_leaderboard_ids/`.

## Caveats

- **Not an upper bound on the proxy task.** DREAM-RNN predicts the literally-measured quantity (reporter expression); the foundation models predict a native-context proxy. The comparison is fair as a *ranking* (what the plots show) and DREAM-RNN is the in-distribution reference — but it is not "the same model evaluated on the same inputs." The caption carries this.
- **Native / yeast_exp stratum.** DREAM-RNN was trained on random promoters; native-derived test sequences are themselves harder for it, so its reference value on `yeast_exp` is not a hard ceiling.
- **License.** Confirm the de-Boer-Lab repo license permits vendoring the blocks + `plasmid.json` into this repo before committing.
- **Adapter prefix assert.** All 71,103 sequences start with the 17 bp adapter (verified), but keep an assert in the reflank step so a malformed input can't be silently corrupted.
