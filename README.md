# Benchmark Yeast Sequence-to-Expression Models

> [!NOTE]
> In case of any questions, reach out to mail@timonschneider.de — always happy to help!

A collection of datasets and scripts to benchmark models that predict gene
expression from DNA sequence in *S. cerevisiae*. The goal is a single,
reproducible way to compare sequence-to-expression models across a shared
set of tasks — eQTL classification, MPRA generalization, and native-genome
track prediction.

**Status:** under construction. Currently live:

- `caudal_eqtl` — Caudal et al. *cis*-eQTL classification (Shorkie, Yorzoi
  adapters). Full spec: [`benchmarks/caudal_eqtl.md`](benchmarks/caudal_eqtl.md).

In flight: Kita et al. eQTL, Rafi et al. MPRA (promoter), Shalem et al. MPRA
(terminator), native-genome track-prediction comparison.

## Install

The benchmark framework uses [`uv`](https://docs.astral.sh/uv/). Model-specific
dependencies (PyTorch, Shorkie weights, Yorzoi) are isolated behind extras so
the core install stays light.

```bash
# Minimal install: core benchmark framework + data loaders (no model deps)
uv sync

# Add specific model dependencies as needed
uv sync --extra shorkie   # PyTorch + h5py, for the Shorkie adapter
uv sync --extra yorzoi    # yorzoi + flash-attn, for the Yorzoi adapter
uv sync --extra all       # both models
```

## Running the benchmark

The repo ships a unified CLI, `ybench`, driven by a YAML run-spec. The
committed `configs/default.yaml` is the canonical run — it captures
which `(model, task)` pairs to evaluate with what per-run settings, and
is the single source of truth for the numbers we report.

```bash
# List registered models and tasks
uv run ybench list

# Preview the planned runs without executing
uv run ybench run --config configs/default.yaml --dry-run

# Execute every (model, task) pair in the config
uv run ybench run --config configs/default.yaml

# Filter to a single model or task
uv run ybench run --config configs/default.yaml --model shorkie
uv run ybench run --config configs/default.yaml --task  caudal_eqtl
```

### Output layout

One directory per `(model, task)` pair, under the config's `out_dir`:

```
results/default/
  shorkie__caudal_eqtl/
    negset_{1..4}_scores.npy   # per-iteration raw scores
    negset_{1..4}_labels.npy   # per-iteration labels
    negset_{1..4}_pairs.tsv    # per-pair metadata (pair_id, distances)
    summary.json               # per-iter + aggregate AUROC / AUPRC
    run_metadata.json          # config hash, git commit, timestamp
  yorzoi__caudal_eqtl/
    …
```

`run_metadata.json` captures everything needed to reproduce that directory's
numbers — the config hash, repo git commit, resolved model/task configs,
and timestamp. Raw scores + labels + pair metadata are persisted so
post-hoc analyses (distance-stratification, signed-vs-absolute comparison,
etc.) don't require re-running the model.

### Adding a new model or task

- **Model:** add one entry to `MODELS` in `src/yeastbench/registry.py`
  with signature `(task, device, **model_config) -> VariantEffectScorer`.
  Then reference it by name in the YAML.
- **Task:** add one entry to `TASKS` with signature
  `(**task_config) -> Benchmark`. Then reference it in the YAML.

No new runner scripts, no new CLI wiring.

## Repository layout

```
benchmarks/                benchmark specs (one markdown per task)
configs/                   YAML run-specs (committed canonical runs)
data/
  raw/                     raw upstream files (FASTA, GTF, GWAS, gVCF)
  processed/               versioned processed distributions per task
  models/                  downloaded model weights + targets sheets
scripts/eqtl/              one-off data-preparation scripts
src/yeastbench/
  adapters/                model adapters (Shorkie, Yorzoi, …)
  benchmarks/              task implementations (eQTL, MPRA, …)
  models/                  vendored pure-PyTorch model ports
  registry.py              MODELS + TASKS registries
  cli.py                   the `ybench` CLI
  config.py                YAML config loader
```

## Roadmap

- [x] Caudal et al. eQTL eval (Shorkie, Yorzoi)
- [ ] Implement `Benchmark.plot()` with ROC / PR + random baseline + distance stratification
- [ ] Measured-RNA-seq oracle baseline (upper-bound reference for eQTL benchmarks)
- [ ] Kita et al. eQTL eval
- [ ] Rafi et al. MPRA (promoter)
- [ ] Shalem et al. MPRA (terminator)

## Contact

Reach out to mail@timonschneider.de in case you have questions, need help,
or want to chat.
