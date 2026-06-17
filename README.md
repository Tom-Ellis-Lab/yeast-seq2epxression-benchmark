# ybench: a benchmark for fungal sequence-to-expression models

![image](img/readme_banner.svg)

`ybench` is a comprehensive set of ten tasks to benchmark yeast-focussed sequence-to-function (S2F) models like Shorkie and Yorzoi. Each task covers a different part of the genomes regulatory complexity from eQTLs, promoter/UTR/terminator [MPRAs](https://www.google.com/search?q=mpra+massively+parallel+reporter+assay), reporter gene insertions in different genomic locations to investigate context effects as well as exogenous but linearised bacterial DNA in yeast and structurally rearranged chromosomes.

## Table of Contents

- [Benchmark Tasks](#benchmark-tasks)
- [Models](#models)
- [Install](#install)
- [Getting the data](#getting-the-data)
- [Running the benchmark](#running-the-benchmark)
- [Extending the benchmark](#extending-the-benchmark)
- [Repository layout](#repository-layout)
- [Roadmap](#roadmap)
- [Contact](#contact)

## Benchmark Tasks

Please find a more comprehensive overview in [docs/benchmarks/](docs/benchmarks).

| Benchmark | Description | Task | Primary metric |
| --- | --- | --- | --- |
| [Caudal eQTL](docs/benchmarks/caudal_eqtl.md) | single-nucleotide *cis*-eQTLs vs matched controls | binary classification | AUROC / AUPRC (mean ± SEM over 4 negative sets) |
| [Kita eQTL](docs/benchmarks/kita_eqtl.md) | single-nucleotide *cis*-eQTLs (independent panel) | binary classification | AUROC / AUPRC |
| [Rafi / deBoer MPRA (promoter)](docs/benchmarks/rafi_mpra_promoter.md) | ~71k random 80 bp promoters in a dual-reporter plasmid GPRA | regression (marginalized logSED) | Pearson / Spearman |
| [Shalem MPRA (terminator)](docs/benchmarks/shalem_mpra_terminator.md) | designed 3′-end / terminator variants (cleavage & termination) | regression (marginalized logSED) | Pearson *r* (Spearman alongside) |
| [Chen synonymous MPRA](docs/benchmarks/chen_synonymous.md) | synonymous-codon effect on mRNA abundance | regression, 3 libraries | Pearson *r* + Spearman ρ on `log2(R/D)` | 
| [Wu RFP insertions](docs/benchmarks/wu_rfpins.md) | fixed RFP cassette across ORF-deletion loci (position effect) | regression | Pearson *r* + Spearman ρ (+ tail AUROC) | 
| [Hong IGR insertions](docs/benchmarks/hong_igr.md) | fixed reporter across intergenic loci (position effect) | regression | Spearman ρ on IntProp | 
| [Brooks SCRaMBLE](docs/benchmarks/brooks_scramble.md) | SCRaMBLE rearrangement (altered neighbour / downstream context) | coverage-track LFC | direction balanced accuracy, then Spearman / Pearson | 
| [Cuperus 5′-UTR](docs/benchmarks/cuperus_mpra_5utr.md) | random 50 bp 5′-UTRs (Kozak, uORFs, structure) | regression (HIS3 reporter) | Spearman/Pearson + partial-corr over Kozak features | 
| [Meneu foreign DNA](docs/benchmarks/meneu_foreign_dna.md) | whole bacterial chromosomes integrated in yeast (far-OOD sequence) | zero-shot coverage-track prediction | per-window Pearson + JS divergence (+ fold-change error) |

## Models

| Model | Description | Paper | Status |
| --- | --- | --- | --- |
| Shorkie | Language model based S2F model | [Predicting dynamic expression patterns in budding yeast with a fungal DNA language model](https://www.biorxiv.org/content/10.1101/2025.09.19.677475v1) | done |
| Yorzoi | Borzoi-based S2F model | [Yorzoi: Predicting RNA-seq coverage from DNA sequence in yeast](https://www.biorxiv.org/content/10.1101/2025.09.20.677345v1) | done |
| ExoShorkie | Shorkie finetuned on exo. genomes | [ExoShorkie: Predicting RNA-seq coverage of exogenous genomes in yeast by transfer learning](https://www.biorxiv.org/content/10.64898/2026.01.25.701486v1) | in progress |

## Install

The benchmark framework uses [`uv`](https://docs.astral.sh/uv/). Model-specific
dependencies (PyTorch, Shorkie weights, Yorzoi) are isolated behind extras:

```bash
# Minimal install: core benchmark framework + data loaders (no model deps)
uv sync

# Add specific model dependencies as needed
uv sync --extra shorkie   # PyTorch + h5py, for the Shorkie adapter
uv sync --extra yorzoi    # yorzoi + flash-attn, for the Yorzoi adapter
uv sync --extra all       # both models
```

## Getting the data

Task data and model weights are **not** in git. `ybench data` downloads them
from a mirror (HuggingFace or GCS) and checksum-verifies every file against a
committed lock (`src/yeastbench/data/manifest.lock.json`).

```bash
uv sync --extra data            # adds huggingface_hub (HF backend)

# Pull everything a config's runs need (recommended — same selection as `run`)
uv run ybench data get --config configs/default.yaml

# Or pull explicit subsets / everything
uv run ybench data get --tasks cuperus_utr,caudal_eqtl --models shorkie
uv run ybench data get                     # all artifacts
uv run ybench data get --dry-run           # show the plan, fetch nothing

# Inspect and check
uv run ybench data list                    # every artifact, its mirrors, license
uv run ybench data status                  # what's present locally vs declared
uv run ybench data verify                  # checksum local files against the lock
```

`get` is idempotent (skips files already present and valid), picks the first
reachable mirror (HF first, then GCS; override with `--from hf|gcs`), and writes
atomically. Shorkie weights, Yorzoi, and CodonTransformer resolve from their
public homes; the per-task processed data comes from the project mirror. Add
`--json` to `list`/`status` for machine-readable output.

The HF mirror is free and needs no account — it's the default. The GCS mirror
(`gs://yeast-seq2expression-benchmark`) is **requester pays**, so `--from gcs`
needs your own GCP project to bill: pass `--billing-project <project>` or set
`YBENCH_GCS_BILLING_PROJECT`. If you don't have a project, just use the HF
default.

**How much you'll download.** The full set is **~690 MB**: Shorkie weights
(~440 MB), all task data (~215 MB), and the R64 reference genomes (~34 MB). A
single task is much smaller — from ~11 KB (`hong`) to ~132 MB (`brooks_scramble`)
— so scope your `get` to the config or `--tasks`/`--models` you actually need.
Yorzoi and CodonTransformer aren't in that figure: they're pulled into the
HuggingFace cache the first time you run those models (additional, model-sized).
Run `ybench data status` for an exact present-vs-total byte count, or
`ybench data get --dry-run` to see the size before fetching.

> Maintainers: `ybench data lock` re-freezes the checksum lock from a local
> copy, and `ybench data publish --to hf|gcs` uploads the redistributable
> artifacts to a mirror (dry-run unless `--yes`). Publishing to GCS needs a
> billing project (`--billing-project` / `YBENCH_GCS_BILLING_PROJECT`).

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

# Regenerate plots for an existing run without re-scoring
uv run ybench replot results/default/shorkie__caudal_eqtl
```

See [docs/cli.md](docs/cli.md) for the full CLI reference — all flags, GPU/device
selection, the progress banner, the output layout, and the `ybench data` commands.

## Extending the benchmark

Tasks and models are plug-ins, wired through a registry of protocols,
benchmarks, and adapters — adding either needs no new runner scripts or CLI
wiring. See [docs/extending.md](docs/extending.md) for how to add a benchmark
or a model.

## Repository layout

```
docs/
  benchmarks/              benchmark specs (one markdown per task)
  cli.md                   full CLI reference
  extending.md             how to add a benchmark or a model
configs/                   YAML run-specs (committed canonical runs)
data/
  raw/                     raw upstream files (FASTA, GTF, GWAS, gVCF, MPRA)
  processed/               versioned processed distributions per task
  models/                  downloaded model weights + targets sheets
scripts/                   one-off data-preparation scripts
src/yeastbench/
  adapters/
    protocols.py           protocol definitions (VariantEffectScorer, …)
    _genome.py             shared FASTA/GTF + one-hot utilities
    shorkie_*.py           one adapter per (Shorkie, protocol) pair
    yorzoi_*.py            one adapter per (Yorzoi,  protocol) pair
  benchmarks/
    base.py                Benchmark[AdapterT, ResultT] ABC
    eqtl.py                EQTLClassificationBenchmark
    mpra.py                MPRA{Regression,Marginalized}Benchmark
  models/                  vendored pure-PyTorch model ports (Shorkie)
  registry.py              MODELS + TASKS registries
  cli.py                   the `ybench` CLI
  config.py                YAML config loader
tests/                     pytest suite (86 tests)
```

## Roadmap

See [`ROADMAP.md`](ROADMAP.md).

## Contact

> [!NOTE]
> In case of any questions, reach out to mail@timonschneider.de — always happy to help!

