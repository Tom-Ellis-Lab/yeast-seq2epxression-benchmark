# ybench: a benchmark for fungal sequence-to-expression models

![image](img/readme_banner.svg)

`ybench` is a comprehensive set of ten tasks to benchmark yeast-focussed sequence-to-function (S2F) models like Shorkie and Yorzoi. Each task covers a different part of the genomes regulatory complexity from eQTLs, promoter/UTR/terminator [MPRAs](https://www.google.com/search?q=mpra+massively+parallel+reporter+assay), reporter gene insertions in different genomic locations to investigate context effects as well as exogenous but linearised bacterial DNA in yeast and structurally rearranged chromosomes.

## Table of Contents

- [Benchmark Tasks](#benchmark-tasks)
- [Models](#models)
- [Quickstart](#quickstart)
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

## Quickstart

Install the framework with every model's dependencies, fetch the data, then run
the canonical `configs/default.yaml`:

```bash
uv sync --extra all                                   # framework + all models + data backend
uv run ybench data get --config configs/default.yaml  # download data + weights
uv run ybench run --config configs/default.yaml       # score every (model, task) pair
```

Results land in `results/default/<model>__<task>/`. For everything else — the
other models, partial data pulls, GPU selection, every flag, and the output
layout — see the [CLI reference](docs/cli.md).

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
  ROADMAP.md               v1/v2 status checklist
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

See [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Contact

> [!NOTE]
> In case of any questions, reach out to mail@timonschneider.de — always happy to help!

