# Benchmark Yeast Sequence-to-Expression Models

> [!NOTE]
> In case of any questions, reach out to mail@timonschneider.de — always happy to help!

A collection of datasets and scripts to benchmark models that predict gene
expression from DNA sequence in *S. cerevisiae*. The goal is a single,
reproducible way to compare sequence-to-expression models across a shared
set of tasks — eQTL classification, MPRA generalization, and native-genome
track prediction.

**Status:** under construction.

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
committed lock (`src/yeastbench/data/manifest.lock.json`). See
[`specs/data-storage.md`](specs/data-storage.md) for the design.

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

### Progress, hardware, and GPU selection

`run` opens with a banner — config hash, resolved device (GPU name + free
VRAM + `CUDA_VISIBLE_DEVICES` when CUDA is available), a data-readiness check,
and the planned pairs — then prints a `[i/N]` line per pair with a running
mean-pair-time **ETA** so you can see at a glance what's done and how long is
left:

```
hardware:      cuda:0  NVIDIA A100-80GB  (79.2/80.0 GB free)
data:          ✓ ready  19/19 files (474.0MB)
runs:          12 pair(s)

[ 1/12] shorkie × caudal_eqtl → results/default/shorkie__caudal_eqtl
  …
  ✓ done in 41s · elapsed 41s · mean 41s/pair · ETA ~7m32s
```

Before executing, `run` pre-flights the data: a real run **stops** if any
required file is missing or stale (with the `ybench data get` command to fix
it); `--no-data-check` skips that. Pick the GPU with `--gpu 2` (shorthand for
`--device cuda:2`) or `--device cpu`. Runs are sequential — one GPU at a time;
to isolate a physical GPU on a shared box use `CUDA_VISIBLE_DEVICES`.

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
    primary_roc_pr.png         # ROC + PR on full set (|score|, mean ± SEM, baselines)
    close_only_roc_pr.png      # same, filtered to pos_distance_to_tss ≤ 2 kb
    distance_stratified.png    # AUROC / AUPRC per distance-to-TSS bin
  yorzoi__caudal_eqtl/
    …
```

`run_metadata.json` captures everything needed to reproduce that directory's
numbers — the config hash, repo git commit, resolved model/task configs,
and timestamp. Raw scores + labels + pair metadata are persisted so
post-hoc analyses (distance-stratification, signed-vs-absolute comparison,
etc.) don't require re-running the model.

## Extending the benchmark

The codebase is built around two orthogonal abstractions, wired together
by a registry:

- **Protocols** (`src/yeastbench/adapters/protocols.py`) — small Python
  `Protocol`s describing what a model must implement to run a given
  *type* of benchmark. Current protocols:
  - `VariantEffectScorer.score_variants(variants) -> np.ndarray`
    — for eQTL-style benchmarks.
  - `MarginalizedSequenceExpressionPredictor.predict_marginalized_expressions(seqs) -> np.ndarray`
    — for native-position marginalized MPRA scoring.
- **Benchmarks** (`src/yeastbench/benchmarks/`) — a `Benchmark` subclass
  per task type. Each declares `adapter_protocol` (which protocol it
  consumes) and implements `evaluate`, `plot`, `save_results`,
  `load_results`, `summary_dict`, and `headline`.
- **Adapters** (`src/yeastbench/adapters/`) — one class per
  `(model, protocol)` pair. Implements the protocol by wrapping the
  model's forward pass, tokenization, and post-processing.

The CLI's `_run_pair` is task-agnostic: `task = TASKS[name](...)`,
`adapter = MODELS[name](task, device, ...)`, then
`task.evaluate(adapter) → task.plot → task.save_results`.

### Adding a new benchmark

Most new benchmarks reuse an existing protocol. The workflow:

1. **Pick or add a protocol.** Can one of the existing protocols score
   your task? If yes, reuse it. If no — the task needs a
   semantically-different operation — add a new `@runtime_checkable`
   `Protocol` in `adapters/protocols.py`.
2. **Write the Benchmark class** in `src/yeastbench/benchmarks/<name>.py`:
   - Subclass `Benchmark[AdapterT, ResultT]` with your adapter protocol
     and results dataclass.
   - Set `adapter_protocol: ClassVar[type] = YourProtocol`.
   - Implement `__init__(<task_config_fields>, info)`,
     `evaluate(adapter) -> Results`, `plot`, `save_results`,
     `load_results`, `summary_dict`, `headline`.
3. **Register** the task in `src/yeastbench/registry.py`:
   ```python
   def _build_my_task(path_a, path_b) -> Benchmark:
       return MyBenchmark(..., info=BenchmarkInfo(name="my_task", ...))

   TASKS["my_task"] = _build_my_task
   ```
4. **If you added a new protocol**, extend each model's adapter map
   (see "Adding a new model" below) with an implementation for that
   protocol.
5. **Reference the task in `configs/default.yaml`** under both
   `tasks_config:` (its constructor kwargs) and any `runs:` that should
   include it.
6. **Write a spec** in `benchmarks/<name>.md` and add tests in
   `tests/test_<name>.py`.

### Adding a new model

1. **Implement one adapter class per protocol the model should support**,
   in `src/yeastbench/adapters/<model>_<task_type>.py`. Each adapter
   wraps the model's forward pass + any pre/post-processing, and
   exposes the single method required by its protocol.
2. **Register the model** in `src/yeastbench/registry.py` by adding a
   protocol → builder dict:
   ```python
   def _mymodel_eqtl(device, fasta_path, gtf_path, **cfg):
       from yeastbench.adapters.mymodel_eqtl import MyModelScorer
       return MyModelScorer(fasta_path, gtf_path, device, **cfg)

   MYMODEL_ADAPTERS: dict[type, tuple[Callable, bool]] = {
       VariantEffectScorer: (_mymodel_eqtl, True),  # True = needs FASTA/GTF
       # add more protocol entries as you add adapters
   }

   def _build_mymodel(task, device, **cfg):
       return _dispatch(MYMODEL_ADAPTERS, task, device, **cfg)

   MODELS["mymodel"] = _build_mymodel
   ```
   The `needs_refs` flag controls whether the dispatcher passes the
   task's `fasta_path`/`gtf_path` to the adapter (true for
   genomic-context tasks; false for protocol-only adapters that need
   neither reference, e.g. the Brooks coverage track).
3. **Reference the model in `configs/default.yaml`** under `runs:`
   with the model-specific kwargs it accepts (checkpoint paths, batch
   size, `use_rc`, etc.).
4. **Optional**: add a `[project.optional-dependencies]` entry for any
   model-specific packages (e.g., a HuggingFace wheel), so users can
   install just the adapter they need with `uv sync --extra mymodel`.

No new runner scripts, no new CLI wiring — both tasks and models are
fully plug-in.

## Repository layout

```
benchmarks/                benchmark specs (one markdown per task)
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

Reach out to mail@timonschneider.de in case you have questions, need help,
or want to chat.
