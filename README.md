# Benchmark Yeast Sequence-to-Expression Models

> [!NOTE]
> In case of any questions, reach out to mail@timonschneider.de — always happy to help!

A collection of datasets and scripts to benchmark models that predict gene
expression from DNA sequence in *S. cerevisiae*. The goal is a single,
reproducible way to compare sequence-to-expression models across a shared
set of tasks — eQTL classification, MPRA generalization, and native-genome
track prediction.

**Status:** under construction. Currently live (Shorkie + Yorzoi adapters):

- `caudal_eqtl` — Caudal et al. *cis*-eQTL classification.
  Spec: [`benchmarks/caudal_eqtl.md`](benchmarks/caudal_eqtl.md).
- `rafi_mpra_promoter` — Rafi / deBoer DREAM MPRA, fixed-context
  scoring (embed 110 bp into the 5 kb plasmid, sum YFP-CDS bins).
  Spec: [`benchmarks/rafi_mpra_promoter.md`](benchmarks/rafi_mpra_promoter.md).
- `rafi_mpra_marginalized` — same test set, marginalized native-position
  scoring (insert into 22 host genes at 180 bp upstream, logSED over
  exons, mean across genes).

In flight: Kita et al. eQTL, Shalem et al. MPRA (terminator),
Cuperus et al. 5′ UTR, native-genome track-prediction comparison.

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

# Regenerate plots for an existing run without re-scoring
uv run ybench replot results/default/shorkie__caudal_eqtl
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
  - `SequenceExpressionPredictor.predict_expressions(seqs) -> np.ndarray`
    — for fixed-context MPRA scoring.
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
   genomic-context tasks, false for fixed-context MPRA, etc.).
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
