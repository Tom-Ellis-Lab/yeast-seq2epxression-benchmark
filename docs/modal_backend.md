# Modal GPU compute backend for `ybench`

A runtime backend that lets a user with no local NVIDIA GPU run the full
`configs/default.yaml` benchmark faithfully, including `yorzoi` (the one model
that cannot run without a GPU). Self-contained design doc.

## TL;DR — recommended approach

Lift-and-shift the whole run into one Modal GPU container (Architecture A), with
two grafts from the fan-out design (B): a **separate CPU seed step** so the
~1 GB data/weights download is never billed at GPU rates, and a **durable
results Volume** so a late crash doesn't lose finished pairs. The local machine
installs only the `modal` client and calls `ybench modal run -c
configs/default.yaml`; the container runs the unmodified `ybench data get` +
`ybench run --device cuda` (auto-compare included) and the laptop gets back the
identical results tree. This beats the per-pair fan-out (B) because the default
goal is "run it once, faithfully, with minimal code and friction" — fan-out's
tier routing, spawn/gather, and partial-failure plumbing buy wall-clock and
resumability we don't need yet for a ~$2–4, sub-2-GPU-hour run. It beats the
forward-pass offload (C) decisively: C's RPC seam is destroyed by the
marginalization fan-out (tens of thousands of synchronous round-trips,
idle-billed GPU, ~$8–30/run), and it touches `registry.py` and scoring-adjacent
reduction order, which collides with the maintainer's bit-exactness sensitivity.
The hybrid keeps every line of scoring code untouched and runs it remotely,
verbatim.

## How it works

End to end:

1. **Local CLI** (`ybench modal …`, a Typer sub-app) reads the config bytes,
   ensures the Modal app is reachable, and dispatches.
2. **One-time seed** (`ybench modal data get`) runs a **CPU** function that
   executes the real `ybench data get -c <cfg>` inside the container, writing
   locked artifacts to a `ybench-data` Volume and warming the HF cache
   (`yorzoi`, `CodonTransformer`) into a `.hf` subdir of that same volume
   (`HF_HOME=/repo/data/.hf`), then `commit()`s. Idempotent and checksum-gated,
   so later runs skip.
3. **Run** (`ybench modal run`) calls one **GPU** function that, with cwd
   `=/repo`, shells out to `ybench run -c <cfg> --device cuda` (which also fires
   the auto-compare), writes to the `ybench-results` Volume, `commit()`s, and
   returns only a small manifest (the list of `<model>__<task>` dirs produced).
4. **Results back**: the local command downloads the results tree from the
   Volume with `modal volume get --force ybench-results / <out>` (default `out`
   is `results/`, recreating `results/default/…`). `ybench modal pull` is the
   same download exposed standalone. We download from the Volume rather than
   returning a tarball because a full run's `.npy`/plot tree is tens of MB and
   Modal doesn't guarantee large function return values pass through; the Volume
   path has no such limit and is also durable if the client disconnects.

Why cwd must be `/repo`: `default_data_root()` walks up to the nearest
`pyproject.toml`, and every benchmark/adapter opens its config paths
(`data/tasks/...`, `data/models/shorkie/checkpoints/f*.h5`) relative to cwd —
not through `--data-root`. The data Volume is mounted at `/repo/data` so those
relative opens resolve. `.git` is baked in so `run_metadata.json` records a real
`git_commit`.

### Modal app sketch (`src/yeastbench/modal/app.py`)

API-correct against the SDK research; illustrative, not final.

```python
import modal

FLASH_ATTN_WHEEL = (
    "https://github.com/adithyaxx/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3%2Bcu13torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)  # the exact wheel already pinned in pyproject [tool.uv.sources]

app = modal.App("ybench")

# GPU image: CUDA 13 base (so the cu13 flash-attn wheel can load) + cp312 + cu130 torch.
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])                      # clear the base image entrypoint
    .apt_install("git")
    # force the cu13 torch build first so the wheel's ABI matches (PyPI default torch is cu12)
    .uv_pip_install("torch==2.11.*", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("uv")                # for the project install below
    # bake the repo so the build step can install the package and so .git is present
    .add_local_dir(".", "/repo", copy=True,
                   ignore=["data/**", "results/**", "**/__pycache__", ".venv/**"])
    # install the project's [all] extra -> uv reads [tool.uv.sources] and resolves the pinned
    # flash-attn wheel + yorzoi==0.2.1; transformers/CodonTransformer are codon_transformer's
    # undeclared deps and must be added explicitly.
    .run_commands(
        "cd /repo && uv pip install --system '.[all]' transformers && "
        f"uv pip install --system '{FLASH_ATTN_WHEEL}' "
        "git+https://github.com/Adibvafa/CodonTransformer"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/repo/data/.hf"})
)

# CPU image for the seed (torch-free): just the data backend.
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .add_local_dir(".", "/repo", copy=True, ignore=["data/**", "results/**"])
    .run_commands("cd /repo && pip install '.[data]'")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/repo/data/.hf"})
)

data_vol    = modal.Volume.from_name("ybench-data",    create_if_missing=True)
results_vol = modal.Volume.from_name("ybench-results", create_if_missing=True)

# Two volumes, not three: the HF cache rides inside the data volume via
# HF_HOME=/repo/data/.hf (set in .env above), so it shares the inputs' lifecycle
# and needs no separate mount.
VOLUMES = {"/repo/data": data_vol, "/repo/results": results_vol}

# Optional; not needed on the default public path. See "Data & weights".
HF_SECRET = modal.Secret.from_name("huggingface")  # only to dodge anonymous HF rate limits

import subprocess, pathlib

def _write_config(config_bytes, config_name):  # under the original name → config_hash + path parity
    rel = f"configs/{config_name}"
    pathlib.Path("/repo", rel).write_bytes(config_bytes)
    return rel

@app.function(image=cpu_image, volumes=VOLUMES, timeout=2 * 3600)
def seed(config_bytes: bytes, config_name: str):
    rel = _write_config(config_bytes, config_name)
    subprocess.run(["ybench", "data", "get", "--config", rel], cwd="/repo", check=True)
    data_vol.commit()   # persists locked data/ AND the /repo/data/.hf cache

@app.function(image=gpu_image, gpu="A10", volumes=VOLUMES, timeout=4 * 3600)
def run_benchmark(config_bytes, config_name, out_dir="results/default", device="cuda") -> list[str]:
    data_vol.reload()   # pick up the seed's committed data + HF cache
    rel = _write_config(config_bytes, config_name)
    subprocess.run(["ybench", "data", "get", "--config", rel],
                   cwd="/repo", check=True)                            # no-op if seeded
    subprocess.run(["ybench", "run", "--config", rel, "--device", device],
                   cwd="/repo", check=True)                            # includes auto-compare
    results_vol.commit()
    produced = pathlib.Path("/repo", out_dir)
    return sorted(p.name for p in produced.iterdir() if p.is_dir())    # small manifest, not the tree
```

### Local CLI surface (`src/yeastbench/modal/cli.py`)

Mirrors `src/yeastbench/data/cli.py`; mounted in `cli.py` with two guarded lines
next to the existing `app.add_typer(data_app, name="data")`.

```
ybench modal data get -c configs/default.yaml      # one-time CPU seed of the data/HF volume
ybench modal run      -c configs/default.yaml \    # full run on a GPU; streams container logs
                      [--gpu A10] [--out results] [--detach]
ybench modal pull     [--out results]              # download the results volume (no run needed)
ybench modal status                                # what's on the data/results volumes
```

`ybench modal run` does, roughly:

```python
import modal, subprocess
from yeastbench.modal.app import app, run_benchmark, seed

with modal.enable_output(), app.run():            # streams remote stdout to the terminal
    seed.remote(cfg, name)                         # CPU seed (idempotent)
    fn = run_benchmark.with_options(gpu=gpu)       # runtime GPU-tier override
    produced = fn.remote(cfg, name, out_dir=out_dir, device="cuda")
subprocess.run(["modal", "volume", "get", "--force", "ybench-results", "/", out])  # download tree
```

The local commands import `modal` lazily so `ybench --help` still works without
the extra installed.

## Data & weights

Measured default-run footprint: **~562 MB locked to `data/`** (Shorkie's 8 `.h5`
folds = 440 MB dominate; cuperus 39 MB; refs 34 MB; rafi 25 MB; dream_rnn
16.5 MB) **+ ~417 MB HF cache** (`yorzoi` 74 MB, `CodonTransformer` 343 MB) ≈
**~0.98 GB**.

**Two volumes, not one or three.** The layout is one `ybench-data` volume
mounted at `/repo/data` (holding the locked files *and* the HF cache, via
`HF_HOME=/repo/data/.hf`) plus one `ybench-results` volume at `/repo/results`.
It's two, not three, because the HF cache is relocated with an env var rather
than its own mount, so it shares the inputs' seed-once / read-many lifecycle.
It's two, not one, because the unmodified `default.yaml` names two distinct
paths (`data/` and `results/`), a Modal volume mounts as one tree at one path,
and `/repo` itself is occupied by image-baked code (a volume there would shadow
it) — so each path gets its own mount and the config stays byte-identical
(preserving `config_hash`). Collapsing to one volume would mean rewriting the
config paths (breaks hash parity) or moving the code onto the volume (loses
image reproducibility).

**Strategy: seed once into a Volume; don't bake, don't fetch-every-run.** Baking
couples data to the image and re-pulls Shorkie's 440 MB on every rebuild;
fetching every run wastes ~1 GB egress and (if done on the GPU function)
GPU-billed minutes. The `seed` function runs the canonical `ybench data get`, so
it inherits the lock's SHA-256 verification and atomic `.part`+`os.replace`
writes; on later runs it sees checksum-OK files and skips. The HF cache lands
inside the **same** data volume via `HF_HOME=/repo/data/.hf`, persisting
`yorzoi`/`CodonTransformer` so `Borzoi.from_pretrained` /
`BigBirdForMaskedLM.from_pretrained` don't re-download.

**Secrets: none on the default path.** Every default artifact is anonymous and
public:
- Task data + `refs` + `dream_rnn` → HF dataset repo
  `tom-ellis-lab/yeast-seq2expression-data` (free, no auth).
- `shorkie` → authors' public GCS bucket over plain HTTPS
  (`storage.googleapis.com/seqnn-share/...`), `params.json` from
  `raw.githubusercontent.com`.
- `yorzoi`, `codon_transformer` → public HF model repos.

`_choose_mirror` tries HF first and only touches the requester-pays GCS bucket
if you force `--from gcs`, so the costed path is never hit by default.
**Avoiding GCS requester-pays cost: do nothing** — just keep `huggingface_hub`
installed (it is, via `.[data]`/`.[all]`) so the HF backend is "available" and
chosen first. Only force `--from gcs` if HF is down, and only then provide a
`gcp` secret (service-account JSON written to a temp file +
`GOOGLE_APPLICATION_CREDENTIALS`) plus `YBENCH_GCS_BILLING_PROJECT`. Optional
`huggingface` secret (`HF_TOKEN`) is worth adding only to dodge anonymous HF
rate limits on the cold seed; it is not required.

## Faithfulness

The container runs the literal `ybench run -c <cfg> --device cuda` against
checksum-verified artifacts, on the exact pinned flash-attn `cu13/torch2.11/cp312`
ABI. The eval loop, `save_results()`, `plot()`, and the auto-`compare()` are the
unmodified repo code, so the output tree is identical: per-pair `summary.json` /
`run_metadata.json` (with a real `git_commit` and the same `config_hash`,
because the user's YAML bytes are written verbatim), the
`*_scores.npy`/`*_labels.npy`/`*.tsv`/`*.png` per benchmark, and
`compare/summary.csv`/`summary.md`/`per_task/*`.

**This is a runtime backend, not a refactor of scoring code.** No scoring path,
reduction order, adapter, or `registry.py` factory is touched — that is the
whole point of choosing lift-and-shift over the offload design (C), which would
have re-implemented per-model readout reductions remotely and put bit-exactness
at risk. The maintainer's "preserve score bit-exactness" constraint is satisfied
by construction here: the same bytes of code compute the scores.

**What can differ, honestly:** GPU nondeterminism plus the project's documented
"eval not bit-reproducible (~4–5 decimal)" contract means results match a local
GPU run to ~4–5 decimals, not bit-for-bit. A different physical GPU model than
the one used for the published numbers adds tiny additional drift
(autocast/TF32/cuDNN algorithm selection). This is inherent to *any* GPU
backend, local or remote, and is independent of Modal. If exactness matters for
a given diff, pin the GPU tier (e.g. always `A10`) so successive Modal runs at
least agree with each other.

## Packaging

Purely additive, mirroring the data-backend convention.

- **`pyproject.toml`**: add `modal = ["modal>=1.5.1"]` under
  `[project.optional-dependencies]` (the code uses 1.x-only APIs:
  `uv_pip_install`, `add_local_dir(ignore=)`, `enable_output`, `with_options`).
  **Do not** fold it into `all` — `all` is "everything `default.yaml` needs" for
  a local run; Modal is opt-in infra. No `[tool.uv.sources]` entry needed (Modal
  is a normal PyPI wheel).
- **`src/yeastbench/cli.py`**: register the sub-app only when `modal` is
  installed, next to the existing `add_typer`. We key off `find_spec` rather than
  `try/except ModuleNotFoundError` so a real import error *inside* the sub-app
  isn't silently swallowed, and so `modal` itself isn't imported at startup:
  ```python
  if importlib.util.find_spec("modal") is not None:
      from yeastbench.modal.cli import app as modal_app
      app.add_typer(modal_app, name="modal")
  ```
- **New subpackage** `src/yeastbench/modal/`: `__init__.py`, `cli.py` (Typer
  sub-app), `app.py` (App/images/volumes/functions), and a pure `plan.py`
  (config → dispatch plan: pairs, gpu tier, out dir) that is CPU-unit-testable
  with no `modal` import.
- **No edits** to `registry.py`, `protocols.py`, benchmarks, or adapters.

**Tests** (`tests/test_modal.py`):
- `pytest.importorskip("modal")` to skip when the extra is absent (same as
  `importorskip("torch")` elsewhere).
- Pure-CPU asserts on `plan.build_plan(cfg)` (pairs, out_dir guard) — no remote
  call (mirrors `tests/test_data_backends.py` asserting on `_build_cmd`).
- _(Deferred)_ Real dispatch behind `@pytest.mark.integration` + `@pytest.mark.gpu`
  + a `YBENCH_MODAL=1` `skipif` — not yet implemented; the live path is exercised
  manually via the Phase 0 spike and a real `ybench modal run` for now.

**CI**: the only workflow is `lint.yml` running `ruff F401`. It executes no
pytest, so the Modal/GPU/network tests never run in CI by construction. The sole
CI obligation for the new files is staying free of unused imports.

**Docs**: this doc, a pointer in `docs/cli.md`, and a line in the README
architecture/quickstart.

## Implementation plan

Phased; each phase has a verifiable milestone. The walking skeleton is
**Phase 3**.

- **Phase 0 — Image/driver spike (de-risk first).** No repo changes. A throwaway
  `modal run` script with `gpu_image` and `gpu="A10"` that prints
  `torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available()`, and
  does `import flash_attn` + `from yorzoi.model.borzoi import Borzoi`.
  *Milestone:* on an A10, CUDA is available, torch is `2.11.* / cu130`, and both
  `flash_attn` and `yorzoi` import without ABI errors. **If this fails, stop** —
  the whole "run yorzoi remotely" promise depends on it; fall back to a GPU image
  that drops the `yorzoi` extra and document yorzoi as unsupported.

- **Phase 1 — Additive packaging skeleton.** Add the `modal` extra; create
  `src/yeastbench/modal/{__init__,cli,app,plan}.py`; add the guarded `add_typer`
  in `cli.py`; write the pure `plan.py` + its CPU test.
  *Milestone:* `ybench --help` works with and without `modal` installed; `ybench
  modal --help` lists `data get` / `run` / `pull` / `status`; `pytest
  tests/test_modal.py` (unit half) green.

- **Phase 2 — Data seed.** Implement `seed()` + `ybench modal data get`; create
  the two Volumes (`ybench-data`, `ybench-results`).
  *Milestone:* after `ybench modal data get -c configs/default.yaml`, `modal
  volume ls ybench-data` shows the locked tree and a re-run is a checksum no-op
  (prints "already present").

- **Phase 3 — Walking skeleton (smallest end-to-end).** Run **one CPU baseline
  pair** first: dispatch only `cai__chen_synonymous` through `run_benchmark` (it
  needs no GPU but proves the container/cwd/volume/download path). Then **one
  GPU pair**: `dream_rnn__rafi_mpra_marginalized` (small, fast) or a single
  Shorkie eqtl task. Pull both, diff `summary.json` against a known-good local
  run.
  *Milestone:* `results/default/cai__chen_synonymous/` and the GPU pair's dir
  appear on the laptop; metrics match local to ~4–5 decimals;
  `run_metadata.json` has a real `git_commit` and matching `config_hash`.

- **Phase 4 — Full default run.** `ybench modal run -c configs/default.yaml`
  runs all pairs + auto-compare; Volume commit + `modal volume get` download;
  `ybench modal pull` as the standalone re-download.
  *Milestone:* full `results/default/` including
  `compare/summary.csv`/`summary.md`/`per_task/*` lands locally; yorzoi pairs
  succeed.

- **Phase 5 — Harden + document.** Raise/confirm `timeout`, add `ybench modal
  status`, `gpu=["A10","L40S"]` fallback list, this doc finalized, README
  pointer, integration test behind `YBENCH_MODAL=1`.
  *Milestone:* a cold first-time user can go from `modal setup` to a pulled
  results tree following only this doc.

- **(Deferred, v2) Phase 6 — Fan-out.** If wall-clock or partial-failure
  isolation becomes real pain, graft Architecture B: `score_pair_*` per GPU
  tier, `.spawn()`/`.map(return_exceptions=True)`, disjoint-subdir commits,
  `--only-missing` resume. Keep the lift-and-shift path as the default.

## Cost & ops

- **$/run:** ~**$1.10–$3.50** on one A10 (~$1.10/hr) for the full ~19-pair
  default run, billed per-second. Shorkie (8 folds × 16,384 bp × RC across 8
  tasks) and yorzoi (8 Borzoi tasks) dominate; cai/codon_transformer/dream_rnn
  are minutes. Sequential on one GPU, so GPU-hours ≈ wall-clock (~1–2 hr). L4
  (~$0.80/hr, 24 GB) is a cheaper drop-in if it holds the batch sizes.
- **Seed cost:** the ~1 GB download runs on a **CPU** function, so it's
  near-zero (~$0.05) and never billed at GPU rates — this is the main graft from
  B.
- **Image build:** one-time, several minutes (CUDA 13 + cu130 torch + flash-attn
  + transformers/CodonTransformer); cached after, free compute.
- **Volume/idle:** Volume storage is the only standing cost and is negligible at
  ~1 GB. The GPU container is billed only while alive; the lift-and-shift design
  holds it for the whole run, so don't leave it idle between phases — use
  `--detach` for long runs and let it exit on completion.
- **Free credit:** Modal's Starter plan gives ~$30/month, so a full run is
  effectively free.
- **GPU tier:** default `A10`; set `gpu=["A10", "L40S"]` as an ordered fallback
  if A10 is scarce. Avoid region pinning (adds a 1.5–1.75× multiplier) unless
  data locality is needed.
- **Auth (one-time, brand-new user):** `uv pip install modal` then `modal setup`
  (browser flow, writes `~/.modal.toml`). Headless/CI: set `MODAL_TOKEN_ID` /
  `MODAL_TOKEN_SECRET`. No other credentials are needed for the default
  public-data path.

## Open questions / decisions for the maintainer

1. **Volume-seed vs runtime-fetch.** Recommended: seed once (CPU) into a Volume.
   The alternative — fetch inside the GPU function every run — is simpler (no
   seed command) but re-downloads ~1 GB on every cold start and burns
   GPU-billed minutes. Pick runtime-fetch only if you never want a persistent
   Volume to manage.
2. **Lift-and-shift (one container) vs fan-out (per-pair).** Recommended:
   lift-and-shift for v1; fan-out deferred to v2. Fan-out earns its keep only
   with repeated/partial runs (resume one model, recover from one OOM) and
   shorter wall-clock — it costs ~the same money but a lot more orchestration
   code. Decide whether pair-level resumability is a day-1 requirement.
3. **Repo into image: `add_local_dir(copy=True)` vs `git clone` at a tag.**
   `copy=True` of the working tree runs *your current* code and bakes `.git`
   (faithful `git_commit`), but a code edit forces an image rebuild. `git clone
   --depth 1 <tag>` is reproducible and decouples builds from the working tree
   but only ever runs the released tag. Which matters more —
   iterate-on-uncommitted-code, or pinned reproducibility?
4. **Results return: tarball vs Volume-only.** *Decided → Volume-only.* The GPU
   function commits results to the Volume and returns only a small manifest; the
   client downloads with `modal volume get`. Returning a tarball was rejected
   because a full run's tree is tens of MB and Modal doesn't guarantee large
   return values pass through — and the Volume path is durable against a client
   disconnect anyway.
5. **codon_transformer deps in the image only, or declared in `pyproject.toml`?**
   `transformers` and `CodonTransformer` are currently undeclared deps of that
   adapter and the image adds them manually. This is a pre-existing gap the
   Modal image papers over — do you also want to fix it upstream in the extras
   so a local install of `codon_transformer` works too?
6. **Also offer Docker/RunPod?** A `ComputeBackend` Protocol mirroring the data
   `Backend` Protocol would let additional compute backends slot in later.
   Worth the small upfront abstraction now, or YAGNI until a second backend is
   actually requested?
7. **HF-only vs GCS.** Recommended: HF-only by default (free, no auth, chosen
   first). Expose `--from gcs` + the `gcp` secret only as a documented fallback.
   Confirm you don't want GCS as a first-class Modal path (it would require
   shipping the service-account secret and a billing project).

## Risks & gotchas

- **flash-attn wheel / CUDA-driver match on Modal's GPUs (the top risk).** The
  pinned wheel is `cu13 / torch2.11 / cxx11abiTRUE / cp312 / linux_x86_64`, so
  the image must be CUDA 13 base + `add_python="3.12"` + torch from
  `download.pytorch.org/whl/cu130`, and Modal's GPU host driver must be
  CUDA-13-capable (toolkit ≤ host driver). Two sub-risks: (a) `whl/cu130` may
  not yet publish a torch `2.11.*` build — verify in Phase 0; (b) if any layer
  drifts off that ABI, yorzoi's import fails at load. Mitigation: Phase 0 is a
  hard gate; if it fails, ship a GPU image without the `yorzoi` extra and
  document yorzoi as the one unsupported model. Note the wheel is a real GitHub
  release asset (not a local-only file), so passing the URL to `uv_pip_install`
  is sound.
- **Image build time and size.** CUDA 13 devel base + cu130 torch + flash-attn +
  transformers/CodonTransformer is a multi-GB, multi-minute first build. Use the
  CUDA `devel` (not `runtime`) tag so the wheel's CUDA deps resolve; cache
  aggressively; keep `data/`/`results/` out of the image (`ignore=`) so the
  local `data/` never inflates a layer.
- **Cold starts.** First call per cold container reloads weights (Shorkie's 8
  folds, Borzoi) from the Volume — seconds, not the bottleneck for a sequential
  run, but it's paid again if the container scales to zero between phases. Use
  `--detach` for the full run so it isn't interrupted by client disconnect.
- **cwd / relative-path trap.** If the function doesn't run with cwd `=/repo`,
  or the data Volume isn't mounted exactly at `/repo/data`, `default_data_root()`
  and the adapters' relative opens break. Pin `cwd="/repo"` in every
  `subprocess.run`.
- **Single-container timeout/OOM.** Default 300 s would kill a multi-pair run —
  set `timeout=4*3600` (max is 24 h). A10 VRAM must hold Shorkie's batch at 16 kbp and
  dream_rnn's large batch; if a pair OOMs mid-loop the run aborts. The per-pair
  results are written as the run progresses, but the Volume `commit()` only
  happens after `ybench run` returns — so on a hard crash mid-run, rely on
  Modal's periodic background commits (or add an intermediate commit) to keep
  finished pairs; a crashed pair just won't appear (and `compare` silently drops
  a task that falls below 2 models — surface this in `ybench modal status`).
- **Requester-pays surprise.** Only reachable via `--from gcs`. Keep
  `huggingface_hub` installed so HF is chosen first; document that `--from gcs`
  needs a billing project and a `gcp` secret. Don't make GCS the default.
- **Determinism.** Results match a local GPU run to ~4–5 decimals (project
  contract + GPU nondeterminism), not bit-for-bit, and a different GPU model
  adds small drift. This is a property of GPU eval, not of Modal, and does not
  touch scoring code — but call it out to anyone expecting byte-identical
  `*.npy`. Pin one GPU tier for run-to-run agreement.
