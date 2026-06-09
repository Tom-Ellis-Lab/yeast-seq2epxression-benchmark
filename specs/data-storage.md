# Data storage & download

How models and task data get stored, mirrored, and pulled down. The goal:
anyone (human or agent) can run `ybench data get` and end up with exactly the
files a benchmark run needs, verified against checksums, from a backend they
can actually reach.

This spec covers **v1** (download published benchmark-ready data + model
weights). It also describes the **v2** raw→build→reproduce path so the data
model leaves room for it — but v2 is *designed here, not implemented yet*.

## Goals

- One command pulls the data for a chosen set of models/tasks.
- Agent-ready: a machine-readable manifest + `--json` introspection means an
  agent can discover every artifact, its mirrors, and its checksum without
  reverse-engineering any script.
- Two interchangeable backends (HuggingFace + GCS), so a download never depends
  on a single host or on credentials the user doesn't have.
- Selection reuses the *run* config — the thing that says which `(model, task)`
  pairs you care about already says which data you need.
- Integrity and idempotency are non-negotiable: checksums, skip-if-present,
  atomic writes.

## Non-goals (v1)

- Downloading **raw** upstream paper data (GEO/figshare/1002genomes/…).
- Running the per-task `build_*` preprocessing to **reproduce** processed data.
- A `publish` path that pushes to the mirrors (maintainer tooling; stubbed).

These are v2. The manifest schema reserves fields for them (`raw`, `build`,
`redistributable`) so adding v2 is additive, not a rewrite.

## The one idea: the lock is the source of truth

Artifacts are **declared in code** (`src/yeastbench/data/manifest.py`, the
companion to `registry.py`) and **frozen to a checksum lock**
(`src/yeastbench/data/manifest.lock.json`, committed, ships with the wheel).

- The *declaration* says what an artifact is, where it lives (`dest`), which
  tasks/models need it, and which mirrors host it.
- The *lock* records, per file, its path + sha256 + size. It is generated from
  the local copy (`ybench data lock`) and is the authoritative file list.

Everything downstream reads the lock:

- `get` knows exactly which files to fetch and what each should hash to — so
  integrity is free and no remote directory-listing is required.
- `verify` / `status` compare local files against the lock.
- versioning falls out: tagging a lock pins a benchmark release to exact bytes.

Declaring in code (not a hand-edited YAML) means adding a task **forces**
declaring its data — a test asserts every `TASKS`/`MODELS` key is covered.

## Data model

```python
class Kind(Enum):            # what an artifact is
    TASK_PROCESSED           # benchmark-ready files for one task
    MODEL_WEIGHTS            # checkpoints / params for one model
    SHARED_REFERENCE         # genome FASTA/GTF used by many tasks

class BackendKind(Enum):     # how to fetch a mirror
    HF, GCS, HTTP            # (v2 adds: FIGSHARE, ZENODO, SOURCE)

@dataclass(frozen=True)
class Mirror:
    backend: BackendKind
    base: str                # repo id (HF) or gs:// prefix or https:// base
    repo_type: str = "dataset"          # HF only: dataset | model
    member_remote: dict[str, str] = {}  # local relpath -> remote relpath/URL override

@dataclass(frozen=True)
class Artifact:
    id: str                  # stable; matches task/model names where possible
    kind: Kind
    dest: str                # dir relative to the data root (e.g. data/tasks/cuperus_mpra_5utr)
    mirrors: tuple[Mirror, ...]          # priority order
    needed_by: tuple[str, ...] = ()      # task/model registry names
    requires: tuple[str, ...] = ()       # other artifact ids (e.g. shared refs)
    include: tuple[str, ...] = ("**/*",) # globs under dest the lock should track
    exclude: tuple[str, ...] = ("**/_cache/**",)
    license: License = License.UNKNOWN
    redistributable: bool = False        # may we re-host on our mirrors? (v2 publish honors this)
    raw: RawSpec | None = None           # v2: raw tier + build script + seed
```

`member_remote` handles mirrors whose layout differs from ours. Example: Shorkie
weights live on the authors' public bucket as `f0/model_best.h5` but land
locally as `f0.h5`, so the HTTP mirror maps `{"f0.h5": "f0/model_best.h5", …}`.
For our own HF/GCS mirrors the layout matches `dest`, so the map is empty.

## Backends

A two-method protocol; one implementation per `BackendKind`.

```python
class Backend(Protocol):
    def available(self) -> bool: ...                      # creds/tools present?
    def fetch(self, mirror, remote, dest_file) -> None:   # atomic + resumable
```

- **HTTP** (stdlib `urllib`) — public URLs, no auth. Resolves the Shorkie
  authors' bucket today, so it's the backend that proves `get` end-to-end.
- **HF** (`huggingface_hub`) — free, no-auth public mirror; the default for
  external users and CI. Also warms the HF cache for runtime-loaded models
  (Yorzoi, CodonTransformer).
- **GCS** — shells out to `gcloud storage cp` (fallback `gsutil cp`), matching
  the existing `scripts/brooks/…` pattern. No heavy Python GCS dep; uses the
  user's existing gcloud auth. Holds large/private files and is the secondary
  mirror. The bucket (`gs://yeast-seq2expression-benchmark`, project
  `clex-415415`, US multi-region) is **requester pays**, so every read and write
  must name a billing project: `--billing-project <project>` or the
  `YBENCH_GCS_BILLING_PROJECT` env var (flag wins; there is no gcloud-config
  fallback, so we never silently bill an ambient project). Without one the
  backend errors before shelling out and points you at `--from hf`.

The resolver walks an artifact's `mirrors` in order, skipping any whose backend
is unavailable or excluded by `--from`, and fetches every file in the lock.
HF-first means the happy path needs no credentials.

## Mirror strategy

**HF-first, GCS-secondary.** HF downloads are free and auth-free → best default
for outside users, CI, and agents. GCS is the maintainer/large-file backend and
the home for anything not suited to HF; because its bucket is **requester pays**,
a GCS download isn't free — the caller brings their own billing project and pays
egress. That's deliberate: HF stays the zero-cost default and GCS is the
bring-your-own-project path. Both mirror the same bytes; the lock's checksums are
the cross-mirror parity guard (`verify` against either must agree).

## CLI (`ybench data`)

```
ybench data get    [--config configs/x.yaml | --tasks a,b --models shorkie]
                   [--from hf|gcs|http] [--data-root DIR] [--dry-run]
ybench data verify [--tasks …] [--models …] [--data-root DIR]
ybench data status [--json] [--data-root DIR]
ybench data list   [--json]
ybench data lock   [--tasks …]          # maintainer: (re)compute checksums from local files
ybench data build  <task>               # v2: raw -> processed (stub: errors "not in v1")
```

- **Selection** mirrors `ybench run`: `--config` loads the run spec and pulls
  exactly the artifacts its `(model, task)` pairs need, closing over `requires`
  (shared refs). `--tasks`/`--models` are explicit overrides. Same
  `load_config().filtered()` path, so download and run never disagree.
- **`--dry-run`** prints the plan — every file, its chosen mirror, expected size
  — without fetching. This is the agent's "what will happen" probe.
- **Idempotency**: a file already present whose sha256 matches the lock is
  skipped. Re-running `get` converges; it never re-downloads valid files.
- **Atomicity**: fetch to a temp path, verify the hash, then move into place.
  An interrupted `get` never leaves a half-written file that looks valid.

## What's declared (v1)

- **Shared references** — `R64-1-1.fa`, `R64-1-1.115.gtf`, `R64-5-1.fa`
  (`data/tasks/`), required by ~6 tasks; declared once, deduped on fetch.
- **Task-processed** — one artifact per task in `TASKS`, `dest =
  data/tasks/<task-dir>`, mirrors = our HF dataset repo + GCS prefix. Lockable
  and verifiable against local files now; `get` from the mirrors is live once
  they're published.
- **Model weights**:
  - `shorkie` — `f0..f7.h5` + `params.json` from the authors' **public** bucket
    (`https://storage.googleapis.com/seqnn-share/shorkie/f{i}/model_best.h5`)
    and the `shorkie-pytorch` repo. Public → no re-hosting, `get` works today.
  - `yorzoi`, `codon_transformer` — HF hub models the adapters load at runtime;
    `get` warms the HF cache rather than placing files under `data/`.

## Licensing / re-hosting (drives v2 `publish`)

Each artifact carries `license` + `redistributable`. Our own data (Brooks) and
CC-BY sources (figshare/Zenodo) may be mirrored; GEO and the 1002 Genomes gVCF
are fetched from origin and **not** re-hosted. v1 doesn't publish, but the flags
are set now so the v2 `publish` step can refuse to mirror anything it shouldn't.

## Pilot → rollout

1. **Machinery**: manifest + lock + HF/GCS/HTTP backends + `data
   get/verify/status/list/lock`.
2. **Prove `get` end-to-end on Shorkie weights** — the only artifact with a live
   public mirror today; downloads and checksum-verifies for real.
3. **Cuperus as the task-data template** — declared, locked, and verified
   against local files; `get` goes live when its mirror is published.
4. **Copy the pattern** to the other 11 tasks; run `data lock` to freeze the
   full lock; commit it.

Publishing to the HF/GCS mirrors and the v2 raw/build/reproduce path are
separate, later PRs.

## Publishing (maintainer runbook)

Both mirrors are populated from the same locked local bytes. After (re)building
any distribution, re-freeze the lock, then publish to each backend:

```bash
uv sync --extra data
uv run ybench data lock                      # checksums from local files → lock
uv run ybench data publish --to hf           # dry run: prints the plan
uv run ybench data publish --to hf  --yes    # needs `huggingface-cli login` / HF_TOKEN
# GCS bucket is requester pays — name a billing project (flag or env):
export YBENCH_GCS_BILLING_PROJECT=clex-415415
uv run ybench data publish --to gcs          # dry run
uv run ybench data publish --to gcs --yes    # needs gcloud auth + billing project
```

`publish` re-checks every file against the lock before uploading and refuses to
push anything marked non-redistributable (Shorkie weights stay on the authors'
public bucket; raw GEO/1002genomes inputs are v2 source-only). Commit the lock
whenever it changes — it's the contract the download + the published bytes both
honor.

## Fresh-install acceptance test

The release gate before `dev` → `main`: a fresh checkout with an empty `data/`
must pull everything and run the full matrix. It's GPU-heavy and needs the
published mirror, so it's opt-in (`tests/test_fresh_install.py`, skipped unless
`YBENCH_FRESH_INSTALL=1`):

```bash
# on a GPU box, fresh checkout, mirror published:
uv sync --extra all
YBENCH_FRESH_INSTALL=1 uv run pytest -m integration tests/test_fresh_install.py
```

It pulls all artifacts, verifies checksums, then runs `configs/default.yaml`,
`configs/brooks.yaml`, and `configs/meneu.yaml` and asserts a `summary.json`
landed for every `(model, task)` pair.

## v2 sketch (planned, not built)

- `raw` tier per artifact: upstream URIs (GEO/figshare/Zenodo/buckets) +
  `build` (the existing `scripts/<task>/build_*.py`) + `build_seed`.
- `ybench data get --raw <task>` fetches raw inputs; `ybench data build <task>`
  runs the seeded build and **asserts the output sha256 equals the lock** —
  turning "reproducible from raw" into a CI test.
- Reproduce is **bit-exact**: every sampling step in the build scripts gets a
  pinned seed, ordering is made deterministic, float formatting is fixed. The
  eqtl negative-sampling and Shalem DEE2 sampling are the known nondeterminism
  to fix first.
