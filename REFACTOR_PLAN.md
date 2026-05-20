# Refactor plan — model classes for Yorzoi & Shorkie

**Branch:** `refactor/adapter-model-classes`
**Scope:** PART 1 of the post-Brooks refactor — structural cleanup only.
**Explicitly OUT of scope:** the correctness sweep (inverse Borzoi
transform + per-base unbin for every adapter). That ships as a
separate follow-up PR; this one **must produce bit-identical numbers**
on every existing benchmark.

## Motivation

Today, the adapter folder holds ~2,500 LOC across 12 files (6 per model).
Each Yorzoi adapter re-implements:

- `from_pretrained` (HF repo loading)
- the RC-averaging forward with plus↔minus strand swap (`_full_swap_idx`
  construction + the autocast-wrapped `_forward_full_tracks` body)
- a near-identical batch loop calling `model(x)` once per batch

Each Shorkie adapter re-implements:

- `from_checkpoints` (load 8 H5 folds via `Shorkie.from_tf_checkpoint`)
- the 8-fold ensemble loop with optional RC averaging
- the same model-to-device + eval boilerplate

Adding a new (model × task) pair currently means copy-pasting one of
these and changing the task-specific bits (input embedding, output
aggregation). The intended architecture is **models + benchmarks + thin
adapters connecting them** — the model-side machinery should live once.

## Goal

After the refactor, each adapter is responsible **only** for the
task-specific work:

1. Build the model input (encode a sequence, embed a cassette, etc.).
2. Call the model wrapper's batched forward.
3. Aggregate the model's output for the task's protocol.
4. Return the protocol's expected shape.

The wrapper classes own `from_pretrained` / `from_checkpoints`, RC
averaging, ensemble averaging, device + autocast handling, and the
batched forward path.

Concretely: every `from_pretrained` and every `_forward_full_tracks`
method disappears from the adapters.

## Proposed layout

The existing `src/yeastbench/models/shorkie.py` defines `class
Shorkie(nn.Module)` (the pytorch model). The new wrapper class is a
*different* concept (benchmark-side machinery, owns device + RC +
ensemble). To avoid a naming clash:

```
src/yeastbench/models/
├── __init__.py                       (new, empty or re-exports)
├── shorkie/
│   ├── __init__.py                   re-exports `ShorkieModule` and `Shorkie`
│   ├── nn.py                         current shorkie.py moved here, `class Shorkie` renamed to `ShorkieModule`
│   └── wrapper.py                    new `class Shorkie` (the wrapper)
└── yorzoi/
    ├── __init__.py                   re-exports `Yorzoi`
    └── wrapper.py                    new `class Yorzoi` (the wrapper)
```

**Renaming `class Shorkie(nn.Module)` to `ShorkieModule`** is the
slightly disruptive step. All call sites are inside `src/yeastbench/
adapters/shorkie_*.py` (10 import lines) and will be rewritten anyway
during the refactor, so the rename has near-zero collateral.

**Alternative considered**: keep the nn module's class name as
`Shorkie` and call the wrapper `ShorkieEnsemble`. Cleaner from a
"don't break things" angle, but ugly to use — the wrapper is the thing
adapters interact with, so it should have the natural name. **Picked
the rename.**

For Yorzoi the existing code imports `Borzoi` from the third-party
`yorzoi` package (`from yorzoi.model.borzoi import Borzoi`) — there's
no existing `yeastbench.models.yorzoi` module, so no rename needed.

## Wrapper class APIs

### `Yorzoi`

```python
# src/yeastbench/models/yorzoi/wrapper.py

class Yorzoi:
    """Benchmark-side wrapper around `yorzoi.model.borzoi.Borzoi`.
    Owns device, autocast, RC averaging with strand swap, and the
    batched forward. Adapters consume `forward_tracks_binned` and do
    task-specific aggregation on top."""

    SEQ_LEN: ClassVar[int] = 4992
    OUTPUT_BINS: ClassVar[int] = 300
    BIN_WIDTH: ClassVar[int] = 10
    CROP_BP_EACH_SIDE: ClassVar[int] = 996
    N_PLUS_TRACKS: ClassVar[int] = 81
    N_TRACKS_TOTAL: ClassVar[int] = 162

    def __init__(
        self,
        model: Any,  # yorzoi.model.borzoi.Borzoi
        device: str | torch.device = "cuda",
        use_rc: bool = True,
        autocast: bool = True,
    ) -> None: ...

    @classmethod
    def from_pretrained(
        cls,
        hf_repo: str,
        device: str = "cuda",
        use_rc: bool = True,
        autocast: bool = True,
    ) -> "Yorzoi": ...

    def forward_tracks_binned(
        self,
        x: torch.Tensor,  # (B, SEQ_LEN, 4) channels-last
    ) -> torch.Tensor:
        """One forward through the model with RC averaging.
        Returns (B, N_TRACKS_TOTAL, OUTPUT_BINS) — still binned at
        BIN_WIDTH bp, still in Borzoi's transformed output space.
        Adapter is responsible for any inverse transform / unbin
        (those land in the SEPARATE correctness PR)."""
```

Concretely, the body of `forward_tracks_binned` is the existing
`_forward_full_tracks` method copy-pasted verbatim (it's already
identical across 3 adapters today).

### `Shorkie`

```python
# src/yeastbench/models/shorkie/wrapper.py

class Shorkie:
    """8-fold ensemble wrapper around `ShorkieModule`. Owns device,
    RC averaging, ensemble averaging, and the batched forward."""

    SEQ_LEN: ClassVar[int] = 16384
    OUTPUT_BINS: ClassVar[int] = 896
    BIN_WIDTH: ClassVar[int] = 16
    CROP_BP_EACH_SIDE: ClassVar[int] = 1024

    def __init__(
        self,
        folds: list["ShorkieModule"],
        device: str | torch.device = "cuda",
        use_rc: bool = True,
    ) -> None: ...

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        device: str = "cuda",
        use_rc: bool = True,
    ) -> "Shorkie": ...

    def forward_tracks_binned(
        self,
        x: torch.Tensor,  # (B, 4, SEQ_LEN) channels-first
        track_subset: list[int] | None = None,
    ) -> torch.Tensor:
        """Mean across folds (+ RC averaging). Returns:
          - (B, OUTPUT_BINS, n_tracks_total) if track_subset is None
          - (B, OUTPUT_BINS, len(track_subset)) otherwise
        Adapters that always reduce to a track mean immediately may
        prefer the next helper instead."""

    def forward_track_mean_binned(
        self,
        x: torch.Tensor,
        track_subset: list[int],
    ) -> torch.Tensor:
        """Mean across folds + RC + track_subset, returned binned at
        BIN_WIDTH bp. Shape (B, OUTPUT_BINS). This is what every
        existing shorkie_* adapter does next, in slightly varying
        ways."""
```

The `forward_track_mean_binned` shortcut isn't strictly necessary
(adapters could `.mean(dim=2)` themselves on `forward_tracks_binned`'s
output), but every existing adapter does this immediately and it
saves repeated index_select bookkeeping.

## What gets removed from adapters

After the refactor, each adapter loses:

- The `model` / `device` / `use_rc` / `autocast` constructor params
  (now constructor takes a `Yorzoi` or `Shorkie` instance).
- The `_full_swap_idx` precomputation.
- The `_forward_full_tracks` / `_forward` method.
- The autocast context construction.
- The `from_pretrained` / `from_checkpoints` classmethod (replaced by
  a one-liner forwarding to the wrapper's classmethod).

Each adapter gains a `model: Yorzoi` (or `Shorkie`) attribute and
calls `self.model.forward_tracks_binned(x)` where it used to call
`self._forward(x)` or `self._forward_full_tracks(x)`.

Rough size estimate: each adapter shrinks by ~50–80 LOC. Total
reduction across 12 adapters: **~600–900 LOC**. The wrapper classes
add ~150 LOC, so net ≈ **−500 LOC**.

See **Sizing & scaling** below for actual measurements + how LOC
scales as new models and benchmarks are added.

## Sizing & scaling

### Measured removable LOC, per adapter

Counted by walking each adapter and tallying the three sections that
move into the wrapper class: the model/device/autocast setup inside
`__init__`, the body of `from_pretrained` / `from_checkpoints`, and
the body of `_forward` / `_forward_full_tracks` (Yorzoi) or the
ensemble loop (Shorkie). Numbers are conservative — they don't count
the few lines that get *simpler* but stay in the adapter
(`predict_*` no longer needs to construct its own autocast context,
no longer needs to refer to `self.use_rc`, etc.).

| Adapter | LOC today | Removable | Net | Notes |
| --- | ---: | ---: | ---: | --- |
| `yorzoi_brooks.py` | 214 | 61 | 153 | full RC + swap; transform inv stays |
| `yorzoi_eqtl.py` | 249 | 42 | 207 | no separate `_forward`; inline today |
| `yorzoi_mpra.py` | 153 | 33 | 120 | smallest; inline forward today |
| `yorzoi_mpra_marginalized.py` | 278 | 68 | 210 | |
| `yorzoi_shalem.py` | 261 | 69 | 192 | |
| `yorzoi_wu.py` | 161 | 64 | 97 | |
| `shorkie_brooks.py` | 166 | 51 | 115 | |
| `shorkie_eqtl.py` | 227 | 60 | 167 | |
| `shorkie_mpra.py` | 136 | 51 | 85 | |
| `shorkie_mpra_marginalized.py` | 259 | 68 | 191 | |
| `shorkie_shalem.py` | 254 | 70 | 184 | |
| `shorkie_wu.py` | 154 | 65 | 89 | |
| **Total** | **2,512** | **702** | **1,810** | |

Add back ~8 LOC per adapter for the thin forwarder
(`from_pretrained` becomes `cls(wrapper, **task_kwargs)` and the new
`__init__` takes a `wrapper` instead of building one): **+96 LOC**.

Add ~150 LOC for the two wrapper classes (75 each — `from_pretrained`,
`forward_tracks_binned`, RC machinery, ensemble loop): **+150 LOC**.

Add ~80 LOC for the new wrapper unit tests
(`test_yorzoi_wrapper.py` + `test_shorkie_wrapper.py`): **+80 LOC**.

| | Today | After refactor | Δ |
| --- | ---: | ---: | ---: |
| Adapters | 2,512 | 1,906 | −606 |
| Wrappers | 0 | 150 | +150 |
| Tests | (existing) | +80 | +80 |
| **Net repo change** | | | **−376 LOC** |

Less than my earlier hand-wavy "−500" but still a real win. The
under-shoot comes mostly from the two Yorzoi adapters
(`yorzoi_eqtl.py`, `yorzoi_mpra.py`) that don't have a dedicated
`_forward_full_tracks` method today and inline the forward in
`predict_*`. Those *do* clean up nicely (the autocast context goes
away, the `use_rc` flag handling goes away) but the LOC drop is
modest because they were already simpler.

### Scaling — how LOC grows with new models and benchmarks

Let `M` = number of models (today: 2), `T` = number of tasks/
benchmarks (today: 6). Marginal cost of adding one new entry:

| Action | Today (copy-adapter pattern) | After refactor |
| --- | --- | --- |
| New model (covers all existing T) | `T` new adapters @ ~200 LOC ≈ **+1200 LOC** | 1 wrapper @ ~75 LOC + `T` thin adapters @ ~100 LOC ≈ **+675 LOC** |
| New benchmark (covers all existing M) | `M` new adapters @ ~200 LOC ≈ **+400 LOC** | `M` thin adapters @ ~100 LOC ≈ **+200 LOC** |
| New (model, task) pair | 1 adapter @ ~200 LOC ≈ **+200 LOC** | 1 thin adapter @ ~100 LOC ≈ **+100 LOC** |

Rough total LOC formula:

| | Today | After refactor |
| --- | --- | --- |
| Approx. adapter+wrapper LOC | `~200 × M × T` | `~75 × M + ~100 × M × T` |

Concrete points:

- **Current state** (M=2, T=6): today ≈ 2400; refactored ≈ 1350.
- **+1 model** (M=3, T=6, e.g. ExoShorkie or a from-scratch model):
  today ≈ 3600; refactored ≈ 2025. **Saves ~1.5k LOC** at this scale.
- **+1 benchmark** (M=2, T=7, e.g. an OOD/exogenous track-prediction
  task): today ≈ 2800; refactored ≈ 1550. Saves ~1.2k LOC.

The slope of LOC growth is roughly halved per (model × task) cell
after the refactor. Beyond the LOC win, the more important quality is
that **adding a new model becomes O(one new wrapper + one thin
adapter per existing task)** instead of "rewrite 6 from_pretrained
methods + 6 forward methods + 6 RC swap precomputations." Same for
adding a benchmark.

What this refactor doesn't compress:

- **Per-task scaffold files** (`_deboer_scaffold.py`, `_shalem_scaffold.py`,
  `_wu_scaffold.py`, etc.) are shared between models for the same task
  already. They don't scale with `M` and don't get smaller here.
- **The per-task `predict_*` aggregation logic**. Each task has a
  genuinely different output formula (variant LFC over exon bins,
  marginalised logSED across host genes, per-base CDS sum for LFC,
  scalar over YFP bins, ...). That's task complexity, not boilerplate,
  and stays in the adapter.

If we later want to compress further, the next step (out of scope for
this PR) is parametrising one `MarginalizedExpressionPredictor`
adapter over many tasks, which would reduce inter-task duplication on
top of this PR's inter-model deduplication. ROADMAP records this as a
separate follow-up.

## Registry impact

The registry's `_build_*` factory functions need to construct a
wrapper once and pass it into each adapter. Two options:

### Option R1 — wrapper constructed inside the adapter's `from_pretrained`

```python
class YorzoiBrooksPredictor:
    @classmethod
    def from_pretrained(cls, hf_repo, device, use_rc, autocast,
                        track_mode="matched", batch_size=16):
        wrapper = Yorzoi.from_pretrained(hf_repo, device, use_rc, autocast)
        return cls(wrapper, track_mode=track_mode, batch_size=batch_size)
```

Pro: registry doesn't change. Con: if a config runs N tasks on the
same model, the wrapper (= the model weights on GPU) is loaded N
times.

### Option R2 — registry caches the wrapper per (model, device, kwargs) tuple

```python
def _build_yorzoi(task, device, **cfg):
    wrapper = _ensure_yorzoi(device=device, **{k: cfg.pop(k, ...) for k in WRAPPER_KEYS})
    adapter_factory, needs_refs = YORZOI_ADAPTERS[task.adapter_protocol]
    return adapter_factory(wrapper=wrapper, task=task, **cfg)
```

Pro: GPU loaded once per `ybench run` invocation. Con: requires
adapter factory signature change.

**Picked R1 for this PR** — keeps the refactor mechanical, no behavior
change. Caching can be a separate change later if config runs grow
large enough to need it.

## Order of operations

Each step is independently testable and produces bit-identical numbers
on the affected adapter's benchmark.

1. **Move `models/shorkie.py` → `models/shorkie/nn.py`** + rename
   `class Shorkie` → `class ShorkieModule`. Add
   `models/shorkie/__init__.py` re-exporting `ShorkieModule`. Update
   all 10 imports in `src/yeastbench/adapters/shorkie_*.py`. Run full
   test suite — should pass with **zero behavior change**.

2. **Add `models/yorzoi/__init__.py` + `wrapper.py`** with the
   `Yorzoi` class. Body of `forward_tracks_binned` lifted verbatim
   from `yorzoi_shalem.py:_forward_full_tracks` (which is identical
   to the body in `yorzoi_wu.py` and `yorzoi_brooks.py`).

3. **Migrate one Yorzoi adapter — pick `yorzoi_mpra.py`** (simplest;
   no RC swap, no track routing, smallest adapter). Update its
   `from_pretrained` to load via `Yorzoi.from_pretrained` and store
   the wrapper instead of the raw model. Replace `self.model(x)` with
   `self.model.forward_tracks_binned(x)`. Run the DREAM MPRA
   benchmark; bit-identical numbers required.

4. **Migrate remaining Yorzoi adapters** one at a time, running each
   adapter's benchmark after migration: `yorzoi_eqtl` → `yorzoi_wu`
   → `yorzoi_shalem` → `yorzoi_mpra_marginalized` → `yorzoi_brooks`.
   Bit-identical numbers required at every step.

5. **Add `models/shorkie/wrapper.py`** with the `Shorkie` class. Body
   of `forward_tracks_binned` lifted from `shorkie_mpra.py` (the
   8-fold ensemble loop + RC averaging is identical across all 6
   shorkie adapters).

6. **Migrate Shorkie adapters** one at a time: `shorkie_mpra` →
   `shorkie_eqtl` → `shorkie_wu` → `shorkie_shalem` →
   `shorkie_mpra_marginalized` → `shorkie_brooks`. Bit-identical
   numbers at every step.

7. **Final pass — re-run the full default config** end-to-end on GPU
   and diff every `summary.json` against the pre-refactor results to
   confirm bit-identical end-to-end. Document the diff (should be
   empty) in the commit message and PR description.

12 adapters × ~5 minutes GPU each = ~1 hour of GPU time across the
whole refactor, plus the implementation work.

## Tests

- **Existing adapter tests** must keep passing at every step (the
  `MockAdapter`s in `test_*.py` don't go through the wrapper, but
  the registry tests construct the real adapter and exercise the
  factory chain).
- **New wrapper-level tests**: small, focused (`tests/test_yorzoi_wrapper.py`,
  `tests/test_shorkie_wrapper.py`):
  - Constructor accepts device + use_rc + autocast.
  - `from_pretrained` / `from_checkpoints` round-trip.
  - `forward_tracks_binned` returns the right shape on a small batch.
  - RC averaging matches a hand-computed reference on a synthetic
    input (flip + index-swap + average).
- **No new integration tests needed** — the benchmark tests already
  cover end-to-end correctness via the adapters.

## Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Bit-identical numbers drift due to subtle reordering of ops (e.g. RC averaging before vs after track-subset selection) | Lift `_forward_full_tracks` body verbatim; don't restructure. Re-run each adapter's benchmark after migration. |
| Renaming `class Shorkie` breaks imports we don't catch | The 10 import lines are all in `src/yeastbench/adapters/shorkie_*.py` and all get rewritten anyway. Grep for `from yeastbench.models.shorkie import Shorkie` across the whole repo as a final check. |
| Registry kwargs leaked into adapter `__init__` (e.g. `autocast=` passed to an adapter that no longer accepts it) | Each adapter migration includes signature cleanup. CI / pytest catches this. |
| Per-task overrides (e.g. Yorzoi MPRA uses `track_subset=` defaulted to all 81 tracks; Yorzoi Brooks uses `track_mode="matched"`) | These are task-specific, stay on the adapter — wrapper exposes the full track tensor, adapter picks. |
| `Yorzoi.from_pretrained` called multiple times in one `ybench run` (option R1) | Acknowledged; deferred to a separate caching change. |

## Out of scope (for this PR)

- **Correctness sweep** — inverse Borzoi transform + per-base unbin
  across all adapters. ROADMAP records the audit; that's a separate
  PR with substantive number changes to argue about, and it builds on
  the model classes this PR introduces (the inverse transform will
  live in `Yorzoi.forward_per_base_raw_counts(...)`).
- **Registry-level wrapper caching** (option R2). Defer.
- **Adding new (model × task) pairs.** This PR is structural only.
- **Cross-task adapter consolidation** (e.g. parametrising one
  `MarginalizedExpressionPredictor` adapter over many tasks). Out of
  scope; would compress more but requires task-specific scaffold
  abstraction that's a separate design conversation.

## Acceptance criteria

- Full test suite green.
- `uv run ybench run --config configs/default.yaml` produces summary
  JSONs bit-identical to the pre-refactor versions (down to the same
  float precision).
- Adapter folder LOC drops by ≥500.
- Each adapter file no longer contains `from_pretrained` for the
  underlying nn module, RC-averaging code, or autocast context
  construction.
- PR description includes the LOC diff per file and the bit-identical
  guarantee.
