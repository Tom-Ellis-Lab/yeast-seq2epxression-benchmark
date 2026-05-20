# Adapter refactor — remaining follow-ups

**Part 1 — model wrapper classes (`Yorzoi`, `Shorkie`) + thin adapters:
done in PR #3** (branch `refactor/adapter-model-classes`, merged into
`dev`). All 12 task adapters now take a wrapper instance and call its
batched forward; the wrappers own `from_pretrained` /
`from_checkpoints`, RC averaging, the 8-fold ensemble loop, autocast,
and the strand swap. Net LOC: −127. Bit-identical numbers; 144/144
tests green at every step.

This file now tracks **what's left** for the adapter/model layer.
Cross-references to ROADMAP where the same item is already recorded.

## Open follow-ups

### Correctness sweep — inverse Borzoi transform + per-base unbin

ROADMAP: `## Code-structure refactor (post-Brooks)` →
`### Correctness sweep — always evaluate on the untransformed,
unbinned scale`. Audit table there shows every Yorzoi adapter (except
`yorzoi_brooks`) still aggregates over Borzoi's transformed binned
output; Shorkie has no transform but bins at 16 bp.

Plan, now that the wrappers exist:

- Add `Yorzoi.forward_per_base_raw_counts(x, track_subset=None) →
  (B, n_tracks, BIN_WIDTH × OUTPUT_BINS)` that calls
  `forward_tracks_binned`, applies `_borzoi_inv_transform` per bin,
  and unbins to per-base via `np.repeat / BIN_WIDTH`.
- Symmetric `Shorkie.forward_per_base_raw_counts` — same shape, just
  no inverse transform (Shorkie's Poisson + softplus head outputs raw
  counts already).
- Migrate each non-Brooks adapter to consume the per-base API. Each
  becomes a CDS-bp slice → sum rather than a CDS-bin slice → sum;
  re-run benchmarks; document the number changes.

Expected: rank metrics (Spearman / AUROC / dir-acc) largely
unchanged; Pearson r and scalar magnitude predictions can shift
meaningfully (matches the Brooks Tier-1 / Tier-2 behaviour after
that PR's transform-fix).

Substantive number changes mean this PR needs a real review
discussion separate from the structural one — splitting it out from
the structural refactor was deliberate.

### Wrapper caching at the registry level

ROADMAP doesn't have this as a separate item yet. Currently each
adapter's `from_pretrained` builds its own wrapper instance, so a
config running N tasks on the same model loads weights N times.
Acceptable for the default config (3–6 tasks per model on different
devices anyway), but worth caching once a single
`ybench run` invocation routinely repeats model loads.

Sketch:

```python
def _build_yorzoi(task, device, **cfg):
    wrapper = _ensure_yorzoi(device=device, **{k: cfg.pop(k, ...) for k in WRAPPER_KEYS})
    adapter_factory, _ = YORZOI_ADAPTERS[task.adapter_protocol]
    return adapter_factory(wrapper=wrapper, task=task, **cfg)
```

Requires factory-signature changes (adapter takes `wrapper=` instead
of `**kwargs` rooted at the wrapper's classmethod). Low priority.

### Cross-task adapter consolidation

Not yet on ROADMAP. The intra-task duplication is already gone, but
adapters that solve structurally similar tasks across different
benchmarks (e.g. `yorzoi_shalem` vs `yorzoi_mpra_marginalized` —
both marginalised-over-host-genes logSED) still re-implement the
REF caching + ALT splicing loop separately. Could parametrise one
`MarginalizedExpressionPredictor` adapter class per model over many
tasks (route by a `Scaffold` / `Site` argument).

Defer until either (a) we add a third task in the marginalised
family, or (b) we discover a bug that needs fixing in N parallel
adapters at once.

## Sizing reference (post-Part-1)

For quick reference when adding new things.

| Slice | LOC |
| --- | ---: |
| 12 task adapters | 2,107 |
| 2 model wrappers | 278 |
| Adapter shared utilities (`_*_constants.py`, `_genome.py`, scaffolds) | ~1,200 (unchanged) |

Approximate marginal cost of new entries (after Part 1):

| Action | LOC |
| --- | ---: |
| New model (covers existing T) | 1 wrapper (~75 LOC) + T thin adapters (~100 LOC each) |
| New benchmark (covers existing M) | M thin adapters (~100 LOC each) |
| New (model, task) pair | 1 thin adapter (~100 LOC) |

The two-method Shorkie wrapper API (`forward_tracks_binned` vs
`forward_track_mean_binned`) was deliberate — the existing adapters
use both accumulation orderings, and preserving each one's exact
floating-point sequence kept the refactor bit-identical. Future
adapters should prefer `forward_tracks_binned` and reduce in the
adapter unless memory pressure or batch size argues otherwise.
