# Per-base correctness sweep — adapter migration

Move every genomic-model adapter onto **per-base, untransformed raw-count**
predictions, per the ROADMAP "always evaluate on the untransformed, unbinned
scale" section. The model wrappers own the per-base forwards:

- `Yorzoi.forward_tracks_perbase` — inverts the Borzoi transform per forward
  pass *before* RC-averaging (the inverse is nonlinear), then unbins 10 bp →
  per-base, returning `(B, 162, out_len)` raw per-track counts.
- `Shorkie.forward_track_mean_perbase` — just unbins 16 bp → per-base (the
  Poisson/softplus head is already raw; no inverse, order irrelevant).

Adapters do track aggregation + base-sums on raw counts. Each pair's
conversion **re-baselines** that task (rank metrics ~stable; Pearson /
absolute-magnitude predictions shift) and needs a GPU re-run to re-record
headline numbers.

## Status (each item = both `shorkie_` and `yorzoi_` of the pair)

- [x] **brooks** — PR #12. Yorzoi re-baselined (Δr ≈ −0.001; headline
  unchanged — LFC/correlation metric is invariant to the rescaling); Shorkie
  unchanged (no transform).
- [ ] **eqtl** — in progress. Caudal + Kita; per-base exon-base sum +
  strand-matched track mean. Re-baseline pending.
- [ ] **mpra_marginalized** — via the `MarginalizedLogSED` engine readout
  (Rafi DREAM).
- [ ] **shalem** — via the engine readout.
- [ ] **wu** — strand-matched 81-track mean × CDS-base sum (mCherry; an
  absolute-magnitude prediction → larger shift expected).
- [ ] **chen_marginalized** — standalone (not on the engine; keeps its own
  variant×host batching + numpy REF-log).
- [ ] **hong** — IGR-insertion; diagnostic track groups.

Wrapper groundwork (per-base forwards) + brooks: PR #12.
