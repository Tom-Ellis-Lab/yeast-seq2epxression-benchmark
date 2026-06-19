"""Tests for the per-base raw-count wrapper forwards.

Pins the Yorzoi inverse-transform port against the upstream Yorzoi source,
proves the inverse is applied per-element *before* any RC / track mean
(the correctness fix), and checks the Brooks adapters consume per-base raw
counts. The Brooks re-baseline (real-data metric shift) is verified
separately on GPU; here we prove the *math/order* with stubs.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yeastbench.models.yorzoi.wrapper import (
    BIN_WIDTH as Y_BIN,
    N_PLUS_TRACKS,
    N_TRACKS_TOTAL,
    OUTPUT_BINS as Y_OUT_BINS,
    SEQ_LEN as Y_SEQ_LEN,
    Yorzoi,
    _borzoi_transform_inv,
    _unbin_per_base,
)


def _borzoi_transform(x: "torch.Tensor") -> "torch.Tensor":
    """Forward transform (no scale) — matches yorzoi/dataset.py target prep."""
    expd = torch.pow(x, 0.75)
    return torch.where(
        expd <= 384, expd, torch.minimum(expd, 384 + torch.sqrt(expd - 384))
    )


# ── inverse-transform math ────────────────────────────────────


class TestInverseTransform:
    def test_round_trip(self):
        raw = torch.tensor([0.0, 1.0, 5.0, 100.0, 383.0, 700.0, 5000.0])
        back = _borzoi_transform_inv(_borzoi_transform(raw))
        torch.testing.assert_close(back, raw, rtol=1e-4, atol=1e-2)

    def test_matches_upstream(self):
        u = pytest.importorskip("yorzoi.utils")
        y = torch.tensor([0.0, 10.0, 384.0, 400.0, 1000.0, 5000.0])
        torch.testing.assert_close(
            _borzoi_transform_inv(y), u._borzoi_transform_inv(y)
        )


def test_jensen_gap_motivates_order():
    """Why the inverse must precede the mean: mean-of-inverses recovers the
    raw mean exactly; inverse-of-(transformed)-mean is biased."""
    r1, r2 = torch.tensor(10.0), torch.tensor(2000.0)
    t1, t2 = _borzoi_transform(r1), _borzoi_transform(r2)
    mean_of_inv = 0.5 * (_borzoi_transform_inv(t1) + _borzoi_transform_inv(t2))
    inv_of_mean = _borzoi_transform_inv(0.5 * (t1 + t2))
    torch.testing.assert_close(mean_of_inv, torch.tensor(1005.0), rtol=1e-4, atol=1e-1)
    assert abs(float(inv_of_mean) - 1005.0) > 1.0


def test_gene_exon_base_positions():
    from yeastbench.adapters._genome import Gene, gene_exon_base_positions

    # Exon 1-based [101, 110]; window_start=0, crop=50 → cropped output covers
    # genomic [50, 150); exon 0-based [100, 110) → output offsets [50, 60).
    gene = Gene(
        chrom_roman="I", strand="+", tss=101,
        gene_start=101, gene_end=110, exons=((101, 110),),
    )
    base = gene_exon_base_positions(gene, 0, 50, 100)
    assert base.tolist() == list(range(50, 60))

    # Exon partially outside the crop is clamped; fully-outside exon drops.
    gene2 = Gene(
        chrom_roman="I", strand="+", tss=40, gene_start=40, gene_end=55,
        exons=((40, 55), (200, 210)),
    )
    base2 = gene_exon_base_positions(gene2, 0, 50, 100)
    # exon [39,55) → offsets [max(0,-11), 5) = [0,5); second exon out of range.
    assert base2.tolist() == list(range(0, 5))


def test_unbin_recovers_bin_total():
    binned = torch.tensor([[3.0, 10.0]])  # (1 track, 2 bins)
    per_base = _unbin_per_base(binned, 4)  # (1, 8)
    assert per_base.shape == (1, 8)
    torch.testing.assert_close(per_base[0, :4].sum(), torch.tensor(3.0))
    torch.testing.assert_close(per_base[0, 4:].sum(), torch.tensor(10.0))


# ── wrapper per-base forward ──────────────────────────────────


class _StubBorzoi:
    """Underlying-model stub: returns a fixed transformed (B,162,bins) output."""

    def __init__(self, out_transformed):
        self._out = out_transformed

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, x):
        return self._out.expand(x.shape[0], N_TRACKS_TOTAL, Y_OUT_BINS)


def test_wrapper_inverts_per_track_then_unbins():
    # Distinct raw value per track → transformed output handed to the wrapper.
    raw = torch.arange(1, N_TRACKS_TOTAL + 1, dtype=torch.float32).reshape(
        1, N_TRACKS_TOTAL, 1
    ).expand(1, N_TRACKS_TOTAL, Y_OUT_BINS)
    out_t = _borzoi_transform(raw).contiguous()
    w = Yorzoi(_StubBorzoi(out_t), device="cpu", use_rc=False, autocast=False)

    perbase = w.forward_tracks_perbase(torch.zeros(1, Y_SEQ_LEN, 4))
    assert perbase.shape == (1, N_TRACKS_TOTAL, Y_OUT_BINS * Y_BIN)
    # Summing a bin's BIN_WIDTH bases recovers that track's raw bin total.
    recovered = perbase[0, :, :Y_BIN].sum(dim=1)
    torch.testing.assert_close(recovered, raw[0, :, 0], rtol=1e-4, atol=1e-2)


def test_brooks_track_mean_is_on_raw_counts():
    """End-to-end: yorzoi_brooks means strand-matched tracks on the *raw*
    per-base predictions (inverse before track-mean), so a bin's base-sum
    equals the mean of the per-track raw bin totals — not the biased
    inverse-of-transformed-mean."""
    from yeastbench.adapters.yorzoi_brooks import YorzoiBrooksPredictor

    # Plus-strand tracks 0..80 get alternating distinct raws; "all" mode
    # averages exactly those for a + strand sample.
    raw = torch.zeros(1, N_TRACKS_TOTAL, Y_OUT_BINS)
    plus_raws = torch.where(
        torch.arange(N_PLUS_TRACKS) % 2 == 0,
        torch.tensor(20.0), torch.tensor(2000.0),
    )  # mean = 1010
    raw[0, :N_PLUS_TRACKS, :] = plus_raws[None, :, None]
    out_t = _borzoi_transform(raw).contiguous()

    model = Yorzoi(_StubBorzoi(out_t), device="cpu", use_rc=False, autocast=False)
    adapter = YorzoiBrooksPredictor(model, track_mode="all", batch_size=1)

    seq = "ACGT" * (Y_SEQ_LEN // 4)
    out = adapter.predict_coverage_batch([seq], ["+"], [None])  # (1, 3000)
    assert out.shape == (1, Y_OUT_BINS * Y_BIN)

    expected_bin_total = float(plus_raws.mean())  # 1010
    got = float(out[0, :Y_BIN].sum())
    assert abs(got - expected_bin_total) < 1.0, (got, expected_bin_total)


# ── Chen marginalized per-base CDS readout ────────────────────


class _StubPerbaseChenModel:
    """Stand-in for the model wrapper exposing the per-base forwards the
    Chen adapters call. Base position ``p`` carries value ``p`` on every
    track, so a CDS base-sum is a known arithmetic series and the strand-
    matched track mean (Yorzoi) is trivial to predict."""

    def __init__(self, out_len, n_tracks=N_TRACKS_TOTAL):
        self.device = torch.device("cpu")
        self._ol = out_len
        self._nt = n_tracks

    def forward_tracks_perbase(self, x):  # Yorzoi path
        base = torch.arange(self._ol, dtype=torch.float32)
        return base[None, None, :].expand(x.shape[0], self._nt, self._ol).contiguous()

    def forward_track_mean_perbase(self, x, track_subset):  # Shorkie path
        base = torch.arange(self._ol, dtype=torch.float32)
        return base[None, :].expand(x.shape[0], self._ol).contiguous()


def _fake_chen_ctx(cds_base_lo, cds_base_hi):
    from yeastbench.adapters._chen_marginalized import ChenHostContext, HostMeta

    host = HostMeta(
        gene_id="G", gene_name="G", chrom="I", strand="+",
        cds_start=1, cds_end=2, tier="high", dee2_tpm=1.0,
    )
    return ChenHostContext(
        host=host, window_start=0, seq_len=10, window_seq="",
        cds_bin_lo=0, cds_bin_hi=1, cds_base_lo=cds_base_lo, cds_base_hi=cds_base_hi,
        var_start_in_window=0, var_needs_revcomp=False,
    )


def test_yorzoi_chen_exon_sums_on_raw_base_positions():
    """YorzoiChenPredictor._predict_exon_sums reads forward_tracks_perbase,
    sums the CDS base positions, then means strand-matched tracks — all on
    raw counts (inverse applied upstream in the wrapper)."""
    from yeastbench.adapters.yorzoi_chen_marginalized import YorzoiChenPredictor

    out_len = Y_OUT_BINS * Y_BIN
    pred = YorzoiChenPredictor.__new__(YorzoiChenPredictor)  # skip model/data wiring
    pred.model = _StubPerbaseChenModel(out_len)
    contexts = [_fake_chen_ctx(10, 25)]  # 15 CDS bases
    track_slices = [(0, N_PLUS_TRACKS)]
    sums = pred._predict_exon_sums(torch.zeros(1, Y_SEQ_LEN, 4), contexts, track_slices)
    assert float(sums[0]) == pytest.approx(float(sum(range(10, 25))))


def test_shorkie_chen_exon_sums_on_raw_base_positions():
    from yeastbench.adapters.shorkie_chen_marginalized import ShorkieChenPredictor
    from yeastbench.models.shorkie.wrapper import BIN_WIDTH as S_BIN, OUTPUT_BINS as S_OB

    out_len = S_OB * S_BIN
    pred = ShorkieChenPredictor.__new__(ShorkieChenPredictor)
    pred.model = _StubPerbaseChenModel(out_len)
    pred._track_idx_gpu = torch.tensor([0, 1], dtype=torch.long)
    contexts = [_fake_chen_ctx(10, 25)]
    sums = pred._predict_exon_sums(torch.zeros(1, 4, pred.model._ol), contexts)
    assert float(sums[0]) == pytest.approx(float(sum(range(10, 25))))
