"""Characterization + unit tests for the shared marginalized-logSED engine.

The characterization test pins the public output of the Shorkie/Yorzoi
Shalem + MPRA adapters using a deterministic stub model and a tiny
synthetic genome. The ``EXPECTED`` / ``EXPECTED_MPRA`` values are the
**per-base, untransformed** baseline: the engine now reads the wrapper's
per-base forwards (Yorzoi inverts the Borzoi transform per-pass before
RC-averaging, then unbins; Shorkie is unbin-only) and the exon readout
selects exact base positions. These goldens were re-recorded when the
family was converted off the transformed-binned scale — they are an
intentional re-baseline, not a no-op refactor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ── Deterministic stub model ──────────────────────────────────


class StubModel:
    """A deterministic stand-in for the Shorkie/Yorzoi wrappers.

    ``forward_track_mean_binned`` / ``forward_tracks_binned`` return a
    fixed function of the (spliced) input one-hot, so the adapter's REF
    vs ALT difference is exercised without real weights or a GPU.
    """

    def __init__(self, seq_len, crop, bin_width, output_bins, n_tracks, layout,
                 global_weight=0.0):
        self.device = torch.device("cpu")
        self._sl = seq_len
        self._crop = crop
        self._bw = bin_width
        self._ob = output_bins
        self._nt = n_tracks
        self._layout = layout  # "cf" (B,4,L) or "cl" (B,L,4)
        # A nonzero global_weight makes every output bin depend on the whole
        # window (crude receptive field), so an insert outside the readout
        # bins still moves the output — needed to exercise the MPRA path.
        self._gw = global_weight

    def _signal(self, x: "torch.Tensor") -> "torch.Tensor":
        w = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=x.dtype, device=x.device)
        if self._layout == "cf":
            sig = (x * w[None, :, None]).sum(1)  # (B, L)
        else:
            sig = (x * w[None, None, :]).sum(-1)  # (B, L)
        cropped = sig[:, self._crop : self._crop + self._ob * self._bw]
        binned = cropped.reshape(sig.shape[0], self._ob, self._bw).mean(-1)  # (B, OB)
        if self._gw:
            binned = binned + self._gw * sig.mean(dim=1, keepdim=True)
        return binned

    def forward_track_mean_binned(self, x, track_subset):
        return self._signal(x)  # (B, OB)

    def forward_tracks_binned(self, x, track_subset=None):
        base = self._signal(x)  # (B, OB)
        if track_subset is not None:  # Shorkie all-track variant: (B, OB, n)
            nt = int(len(track_subset))
            scale = 1.0 + 0.01 * torch.arange(nt, dtype=base.dtype, device=base.device)
            return base[:, :, None] * scale[None, None, :]
        # Yorzoi: (B, 162, OB), per-track scale so + and - strands differ
        scale = 1.0 + 0.01 * torch.arange(self._nt, dtype=base.dtype, device=base.device)
        return base[:, None, :] * scale[None, :, None]

    # ── per-base forwards (what the converted engine actually calls) ──
    # The stub has no Borzoi transform, so its per-base output is simply the
    # binned output unbinned (÷ bin_width, repeat) — matching the wrapper
    # contract that summing a bin's BIN_WIDTH bases recovers the bin total.
    def _unbin(self, binned):
        return binned.repeat_interleave(self._bw, dim=-1) / float(self._bw)

    def forward_track_mean_perbase(self, x, track_subset):
        return self._unbin(self.forward_track_mean_binned(x, track_subset))

    def forward_tracks_perbase(self, x):
        return self._unbin(self.forward_tracks_binned(x))


# ── Synthetic genome that yields valid Shalem contexts ─────────


def build_synthetic_shalem_genome(out_dir: Path) -> tuple[Path, Path, Path]:
    """Write a FASTA + GTF + host-genes JSON for one chromosome 'I' with
    CYC1 (filler donor), one + and one - strand host gene. Returns
    (fasta_path, gtf_path, host_genes_json)."""
    import json

    import pysam

    chrom_len = 30000
    rng = np.random.default_rng(0)
    seq = list("".join(rng.choice(list("ACGT"), size=chrom_len)))

    # CYC1 (YJR048W): CDS 2001..2400 (1-based), then a UTR with two TATTTA
    # motifs so build_filler's deletion leaves >=200 bp of mutant UTR.
    utr = "TATTTA" + "GGGGGGGGGG" + "TATTTA" + ("C" * 478)  # 500 bp
    assert len(utr) == 500
    for i, b in enumerate(utr):
        seq[2400 + i] = b  # 0-based 2400.. == 1-based 2401..

    fasta_seq = "".join(seq)
    fa = out_dir / "genome.fa"
    fa.write_text(f">I\n{fasta_seq}\n")
    pysam.faidx(str(fa))

    # GTF: gene + single exon for each of CYC1, H+, H-
    def rows(gid, start, end, strand):
        return (
            f'I\tt\tgene\t{start}\t{end}\t.\t{strand}\t.\tgene_id "{gid}";\n'
            f'I\tt\texon\t{start}\t{end}\t.\t{strand}\t.\tgene_id "{gid}";\n'
        )

    gtf = out_dir / "genes.gtf"
    gtf.write_text(
        rows("YJR048W", 2001, 2400, "+")   # CYC1 (filler donor)
        + rows("YAL001W", 5000, 6000, "+")  # Shalem host +
        + rows("YBR001C", 10000, 11000, "-")  # Shalem host -
        + rows("YOL056W", 8000, 8500, "+")  # MPRA host + (in HOST_GENES)
        + rows("YLR218C", 15000, 15500, "-")  # MPRA host - (in HOST_GENES)
    )

    host_json = out_dir / "host_genes.json"
    host_json.write_text(json.dumps({"genes": [
        {"gene_id": "YAL001W", "strand": "+"},
        {"gene_id": "YBR001C", "strand": "-"},
    ]}))
    return fa, gtf, host_json


@pytest.fixture
def shalem_genome(tmp_path: Path) -> tuple[Path, Path, Path]:
    return build_synthetic_shalem_genome(tmp_path)


# Two fixed 150 bp test oligos.
OLIGOS = [
    ("ACGT" * 37 + "AC"),
    ("TTGGCCAA" * 18 + "TTGGCC"),
]
assert all(len(o) == 150 for o in OLIGOS)


def _build_shorkie_shalem(stub, fa, gt, host_json):
    from yeastbench.adapters.shorkie_shalem import ShorkieShalemPredictor

    return ShorkieShalemPredictor(
        stub, fasta_path=fa, gtf_path=gt, host_genes_json=host_json,
        track_subset=[0, 1, 2, 3], batch_size=8,
    )


def _build_yorzoi_shalem(stub, fa, gt, host_json):
    from yeastbench.adapters.yorzoi_shalem import YorzoiShalemPredictor

    return YorzoiShalemPredictor(
        stub, fasta_path=fa, gtf_path=gt, host_genes_json=host_json,
        batch_size=8,
    )


# Per-base untransformed baseline (deterministic stub + genome).
EXPECTED = {
    "shorkie": [-0.00012493133544921875, -3.123283386230469e-05],
    "yorzoi": [-8.678436279296875e-05, 0.0],
}


class TestShalemCharacterization:
    def test_shorkie_shalem_matches_golden(self, shalem_genome):
        fa, gt, host_json = shalem_genome
        stub = StubModel(16384, 1024, 16, 896, n_tracks=4, layout="cf")
        adapter = _build_shorkie_shalem(stub, fa, gt, host_json)
        out = adapter.predict_terminator_marginalized(OLIGOS)
        if EXPECTED["shorkie"] is None:
            print("RECORD shorkie:", repr(out.tolist()))
            pytest.skip("recording golden")
        np.testing.assert_array_equal(out, np.array(EXPECTED["shorkie"]))

    def test_yorzoi_shalem_matches_golden(self, shalem_genome):
        fa, gt, host_json = shalem_genome
        stub = StubModel(4992, 996, 10, 300, n_tracks=162, layout="cl")
        adapter = _build_yorzoi_shalem(stub, fa, gt, host_json)
        out = adapter.predict_terminator_marginalized(OLIGOS)
        if EXPECTED["yorzoi"] is None:
            print("RECORD yorzoi:", repr(out.tolist()))
            pytest.skip("recording golden")
        np.testing.assert_array_equal(out, np.array(EXPECTED["yorzoi"]))


# ── MPRA-marginalized characterization ────────────────────────

MPRA_SEQS = [
    "T" * 17 + "ACGT" * 20 + "A" * 13,
    "T" * 17 + "TTGGCCAA" * 10 + "A" * 13,
]
assert all(len(s) == 110 for s in MPRA_SEQS)


def _build_shorkie_mpra(stub, fa, gt):
    from yeastbench.adapters.shorkie_mpra_marginalized import (
        ShorkieMPRAMarginalizedPredictor,
    )

    return ShorkieMPRAMarginalizedPredictor(
        stub, fasta_path=fa, gtf_path=gt, track_subset=[0, 1, 2, 3], batch_size=8,
    )


def _build_yorzoi_mpra(stub, fa, gt):
    from yeastbench.adapters.yorzoi_mpra_marginalized import (
        YorzoiMPRAMarginalizedPredictor,
    )

    return YorzoiMPRAMarginalizedPredictor(
        stub, fasta_path=fa, gtf_path=gt, batch_size=8,
    )


EXPECTED_MPRA = {
    "shorkie": [5.7220458984375e-06, 5.7220458984375e-06],
    "yorzoi": [1.71661376953125e-05, 1.71661376953125e-05],
}


class TestMPRACharacterization:
    def test_shorkie_mpra_matches_golden(self, shalem_genome):
        fa, gt, _ = shalem_genome
        stub = StubModel(16384, 1024, 16, 896, n_tracks=4, layout="cf", global_weight=0.5)
        adapter = _build_shorkie_mpra(stub, fa, gt)
        out = adapter.predict_expression_scores(MPRA_SEQS)
        if EXPECTED_MPRA["shorkie"] is None:
            print("RECORD mpra shorkie:", repr(out.tolist()))
            pytest.skip("recording golden")
        np.testing.assert_array_equal(out, np.array(EXPECTED_MPRA["shorkie"]))

    def test_yorzoi_mpra_matches_golden(self, shalem_genome):
        fa, gt, _ = shalem_genome
        stub = StubModel(4992, 996, 10, 300, n_tracks=162, layout="cl", global_weight=0.5)
        adapter = _build_yorzoi_mpra(stub, fa, gt)
        out = adapter.predict_expression_scores(MPRA_SEQS)
        if EXPECTED_MPRA["yorzoi"] is None:
            print("RECORD mpra yorzoi:", repr(out.tolist()))
            pytest.skip("recording golden")
        np.testing.assert_array_equal(out, np.array(EXPECTED_MPRA["yorzoi"]))


# ── Unit tests for the model-coverage strategies ──────────────


class TestModelCoverageReadout:
    def test_shorkie_readout_is_exon_bin_sum(self):
        from yeastbench.adapters._marginalized_logsed import ShorkieCoverage

        cov = ShorkieCoverage.__new__(ShorkieCoverage)  # skip model wiring
        out = torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5)  # (B=2, bins=5)
        bins_t = torch.tensor([1, 3], dtype=torch.long)
        # row 0: bins 1,3 -> 1 + 3 = 4
        assert float(cov.readout(out, 0, bins_t, "+")) == 4.0
        # row 1: 6 + 8 = 14
        assert float(cov.readout(out, 1, bins_t, "-")) == 14.0

    def test_yorzoi_readout_strand_matched_mean(self):
        from yeastbench.adapters._marginalized_logsed import YorzoiCoverage

        cov = YorzoiCoverage.__new__(YorzoiCoverage)
        # (B=1, 162 tracks, bins=4); bin-sum over bins [0,1]
        out = torch.zeros(1, 162, 4, dtype=torch.float32)
        out[0, :, 0] = 1.0  # every track has bin0 = 1 -> bin-sum over {0,1} = 1
        bins_t = torch.tensor([0, 1], dtype=torch.long)
        # + strand: mean over tracks 0..80 of (1.0) = 1.0
        assert float(cov.readout(out, 0, bins_t, "+")) == pytest.approx(1.0)
        # make - strand tracks distinct
        out[0, 81:162, 0] = 3.0
        assert float(cov.readout(out, 0, bins_t, "-")) == pytest.approx(3.0)


def test_logsed_primitive_sign():
    # log2(alt+1) - log2(ref+1): alt>ref -> positive
    alt = torch.tensor([3.0, 0.0])
    ref = torch.tensor([1.0, 3.0])
    logsed = torch.log2(alt + 1.0) - torch.log2(ref + 1.0)
    assert logsed[0] > 0 and logsed[1] < 0


class TestPerBaseForward:
    """The converted Coverage strategies read the wrapper's **per-base**
    forwards: output length is ``OUTPUT_BINS*BIN_WIDTH`` and summing a
    bin's ``BIN_WIDTH`` bases recovers that bin's total — i.e. the readout
    operates on raw per-base counts, not transformed bins. (Inverse-
    before-mean order is proven wrapper-side in test_perbase_wrappers.py;
    here we pin that the engine consumes the per-base axis.)"""

    @staticmethod
    def _stub(layout, n_tracks):
        # seq_len >= crop + output_bins*bin_width = 8 + 40 = 48
        return StubModel(
            seq_len=64, crop=8, bin_width=4, output_bins=10,
            n_tracks=n_tracks, layout=layout,
        )

    def test_shorkie_forward_is_perbase_and_unbins(self):
        from yeastbench.adapters._marginalized_logsed import ShorkieCoverage

        stub = self._stub("cf", 4)
        cov = ShorkieCoverage(stub, [0, 1, 2, 3])
        x = torch.rand(2, 4, 64)
        binned = stub.forward_track_mean_binned(x, cov._track_idx_t)  # (2, 10)
        perbase = cov.forward(x)  # (2, 40)
        assert perbase.shape == (2, 10 * 4)
        torch.testing.assert_close(perbase.reshape(2, 10, 4).sum(-1), binned)

    def test_yorzoi_forward_is_perbase_and_unbins(self):
        from yeastbench.adapters._marginalized_logsed import YorzoiCoverage

        stub = self._stub("cl", 162)
        cov = YorzoiCoverage(stub)
        x = torch.rand(2, 64, 4)
        binned = stub.forward_tracks_binned(x)  # (2, 162, 10)
        perbase = cov.forward(x)  # (2, 162, 40)
        assert perbase.shape == (2, 162, 10 * 4)
        torch.testing.assert_close(perbase.reshape(2, 162, 10, 4).sum(-1), binned)
