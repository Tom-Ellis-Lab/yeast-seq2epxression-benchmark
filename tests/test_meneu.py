"""Tests for the Meneu foreign-DNA tiled-coverage benchmark.

The benchmark tiles a whole contig to the *adapter's* receptive field at run
time, stitches the per-tile central predictions into one per-base track, and
scores it per contig against a measured RNA-seq coverage sidecar
(``meneu_cov_<contig>.npz``, arrays ``seq``/``fwd``/``rev``). The tests build a
tiny self-contained set of ``.npz`` sidecars and run the whole path against a
pure-numpy mock ``TiledCoverageTrackPredictor`` — no GPU, no torch.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from yeastbench.adapters.protocols import TiledCoverageTrackPredictor
from yeastbench.benchmarks.base import BenchmarkInfo
from yeastbench.benchmarks.meneu import (
    COV_PREFIX,
    MeneuForeignDNABenchmark,
    tile_contig,
)
from yeastbench.registry import TASKS

INFO = BenchmarkInfo(name="test_meneu", version="test", description="t",
                     distribution_uri="")

# Mock model geometry — tiny so the fixtures are cheap. stride == out_len.
SEQ_LEN = 20
CROP = 4
OUT_LEN = SEQ_LEN - 2 * CROP            # 12 — the per-tile central region

# Two contigs. Lengths chosen so each gives exactly one 5 kb eval window
# (L // EVAL_WINDOW == 1). Mmmyco's length is NOT a multiple of the stride, so
# its final tile's central region is clipped on stitching (exercises a short
# last tile). Both are fully covered exactly once.
CONTIGS = {"Mpneumo": 6000, "Mmmyco": 5500}

_BASE_CODE = {"A": 1.0, "C": 2.0, "G": 3.0, "T": 4.0, "N": 0.0}


def _contig_seq(rng: np.random.Generator, length: int) -> str:
    return "".join(rng.choice(list("ACGT"), size=length))


def _truth_from_seq(seq: str) -> np.ndarray:
    """Per-base 'measured coverage' = base code + a low positional ramp,
    so it is non-constant within every 5 kb window (Pearson defined)."""
    codes = np.array([_BASE_CODE[b] for b in seq], dtype=np.float32)
    ramp = (np.arange(len(seq), dtype=np.float32) % 7) * 0.1
    return codes + ramp


def _split_fwd_rev(truth: np.ndarray, rng: np.random.Generator
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Split an unstranded truth into fwd/rev that sum back to it; the
    benchmark uses fwd + rev as the unstranded label."""
    frac = rng.uniform(0.2, 0.8, size=truth.shape).astype(np.float32)
    fwd = (truth * frac).astype(np.float32)
    rev = (truth - fwd).astype(np.float32)
    return fwd, rev


@pytest.fixture
def meneu_dir(tmp_path: Path) -> Path:
    """Write tiny window-agnostic ``meneu_cov_<contig>.npz`` sidecars
    (seq + fwd + rev) and return the directory holding them."""
    rng = np.random.default_rng(0)
    for chrom, L in CONTIGS.items():
        full = _contig_seq(rng, L)
        truth = _truth_from_seq(full)
        fwd, rev = _split_fwd_rev(truth, rng)
        np.savez(
            tmp_path / f"{COV_PREFIX}{chrom}.npz",
            seq=np.frombuffer(full.encode("ascii"), dtype=np.uint8),
            fwd=fwd, rev=rev,
        )
    return tmp_path


class _MockAdapter:
    """Pure-numpy ``TiledCoverageTrackPredictor``. For each tile it reads
    the central region of the input seq and returns a DETERMINISTIC,
    NON-constant per-base vector (the same base-code + ramp the truth was
    built from, lightly perturbed) — so per-window Pearson is well-defined
    and tracks the truth. Returns shape ``(B, OUT_LEN)`` in raw per-base
    unstranded count units."""

    seq_len = SEQ_LEN
    crop_bp_each_side = CROP
    batch_size = 7                          # arbitrary, exercises chunking
    varies_by_strain = False

    def predict_coverage_batch(self, seqs, strands, strains=None):
        out = np.empty((len(seqs), OUT_LEN), dtype=np.float64)
        for i, s in enumerate(seqs):
            central = s[CROP:CROP + OUT_LEN]
            codes = np.array([_BASE_CODE.get(b, 0.0) for b in central],
                             dtype=np.float64)
            ramp = (np.arange(OUT_LEN, dtype=np.float64) % 7) * 0.1
            # Scale by 3 so pred and true differ in magnitude (the
            # depth-normalization / fold-change path is exercised), but
            # the shape co-varies, so raw Pearson is high.
            out[i] = 3.0 * (codes + ramp) + 0.5
        return out


assert isinstance(_MockAdapter(), TiledCoverageTrackPredictor)


def _contig_len(data_dir: Path, chrom: str) -> int:
    with np.load(data_dir / f"{COV_PREFIX}{chrom}.npz") as d:
        return len(d["fwd"])


_METRIC_KEYS = (
    "shape_pearson", "shape_js", "mag_fc_mean", "mag_fc_sd",
    "n_windows_kept", "n_windows_pearson", "n_windows_total",
)


# ── tile_contig (pure run-time tiling) ────────────────────────


class TestTileContig:
    def test_central_regions_tile_contiguously(self):
        seq = "ACGT" * 25                     # 100 bp, no Ns
        window, crop = 20, 4
        stride = window - 2 * crop            # 12
        tiles = tile_contig(seq, window, crop)
        assert len(tiles) == (len(seq) + stride - 1) // stride
        # center_starts step by stride from 0, covering [0, L)
        assert [t.center_start for t in tiles] == list(range(0, len(seq), stride))
        # every input window is exactly `window` bp
        assert all(len(t.seq) == window for t in tiles)

    def test_ends_are_n_padded(self):
        seq = "ACGT" * 25
        window, crop = 20, 4
        tiles = tile_contig(seq, window, crop)
        # first tile: window opens `crop` bp before the contig -> leading Ns
        assert tiles[0].window_start == -crop
        assert tiles[0].seq[:crop] == "N" * crop
        assert tiles[0].seq[crop] == seq[0]
        assert tiles[0].seq.count("N") == crop      # only the left pad
        # last tile: window runs past the end -> trailing Ns
        assert tiles[-1].seq.endswith("N")

    def test_no_padding_when_window_equals_stride(self):
        # crop=0 -> stride==window, interior tiles are exact contig slices
        tiles = tile_contig("A" * 30, window=10, crop=0)
        assert [t.window_start for t in tiles] == [0, 10, 20]
        assert all(t.seq.count("N") == 0 for t in tiles)


# ── benchmark ─────────────────────────────────────────────────


class TestMeneuBenchmark:
    def test_init_discovers_contigs(self, meneu_dir):
        b = MeneuForeignDNABenchmark(meneu_dir, INFO)
        assert set(b.contigs) == set(CONTIGS)

    def test_init_requires_sidecars(self, tmp_path):
        with pytest.raises(AssertionError):
            MeneuForeignDNABenchmark(tmp_path, INFO)     # empty dir

    def test_evaluate_metrics_present_and_finite(self, meneu_dir):
        b = MeneuForeignDNABenchmark(meneu_dir, INFO)
        res = b.evaluate(_MockAdapter())
        assert set(res.contigs) == set(CONTIGS)
        assert res.window_len == SEQ_LEN          # taken from the adapter
        for c in CONTIGS:
            pc = res.per_contig[c]
            for k in _METRIC_KEYS:
                assert k in pc, f"{k} missing for {c}"
            assert np.isfinite(pc["shape_pearson"])
            assert np.isfinite(pc["shape_js"])
            assert np.isfinite(pc["mag_fc_mean"])
            assert np.isfinite(pc["mag_fc_sd"])
            assert pc["n_windows_total"] == 1
            assert pc["n_windows_kept"] >= 1
            # The mock prediction is always non-constant and non-zero, so
            # every kept window yields a finite Pearson — none dropped.
            assert pc["n_windows_pearson"] == pc["n_windows_kept"]
            # Mock shape co-varies with truth → high raw Pearson.
            assert pc["shape_pearson"] > 0.9

    def test_fold_change_runs_over_all_windows(self, meneu_dir):
        """Magnitude FC must cover ALL eval windows (option A), including a
        true-silent window where the model predicts coverage — that is a real
        mis-allocation and must register. The shape floor gates only the shape
        metrics; it must NOT shrink the magnitude window set."""
        b = MeneuForeignDNABenchmark(meneu_dir, INFO)
        W = b.EVAL_WINDOW
        rng = np.random.default_rng(1)
        # Window 0: genuine non-flat signal. Window 1: true all-zero (floored
        # out of the shape metrics) but the model hallucinates flat coverage.
        true = np.concatenate([rng.uniform(1.0, 10.0, W), np.zeros(W)])
        pred = np.concatenate([rng.uniform(1.0, 10.0, W), np.full(W, 5.0)])
        sc = b._score_contig(true, pred)
        assert sc["n_windows_total"] == 2
        assert sc["n_windows_kept"] == 1          # silent-true window not kept for shape
        # Recompute the per-window FC; the metric must average BOTH windows
        # (the old kept-only behaviour would have used window 0 alone).
        scale = true.sum() / pred.sum()
        pdn = (pred * scale).reshape(2, W)
        t = true.reshape(2, W)
        fc = np.log2((pdn.sum(axis=1) + 1) / (t.sum(axis=1) + 1))
        assert fc[1] > 0                          # hallucination on silent truth penalised
        assert sc["mag_fc_mean"] == pytest.approx(float(fc.mean()))
        assert sc["mag_fc_sd"] == pytest.approx(float(fc.std()))
        # The all-window mean differs from the kept-only (window-0) value.
        assert sc["mag_fc_mean"] != pytest.approx(float(fc[0]))

    def test_tiling_covers_every_base(self, meneu_dir):
        b = MeneuForeignDNABenchmark(meneu_dir, INFO)
        res = b.evaluate(_MockAdapter())
        for c, L in CONTIGS.items():
            pred = res.stitched_pred[c]
            assert pred.shape == (L,)
            # The stitched track inits to np.zeros(L), so an unwritten
            # ("hole") position stays 0.0 — which np.isfinite would NOT
            # catch. The mock never emits 0 in the interior (min value 3.5),
            # so requiring every position > 0 actually catches a base the
            # tiling failed to cover.
            assert np.all(pred > 0), (
                f"{c}: stitching left an unwritten 0.0 hole at indices "
                f"{np.flatnonzero(pred == 0).tolist()[:5]}"
            )
            assert L == _contig_len(meneu_dir, c)

    def test_save_load_roundtrip_exact(self, meneu_dir):
        b = MeneuForeignDNABenchmark(meneu_dir, INFO)
        res = b.evaluate(_MockAdapter())
        out = meneu_dir / "out"
        b.save_results(res, out)
        loaded = b.load_results(out)
        assert loaded.window_len == res.window_len
        assert sorted(loaded.contigs) == sorted(res.contigs)
        for c in CONTIGS:
            np.testing.assert_array_almost_equal(
                loaded.stitched_pred[c], res.stitched_pred[c],
            )
            for k in _METRIC_KEYS:
                assert loaded.per_contig[c][k] == res.per_contig[c][k], (
                    f"{c}.{k} differs on round-trip"
                )
            # Metrics must NOT be dropped to NaN on load.
            assert np.isfinite(loaded.per_contig[c]["shape_pearson"])

    def test_plot_writes_a_file_per_contig(self, meneu_dir):
        b = MeneuForeignDNABenchmark(meneu_dir, INFO)
        res = b.evaluate(_MockAdapter())
        out = meneu_dir / "p"
        b.plot(res, out)
        assert out.is_dir()
        imgs = list(out.glob("*.png")) + list(out.glob("*.svg"))
        assert len(imgs) >= len(CONTIGS)

    def test_summary_dict_is_flat_scalars(self, meneu_dir):
        b = MeneuForeignDNABenchmark(meneu_dir, INFO)
        res = b.evaluate(_MockAdapter())
        s = b.summary_dict(res)
        for c in CONTIGS:
            for k in _METRIC_KEYS:
                key = f"{c}_{k}"
                assert key in s, f"{key} missing from summary"
                assert isinstance(s[key], (int, float))

    def test_headline_names_each_contig(self, meneu_dir):
        b = MeneuForeignDNABenchmark(meneu_dir, INFO)
        res = b.evaluate(_MockAdapter())
        h = b.headline(res)
        assert isinstance(h, str)
        for c in CONTIGS:
            assert c in h


# ── registry ──────────────────────────────────────────────────


class TestMeneuRegistry:
    def test_single_task_registered(self):
        assert "meneu_foreign_dna" in TASKS
        # The window-split twin is gone — one task serves both models now.
        assert "meneu_foreign_dna_shorkie" not in TASKS

    def test_factory_builds_benchmark(self, meneu_dir):
        task = TASKS["meneu_foreign_dna"](data_path=meneu_dir)
        assert isinstance(task, MeneuForeignDNABenchmark)
        assert task.adapter_protocol is TiledCoverageTrackPredictor


# ── bit-exactness regression guard (real shipped data) ────────
#
# Runtime tiling must reproduce the original pre-cut distribution exactly.
# Each digest is sha256 over the center-ordered tile `seq` strings for one
# (contig, window); the counts/digests were captured from the legacy TSVs
# before they were removed. Skipped where the shipped data isn't present
# (e.g. CI without a data fetch).

_REAL_DIR = Path("data/tasks/meneu_foreign_dna")
_REAL_CROP = {4992: 996, 16384: 1024}            # Yorzoi, Shorkie receptive fields
_GOLDEN: dict[tuple[str, int], tuple[int, str]] = {
    ("Mmmyco", 4992): (
        408, "01d96e13ce2b7788d7588c14a226e794cff7ad52f222ce02c4ceeba1d1b5237f"),
    ("Mmmyco", 16384): (
        86, "4bd0478db1ac10950c5df8f451c11241429285a0cd6a556985236806c8e63177"),
    ("Mpneumo", 4992): (
        273, "5e6eddbaeea817e45595e579111e1a8024d174aa7d354ea37f677d248a9ac0a3"),
    ("Mpneumo", 16384): (
        58, "53d4ad3f1f6d1abbc8c2e6bf3be9864c0bf32b6e272129bd25d5b9189fb3e07b"),
}


@pytest.mark.parametrize("contig,window", sorted(_GOLDEN))
def test_runtime_tiling_matches_shipped_distribution(contig, window):
    npz = _REAL_DIR / f"{COV_PREFIX}{contig}.npz"
    if not npz.exists():
        pytest.skip(f"{npz} not present (data not fetched)")
    with np.load(npz) as d:
        seq = d["seq"].tobytes().decode("ascii")
    tiles = tile_contig(seq, window, _REAL_CROP[window])
    n_expected, digest_expected = _GOLDEN[(contig, window)]
    assert len(tiles) == n_expected
    digest = hashlib.sha256("".join(t.seq for t in tiles).encode()).hexdigest()
    assert digest == digest_expected
