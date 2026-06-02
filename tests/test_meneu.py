"""Tests for the Meneu foreign-DNA tiled-coverage benchmark.

The benchmark tiles a whole contig end-to-end, stitches the per-tile
central predictions into one per-base track, and scores it per contig
against a measured RNA-seq coverage sidecar (``meneu_cov_<contig>.npz``,
arrays ``fwd``/``rev``). The tests build a tiny self-contained TSV +
matching tiny ``.npz`` sidecars and run the whole path against a
pure-numpy mock ``TiledCoverageTrackPredictor`` — no GPU, no torch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yeastbench.adapters.protocols import TiledCoverageTrackPredictor
from yeastbench.benchmarks.base import BenchmarkInfo
from yeastbench.benchmarks.meneu import MeneuForeignDNABenchmark
from yeastbench.registry import TASKS

INFO = BenchmarkInfo(name="test_meneu", version="test", description="t",
                     distribution_uri="")

# Mock model geometry — tiny so the fixtures are cheap. stride == out_len.
SEQ_LEN = 20
CROP = 4
OUT_LEN = SEQ_LEN - 2 * CROP            # 12 — the per-tile central region

# Two contigs. Lengths chosen so each gives exactly one 5 kb eval window
# (L // EVAL_WINDOW == 1) and the *second* contig's length is not a
# multiple of OUT_LEN, so its final tile is clipped (exercises stitching
# of a short last tile). Both are fully covered exactly once.
CONTIGS = {"Mpneumo": 6000, "Mmmyco": 5496}

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
def meneu_tsv(tmp_path: Path) -> Path:
    """Write a tiny TSV (both contigs tiled) + matching tiny
    ``meneu_cov_<contig>.npz`` sidecars next to it. Returns the TSV path."""
    rng = np.random.default_rng(0)
    rows = []
    for chrom, L in CONTIGS.items():
        full = _contig_seq(rng, L)
        truth = _truth_from_seq(full)
        fwd, rev = _split_fwd_rev(truth, rng)
        np.savez(tmp_path / f"meneu_cov_{chrom}.npz", fwd=fwd, rev=rev)
        # Tiles ordered by center_start, stepping by the stride (== OUT_LEN).
        for tile_i, center_start in enumerate(range(0, L, OUT_LEN)):
            window_start = center_start - CROP
            # Slice the window from the contig, N-padding the contig ends.
            chars = []
            for j in range(window_start, window_start + SEQ_LEN):
                chars.append(full[j] if 0 <= j < L else "N")
            rows.append({
                "tile_id": f"{chrom}_{tile_i}",
                "chrom": chrom,
                "strain": f"strain_{chrom}",
                "window_start": window_start,
                "center_start": center_start,
                "window_len": SEQ_LEN,
                "crop_bp_each_side": CROP,
                "seq": "".join(chars),
            })
    p = tmp_path / "mini_meneu.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return p


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


def _contig_len(tsv: Path, chrom: str) -> int:
    d = np.load(tsv.parent / f"meneu_cov_{chrom}.npz")
    return len(d["fwd"])


_METRIC_KEYS = (
    "shape_pearson", "shape_js", "mag_fc_mean", "mag_fc_sd",
    "n_windows_kept", "n_windows_total",
)


# ── benchmark ─────────────────────────────────────────────────


class TestMeneuBenchmark:
    def test_init_loads_and_validates_schema(self, meneu_tsv):
        b = MeneuForeignDNABenchmark(meneu_tsv, INFO)
        assert b.window_len == SEQ_LEN
        assert (b.df.seq.str.len() == SEQ_LEN).all()
        assert set(b.df.chrom.unique()) == set(CONTIGS)

    def test_init_rejects_inconsistent_window_len(self, meneu_tsv):
        df = pd.read_csv(meneu_tsv, sep="\t")
        df.loc[0, "window_len"] = SEQ_LEN + 1     # mismatch
        df.to_csv(meneu_tsv, sep="\t", index=False)
        with pytest.raises(AssertionError):
            MeneuForeignDNABenchmark(meneu_tsv, INFO)

    def test_init_rejects_missing_column(self, meneu_tsv, tmp_path):
        df = pd.read_csv(meneu_tsv, sep="\t").drop(columns=["center_start"])
        bad = tmp_path / "bad.tsv"
        df.to_csv(bad, sep="\t", index=False)
        with pytest.raises(AssertionError):
            MeneuForeignDNABenchmark(bad, INFO)

    def test_evaluate_metrics_present_and_finite(self, meneu_tsv):
        b = MeneuForeignDNABenchmark(meneu_tsv, INFO)
        res = b.evaluate(_MockAdapter())
        assert set(res.contigs) == set(CONTIGS)
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
            # Mock shape co-varies with truth → high raw Pearson.
            assert pc["shape_pearson"] > 0.9

    def test_tiling_covers_every_base(self, meneu_tsv):
        b = MeneuForeignDNABenchmark(meneu_tsv, INFO)
        res = b.evaluate(_MockAdapter())
        for c, L in CONTIGS.items():
            pred = res.stitched_pred[c]
            assert pred.shape == (L,)
            # The mock never emits a 0 in the interior, so any unwritten
            # position would stay at the init value — assert the stitched
            # track is fully populated (no NaN, finite everywhere).
            assert np.all(np.isfinite(pred))
            assert L == _contig_len(meneu_tsv, c)

    def test_save_load_roundtrip_exact(self, meneu_tsv, tmp_path):
        b = MeneuForeignDNABenchmark(meneu_tsv, INFO)
        res = b.evaluate(_MockAdapter())
        out = tmp_path / "out"
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

    def test_plot_writes_a_file_per_contig(self, meneu_tsv, tmp_path):
        b = MeneuForeignDNABenchmark(meneu_tsv, INFO)
        res = b.evaluate(_MockAdapter())
        out = tmp_path / "p"
        b.plot(res, out)
        assert out.is_dir()
        imgs = list(out.glob("*.png")) + list(out.glob("*.svg"))
        assert len(imgs) >= len(CONTIGS)

    def test_summary_dict_is_flat_scalars(self, meneu_tsv):
        b = MeneuForeignDNABenchmark(meneu_tsv, INFO)
        res = b.evaluate(_MockAdapter())
        s = b.summary_dict(res)
        for c in CONTIGS:
            for k in _METRIC_KEYS:
                key = f"{c}_{k}"
                assert key in s, f"{key} missing from summary"
                assert isinstance(s[key], (int, float))

    def test_headline_names_each_contig(self, meneu_tsv):
        b = MeneuForeignDNABenchmark(meneu_tsv, INFO)
        res = b.evaluate(_MockAdapter())
        h = b.headline(res)
        assert isinstance(h, str)
        for c in CONTIGS:
            assert c in h


# ── registry ──────────────────────────────────────────────────


class TestMeneuRegistry:
    def test_tasks_registered(self):
        assert "meneu_foreign_dna" in TASKS
        assert "meneu_foreign_dna_shorkie" in TASKS

    def test_factory_builds_benchmark(self, meneu_tsv):
        task = TASKS["meneu_foreign_dna"](data_path=meneu_tsv)
        assert isinstance(task, MeneuForeignDNABenchmark)
        assert task.adapter_protocol is TiledCoverageTrackPredictor

    def test_shorkie_factory_builds_same_class(self, meneu_tsv):
        task = TASKS["meneu_foreign_dna_shorkie"](data_path=meneu_tsv)
        assert isinstance(task, MeneuForeignDNABenchmark)
        assert task.info.name == "meneu_foreign_dna_shorkie"
        assert task.adapter_protocol is TiledCoverageTrackPredictor
