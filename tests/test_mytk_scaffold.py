"""CPU tests for the MYTK adapter scaffold (no model, no GPU, no real data).

Covers the two MYTK-specific pieces: the 4-record payload loader (with
readout-by-substring) and the point-insertion context builder. Uses a
synthetic payload FASTA + a synthetic mini genome, so it runs anywhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pysam
import pytest

from yeastbench.adapters._mytk_scaffold import (
    build_mytk_context,
    load_payload_geometries,
)
from yeastbench.adapters.protocols import PromoterIntegrationConstruct

# Synthetic parts. Flanks are homopolymer / AC-repeat so the mixed-base
# mScarlet 60-mer cannot occur in them or span a junction by accident.
CONS = "ACACACACAC"          # 10
TTDH1 = "T" * 15
CONE = "CACACACACA"          # 10
MSCARLET = "ATG" + "GCATTACGAGTCATGCCATTAGGCATTGACGTACGGATCATTAGGCATTGACGTACG"  # 60
PROMOTERS = {"pTDH3": "G" * 30, "pRPL18B": "G" * 50, "pREV1": "G" * 70}


def _payload(promoter_seq: str) -> str:
    return CONS + promoter_seq + MSCARLET + TTDH1 + CONE


def _write_fasta(path: Path, records: dict[str, str]) -> Path:
    path.write_text("".join(f">{n}\n{s}\n" for n, s in records.items()))
    return path


@pytest.fixture
def payloads_fasta(tmp_path: Path) -> Path:
    records = {
        f"payload_{pid}_mScarlet_tTDH1": _payload(seq)
        for pid, seq in PROMOTERS.items()
    }
    records["mScarlet"] = MSCARLET
    return _write_fasta(tmp_path / "payloads.fasta", records)


# ── load_payload_geometries ───────────────────────────────────


class TestLoadGeometries:
    def test_resolves_all_promoters(self, payloads_fasta):
        geoms = load_payload_geometries(payloads_fasta)
        assert set(geoms) == set(PROMOTERS)

    def test_readout_span_located_by_substring(self, payloads_fasta):
        geoms = load_payload_geometries(payloads_fasta)
        for pid, prom_seq in PROMOTERS.items():
            g = geoms[pid]
            assert g.readout_len == len(MSCARLET)
            # offset = ConS + promoter (differs per promoter) → substring works
            assert g.readout_start == len(CONS) + len(prom_seq)
            assert g.payload[g.readout_start : g.readout_start + g.readout_len] == MSCARLET

    def test_missing_readout_record_raises(self, tmp_path):
        fa = _write_fasta(tmp_path / "no_ref.fasta", {
            "payload_pTDH3_x": _payload(PROMOTERS["pTDH3"]),
        })
        with pytest.raises(ValueError, match="readout reference"):
            load_payload_geometries(fa, promoter_ids=("pTDH3",))

    def test_promoter_matching_zero_records_raises(self, tmp_path):
        fa = _write_fasta(tmp_path / "miss.fasta", {
            "payload_pTDH3_x": _payload(PROMOTERS["pTDH3"]),
            "mScarlet": MSCARLET,
        })
        with pytest.raises(ValueError, match="expected exactly 1"):
            load_payload_geometries(fa, promoter_ids=("pREV1",))

    def test_readout_not_unique_raises(self, tmp_path):
        # mScarlet appears twice in the payload → ambiguous readout.
        dup = CONS + PROMOTERS["pTDH3"] + MSCARLET + TTDH1 + MSCARLET + CONE
        fa = _write_fasta(tmp_path / "dup.fasta", {
            "payload_pTDH3_x": dup,
            "mScarlet": MSCARLET,
        })
        with pytest.raises(ValueError, match="occurs 2"):
            load_payload_geometries(fa, promoter_ids=("pTDH3",))


# ── build_mytk_context ────────────────────────────────────────


@pytest.fixture
def mini_genome(tmp_path: Path) -> pysam.FastaFile:
    """One 20 kb chromosome 'V' of deterministic ACGT filler."""
    rng = np.random.default_rng(0)
    chrom = "".join(rng.choice(list("ACGT"), 20_000))
    fa = tmp_path / "mini.fa"
    fa.write_text(">V\n" + "\n".join(chrom[i : i + 60] for i in range(0, len(chrom), 60)) + "\n")
    pysam.faidx(str(fa))
    return pysam.FastaFile(str(fa))


PARAMS = dict(seq_len=400, crop_bp_each_side=40, bin_width=10, output_bins=32)


class TestBuildContext:
    def _construct(self, promoter="pTDH3", coord=10_000):
        return PromoterIntegrationConstruct(
            locus_id="Int.1", chrom="V", integration_coord=coord, promoter=promoter,
        )

    def test_window_and_readout(self, payloads_fasta, mini_genome):
        geoms = load_payload_geometries(payloads_fasta)
        ctx = build_mytk_context(self._construct(), geoms, mini_genome, **PARAMS)
        assert ctx is not None
        assert len(ctx.window_seq) == PARAMS["seq_len"]
        # The full payload sits forward (+ strand) in the window…
        payload = geoms["pTDH3"].payload
        assert ctx.window_seq[ctx.payload_start_in_window : ctx.payload_start_in_window + len(payload)] == payload
        # …and the mScarlet readout maps to a non-empty per-base region.
        assert ctx.readout_base_positions.size > 0

    def test_cassette_centered(self, payloads_fasta, mini_genome):
        geoms = load_payload_geometries(payloads_fasta)
        ctx = build_mytk_context(self._construct(), geoms, mini_genome, **PARAMS)
        payload_len = len(geoms["pTDH3"].payload)
        mid = ctx.payload_start_in_window + payload_len // 2
        assert abs(mid - PARAMS["seq_len"] // 2) <= 1

    def test_readout_region_is_mscarlet(self, payloads_fasta, mini_genome):
        """The per-base readout positions, mapped back through the crop,
        land on the mScarlet bases — not the promoter or marker."""
        geoms = load_payload_geometries(payloads_fasta)
        ctx = build_mytk_context(self._construct(), geoms, mini_genome, **PARAMS)
        crop = PARAMS["crop_bp_each_side"]
        # readout_base_positions are in cropped-output coords; shift back to
        # window coords by +crop and confirm they index mScarlet bases.
        win_positions = ctx.readout_base_positions + crop
        readout = "".join(ctx.window_seq[p] for p in win_positions)
        assert readout in MSCARLET  # contiguous slice of the mScarlet CDS

    def test_per_promoter_offsets_differ(self, payloads_fasta, mini_genome):
        geoms = load_payload_geometries(payloads_fasta)
        starts = {
            pid: build_mytk_context(
                self._construct(promoter=pid), geoms, mini_genome, **PARAMS,
            ).readout_start_in_window
            for pid in PROMOTERS
        }
        # Different promoter lengths → different readout offsets in payload,
        # but all still produce a valid centred window.
        assert len(set(starts.values())) > 1

    def test_chrom_too_short_returns_none(self, payloads_fasta, tmp_path):
        """A chromosome shorter than the window (even with the payload
        added) can't form a full input → None."""
        geoms = load_payload_geometries(payloads_fasta)
        short = "".join(np.random.default_rng(1).choice(list("ACGT"), 200))
        fa = tmp_path / "short.fa"
        fa.write_text(">V\n" + short + "\n")
        pysam.faidx(str(fa))
        g = pysam.FastaFile(str(fa))
        ctx = build_mytk_context(self._construct(coord=100), geoms, g, **PARAMS)
        assert ctx is None  # 200 bp + 125 bp payload < 400 bp window
