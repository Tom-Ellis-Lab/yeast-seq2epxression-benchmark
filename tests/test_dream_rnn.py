"""Tests for the DREAM-RNN supervised baseline (Rafi/deBoer MPRA).

See ``benchmarks/rafi_mpra_promoter.md``. The architecture/predictor tests run
on CPU with random weights; the strict-weight-load test is skipped unless the
published checkpoint has been staged at ``data/models/dream_rnn/``.
"""

from __future__ import annotations

# Imports of torch-backed yeastbench modules must follow the importorskip guard
# so this whole module skips cleanly when torch isn't installed.
# ruff: noqa: E402

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from yeastbench.adapters.dream_rnn_rafi import DreamRnnRafiPredictor
from yeastbench.adapters.protocols import SequenceExpressionScorer
from yeastbench.models.dream_rnn import DreamRnn, DreamRnnExpressionPredictor
from yeastbench.models.dream_rnn.predictor import LEFT_ADAPTER, SEQ_SIZE

WEIGHTS = Path("data/models/dream_rnn/model_best.pth")
TAIL = "GGTTACGGCTGTT"  # constant 13 bp tail of every Rafi sequence


def _synthetic_plasmid(tmp_path: Path) -> Path:
    """A minimal plasmid string: ≥150 bp upstream ending in the left adapter,
    then the 80-N insert slot, then some 3' sequence."""
    upstream = ("ACGT" * 40)[: SEQ_SIZE - len(LEFT_ADAPTER)] + LEFT_ADAPTER
    assert len(upstream) == SEQ_SIZE
    plasmid = upstream + "N" * 80 + "ACGT" * 10
    p = tmp_path / "plasmid.json"
    p.write_text(json.dumps(plasmid))
    return p


def _make_seq(insert: str) -> str:
    seq = LEFT_ADAPTER + insert + TAIL
    assert len(seq) == 17 + len(insert) + 13
    return seq


def test_net_assembles_and_forward_shape():
    model = DreamRnn().eval()
    # Single net: BHI first + BHI Bi-LSTM core + Autosome final.
    assert isinstance(model.core.lstm, torch.nn.LSTM)
    assert model.core.lstm.bidirectional
    assert model.final.mapper[0].in_channels == 320
    assert model.final.mapper[0].out_channels == 18
    with torch.no_grad():
        y = model(torch.randn(3, 6, SEQ_SIZE))
    assert y.shape == (3,)


def test_encode_shape_and_channels():
    seq = "A" * SEQ_SIZE
    enc = DreamRnnExpressionPredictor._encode(seq, rev_value=1)
    assert enc.shape == (6, SEQ_SIZE)
    # rows 0-3 one-hot bases, row 4 = is_reverse (all 1 here), row 5 = singleton (0)
    assert torch.all(enc[4] == 1.0)
    assert torch.all(enc[5] == 0.0)
    assert torch.all(enc[:4].sum(dim=0) == 1.0)  # exactly one base per position


def test_reflank_crops_to_150_and_keeps_insert(tmp_path):
    plasmid = _synthetic_plasmid(tmp_path)
    pred = DreamRnnExpressionPredictor(DreamRnn(), plasmid, device="cpu", batch_size=8)
    insert = "ACGT" * 20  # 80 bp
    reflanked = pred._reflank(_make_seq(insert))
    assert len(reflanked) == SEQ_SIZE
    # The insert + tail end up at the 3' end of the window.
    assert reflanked.endswith(insert + TAIL)


def test_reflank_rejects_bad_prefix(tmp_path):
    plasmid = _synthetic_plasmid(tmp_path)
    pred = DreamRnnExpressionPredictor(DreamRnn(), plasmid, device="cpu", batch_size=8)
    with pytest.raises(ValueError, match="adapter"):
        pred._reflank("AAAA" + "ACGT" * 24 + TAIL)  # wrong 17 bp prefix


def test_predict_returns_finite_array(tmp_path):
    plasmid = _synthetic_plasmid(tmp_path)
    pred = DreamRnnExpressionPredictor(
        DreamRnn().eval(), plasmid, device="cpu", batch_size=2
    )
    seqs = [_make_seq("ACGT" * 20), _make_seq("TGCA" * 20), _make_seq("GGGG" * 20)]
    out = pred.predict(seqs)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
    # Within the 18-bin expected-value range.
    assert out.min() >= 0.0 and out.max() <= 17.0


def test_adapter_satisfies_protocol(tmp_path):
    plasmid = _synthetic_plasmid(tmp_path)
    pred = DreamRnnExpressionPredictor(DreamRnn().eval(), plasmid, device="cpu")
    adapter = DreamRnnRafiPredictor(pred)
    assert isinstance(adapter, SequenceExpressionScorer)
    out = adapter.predict_expression_scores([_make_seq("ACGT" * 20)])
    assert out.shape == (1,)


@pytest.mark.skipif(not WEIGHTS.exists(), reason="DREAM-RNN checkpoint not staged")
def test_published_checkpoint_loads_strict():
    """The published 0_1_1_0 state_dict must load into the port with
    strict=True — this is the architecture-equivalence guarantee."""
    model = DreamRnn()
    state_dict = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(state_dict)  # raises on any key/shape mismatch
    assert any(k.startswith("core.lstm") for k in state_dict)
