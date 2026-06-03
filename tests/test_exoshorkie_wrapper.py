"""Unit tests for the ExoShorkie ensemble wrapper's count-space math.

These run anywhere (no checkpoint, no GPU): the students are random-init
``ShorkieModule``s under the ExoShorkie config, so values are meaningless —
what's pinned is the *composition*:

* per-student log-z inverse ``clamp(expm1(z*sigma + mu), 0)`` is applied
  BEFORE any averaging (denorm-then-mean), with each student's own (mu, sigma);
* RC averaging happens in count space (denorm each pass, flip the bin axis of
  the RC pass, then 0.5·(fwd+rc));
* the 6→N ensemble mean and the ÷16 repeat-unbin are exactly the documented ops.

The real correctness anchor is ``scripts/exoshorkie/validate_denorm.py`` (closed
loop vs figshare truth); this file guards against future drift in the wiring.
"""
from __future__ import annotations

import json
import pathlib

import pytest

torch = pytest.importorskip("torch")

from yeastbench.models.exoshorkie import ExoShorkie  # noqa: E402
from yeastbench.models.shorkie.nn import ShorkieModule  # noqa: E402

REPO = pathlib.Path(__file__).parent.parent
PARAMS = REPO / "data" / "models" / "shorkie" / "params.json"

needs_params = pytest.mark.skipif(
    not PARAMS.exists(), reason=f"Shorkie params.json not present at {PARAMS}"
)

STATS = [(3.5, 1.7), (1.3, 1.5)]  # two arbitrary (mu, sigma) pairs


def _exo_config() -> dict:
    with open(PARAMS) as fh:
        return ShorkieModule.exoshorkie_config(json.load(fh)["model"])


@pytest.fixture(scope="module")
def students() -> list[ShorkieModule]:
    torch.manual_seed(0)
    cfg = _exo_config()
    ms = [ShorkieModule(cfg), ShorkieModule(cfg)]
    for m in ms:
        m.eval()
    return ms


def _random_onehot(batch: int = 2) -> "torch.Tensor":
    torch.manual_seed(1)
    seq = torch.randint(0, 4, (batch, 16384))
    x = torch.zeros(batch, 4, 16384)
    x.scatter_(1, seq.unsqueeze(1), 1.0)
    return x


@needs_params
def test_denorm_then_mean_no_rc(students):
    """forward_count_bins (use_rc=False) == mean_g clamp(expm1(z_g*sigma_g+mu_g))."""
    model = ExoShorkie(students, STATS, device="cpu", use_rc=False)
    x = _random_onehot()
    with torch.no_grad():
        got = model.forward_count_bins(x)
        per_student = []
        for i, m in enumerate(students):
            z = m(x).float()
            per_student.append(torch.clamp(torch.expm1(z * STATS[i][1] + STATS[i][0]), min=0.0))
        expected = torch.stack(per_student).mean(dim=0)
    assert got.shape == (2, 896)
    torch.testing.assert_close(got, expected)
    assert (got >= 0).all()  # clamp holds


@needs_params
def test_rc_average_in_count_space(students):
    """use_rc=True averages the denormed fwd and (bin-flipped) RC passes per
    student, in count space — not in z-space."""
    model = ExoShorkie(students, STATS, device="cpu", use_rc=True)
    x = _random_onehot()
    x_rc = x.flip(dims=[1, 2])
    with torch.no_grad():
        got = model.forward_count_bins(x)
        per_student = []
        for i, m in enumerate(students):
            mu, sigma = STATS[i][0], STATS[i][1]
            c_fwd = torch.clamp(torch.expm1(m(x).float() * sigma + mu), min=0.0)
            c_rc = torch.clamp(torch.expm1(m(x_rc).float() * sigma + mu), min=0.0).flip(dims=[-1])
            per_student.append(0.5 * (c_fwd + c_rc))
        expected = torch.stack(per_student).mean(dim=0)
    torch.testing.assert_close(got, expected)


@needs_params
def test_perbase_is_bin_total_over_16(students):
    """forward_perbase == count_bins repeated 16× and divided by 16 (Shorkie/
    Yorzoi per-base convention, comparable scale)."""
    model = ExoShorkie(students, STATS, device="cpu", use_rc=True)
    x = _random_onehot()
    with torch.no_grad():
        bins = model.forward_count_bins(x)
        pb = model.forward_perbase(x)
    assert pb.shape == (2, 14336)
    torch.testing.assert_close(pb, bins.repeat_interleave(16, dim=-1) / 16.0)
    # summing a bin's 16 bases recovers the bin total
    torch.testing.assert_close(pb.reshape(2, 896, 16).sum(dim=-1), bins)


@needs_params
def test_track_mean_perbase_is_alias(students):
    """The Shorkie-scaffold alias ignores track_subset and equals forward_perbase."""
    model = ExoShorkie(students, STATS, device="cpu", use_rc=True)
    x = _random_onehot()
    with torch.no_grad():
        a = model.forward_track_mean_perbase(x, track_subset=None)
        b = model.forward_track_mean_perbase(x, torch.tensor([1, 2, 3]))
        c = model.forward_perbase(x)
    torch.testing.assert_close(a, c)
    torch.testing.assert_close(b, c)


@needs_params
def test_stats_length_mismatch_raises(students):
    with pytest.raises(ValueError, match="length mismatch"):
        ExoShorkie(students, STATS[:1], device="cpu")
