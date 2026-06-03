"""Characterization tests for the ExoShorkie Dense(1) head port.

ExoShorkie = the Shorkie trunk + a single *linear* per-bin coverage head
(``Dense(1)``, no softplus), output squeezed to ``(B, T_out)``. The same
``ShorkieModule`` serves both models; the head is driven by the head
config (units/activation) and the checkpoint's head layer is auto-detected
(``per_bin_f<F>`` for ExoShorkie vs ``dense_<N>`` for Shorkie).

Two levels of trust:

* **Level 1 (always run when the probe .h5 is present):** a real-weights
  characterization. Loads one teacher checkpoint, proves the head is
  sourced from the ``per_bin_f0`` group with the correct transpose (exact
  weight-identity, machine-independent), proves the 170-channel trunk
  still loads under the ExoShorkie config, checks the forward output is
  ``(B, 896)`` with no softplus, and locks the full forward output against
  a committed golden array.

* **Level 2 (``test_matches_tf_reference``, optional, skipped by default):**
  an end-to-end cross-check vs the authors' TensorFlow ``predict.py``. This
  is NOT required for trust. The Shorkie trunk this port reuses is already
  numerically verified against TF upstream (tdsone/shorkie-pytorch,
  ``scripts/compute_equivalence.py``: per-sequence Pearson 1.0, max abs error
  ~1.95e-3 over 996 fold-0 sequences), and ExoShorkie's only delta from that
  trunk is the linear ``Dense(1)`` head — which the Level-1 weight-identity
  test pins exactly, with no nonlinearity to mismatch. The test stays as
  belt-and-suspenders for anyone who wants an ExoShorkie-specific TF run; it
  needs a TF/Baskerville env (not installed here).

Trust chain (no TF needed here): verified trunk code (upstream) + exact head
weight-identity + linear/squeeze semantics (this file) ⇒ correct ExoShorkie.

Which tests run where:
- ``test_*_config_*`` / ``test_head_*`` / ``test_exo_forward_structure*`` need
  no checkpoint and run everywhere (incl. CI); they guard the Shorkie path
  (softplus, no squeeze), the ExoShorkie wiring, and the forward structure.
- The exact-weight Level-1 tests need the 141 MB probe ``.h5`` and SKIP where
  it is absent (e.g. CI) — they are the real correctness anchor and must be
  run in an environment that has the checkpoint.
"""
from __future__ import annotations

import copy
import json
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from yeastbench.models.shorkie.nn import ShorkieModule  # noqa: E402

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data" / "exoshorkie"
REPO = HERE.parent
PARAMS = REPO / "data" / "models" / "shorkie" / "params.json"
PROBE = pathlib.Path(
    "/home/tds122/exoshorkie-weights/probe/Models/M_pneumoniae/cv0/f0/model_finetune.h5"
)
WINDOW = DATA / "test_window_16384.npy"
GOLDEN = DATA / "golden_window_output.npy"
TF_REF = DATA / "tf_reference_window_output.npy"

needs_probe = pytest.mark.skipif(
    not PROBE.exists(), reason=f"ExoShorkie probe checkpoint not present at {PROBE}"
)
needs_params = pytest.mark.skipif(
    not PARAMS.exists(), reason=f"Shorkie params.json not present at {PARAMS}"
)


def _shorkie_model_config() -> dict:
    with open(PARAMS) as fh:
        return json.load(fh)["model"]


def _fixed_window_onehot() -> "torch.Tensor":
    """The committed 16,384 bp window as channels-first one-hot ``(1,4,T)``.

    Shared with the TF-reference generator so both paths score the same
    sequence."""
    seq = np.load(WINDOW)  # (16384,) int8 in {0,1,2,3} = A,C,G,T
    onehot = np.zeros((4, seq.shape[0]), dtype=np.float32)
    onehot[seq, np.arange(seq.shape[0])] = 1.0
    return torch.from_numpy(onehot).unsqueeze(0)


# Module-scoped: loading + one forward of the full trunk is ~minute on CPU.
@pytest.fixture(scope="module")
def exo_model():
    model = ShorkieModule.from_tf_checkpoint(
        ShorkieModule.exoshorkie_config(_shorkie_model_config()), str(PROBE)
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def exo_output(exo_model):
    with torch.no_grad():
        return exo_model(_fixed_window_onehot()).squeeze(0).numpy()  # (896,)


# ──────────────────────────────────────────────────────────────
# Config wiring — no weights; also guards the Shorkie path is unchanged
# ──────────────────────────────────────────────────────────────


@needs_params
def test_exoshorkie_config_overrides_head_only():
    base = _shorkie_model_config()
    base_snapshot = copy.deepcopy(base)
    exo = ShorkieModule.exoshorkie_config(base)

    assert exo["head"] == {"name": "per_bin", "units": 1, "activation": "linear"}
    # ExoShorkie input convention: host one-hot at global col 119 + explicit N channel.
    assert exo["species_channel"] == 119
    assert exo["encode_n_channel"] is True
    # Trunk carried over verbatim.
    assert exo["trunk"] == base["trunk"]
    # Deep copy: the input config is not mutated.
    assert base == base_snapshot
    assert base["head"]["units"] == 5215


@needs_params
def test_head_flags_exoshorkie_vs_shorkie():
    shorkie = ShorkieModule(_shorkie_model_config())
    assert shorkie._head_activation == "softplus"
    assert shorkie._squeeze_head is False
    assert shorkie.head.out_features == 5215
    assert shorkie._species_channel == 114  # Shorkie host channel
    assert shorkie._encode_n_channel is False

    exo = ShorkieModule(ShorkieModule.exoshorkie_config(_shorkie_model_config()))
    assert exo._head_activation == "linear"
    assert exo._squeeze_head is True
    assert exo.head.out_features == 1
    assert exo._species_channel == 119  # ExoShorkie host channel
    assert exo._encode_n_channel is True


@needs_params
def test_exo_forward_structure_without_weights():
    """Runs in CI (no checkpoint): random-init ExoShorkie forward guards the
    output *structure* and the head wiring. (1) Output is squeezed to
    (B, OUTPUT_BINS), float32, finite. (2) The linear branch applies NO
    activation: flipping the head flag to softplus must reproduce exactly
    ``softplus(linear_output)`` — which proves the linear path skips softplus
    and the softplus path applies it. Values themselves are meaningless here;
    correctness of values is the probe-gated, upstream-TF-verified job."""
    torch.manual_seed(0)
    model = ShorkieModule(ShorkieModule.exoshorkie_config(_shorkie_model_config()))
    model.eval()
    seq = torch.randint(0, 4, (16384,))
    x = torch.zeros(2, 4, 16384)
    x[:, seq, torch.arange(16384)] = 1.0
    with torch.no_grad():
        out_linear = model(x)
        assert model._head_activation == "linear"
        model._head_activation = "softplus"  # flip only the head activation
        out_softplus = model(x)

    assert out_linear.shape == (2, 896)  # squeezed: no trailing singleton dim
    assert out_linear.dtype == torch.float32
    assert torch.isfinite(out_linear).all()
    torch.testing.assert_close(
        out_softplus, torch.nn.functional.softplus(out_linear)
    )


# ──────────────────────────────────────────────────────────────
# Level 1 — real-weights characterization
# ──────────────────────────────────────────────────────────────


@needs_probe
@needs_params
def test_head_loaded_from_per_bin_group(exo_model):
    """Head weights come from ``per_bin_f0`` (not the last trunk dense),
    with the TF→PyTorch transpose applied. Exact match, no tolerance."""
    assert isinstance(exo_model.head, torch.nn.Linear)
    assert exo_model.head.weight.shape == (1, 384)
    assert exo_model.head.bias.shape == (1,)

    with h5py.File(PROBE, "r") as f:
        grp = f["model_weights"]["per_bin_f0"]["per_bin_f0"]
        raw_kernel = np.array(grp["kernel:0"])  # (384, 1)
        raw_bias = np.array(grp["bias:0"])      # (1,)

    np.testing.assert_array_equal(exo_model.head.weight.detach().numpy(), raw_kernel.T)
    np.testing.assert_array_equal(exo_model.head.bias.detach().numpy(), raw_bias)


@needs_probe
@needs_params
def test_trunk_conv_dna_loads_under_exo_config(exo_model):
    """The 170-channel trunk still loads through the existing path when the
    head config is ExoShorkie's — i.e. the head swap didn't disturb the
    trunk weight walk."""
    with h5py.File(PROBE, "r") as f:
        raw = np.array(f["model_weights"]["conv1d"]["conv1d"]["kernel:0"])  # (11,170,96)
    expected = np.transpose(raw, (2, 1, 0))  # (96,170,11)
    np.testing.assert_array_equal(exo_model.conv_dna.conv.weight.detach().numpy(), expected)


@needs_probe
@needs_params
def test_forward_shape_and_no_softplus(exo_output):
    assert exo_output.shape == (896,)
    assert exo_output.dtype == np.float32
    assert np.isfinite(exo_output).all()
    # Linear head over z-scored coverage produces negatives; softplus could not.
    assert (exo_output < 0).any(), "negative outputs absent — softplus may be applied"
    # Not a constant track.
    assert exo_output.std() > 1e-3


@needs_probe
@needs_params
def test_forward_matches_golden(exo_output):
    """Regression lock on the full 896-bin output of the ported path.

    NOTE: the golden was recorded from this same port, so this guards against
    *future drift*, not against a present systematic error. Correctness comes
    from the exact weight-identity tests above + the upstream TF-verified
    trunk; regenerate the golden via ``_record_golden.py`` only after those
    pass."""
    golden = np.load(GOLDEN)
    assert golden.shape == exo_output.shape
    np.testing.assert_allclose(exo_output, golden, rtol=1e-4, atol=1e-4)


# ──────────────────────────────────────────────────────────────
# Level 2 — trust anchor vs the authors' TensorFlow predict.py
# ──────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not TF_REF.exists(),
    reason=(
        "TF reference not generated. Produce tf_reference_window_output.npy by "
        "running the authors' ExoShorkie predict.py (OrensteinLab/ExoShorkie @ "
        "88e89f48, TF 2.15 + pinned baskerville) on tests/data/exoshorkie/"
        "test_window_16384.npy, forcing float32 (override the mixed_bfloat16 "
        "policy) so the diff is tight. Needs a TF/Baskerville env — not installed here."
    ),
)
@needs_probe
@needs_params
def test_matches_tf_reference(exo_output):
    ref = np.load(TF_REF)
    assert ref.shape == exo_output.shape
    # TF runs mixed_bfloat16 by default; with the reference forced to float32
    # the port should match tightly. Correlation guards against any layout/
    # transpose error; max-abs-diff guards against scale/offset error.
    r = float(np.corrcoef(exo_output, ref)[0, 1])
    max_abs = float(np.max(np.abs(exo_output - ref)))
    assert r > 0.9999, f"Pearson r={r:.6f} too low vs TF reference"
    assert max_abs < 5e-2, f"max abs diff {max_abs:.4g} too large vs TF reference"
