"""One-off: load the probe ExoShorkie .h5 through the ported PyTorch path,
verify weight-identity against the raw h5, run the fixed window, and print
golden values to bake into tests/test_exoshorkie_head.py."""
import json
import pathlib

import h5py
import numpy as np
import torch

from yeastbench.models.shorkie.nn import ShorkieModule

HERE = pathlib.Path(__file__).parent
PARAMS = "data/models/shorkie/params.json"
PROBE = "/home/tds122/exoshorkie-weights/probe/Models/M_pneumoniae/cv0/f0/model_finetune.h5"

torch.manual_seed(0)
torch.set_num_threads(8)

with open(PARAMS) as fh:
    shorkie_cfg = json.load(fh)["model"]
exo_cfg = ShorkieModule.exoshorkie_config(shorkie_cfg)
print("exo head cfg:", exo_cfg["head"])

model = ShorkieModule.from_tf_checkpoint(exo_cfg, PROBE)
model.eval()

# ── weight-identity vs raw h5 (exact, machine-independent) ──
with h5py.File(PROBE, "r") as f:
    mw = f["model_weights"]
    raw_k = np.array(mw["per_bin_f0"]["per_bin_f0"]["kernel:0"])  # (384,1)
    raw_b = np.array(mw["per_bin_f0"]["per_bin_f0"]["bias:0"])    # (1,)
    raw_conv = np.array(mw["conv1d"]["conv1d"]["kernel:0"])       # (11,170,96)

head_w = model.head.weight.detach().numpy()  # (1,384)
head_b = model.head.bias.detach().numpy()    # (1,)
print("head shape:", model.head.weight.shape, model.head.bias.shape)
print("head kernel exact match:", np.array_equal(head_w, raw_k.T))
print("head bias exact match:", np.array_equal(head_b, raw_b))

conv_w = model.conv_dna.conv.weight.detach().numpy()  # (96,170,11)
print("conv_dna kernel exact match:",
      np.array_equal(conv_w, np.transpose(raw_conv, (2, 1, 0))))

# ── forward on the fixed window ──
seq = np.load(HERE / "test_window_16384.npy")  # (16384,) int8 0..3
onehot = np.zeros((4, seq.shape[0]), dtype=np.float32)
onehot[seq, np.arange(seq.shape[0])] = 1.0
x = torch.from_numpy(onehot).unsqueeze(0)  # (1,4,16384)

with torch.no_grad():
    out = model(x)

print("\nout.shape:", tuple(out.shape), "dtype:", out.dtype)
out = out.squeeze(0).numpy()  # (896,)
print("finite:", bool(np.isfinite(out).all()))
print("has_negative (=> no softplus):", bool((out < 0).any()))
print("n_bins:", out.shape[0])
print("GOLDEN sum: %.6f" % float(out.sum()))
print("GOLDEN mean: %.8f" % float(out.mean()))
print("GOLDEN std: %.8f" % float(out.std()))
print("GOLDEN min: %.8f" % float(out.min()))
print("GOLDEN max: %.8f" % float(out.max()))
print("GOLDEN first8:", [round(float(v), 6) for v in out[:8]])
print("GOLDEN argmax:", int(out.argmax()), "argmin:", int(out.argmin()))
np.save(HERE / "_golden_full.npy", out)
print("saved _golden_full.npy for reference (not committed/asserted)")
