"""ExoShorkie distillation — Step 2: train one per-genome student (pure PyTorch).

Mirrors the authors' `scripts/train_student.py`: the student is our ExoShorkie
module (Shorkie trunk + linear Dense(1) head), trunk initialized from Shorkie
fold-0, head random; trained to match the 40-teacher mean coverage with MSE,
Adam lr 2e-5, batch 64, 20 epochs.

Differences from the authors' script (deliberate):
  * We hold out a validation slice (default 5%) so we can GATE on Pearson r —
    the authors trained on all synthetic windows and report the r>0.98 figure
    in the paper text without an explicit in-script gate.
  * Train forward uses bf16 autocast for memory (matches their mixed_bfloat16);
    the gate is computed in fp32 (model.eval()) for a clean metric. Inference
    downstream stays fp32, matching the port.

Pearson metric = per-window correlation across the 896 bins (student vs teacher
mean), averaged over the held-out windows — the distillation-fidelity number the
paper's r>0.98 refers to.

Usage:
    uv run --extra shorkie python scripts/exoshorkie/train_student.py \
        --targets /home/tds122/exoshorkie-weights/targets/M_pneumoniae.npz \
        --trunk-ckpt data/models/shorkie/checkpoints/f0.h5 \
        --out /home/tds122/exoshorkie-weights/students/M_pneumoniae.pt \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.models.shorkie.nn import ShorkieModule

SHORKIE_PARAMS = "data/models/shorkie/params.json"
OUTPUT_BINS = 896
GATE_PEARSON = 0.98


def load_configs():
    with open(SHORKIE_PARAMS) as fh:
        shorkie_cfg = json.load(fh)["model"]
    return shorkie_cfg, ShorkieModule.exoshorkie_config(shorkie_cfg)


def build_student(shorkie_cfg: dict, exo_cfg: dict, trunk_ckpt: str) -> ShorkieModule:
    """ExoShorkie student: Shorkie fold-0 trunk + fresh random Dense(1) head."""
    student = ShorkieModule(exo_cfg)
    teacher = ShorkieModule.from_tf_checkpoint(shorkie_cfg, trunk_ckpt)  # 5215 head
    trunk_sd = {k: v for k, v in teacher.state_dict().items() if not k.startswith("head.")}
    missing, unexpected = student.load_state_dict(trunk_sd, strict=False)
    # The only params NOT supplied by the trunk should be the head.
    assert all(m.startswith("head.") for m in missing), f"unexpected missing: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    del teacher
    return student


def encode_all(seqs: list[str]) -> np.ndarray:
    """(N, 4, 16384) uint8 one-hot, channels-first."""
    return np.stack([one_hot_encode_channels_first(s) for s in seqs]).astype(np.uint8)


def per_window_pearson(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Mean over rows of the across-bin Pearson r. Shapes (N, 896)."""
    pred = pred.float()
    true = true.float()
    pm = pred - pred.mean(dim=-1, keepdim=True)
    tm = true - true.mean(dim=-1, keepdim=True)
    num = (pm * tm).sum(dim=-1)
    den = torch.sqrt((pm * pm).sum(dim=-1) * (tm * tm).sum(dim=-1)) + 1e-8
    return float((num / den).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--trunk-ckpt", default="data/models/shorkie/checkpoints/f0.h5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    d = np.load(args.targets, allow_pickle=True)
    seqs = d["train_sequences"].tolist()
    preds = d["train_mean_preds"].astype(np.float32)  # (N, 896)
    assert preds.shape[1] == OUTPUT_BINS, preds.shape
    print(f"targets: {len(seqs)} windows, preds {preds.shape}")

    print("encoding …")
    X = encode_all(seqs)  # (N,4,16384) uint8
    Y = preds
    N = len(seqs)

    idx = np.random.permutation(N)
    n_val = max(1, int(round(args.val_frac * N)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    print(f"train {len(train_idx)}  val {len(val_idx)}")

    shorkie_cfg, exo_cfg = load_configs()
    student = build_student(shorkie_cfg, exo_cfg, args.trunk_ckpt).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    use_amp = (device.type == "cuda") and not args.no_amp

    Xt = torch.from_numpy(X)  # uint8 on CPU; move per batch
    Yt = torch.from_numpy(Y)

    def run_batches(indices, train: bool):
        student.train(train)
        total_loss, nb = 0.0, 0
        for bs in range(0, len(indices), args.batch_size):
            bi = indices[bs : bs + args.batch_size]
            xb = Xt[bi].float().to(device)
            yb = Yt[bi].to(device)
            with torch.set_grad_enabled(train), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=use_amp and train
            ):
                out = student(xb)
                loss = F.mse_loss(out, yb)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            total_loss += loss.item()
            nb += 1
        return total_loss / max(1, nb)

    t0 = time.time()
    for ep in range(args.epochs):
        np.random.shuffle(train_idx)
        tr_loss = run_batches(train_idx, train=True)
        # quick val MSE + Pearson each epoch (fp32 eval)
        with torch.no_grad():
            student.eval()
            vp = []
            for bs in range(0, len(val_idx), args.batch_size):
                bi = val_idx[bs : bs + args.batch_size]
                vp.append(student(Xt[bi].float().to(device)).cpu())
            vpred = torch.cat(vp)
            vtrue = Yt[val_idx]
            val_mse = float(F.mse_loss(vpred, vtrue))
            val_r = per_window_pearson(vpred, vtrue)
        print(f"epoch {ep+1:2d}/{args.epochs}  train_mse={tr_loss:.4f}  "
              f"val_mse={val_mse:.4f}  val_pearson={val_r:.4f}  "
              f"({time.time()-t0:.0f}s)")

    passed = val_r >= GATE_PEARSON
    print(f"\nGATE: val_pearson={val_r:.4f}  threshold={GATE_PEARSON}  "
          f"=> {'PASS' if passed else 'FAIL'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": student.state_dict(),
            "exo_config": exo_cfg,
            "meta": {
                "targets": str(args.targets), "epochs": args.epochs,
                "batch_size": args.batch_size, "lr": args.lr,
                "val_frac": args.val_frac, "val_pearson": val_r,
                "val_mse": val_mse, "gate_pass": passed, "seed": args.seed,
            },
        },
        out_path,
    )
    print(f"saved student -> {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    if not passed:
        raise SystemExit(f"Student failed the r>{GATE_PEARSON} gate (got {val_r:.4f}).")


if __name__ == "__main__":
    main()
