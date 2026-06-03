"""Generate per-base coverage predictions for a few Brooks loci, for the
s2fview true-vs-predicted notebook.

ExoShorkie + Shorkie (both 16,384 bp window) are produced here; Yorzoi (4,992 bp
window) is produced by the sibling worktree's env (gen_plot_preds_yorzoi.py).
Each model's prediction is paired with the measured Nanopore coverage cropped to
the model's predicted genomic span, so all tracks align on the CDS.

Picks `--n-loci` well-supported loci spanning down/near-zero/up true-LFC.

    CUDA_VISIBLE_DEVICES="" uv run --extra shorkie python \
        scripts/exoshorkie/gen_plot_preds.py --out results/plot_preds/exo_shorkie.npz
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

csv.field_size_limit(sys.maxsize)

from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._shorkie_constants import (
    CROP_BP_EACH_SIDE,
    SEQ_LEN,
    SHORKIE_T0_RNA_SEQ_TRACK_IDS,
)
from yeastbench.models.exoshorkie import ExoShorkie
from yeastbench.models.shorkie import Shorkie

TSV = "data/tasks/brooks_scramble/brooks_scramble_v1_w16384.tsv"
SHORKIE_PARAMS = "data/models/shorkie/params.json"
SHORKIE_CKPTS = [f"data/models/shorkie/checkpoints/f{i}.h5" for i in range(8)]


def load_rows() -> list[dict]:
    with open(TSV) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def pick_loci(rows: list[dict], n: int) -> list[dict]:
    """n well-supported loci spanning the true-LFC range (down / ~0 / up)."""
    ok = [r for r in rows if r.get("low_support", "True") in ("False", "0", "")]
    ok = [r for r in ok if r["true_lfc"] not in ("", "nan")]
    ok.sort(key=lambda r: float(r["true_lfc"]))
    if len(ok) < n:
        ok = rows[:n]
    idx = np.linspace(0, len(ok) - 1, n).round().astype(int)
    return [ok[i] for i in idx]


def true_cov_cropped(row: dict) -> np.ndarray:
    """Measured per-base coverage over the model's predicted central region."""
    full = np.array([float(x) for x in row["true_cov_alt"].split(",")], dtype=np.float64)
    return full[CROP_BP_EACH_SIDE : CROP_BP_EACH_SIDE + (SEQ_LEN - 2 * CROP_BP_EACH_SIDE)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/plot_preds/exo_shorkie.npz")
    ap.add_argument("--n-loci", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rows = load_rows()
    loci = pick_loci(rows, args.n_loci)
    print("selected loci:", [(r["sample_id"], round(float(r["true_lfc"]), 2)) for r in loci])

    dev = torch.device(args.device)
    exo = ExoShorkie.from_students(device=dev, use_rc=True)
    shorkie = Shorkie.from_checkpoints(SHORKIE_PARAMS, SHORKIE_CKPTS, device=dev, use_rc=True)
    track_t = torch.tensor(SHORKIE_T0_RNA_SEQ_TRACK_IDS, device=dev, dtype=torch.long)

    X = np.stack([one_hot_encode_channels_first(r["alt_seq"]) for r in loci]).astype(np.float32)
    xb = torch.from_numpy(X).to(dev)
    with torch.no_grad():
        exo_pred = exo.forward_perbase(xb).cpu().numpy()                       # (n,14336)
        shorkie_pred = shorkie.forward_track_mean_perbase(xb, track_t).cpu().numpy()

    out: dict = {}
    meta = []
    for i, r in enumerate(loci):
        sid = r["sample_id"]
        ws = int(r["cds_start"]) - int(r["cds_start_in_window"])  # genomic window start
        pred_lo = ws + CROP_BP_EACH_SIDE
        positions = np.arange(pred_lo, pred_lo + (SEQ_LEN - 2 * CROP_BP_EACH_SIDE))
        out[f"{sid}|positions"] = positions
        out[f"{sid}|true"] = true_cov_cropped(r)
        out[f"{sid}|exoshorkie"] = exo_pred[i]
        out[f"{sid}|shorkie"] = shorkie_pred[i]
        meta.append({
            "sample_id": sid, "gene_id": r["gene_id"], "strand": r["strand"],
            "cds_start": int(r["cds_start"]), "cds_end": int(r["cds_end"]),
            "true_lfc": float(r["true_lfc"]), "window": "16384",
        })
    out["meta"] = np.array(json.dumps(meta))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"saved {args.out}  ({len(loci)} loci × [true, exoshorkie, shorkie])")


if __name__ == "__main__":
    main()
