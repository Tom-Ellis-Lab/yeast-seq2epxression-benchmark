"""Yorzoi per-base coverage for the same Brooks loci as gen_plot_preds.py.

Yorzoi uses a 4,992 bp window (central 3,000 bp predicted), so it is generated
separately and aligned to the others by genomic position. Run from the sibling
worktree's env (which has yorzoi + flash-attn + the cached checkpoint):

    cd /home/tds122/yeast-seq2epxression-benchmark
    CUDA_VISIBLE_DEVICES="" uv run --extra yorzoi python \
        /home/tds122/yeast-exoshorkie/scripts/exoshorkie/gen_plot_preds_yorzoi.py \
        --loci-npz /home/tds122/yeast-exoshorkie/results/plot_preds/exo_shorkie.npz \
        --tsv /home/tds122/yeast-exoshorkie/data/tasks/brooks_scramble/brooks_scramble_v1.tsv \
        --out /home/tds122/yeast-exoshorkie/results/plot_preds/yorzoi.npz
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

csv.field_size_limit(sys.maxsize)

from yeastbench.adapters.yorzoi_brooks import YorzoiBrooksPredictor

SEQ_LEN = 4992
CROP = 996
OUT_LEN = SEQ_LEN - 2 * CROP  # 3000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loci-npz", required=True, help="exo_shorkie.npz (for the sample_id list)")
    ap.add_argument("--tsv", required=True, help="brooks_scramble_v1.tsv (4992 window)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    meta = json.loads(str(np.load(args.loci_npz, allow_pickle=True)["meta"]))
    want = {m["sample_id"] for m in meta}

    with open(args.tsv) as fh:
        rows = {r["sample_id"]: r for r in csv.DictReader(fh, delimiter="\t") if r["sample_id"] in want}
    missing = want - set(rows)
    if missing:
        print(f"WARNING: {len(missing)} loci not in 4992 TSV: {sorted(missing)}")

    sids = [m["sample_id"] for m in meta if m["sample_id"] in rows]
    seqs = [rows[s]["alt_seq"] for s in sids]
    strands = [rows[s]["strand"] for s in sids]

    pred = YorzoiBrooksPredictor.from_pretrained(
        "tom-ellis-lab/yorzoi", device=args.device, track_mode="nanopore_all",
    ).predict_coverage_batch(seqs, strands, strains=None)  # (n, 3000)

    out: dict = {}
    for i, sid in enumerate(sids):
        r = rows[sid]
        ws = int(r["cds_start"]) - int(r["cds_start_in_window"])  # genomic window start
        pred_lo = ws + CROP
        out[f"{sid}|positions"] = np.arange(pred_lo, pred_lo + OUT_LEN)
        full = np.array([float(x) for x in r["true_cov_alt"].split(",")], dtype=np.float64)
        out[f"{sid}|true"] = full[CROP : CROP + OUT_LEN]
        out[f"{sid}|yorzoi"] = pred[i]
    out["sids"] = np.array(json.dumps(sids))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"saved {args.out}  ({len(sids)} loci × yorzoi)")


if __name__ == "__main__":
    main()
