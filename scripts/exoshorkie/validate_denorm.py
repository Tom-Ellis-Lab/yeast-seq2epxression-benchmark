"""Closed-loop denormalization check for a distilled ExoShorkie student.

The students predict in z-space (linear head over log-z-scored coverage). To
work in count space we invert the authors' transform:

    target  z = (log1p( sum_16bp(|cov|) ) - mu) / sigma          (forward)
    inverse c = expm1( z * sigma + mu )                          (ISM_student.py:251)

mu/sigma are NOT shipped with the weights; we recompute them per genome from the
public figshare coverage NPZ with the authors' exact recipe
(`compute_logz_stats_multi`, pooling fwd + reversed-rev strands), then check that
running the genome's own student on its *real* windows reproduces the figshare
truth coverage after denorm. This is the single gate that collapses three risks
at once (is the figshare NPZ the space the student inverts to; is a genome-global
mu/sigma a faithful inverse; is the bin alignment right).

We compare the student's 896-bin output directly to the truth re-binned with the
authors' `crop_and_bin_cov` (sum over 16 bp, crop 1024 bp/side) — bin-for-bin, so
the magnitude ratio sum(pred)/sum(truth) is a direct test of (mu, sigma); no
per-base repeat/divide is involved here.

Usage:
    CUDA_VISIBLE_DEVICES=2 uv run --extra shorkie python \
        scripts/exoshorkie/validate_denorm.py \
        --gkey Mpneumo \
        --student /home/tds122/exoshorkie-weights/students/M_pneumoniae.pt \
        --fasta   /home/tds122/exoshorkie-weights/genomes/Mpneumo.fa \
        --fwd-npz /home/tds122/yeast-meneu/data/tasks/meneu_foreign_dna/_cache/Mpneumo_fwd_norm.npz \
        --rev-npz /home/tds122/yeast-meneu/data/tasks/meneu_foreign_dna/_cache/Mpneumo_rev_norm.npz \
        --device cuda:0 --max-windows 300
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.models.shorkie.nn import ShorkieModule

WINDOW_BP = 16384
CROP_BP = 1024
BIN_BP = 16
OUTPUT_BINS = 896  # (16384 - 2*1024) / 16


def crop_and_bin_cov(cov: np.ndarray) -> np.ndarray:
    """Authors' src/data_loader.py:179-184 — crop 1024/side, SUM each 16bp bin."""
    x = np.abs(np.asarray(cov, dtype=np.float64))[CROP_BP : cov.shape[0] - CROP_BP]
    L = (x.shape[0] // BIN_BP) * BIN_BP
    x = x[:L]
    return x.reshape(-1, BIN_BP).sum(axis=1)


def compute_logz_stats(fwd: np.ndarray, rev: np.ndarray) -> tuple[float, float]:
    """Authors' compute_logz_stats_multi, restricted to one chromosome key.

    Pool the cropped+binned forward coverage and the reversed reverse-strand
    coverage, log1p, then global mean/std. Mirrors data_loader.py:186-194 over
    the whole sequence (genome-global, the matched inverse for the 40-fold mean).
    """
    ys = [crop_and_bin_cov(fwd), crop_and_bin_cov(rev[::-1])]
    ys_log = np.log1p(np.concatenate(ys))
    return float(ys_log.mean()), float(ys_log.std() or 1e-8)


def per_row_pearson(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pm = pred - pred.mean(axis=1, keepdims=True)
    tm = true - true.mean(axis=1, keepdims=True)
    num = (pm * tm).sum(axis=1)
    den = np.sqrt((pm * pm).sum(axis=1) * (tm * tm).sum(axis=1)) + 1e-12
    return num / den


def load_student(path: str, device: torch.device) -> ShorkieModule:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ShorkieModule(ckpt["exo_config"])
    model.load_state_dict(ckpt["state_dict"])
    assert model._species_channel == 119 and model._encode_n_channel, "not an ExoShorkie student"
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gkey", default="auto", help="exogenous chromosome key; 'auto' = the non-yeast key")
    ap.add_argument("--student", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--fwd-npz", required=True)
    ap.add_argument("--rev-npz", required=True)
    ap.add_argument("--mu", type=float, default=None, help="override mu (else recompute from this NPZ)")
    ap.add_argument("--sigma", type=float, default=None, help="override sigma (else recompute)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-windows", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    import pysam

    yeast = {f"chr{r}" for r in ["I", "II", "III", "IV", "V", "VI", "VII", "VIII",
             "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]} | {"Mito"}
    fwd_npz = np.load(args.fwd_npz)
    gkey = args.gkey
    if gkey == "auto":
        gkey = next(k for k in fwd_npz.files if k not in yeast)
    fwd = fwd_npz[gkey]
    rev = np.load(args.rev_npz)[gkey]
    fa = pysam.FastaFile(args.fasta)
    seq = fa.fetch(fa.references[0]).upper()
    print(f"{gkey}: seq {len(seq):,} bp | fwd cov {fwd.shape[0]:,} | rev cov {rev.shape[0]:,}")
    assert len(seq) == fwd.shape[0] == rev.shape[0], "seq/coverage length mismatch"

    if args.mu is not None and args.sigma is not None:
        mu, sigma = args.mu, args.sigma
        print(f"using supplied genome-global stats: mu={mu:.4f}  sigma={sigma:.4f}")
    else:
        mu, sigma = compute_logz_stats(fwd, rev)
        print(f"recomputed (this-NPZ) log-z stats: mu={mu:.4f}  sigma={sigma:.4f}")

    n_pos = len(seq) - WINDOW_BP
    if n_pos <= 0:
        raise SystemExit("sequence shorter than one window")
    stride = max(1, n_pos // args.max_windows)
    starts = list(range(0, n_pos + 1, stride))[: args.max_windows]
    print(f"{len(starts)} real windows (stride {stride})")

    device = torch.device(args.device)
    model = load_student(args.student, device)

    pred_bins, truth_fwd_bins, truth_rev_bins = [], [], []
    rc_bins = []
    with torch.no_grad():
        for bs in range(0, len(starts), args.batch_size):
            batch = starts[bs : bs + args.batch_size]
            X = np.stack([one_hot_encode_channels_first(seq[a : a + WINDOW_BP]) for a in batch])
            xb = torch.from_numpy(X).float().to(device)
            z = model(xb).cpu().numpy()  # (B, 896) z-space, forward
            z_rc = model(xb.flip(dims=[1, 2])).cpu().numpy()[:, ::-1]  # RC pass, bins reversed
            pred_bins.append(np.clip(np.expm1(z * sigma + mu), 0, None))
            rc_bins.append(np.clip(np.expm1(z_rc * sigma + mu), 0, None))
            for a in batch:
                truth_fwd_bins.append(crop_and_bin_cov(fwd[a : a + WINDOW_BP]))
                truth_rev_bins.append(crop_and_bin_cov(rev[a : a + WINDOW_BP]))

    pred = np.concatenate(pred_bins)                     # forward-only denormed
    pred_rc = np.concatenate(rc_bins)                    # RC-only denormed
    pred_rcavg = 0.5 * (pred + pred_rc)                  # RC-averaged (count space)
    tf_ = np.stack(truth_fwd_bins)
    tr_ = np.stack(truth_rev_bins)
    t_avg = 0.5 * (tf_ + tr_)

    def report(name, p, t):
        r = per_row_pearson(p, t)
        rho = per_row_pearson(np.argsort(np.argsort(p, 1), 1).astype(float),
                              np.argsort(np.argsort(t, 1), 1).astype(float))
        rlog = per_row_pearson(np.log1p(p), np.log1p(t))
        ratio = p.sum(axis=1) / (t.sum(axis=1) + 1e-9)
        print(f"\n[{name}]  n={len(p)}")
        print(f"  per-window Pearson (count) : mean={r.mean():.4f} median={np.median(r):.4f}")
        print(f"  per-window Pearson (log1p) : mean={rlog.mean():.4f} median={np.median(rlog):.4f}")
        print(f"  per-window Spearman        : mean={rho.mean():.4f} median={np.median(rho):.4f}")
        print(f"  magnitude ratio sum(p)/sum(t): median={np.median(ratio):.3f} "
              f"[p10={np.percentile(ratio,10):.3f} p90={np.percentile(ratio,90):.3f}]")

    report("forward-only vs fwd truth", pred, tf_)
    report("RC-only vs rev truth", pred_rc, tr_)
    report("RC-averaged vs (fwd+rev)/2 truth", pred_rcavg, t_avg)


if __name__ == "__main__":
    main()
