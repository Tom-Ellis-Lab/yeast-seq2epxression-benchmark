"""Post-hoc Yorzoi vs Shorkie comparison on the shared sample set.

Each model evaluates the Brooks SCRaMBLE benchmark against its own
distribution file — Yorzoi at 4992 bp, Shorkie at 16384 bp — and writes
its full per-replicate prediction arrays + sample IDs to
``results/brooks/{model}__brooks_scramble{_shorkie}/``. The benchmark
spec says headline numbers should be reported on the **shared cohort**
(samples present in both result sets), so that we never penalise one
model for being able to evaluate a larger set than the other.

This script:
  1. Loads both result directories.
  2. Computes per-replicate Pearson r, Spearman ρ, dir-acc, the LOO
     reproducibility ceiling, and the calibration metrics on the
     intersection of ``sample_ids``.
  3. Writes the shared-cohort summary to
     ``results/brooks/compare__shared/summary.json`` and a Tier-1
     comparison plot (``shared_tier1.png``).
  4. Also records per-model full-set summaries side by side as
     ``secondary_full_set`` so the gap between shared-cohort and
     full-set numbers is documented.

Run:
    uv run python scripts/brooks/compare_models.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "brooks"
OUT_DIR = RESULTS / "compare__shared"

PSEUDOCOUNT = 1.0
RANGE_EPS = 1e-6
MIN_CALIB = 2          # need >= 2 reps for calibration metrics


def _load(model_dir: Path) -> dict:
    """Load one model's per-replicate prediction arrays + sample IDs."""
    meta = json.loads((model_dir / "samples.json").read_text())
    return {
        "name": model_dir.name,
        "sample_ids": meta["sample_ids"],
        "low_support": np.asarray(meta["low_support"], dtype=bool),
        "pred_lfc_runs": np.load(model_dir / "pred_lfc_runs.npy"),
        "true_lfc_runs": np.load(model_dir / "true_lfc_runs.npy"),
        "n_reps_supported": np.load(model_dir / "n_reps_supported.npy"),
    }


def _metrics(d: dict, idx: np.ndarray) -> dict:
    """Per-replicate r/ρ/dir-acc + LOO ceiling on the row subset ``idx``."""
    pred = d["pred_lfc_runs"][idx]
    true = d["true_lfc_runs"][idx]
    n_reps = d["n_reps_supported"][idx]
    low = d["low_support"][idx]
    nk = pred.shape[1]

    pr_per = np.full(nk, np.nan)
    sr_per = np.full(nk, np.nan)
    da_per = np.full(nk, np.nan)
    ceil_pr_per = np.full(nk, np.nan)
    ceil_da_per = np.full(nk, np.nan)

    finite_t = np.isfinite(true)
    finite_p = np.isfinite(pred)
    for k in range(nk):
        mk = (~low) & finite_t[:, k] & finite_p[:, k]
        if mk.sum() >= 2:
            t_k, p_k = true[mk, k], pred[mk, k]
            pr_per[k] = float(pearsonr(p_k, t_k).statistic)
            sr_per[k] = float(spearmanr(p_k, t_k).statistic)
            da_per[k] = float(balanced_accuracy_score(
                np.sign(t_k).astype(int), np.sign(p_k).astype(int)))
        # LOO ceiling on this replicate
        others = [j for j in range(nk) if j != k]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            loo = np.nanmean(true[:, others], axis=1)
        ck = (~low) & finite_t[:, k] & np.isfinite(loo)
        if ck.sum() >= 2:
            t_k, l_k = true[ck, k], loo[ck]
            ceil_pr_per[k] = float(pearsonr(l_k, t_k).statistic)
            ceil_da_per[k] = float(balanced_accuracy_score(
                np.sign(t_k).astype(int), np.sign(l_k).astype(int)))

    # Sample-mean LFCs for calibration metrics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_true = np.where(n_reps > 0, np.nanmean(true, axis=1), np.nan)
        mean_pred = np.where(n_reps > 0, np.nanmean(pred, axis=1), np.nan)

    calib_mask = (
        (~low) & (n_reps >= MIN_CALIB)
        & np.isfinite(mean_true) & np.isfinite(mean_pred)
    )
    if calib_mask.sum() >= 1:
        hits = 0
        zs = []
        for ii in np.where(calib_mask)[0]:
            runs = true[ii][finite_t[ii]]
            lo, hi = float(runs.min()), float(runs.max())
            if lo <= mean_pred[ii] <= hi:
                hits += 1
            zs.append(abs(mean_pred[ii] - mean_true[ii])
                       / max(hi - lo, RANGE_EPS))
        within = hits / calib_mask.sum()
        mz = float(np.mean(zs))
    else:
        within = mz = float("nan")

    scored = (~low) & (n_reps >= 1)
    return {
        "n_total": int(len(idx)),
        "n_low_support": int(low.sum()),
        "n_scored": int(scored.sum()),
        "n_calibration": int(calib_mask.sum()),
        "n_weak_baseline": int((n_reps == 0).sum()),
        "pearson_r": (float(np.nanmean(pr_per))
                       if np.any(np.isfinite(pr_per)) else float("nan")),
        "spearman_rho": (float(np.nanmean(sr_per))
                          if np.any(np.isfinite(sr_per)) else float("nan")),
        "dir_balanced_acc": (float(np.nanmean(da_per))
                              if np.any(np.isfinite(da_per)) else float("nan")),
        "ceiling_pearson_r": (float(np.nanmean(ceil_pr_per))
                               if np.any(np.isfinite(ceil_pr_per)) else float("nan")),
        "ceiling_dir_balanced_acc": (float(np.nanmean(ceil_da_per))
                                      if np.any(np.isfinite(ceil_da_per)) else float("nan")),
        "pearson_r_per_rep": pr_per.tolist(),
        "spearman_rho_per_rep": sr_per.tolist(),
        "dir_balanced_acc_per_rep": da_per.tolist(),
        "ceiling_pearson_r_per_rep": ceil_pr_per.tolist(),
        "ceiling_dir_balanced_acc_per_rep": ceil_da_per.tolist(),
        "within_range_rate": within,
        "mean_abs_z": mz,
    }


def _plot_shared_per_sample(
    Y: dict, S: dict, y_idx: np.ndarray, s_idx: np.ndarray,
    out_path: Path,
) -> None:
    """Per-sample interval plot on the shared cohort. For each scored
    sample: three small vertical ranges side by side, with mean dots:
        blue  = truth (min..max of JS94 deep-run true LFCs)
        red   = Yorzoi pred (min..max across the 3 JS94 reps)
        green = Shorkie pred (same; range collapses to a single value
                because Shorkie's varies_by_strain=False)
    Samples are sorted left-to-right by mean true LFC."""
    import matplotlib.pyplot as plt

    true = Y["true_lfc_runs"][y_idx]                       # (n_shared, 3)
    pred_y = Y["pred_lfc_runs"][y_idx]
    pred_s = S["pred_lfc_runs"][s_idx]
    n_reps_y = Y["n_reps_supported"][y_idx]
    low_y = Y["low_support"][y_idx]
    low_s = S["low_support"][s_idx]

    finite_t = np.isfinite(true)
    finite_y = np.isfinite(pred_y)
    finite_s = np.isfinite(pred_s)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_true = np.where(n_reps_y > 0, np.nanmean(true, axis=1), np.nan)
        mean_pred_y = np.where(
            finite_y.any(axis=1), np.nanmean(pred_y, axis=1), np.nan
        )
        mean_pred_s = np.where(
            finite_s.any(axis=1), np.nanmean(pred_s, axis=1), np.nan
        )

    # Shared-scored mask: not low_support on either side, at least one
    # replicate truth, and both models produced predictions.
    m = (
        (~low_y) & (~low_s) & (n_reps_y >= 1)
        & np.isfinite(mean_true)
        & np.isfinite(mean_pred_y) & np.isfinite(mean_pred_s)
    )
    K = int(m.sum())
    if K == 0:
        return
    idx_sorted = np.array(sorted(np.where(m)[0], key=lambda i: mean_true[i]))

    # Per-sample ranges (min..max of finite per-replicate values)
    def ranges(arr, finite_mask):
        lo = np.array([
            arr[i][finite_mask[i]].min() if finite_mask[i].any() else np.nan
            for i in idx_sorted
        ])
        hi = np.array([
            arr[i][finite_mask[i]].max() if finite_mask[i].any() else np.nan
            for i in idx_sorted
        ])
        mn = np.array([
            arr[i][finite_mask[i]].mean() if finite_mask[i].any() else np.nan
            for i in idx_sorted
        ])
        return lo, hi, mn

    t_lo, t_hi, t_mn = ranges(true, finite_t)
    y_lo, y_hi, y_mn = ranges(pred_y, finite_y)
    s_lo, s_hi, s_mn = ranges(pred_s, finite_s)

    fig_w = max(8.0, 0.08 * K)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    ax.axhline(0, color="grey", lw=0.3)
    x = np.arange(K, dtype=float)
    off = 0.22                                  # x-offset between the 3 groups
    sz = 8
    ax.vlines(x - off, t_lo, t_hi, colors="#1f77b4", lw=1.0, alpha=0.7)
    ax.scatter(x - off, t_mn, s=sz, c="#1f77b4", label="true")
    ax.vlines(x,        y_lo, y_hi, colors="#d62728", lw=1.0, alpha=0.7)
    ax.scatter(x,        y_mn, s=sz, c="#d62728", label="Yorzoi pred")
    ax.vlines(x + off, s_lo, s_hi, colors="#2ca02c", lw=1.0, alpha=0.7)
    ax.scatter(x + off, s_mn, s=sz, c="#2ca02c", label="Shorkie pred")

    ax.set_xlim(-1, K)
    ax.set_xlabel(f"sample (sorted by mean true LFC, n={K})")
    ax.set_ylabel("log2 LFC (alt / native)")
    ax.set_title(
        "Brooks SCRaMBLE — per-sample LFC ranges (shared cohort)"
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _plot_shared(results: dict, out_path: Path) -> None:
    """Bar-chart-ish comparison on the shared cohort. One row per
    metric: Yorzoi (red) vs Shorkie (blue) with the LOO ceiling marked
    as a grey dashed line."""
    import matplotlib.pyplot as plt

    shared = results["shared_cohort"]
    rows = [
        ("Pearson r",  "pearson_r",          "ceiling_pearson_r"),
        ("Spearman ρ", "spearman_rho",        None),
        ("dir-acc",    "dir_balanced_acc",    "ceiling_dir_balanced_acc"),
    ]
    fig, axes = plt.subplots(len(rows), 1, figsize=(8, 1.6 * len(rows)),
                              squeeze=False)
    for i, (name, key, ceil_key) in enumerate(rows):
        ax = axes[i, 0]
        ys, names = [], []
        for model_name in ("yorzoi", "shorkie"):
            ys.append(shared[model_name][key])
            names.append(model_name)
        colors = ["crimson", "royalblue"]
        bars = ax.barh(names, ys, color=colors, alpha=0.85)
        ax.axvline(0, color="grey", lw=0.5)
        if ceil_key:
            ceil = shared["yorzoi"][ceil_key]   # same across both on shared set
            ax.axvline(ceil, color="grey", lw=1.0, ls="--",
                        label=f"LOO ceiling {ceil:.3f}")
            ax.legend(loc="lower right", fontsize=8)
        for b, v in zip(bars, ys):
            ax.text(v + 0.005, b.get_y() + b.get_height() / 2,
                     f"{v:+.3f}", va="center", fontsize=9)
        ax.set_xlim(min(-0.05, min(ys) - 0.05), max(1.0, *ys) * 1.05 + 0.05)
        ax.set_title(f"{name}  (shared cohort, n_scored={shared['n_scored']})",
                      fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    yorzoi_dir = RESULTS / "yorzoi__brooks_scramble"
    shorkie_dir = RESULTS / "shorkie__brooks_scramble_shorkie"

    Y = _load(yorzoi_dir)
    S = _load(shorkie_dir)

    shared = sorted(set(Y["sample_ids"]) & set(S["sample_ids"]))
    Y_index = {sid: i for i, sid in enumerate(Y["sample_ids"])}
    S_index = {sid: i for i, sid in enumerate(S["sample_ids"])}
    y_idx = np.array([Y_index[sid] for sid in shared])
    s_idx = np.array([S_index[sid] for sid in shared])

    out = {
        "shared_cohort": {
            "sample_ids": shared,
            "n": len(shared),
            "n_scored": int(((~Y["low_support"][y_idx])
                              & (Y["n_reps_supported"][y_idx] >= 1)).sum()),
            "yorzoi": _metrics(Y, y_idx),
            "shorkie": _metrics(S, s_idx),
        },
        "secondary_full_set": {
            "yorzoi": _metrics(Y, np.arange(len(Y["sample_ids"]))),
            "shorkie": _metrics(S, np.arange(len(S["sample_ids"]))),
        },
        "note": (
            "Headline metrics are computed on the intersection of the two "
            "models' sample sets. Full-set metrics for each model are kept "
            "as secondary so the gap is documented."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(out, indent=2))
    _plot_shared(out, OUT_DIR / "shared_tier1.png")
    _plot_shared_per_sample(
        Y, S, y_idx, s_idx, OUT_DIR / "shared_per_sample.png"
    )

    # Console table
    sh = out["shared_cohort"]
    ful = out["secondary_full_set"]
    print(f"\nShared cohort: n={sh['n']}  n_scored={sh['n_scored']}\n")
    print(f"  {'metric':22s}  {'Yorzoi':>10s}  {'Shorkie':>10s}  {'ceiling':>10s}")
    for name, key, ceil_key in [
        ("Pearson r",   "pearson_r",          "ceiling_pearson_r"),
        ("Spearman ρ",  "spearman_rho",        None),
        ("dir-acc",     "dir_balanced_acc",    "ceiling_dir_balanced_acc"),
        ("within-range", "within_range_rate",   None),
        ("|z|",         "mean_abs_z",          None),
    ]:
        c = (f"{sh['yorzoi'][ceil_key]:>10.3f}" if ceil_key else " " * 10)
        print(f"  {name:22s}  {sh['yorzoi'][key]:>+10.3f}  "
              f"{sh['shorkie'][key]:>+10.3f}  {c}")
    print(f"\nSecondary full-set:\n"
          f"  yorzoi  n_scored {ful['yorzoi']['n_scored']:4d}  "
          f"r {ful['yorzoi']['pearson_r']:+.3f}  "
          f"dir-acc {ful['yorzoi']['dir_balanced_acc']:.3f}\n"
          f"  shorkie n_scored {ful['shorkie']['n_scored']:4d}  "
          f"r {ful['shorkie']['pearson_r']:+.3f}  "
          f"dir-acc {ful['shorkie']['dir_balanced_acc']:.3f}\n")
    print(f"wrote {OUT_DIR / 'summary.json'}")
    print(f"wrote {OUT_DIR / 'shared_tier1.png'}")
    print(f"wrote {OUT_DIR / 'shared_per_sample.png'}")


if __name__ == "__main__":
    main()
