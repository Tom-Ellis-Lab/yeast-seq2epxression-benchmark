"""Cuperus 5'-UTR metrics + the translation-only feature set `f`.

This is the benchmark/eval-layer core, independent of any model. It computes:

- ``kozak_design`` — the translation-only features `f`: a drop-first one-hot of
  the 5 nt immediately 5' of the ATG (Kozak positions -1..-5). See the spec's
  "translation-only features `f`" section.
- ``metric1`` — direct correlation of the model score `g` with `growth_rate`
  `E`, overall and per depth bucket (Spearman headline, Pearson alongside).
- ``metric2`` — the model's signal beyond `f`: cross-validated partial
  correlation ``corr(E_r, g_r)`` and incremental R² ``R²(E~f+g) - R²(E~f)``.

The metric primitives are scale-agnostic: pass `g` already on the scale you
want (Spearman is invariant; for Pearson the benchmark passes ``log g``).
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.model_selection import KFold

# Kozak window: positions -1..-5 (the 5 nt immediately 5' of the ATG). One-hot
# with 'A' as the dropped reference (A at -3 is the favorable base), so each
# position contributes 3 dummy columns for C/G/T.
KOZAK_N_POS = 5
_ALT = "CGT"  # non-reference bases


def kozak_design(utrs, n_pos: int = KOZAK_N_POS) -> np.ndarray:
    """Drop-first one-hot of Kozak positions -1..-n_pos (last n_pos nt of each
    UTR, the bases just 5' of the ATG). Shape ``(N, 3*n_pos)``; reference base
    'A' and any position missing on a short fragment map to the all-zero row.
    Columns are ordered ``[-1:C,-1:G,-1:T, -2:C,...]``."""
    seqs = list(utrs)
    X = np.zeros((len(seqs), 3 * n_pos), dtype=float)
    for i, s in enumerate(seqs):
        L = len(s)
        for k in range(1, n_pos + 1):
            if L >= k:
                base = s[L - k]
                if base in _ALT:
                    X[i, (k - 1) * 3 + _ALT.index(base)] = 1.0
    return X


def _corr(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman + Pearson on finite pairs."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return {"n": n, "spearman": float("nan"), "pearson": float("nan")}
    xs, ys = x[m], y[m]
    spearman = float(stats.spearmanr(xs, ys).statistic)
    pearson = (
        float(stats.pearsonr(xs, ys).statistic)
        if xs.std() > 0 and ys.std() > 0
        else float("nan")
    )
    return {"n": n, "spearman": spearman, "pearson": pearson}


def metric1(g, E, buckets=None) -> dict:
    """Direct correlation of model score `g` with `growth_rate` `E`, overall
    and (if `buckets` given) per depth bucket. Returns
    ``{"overall": {...}, "by_bucket": {b: {...}}}``."""
    g = np.asarray(g, float)
    E = np.asarray(E, float)
    out = {"overall": _corr(g, E)}
    if buckets is not None:
        buckets = np.asarray(buckets)
        out["by_bucket"] = {
            int(b): _corr(g[buckets == b], E[buckets == b])
            for b in np.unique(buckets)
        }
    return out


def _oof_resid(y: np.ndarray, F: np.ndarray, folds) -> np.ndarray:
    """Out-of-fold residuals of `y` regressed on ``[1, F]``."""
    Fi = np.column_stack([np.ones(len(y)), F])
    resid = np.empty(len(y))
    for tr, te in folds:
        beta, *_ = np.linalg.lstsq(Fi[tr], y[tr], rcond=None)
        resid[te] = y[te] - Fi[te] @ beta
    return resid


def _oof_r2(y: np.ndarray, X: np.ndarray, folds) -> float:
    """Cross-validated R² of `y` on ``[1, X]``, pooled out-of-fold."""
    Xi = np.column_stack([np.ones(len(y)), X])
    yhat = np.empty(len(y))
    for tr, te in folds:
        beta, *_ = np.linalg.lstsq(Xi[tr], y[tr], rcond=None)
        yhat[te] = Xi[te] @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def metric2(g, E, f, n_folds: int = 5, seed: int = 0) -> dict:
    """The model's signal beyond the translation-only features `f`.

    Cross-validated (same folds for everything): partial correlation
    ``corr(E_r, g_r)`` of the out-of-fold residuals of `E` and `g` on `f`,
    plus incremental R² = ``R²(E~f+g) - R²(E~f)``.
    """
    g = np.asarray(g, float)
    E = np.asarray(E, float)
    f = np.asarray(f, float)
    if f.ndim == 1:
        f = f[:, None]
    m = np.isfinite(g) & np.isfinite(E) & np.all(np.isfinite(f), axis=1)
    g, E, f = g[m], E[m], f[m]
    n = len(E)
    if n < n_folds + 2:
        nan = float("nan")
        return {"n": n, "partial_pearson": nan, "partial_spearman": nan,
                "r2_f": nan, "r2_fg": nan, "incremental_r2": nan}

    folds = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(E))
    E_r = _oof_resid(E, f, folds)
    g_r = _oof_resid(g, f, folds)
    r2_f = _oof_r2(E, f, folds)
    r2_fg = _oof_r2(E, np.column_stack([f, g]), folds)
    return {
        "n": n,
        "partial_pearson": float(stats.pearsonr(E_r, g_r).statistic),
        "partial_spearman": float(stats.spearmanr(E_r, g_r).statistic),
        "r2_f": r2_f,
        "r2_fg": r2_fg,
        "incremental_r2": r2_fg - r2_f,
    }
