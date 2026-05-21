"""Wu et al. RFP-insertion position-effect benchmark.

Zero-shot prediction of OD600-normalised RFP plate-reader intensity for
one constant cassette integrated at 1044 single-ORF deletion loci.  Only
the genomic insertion site varies — this is a position-effect probe (see
``benchmarks/wu_rfpins.md``).

Primary metric: Pearson r + Spearman ρ between predicted and measured
relative fluorescence.  Secondary: two binary tail-classification tasks
— detecting Wu et al.'s *extreme-low* ([0, 5)) and *extreme-high*
([8, 13]) classes from the predicted score (AUROC + AUPRC, rank-based so
no scale alignment is needed).  These are the biologically interesting
extremes the paper highlights.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from yeastbench.adapters._genome import parse_gene_annotations
from yeastbench.adapters._wu_scaffold import WuLocus, resolve_loci
from yeastbench.adapters.protocols import CassetteExpressionPredictor
from yeastbench.benchmarks.base import (
    Benchmark,
    BenchmarkInfo,
    model_color,
)

# Paper's five fixed absolute-cutoff classes (Wu et al. Fig. 1b) — used
# only for the descriptive histogram now.
CLASS_EDGES: tuple[float, ...] = (5.0, 6.0, 7.0, 8.0)
CLASS_NAMES: tuple[str, ...] = (
    "extreme_low", "low", "moderate", "high", "extreme_high",
)
CLASS_COLORS: tuple[str, ...] = ("red", "gold", "green", "violet", "blue")

# Two binary tail-classification tasks (the paper's outermost classes).
EXTREME_LOW_CUTOFF = CLASS_EDGES[0]    # measured < 5.0  → extreme-low
EXTREME_HIGH_CUTOFF = CLASS_EDGES[-1]  # measured ≥ 8.0  → extreme-high

LABEL_COL = "Relative_Fluorescence_Average"
ERROR_COL = "Relative_Fluorescence_Error"
ORF_COL = "ORF_name"


@dataclass(frozen=True)
class WuResults:
    scores: np.ndarray            # (N,) predicted; NaN for unresolved/uncontextualised loci
    labels: np.ndarray            # (N,) measured relative fluorescence
    gene_ids: list[str]           # (N,) ORF systematic names, row-aligned
    dropped_ids: list[str]        # ORFs not resolvable in the GTF
    n_scored: int
    pearson_r: float
    spearman_rho: float
    # Two binary tail tasks (computed on the scored subset)
    low_n_pos: int                # # loci with measured < EXTREME_LOW_CUTOFF
    low_auroc: float
    low_auprc: float
    high_n_pos: int               # # loci with measured ≥ EXTREME_HIGH_CUTOFF
    high_auroc: float
    high_auprc: float


def _binary_scores(
    y_true: np.ndarray, discriminant: np.ndarray
) -> tuple[float, float]:
    """AUROC, AUPRC for a binary task; NaN if a class is absent."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(y_true, discriminant)),
        float(average_precision_score(y_true, discriminant)),
    )


def _metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(scores) & np.isfinite(labels)
    p, m = scores[mask], labels[mask]
    n = int(mask.sum())
    if n < 2:
        return dict(
            n_scored=n, pearson_r=float("nan"), spearman_rho=float("nan"),
            low_n_pos=0, low_auroc=float("nan"), low_auprc=float("nan"),
            high_n_pos=0, high_auroc=float("nan"), high_auprc=float("nan"),
        )
    y_low = (m < EXTREME_LOW_CUTOFF).astype(int)
    y_high = (m >= EXTREME_HIGH_CUTOFF).astype(int)
    # extreme-low loci are expected to have *low* predicted expression,
    # so the positive-class discriminant is −score; extreme-high uses
    # +score directly.
    low_auroc, low_auprc = _binary_scores(y_low, -p)
    high_auroc, high_auprc = _binary_scores(y_high, p)
    return dict(
        n_scored=n,
        pearson_r=float(pearsonr(p, m).statistic),
        spearman_rho=float(spearmanr(p, m).statistic),
        low_n_pos=int(y_low.sum()),
        low_auroc=low_auroc,
        low_auprc=low_auprc,
        high_n_pos=int(y_high.sum()),
        high_auroc=high_auroc,
        high_auprc=high_auprc,
    )


class RFPInsertionBenchmark(Benchmark[CassetteExpressionPredictor, WuResults]):
    adapter_protocol: ClassVar[type] = CassetteExpressionPredictor

    def __init__(
        self,
        cassette_seq: Path,
        labels_path: Path,
        fasta_path: Path,
        gtf_path: Path,
        info: BenchmarkInfo,
    ) -> None:
        self.cassette_seq = Path(cassette_seq)
        self.labels_path = Path(labels_path)
        self._fasta_path = Path(fasta_path)
        self._gtf_path = Path(gtf_path)
        self.info = info

        df = pd.read_csv(self.labels_path)
        for col in (ORF_COL, LABEL_COL):
            assert col in df.columns, f"{col} missing from {self.labels_path}"
        self.gene_ids: list[str] = df[ORF_COL].astype(str).tolist()
        self.labels: np.ndarray = df[LABEL_COL].to_numpy(dtype=float)
        self.errors: np.ndarray = (
            df[ERROR_COL].to_numpy(dtype=float)
            if ERROR_COL in df.columns
            else np.full(len(df), np.nan)
        )

        genes = parse_gene_annotations(self._gtf_path)
        self.loci, self.dropped = resolve_loci(self.gene_ids, genes)

    @property
    def fasta_path(self) -> Path:
        return self._fasta_path

    @property
    def gtf_path(self) -> Path:
        return self._gtf_path

    def evaluate(self, adapter: CassetteExpressionPredictor) -> WuResults:
        resolved_idx = [i for i, lc in enumerate(self.loci) if lc is not None]
        sub_loci: list[WuLocus] = [self.loci[i] for i in resolved_idx]  # type: ignore[misc]

        scores = np.full(len(self.labels), np.nan, dtype=np.float64)
        if sub_loci:
            sub_scores = np.asarray(
                adapter.predict_expressions(sub_loci), dtype=float
            )
            assert len(sub_scores) == len(sub_loci), (
                f"adapter returned {len(sub_scores)} scores for "
                f"{len(sub_loci)} loci"
            )
            scores[resolved_idx] = sub_scores

        m = _metrics(scores, self.labels)
        return WuResults(
            scores=scores,
            labels=self.labels,
            gene_ids=self.gene_ids,
            dropped_ids=self.dropped,
            **m,
        )

    def plot(self, results: WuResults, out_dir: Path) -> None:
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        title_model = out_dir.name.split("__")[0] if "__" in out_dir.name else ""

        mask = np.isfinite(results.scores) & np.isfinite(results.labels)
        pred = results.scores[mask]
        meas = results.labels[mask]

        # (1) scatter pred vs measured
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(meas, pred, s=8, alpha=0.4, rasterized=True)
        if mask.sum() > 1:
            a, b = np.polyfit(meas, pred, 1)
            xs = np.linspace(meas.min(), meas.max(), 50)
            ax.plot(xs, a * xs + b, color="red", linewidth=1, alpha=0.8)
        ax.set_xlabel("measured relative fluorescence (RFP / OD600)")
        ax.set_ylabel("predicted (mCherry-CDS track readout)")
        title = "Wu RFP-insertion position effect"
        if title_model:
            title += f" — {title_model}"
        ax.set_title(
            f"{title}\nn = {results.n_scored}  "
            f"r = {results.pearson_r:.3f}  ρ = {results.spearman_rho:.3f}"
        )
        fig.tight_layout()
        fig.savefig(out_dir / "scatter.png", dpi=150)
        plt.close(fig)

        # (2) Fig. 1b-style measured histogram, coloured by the 5 classes
        fig, ax = plt.subplots(figsize=(7, 4))
        edges = [meas.min()] + list(CLASS_EDGES) + [max(meas.max(), 13.0)]
        for c, color in enumerate(CLASS_COLORS):
            lo, hi = edges[c], edges[c + 1]
            sel = (meas >= lo) & (meas < hi if c < 4 else meas >= lo)
            ax.hist(
                meas[sel], bins=np.arange(np.floor(meas.min()), 14.0, 0.5),
                color=color, alpha=0.8, label=CLASS_NAMES[c],
            )
        ax.set_xlabel("relative fluorescence intensity")
        ax.set_ylabel("number of loci")
        ax.set_title("Measured intensity by Wu class" +
                     (f" — {title_model}" if title_model else ""))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "measured_classes.png", dpi=150)
        plt.close(fig)

        # (3) ROC + PR for the two binary tail tasks
        y_low = (meas < EXTREME_LOW_CUTOFF).astype(int)
        y_high = (meas >= EXTREME_HIGH_CUTOFF).astype(int)
        for name, y, disc, fname in (
            ("extreme-low (< %.0f)" % EXTREME_LOW_CUTOFF, y_low, -pred,
             "roc_pr_extreme_low.png"),
            ("extreme-high (≥ %.0f)" % EXTREME_HIGH_CUTOFF, y_high, pred,
             "roc_pr_extreme_high.png"),
        ):
            if y.sum() == 0 or y.sum() == len(y):
                continue
            fpr, tpr, _ = roc_curve(y, disc)
            prec, rec, _ = precision_recall_curve(y, disc)
            auroc = roc_auc_score(y, disc)
            auprc = average_precision_score(y, disc)
            base = y.mean()
            fig, (axr, axp) = plt.subplots(1, 2, figsize=(11, 5))
            axr.plot(fpr, tpr, color="C0", label=f"AUROC = {auroc:.3f}")
            axr.plot([0, 1], [0, 1], "--", color="grey", label="random")
            axr.set_xlabel("False positive rate")
            axr.set_ylabel("True positive rate")
            axr.set_xlim(0, 1); axr.set_ylim(0, 1)
            axr.set_aspect("equal", "box")
            axr.legend(loc="lower right", fontsize=8); axr.set_title("ROC")
            axp.plot(rec, prec, color="C1", label=f"AUPRC = {auprc:.3f}")
            axp.axhline(base, ls="--", color="grey",
                        label=f"random (base = {base:.3f})")
            axp.set_xlabel("Recall"); axp.set_ylabel("Precision")
            axp.set_xlim(0, 1); axp.set_ylim(0, 1.02)
            axp.set_aspect("equal", "box")
            axp.legend(loc="upper right", fontsize=8)
            axp.set_title("Precision–Recall")
            fig.suptitle(
                f"Wu {name} tail detection"
                + (f" — {title_model}" if title_model else "")
                + f"   (n+={int(y.sum())} / {len(y)})"
            )
            fig.tight_layout()
            fig.savefig(out_dir / fname, dpi=150)
            plt.close(fig)

    def save_results(self, results: WuResults, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "scores.npy", results.scores)
        np.save(out_dir / "labels.npy", results.labels)
        (out_dir / "loci.json").write_text(
            json.dumps(
                {"gene_ids": results.gene_ids, "dropped_ids": results.dropped_ids},
                indent=2,
            )
        )

    def load_results(self, out_dir: Path) -> WuResults:
        out_dir = Path(out_dir)
        scores = np.load(out_dir / "scores.npy")
        labels = np.load(out_dir / "labels.npy")
        meta = json.loads((out_dir / "loci.json").read_text())
        m = _metrics(scores, labels)
        return WuResults(
            scores=scores,
            labels=labels,
            gene_ids=meta["gene_ids"],
            dropped_ids=meta["dropped_ids"],
            **m,
        )

    def summary_dict(self, results: WuResults) -> dict[str, Any]:
        return {
            "n_rows_total": int(len(results.scores)),
            "n_rows_scored": results.n_scored,
            "n_dropped_unresolved": len(results.dropped_ids),
            "pearson_r": results.pearson_r,
            "spearman_rho": results.spearman_rho,
            "extreme_low_n_pos": results.low_n_pos,
            "extreme_low_auroc": results.low_auroc,
            "extreme_low_auprc": results.low_auprc,
            "extreme_high_n_pos": results.high_n_pos,
            "extreme_high_auroc": results.high_auroc,
            "extreme_high_auprc": results.high_auprc,
        }

    def headline(self, results: WuResults) -> str:
        return (
            f"Pearson r = {results.pearson_r:.4f}  "
            f"Spearman ρ = {results.spearman_rho:.4f}  | "
            f"extreme-low AUROC {results.low_auroc:.3f} "
            f"AUPRC {results.low_auprc:.3f}  | "
            f"extreme-high AUROC {results.high_auroc:.3f} "
            f"AUPRC {results.high_auprc:.3f}  (n = {results.n_scored})"
        )

    def headline_metric_labels(self) -> dict[str, str]:
        # Order: continuous-axis metrics first, then the binary-tail
        # AUROC/AUPRC pairs. Skip the count-style `extreme_*_n_pos`
        # numbers — they aren't comparable across models on a bar plot.
        return {
            "pearson_r":           "Pearson r",
            "spearman_rho":        "Spearman ρ",
            "extreme_low_auroc":   "extreme-low AUROC",
            "extreme_high_auroc":  "extreme-high AUROC",
            "extreme_low_auprc":   "extreme-low AUPRC",
            "extreme_high_auprc":  "extreme-high AUPRC",
        }

    def compare_plot_title(self) -> str:
        return "Wu et al. RFP-insertion position effects"

    def compare_plot(
        self,
        model_dirs: Mapping[str, Path],
        out_dir: Path,
    ) -> Path | None:
        """Custom 1×3 compare panel:
          (1) bar — Pearson r + Spearman ρ per model,
          (2) ROC — both tail tasks overlaid per model (solid = extreme-high,
              dashed = extreme-low), with a random diagonal,
          (3) PR  — same overlay scheme, with per-tail base-rate lines.

        Returns the SVG path, or ``None`` if < 2 models have loadable
        scores+labels on disk."""
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        loaded: dict[str, dict[str, np.ndarray]] = {}
        for name, mdir in model_dirs.items():
            mdir = Path(mdir)
            scores_p = mdir / "scores.npy"
            labels_p = mdir / "labels.npy"
            if not (scores_p.exists() and labels_p.exists()):
                continue
            scores = np.load(scores_p)
            labels = np.load(labels_p)
            mask = np.isfinite(scores) & np.isfinite(labels)
            if mask.sum() < 2:
                continue
            loaded[name] = {
                "pred": scores[mask].astype(float),
                "meas": labels[mask].astype(float),
            }
        if len(loaded) < 2:
            return None

        model_names = sorted(loaded.keys())
        colors = {m: model_color(m, model_names) for m in model_names}

        fig, (ax_bar, ax_roc, ax_pr) = plt.subplots(
            1, 3, figsize=(16, 5.2),
            gridspec_kw={"width_ratios": [1.05, 1.0, 1.0]},
        )

        # ── (1) bar: Pearson r + Spearman ρ ──
        groups = ["Pearson r", "Spearman ρ"]
        x = np.arange(len(groups))
        width = 0.8 / len(model_names)
        bar_vals: list[float] = []
        for i, m in enumerate(model_names):
            p, q = loaded[m]["pred"], loaded[m]["meas"]
            vals = [
                float(pearsonr(p, q).statistic),
                float(spearmanr(p, q).statistic),
            ]
            bar_vals.extend(vals)
            offset = (i - (len(model_names) - 1) / 2) * width
            ax_bar.bar(
                x + offset, vals, width=width, label=m, color=colors[m],
            )
        # Set ylim *before* annotating so the label offset is meaningful.
        vmin, vmax = min(bar_vals + [0.0]), max(bar_vals + [0.0])
        span = max(vmax - vmin, 0.05)
        pad = 0.18 * span    # leaves room for value labels above the tallest bar
        ax_bar.set_ylim(vmin - pad, vmax + pad)
        for i, m in enumerate(model_names):
            offset = (i - (len(model_names) - 1) / 2) * width
            vals = bar_vals[i * len(groups) : (i + 1) * len(groups)]
            for xi, v in zip(x + offset, vals):
                ax_bar.text(
                    xi, v + (0.02 * span if v >= 0 else -0.02 * span),
                    f"{v:+.3f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8,
                )
        ax_bar.axhline(0, color="grey", lw=0.5)
        ax_bar.set_xticks(x); ax_bar.set_xticklabels(groups)
        ax_bar.set_ylabel("correlation")
        ax_bar.set_title("Continuous-axis correlations", pad=8)
        ax_bar.legend(loc="best", fontsize=9)

        # ── (2) ROC + (3) PR, both tails overlaid ──
        # tail-name → (label, linestyle, y-mask fn, discriminant sign)
        tails = (
            ("extreme-low",  "--", lambda y: (y < EXTREME_LOW_CUTOFF),  -1),
            ("extreme-high", "-",  lambda y: (y >= EXTREME_HIGH_CUTOFF), +1),
        )
        # Track per-tail base rates so we can draw PR-baseline lines once.
        first = loaded[model_names[0]]["meas"]
        base_rates = {
            "extreme-low":  float((first < EXTREME_LOW_CUTOFF).mean()),
            "extreme-high": float((first >= EXTREME_HIGH_CUTOFF).mean()),
        }

        for m in model_names:
            pred, meas = loaded[m]["pred"], loaded[m]["meas"]
            for tail_name, ls, mask_fn, sign in tails:
                y = mask_fn(meas).astype(int)
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                disc = sign * pred
                fpr, tpr, _ = roc_curve(y, disc)
                prec, rec, _ = precision_recall_curve(y, disc)
                au_roc = roc_auc_score(y, disc)
                au_pr = average_precision_score(y, disc)
                ax_roc.plot(
                    fpr, tpr, color=colors[m], linestyle=ls, lw=1.4,
                    label=f"{m} · {tail_name} (AUROC {au_roc:.3f})",
                )
                ax_pr.plot(
                    rec, prec, color=colors[m], linestyle=ls, lw=1.4,
                    label=f"{m} · {tail_name} (AUPRC {au_pr:.3f})",
                )

        ax_roc.plot([0, 1], [0, 1], color="grey", lw=1.0, ls=":", label="random")
        ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1)
        ax_roc.set_aspect("equal", "box")
        ax_roc.set_xlabel("False positive rate"); ax_roc.set_ylabel("True positive rate")
        ax_roc.set_title("ROC — extreme-low (dashed), extreme-high (solid)", pad=8)
        ax_roc.legend(loc="lower right", fontsize=7)

        for tail_name, ls, _, _ in tails:
            br = base_rates[tail_name]
            ax_pr.axhline(
                br, color="grey", lw=1.0, ls=ls, alpha=0.7,
                label=f"random {tail_name} (base = {br:.3f})",
            )
        ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.02)
        ax_pr.set_aspect("equal", "box")
        ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Precision–Recall — same overlay scheme", pad=8)
        ax_pr.legend(loc="upper right", fontsize=7)

        fig.suptitle(self.compare_plot_title(), fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = out_dir / "plot.svg"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
