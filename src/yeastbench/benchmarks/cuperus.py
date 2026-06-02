"""Cuperus et al. 2017 5'-UTR MPRA benchmark.

Zero-shot prediction of growth-based 5'-UTR expression in the fixed
``CYC1``pr - [50 bp UTR] - ``HIS3`` - ``CYC1``term reporter (no
marginalization — the construct is native yeast sequence). Two libraries
are scored separately (their labels are from different selection
experiments and aren't comparable in absolute scale):

- **random** — all 489,348 sequences; metric 1 reported overall and per
  input-read-depth bucket (1-5).
- **native** — 11,856 real yeast 5'-UTR fragments; metric 1 overall + a
  ``t0 >= 10`` robustness split.

Metric 1: Spearman ρ (headline) + Pearson r between the model's score and
``growth_rate``. Metric 2: the model's signal beyond translation-only
(Kozak) features — cross-validated partial correlation + incremental R²,
computed within the clean depth bucket (``t0 >= 101``).

See ``benchmarks/cuperus_mpra_5utr.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from yeastbench.adapters.protocols import FivePrimeUtrReporterExpressionPredictor
from yeastbench.benchmarks._cuperus_metrics import kozak_design, metric1, metric2
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo

CLEAN_MIN_T0 = 101         # clean bucket (= paper top-5% threshold); metric-2 home
NATIVE_NOISY_MAX_T0 = 10   # native robustness split: t0<10 (noisy) vs t0>=10


@dataclass(frozen=True)
class CuperusLibraryResult:
    name: str
    utrs: list[str]
    scores: np.ndarray        # (N,) model score g; NaN allowed
    labels: np.ndarray        # (N,) growth_rate
    t0: np.ndarray            # (N,) input read depth
    buckets: np.ndarray       # (N,) depth bucket (random: 1-5; native: 1=noisy/2=rest)
    metric1: dict[str, Any]   # {"overall": {...}, "by_bucket": {b: {...}}}
    metric2: dict[str, Any]   # clean-bucket CV partial corr + incremental R²


@dataclass(frozen=True)
class CuperusResults:
    random: CuperusLibraryResult
    native: CuperusLibraryResult


def _eval_library(
    name: str,
    utrs: list[str],
    scores: np.ndarray,
    labels: np.ndarray,
    t0: np.ndarray,
    buckets: np.ndarray,
) -> CuperusLibraryResult:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    t0 = np.asarray(t0)
    buckets = np.asarray(buckets)
    m1 = metric1(scores, labels, buckets=buckets)
    clean = t0 >= CLEAN_MIN_T0
    f = kozak_design(utrs)
    m2 = metric2(scores[clean], labels[clean], f[clean])
    return CuperusLibraryResult(
        name=name, utrs=list(utrs), scores=scores, labels=labels,
        t0=t0, buckets=buckets, metric1=m1, metric2=m2,
    )


def _native_buckets(t0: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(t0) < NATIVE_NOISY_MAX_T0, 1, 2)


class CuperusUTRBenchmark(
    Benchmark[FivePrimeUtrReporterExpressionPredictor, CuperusResults]
):
    adapter_protocol: ClassVar[type] = FivePrimeUtrReporterExpressionPredictor

    def __init__(
        self,
        random_path: Path,
        native_path: Path,
        info: BenchmarkInfo,
        fasta_path: Path | None = None,
    ) -> None:
        self.random_path = Path(random_path)
        self.native_path = Path(native_path)
        # Carried so the registry can forward it to the model adapter (which
        # builds the construct against the genome); the benchmark itself is
        # model-independent and never reads it.
        self.fasta_path = Path(fasta_path) if fasta_path is not None else None
        self.info = info
        self._random = pd.read_csv(self.random_path, sep="\t")
        self._native = pd.read_csv(self.native_path, sep="\t")
        for col in ("UTR", "growth_rate", "t0"):
            assert col in self._random.columns, f"{col} missing from {self.random_path}"
            assert col in self._native.columns, f"{col} missing from {self.native_path}"
        assert "depth_bucket" in self._random.columns, (
            f"depth_bucket missing from {self.random_path} (run build_distribution.py)"
        )

    # ── evaluation ───────────────────────────────────────────────────────
    def _eval(
        self,
        adapter: FivePrimeUtrReporterExpressionPredictor,
        df: pd.DataFrame,
        name: str,
        buckets: np.ndarray,
    ) -> CuperusLibraryResult:
        utrs = df["UTR"].astype(str).tolist()
        scores = np.asarray(adapter.predict_utr_expressions(utrs), dtype=float)
        assert len(scores) == len(utrs), (
            f"adapter returned {len(scores)} scores for {len(utrs)} {name} UTRs"
        )
        return _eval_library(
            name, utrs, scores,
            df["growth_rate"].to_numpy(dtype=float),
            df["t0"].to_numpy(), buckets,
        )

    def evaluate(
        self, adapter: FivePrimeUtrReporterExpressionPredictor
    ) -> CuperusResults:
        rand = self._eval(
            adapter, self._random, "random",
            self._random["depth_bucket"].to_numpy(),
        )
        nat = self._eval(
            adapter, self._native, "native",
            _native_buckets(self._native["t0"].to_numpy()),
        )
        return CuperusResults(random=rand, native=nat)

    # ── persistence ──────────────────────────────────────────────────────
    @staticmethod
    def _save_lib(lib: CuperusLibraryResult, path: Path) -> None:
        pd.DataFrame({
            "UTR": lib.utrs,
            "score": lib.scores,
            "growth_rate": lib.labels,
            "t0": lib.t0,
            "bucket": lib.buckets,
        }).to_csv(path, sep="\t", index=False)

    @staticmethod
    def _load_lib(path: Path, name: str) -> CuperusLibraryResult:
        df = pd.read_csv(path, sep="\t")
        return _eval_library(
            name, df["UTR"].astype(str).tolist(),
            df["score"].to_numpy(dtype=float),
            df["growth_rate"].to_numpy(dtype=float),
            df["t0"].to_numpy(), df["bucket"].to_numpy(),
        )

    def save_results(self, results: CuperusResults, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._save_lib(results.random, out_dir / "random.tsv")
        self._save_lib(results.native, out_dir / "native.tsv")

    def load_results(self, out_dir: Path) -> CuperusResults:
        out_dir = Path(out_dir)
        return CuperusResults(
            random=self._load_lib(out_dir / "random.tsv", "random"),
            native=self._load_lib(out_dir / "native.tsv", "native"),
        )

    # ── summary / headline ───────────────────────────────────────────────
    @staticmethod
    def _lib_summary(lib: CuperusLibraryResult, prefix: str) -> dict[str, Any]:
        o = lib.metric1["overall"]
        d: dict[str, Any] = {
            f"{prefix}_n": o["n"],
            f"{prefix}_spearman": o["spearman"],
            f"{prefix}_pearson": o["pearson"],
            f"{prefix}_metric2_n": lib.metric2["n"],
            f"{prefix}_metric2_partial_spearman": lib.metric2["partial_spearman"],
            f"{prefix}_metric2_partial_pearson": lib.metric2["partial_pearson"],
            f"{prefix}_metric2_incremental_r2": lib.metric2["incremental_r2"],
        }
        for b, r in lib.metric1.get("by_bucket", {}).items():
            d[f"{prefix}_bucket{b}_spearman"] = r["spearman"]
            d[f"{prefix}_bucket{b}_n"] = r["n"]
        return d

    def summary_dict(self, results: CuperusResults) -> dict[str, Any]:
        return {
            **self._lib_summary(results.random, "random"),
            **self._lib_summary(results.native, "native"),
        }

    def headline(self, results: CuperusResults) -> str:
        r, n = results.random, results.native
        r_clean = r.metric1.get("by_bucket", {}).get(5, {}).get("spearman", float("nan"))
        n_rest = n.metric1.get("by_bucket", {}).get(2, {}).get("spearman", float("nan"))
        return (
            f"random: ρ={r.metric1['overall']['spearman']:.4f} "
            f"(clean ρ={r_clean:.4f}, metric2 partial ρ={r.metric2['partial_spearman']:.4f}) "
            f"| native: ρ={n.metric1['overall']['spearman']:.4f} "
            f"(t0≥10 ρ={n_rest:.4f}, metric2 partial ρ={n.metric2['partial_spearman']:.4f})"
        )

    def headline_metric_labels(self) -> dict[str, str]:
        return {
            "random_bucket5_spearman": "random clean ρ",
            "random_metric2_partial_spearman": "random metric2 ρ",
            "native_spearman": "native ρ",
            "native_metric2_partial_spearman": "native metric2 ρ",
        }

    def compare_plot_title(self) -> str:
        return "Cuperus et al. 5′-UTR reporter expression"

    # ── plots ────────────────────────────────────────────────────────────
    def plot(self, results: CuperusResults, out_dir: Path) -> None:
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        title_model = out_dir.name.split("__")[0] if "__" in out_dir.name else ""

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        self._scatter(axes[0], results.random, clean_only=True,
                      title="random — clean bucket (t0≥101)")
        self._bucket_bar(axes[1], results.random,
                         title="random — Spearman ρ by depth bucket")
        self._scatter(axes[2], results.native, clean_only=True,
                      title="native — clean bucket (t0≥101)")
        suptitle = self.info.name + (f" — {title_model}" if title_model else "")
        fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_dir / "cuperus.png", dpi=150)
        plt.close(fig)

    @staticmethod
    def _scatter(ax, lib: CuperusLibraryResult, clean_only: bool, title: str) -> None:
        mask = np.isfinite(lib.scores) & np.isfinite(lib.labels)
        if clean_only:
            mask &= lib.t0 >= CLEAN_MIN_T0
        x = lib.labels[mask]
        y = lib.scores[mask]
        ax.scatter(x, y, s=6, alpha=0.3, rasterized=True)
        if mask.sum() > 1 and np.std(x) > 0:
            a, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, a * xs + b, color="red", lw=1, alpha=0.8)
        from yeastbench.benchmarks._cuperus_metrics import _corr
        c = _corr(y, x)
        ax.set_xlabel("growth_rate")
        ax.set_ylabel("model score g")
        ax.set_title(f"{title}\nn={c['n']}  ρ={c['spearman']:.3f}  r={c['pearson']:.3f}",
                     fontsize=10)

    @staticmethod
    def _bucket_bar(ax, lib: CuperusLibraryResult, title: str) -> None:
        bb = lib.metric1.get("by_bucket", {})
        keys = sorted(bb)
        vals = [bb[k]["spearman"] for k in keys]
        ax.bar([str(k) for k in keys], vals, color="steelblue")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("depth bucket")
        ax.set_ylabel("Spearman ρ")
        ax.set_title(title, fontsize=10)
