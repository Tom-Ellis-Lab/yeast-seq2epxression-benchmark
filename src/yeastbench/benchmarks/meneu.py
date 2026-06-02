"""Meneu et al. foreign-DNA tiled-coverage benchmark.

Zero-shot prediction of RNA-seq-like coverage over whole bacterial
chromosomes (Mpneumo, Mmmyco) introduced into yeast (Meneu et al.,
Science 2025). For each contig the benchmark tiles fixed-stride windows
across the sequence, predicts the central ``seq_len - 2*crop`` region of
each window, and stitches the per-window predictions into one per-base
contig-length profile. Predicted coverage is then scored against the
measured unstranded coverage (``fwd + rev``) in non-overlapping
``EVAL_WINDOW``-bp evaluation windows.

Three metrics per contig, all on the windows whose *true* signal clears
a small variance floor (near-flat true windows are uninformative for
shape):

  * ``shape_pearson`` — median over kept windows of the RAW (scale-
    invariant) Pearson correlation between true and predicted per-base
    coverage.
  * ``shape_js`` — median over kept windows of the Jensen-Shannon
    divergence (bits) between the sum-1-normalised true and predicted
    profiles.
  * ``mag_fc_mean`` / ``mag_fc_sd`` — mean and SD of the per-window
    log2 fold-change error of the (genome-wide depth-normalised)
    predicted window total vs the true window total.

**Units.** Adapters return raw per-base predicted-count units (any
model-specific training transform inverted inside the adapter) and
**unstranded** coverage; the truth is the raw per-base ``fwd + rev``
Nanopore/Illumina pileup. Genome-wide depth normalisation is applied to
the prediction only for the magnitude metric; the shape metrics are
scale-invariant.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from yeastbench.adapters.protocols import TiledCoverageTrackPredictor
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo
from yeastbench.benchmarks.brooks import _js_divergence

# Per-contig coverage truth lives next to the TSV as
# ``meneu_cov_<contig>.npz`` with arrays 'fwd' and 'rev'.
COV_PREFIX = "meneu_cov_"

_PER_CONTIG_KEYS = (
    "shape_pearson", "shape_js", "mag_fc_mean", "mag_fc_sd",
    "n_windows_kept", "n_windows_total",
)


@dataclass(frozen=True)
class MeneuResults:
    window_len: int
    contigs: list[str]
    # contig -> {shape_pearson, shape_js, mag_fc_mean, mag_fc_sd,
    #            n_windows_kept, n_windows_total}
    per_contig: dict[str, dict]
    # contig -> per-base float64 stitched prediction, length == contig L
    stitched_pred: dict[str, np.ndarray]


def _normalize(x: np.ndarray) -> np.ndarray:
    """Sum-to-1 normalise, guarding an all-zero vector (use x+1e-12)."""
    s = x.sum()
    if s <= 0:
        x = x + 1e-12
        s = x.sum()
    return x / s


class MeneuForeignDNABenchmark(
    Benchmark[TiledCoverageTrackPredictor, MeneuResults]
):
    adapter_protocol: ClassVar[type] = TiledCoverageTrackPredictor

    EVAL_WINDOW: ClassVar[int] = 5000
    FLOOR_EPS: ClassVar[float] = 1e-9

    def __init__(self, data_path: Path, info: BenchmarkInfo) -> None:
        self.data_path = Path(data_path)
        self.info = info
        self.data_dir = self.data_path.parent
        df = pd.read_csv(self.data_path, sep="\t")
        for col in ("tile_id", "chrom", "strain", "window_start",
                    "center_start", "window_len", "crop_bp_each_side",
                    "seq"):
            assert col in df.columns, f"{col} missing from {self.data_path}"
        window_len = int(df.window_len.iloc[0])
        assert (df.window_len == window_len).all(), (
            f"all rows of {self.data_path} must share window_len"
        )
        assert (df.seq.str.len() == window_len).all(), (
            f"all seq strings must be {window_len} bp"
        )
        self.window_len = window_len
        self.df = df.reset_index(drop=True)

    def _run_batched(
        self,
        adapter: TiledCoverageTrackPredictor,
        seqs: list[str],
        strands: list[str],
        strains: list[str | None],
        *,
        desc: str,
    ) -> np.ndarray:
        """Chunk inputs into adapter-sized batches and call
        ``predict_coverage_batch`` on each. Returns ``(len(seqs), out_len)``."""
        from tqdm import tqdm

        if not seqs:
            return np.empty((0, 0), dtype=np.float64)
        bs = max(1, int(getattr(adapter, "batch_size", 1)))
        out_chunks: list[np.ndarray] = []
        for start in tqdm(range(0, len(seqs), bs), desc=desc, ncols=80):
            end = min(start + bs, len(seqs))
            arr = adapter.predict_coverage_batch(
                seqs=seqs[start:end],
                strands=strands[start:end],
                strains=strains[start:end],
            )
            out_chunks.append(np.asarray(arr, dtype=np.float64))
        return np.concatenate(out_chunks, axis=0)

    def _load_truth(self, contig: str) -> np.ndarray:
        path = self.data_dir / f"{COV_PREFIX}{contig}.npz"
        with np.load(path) as d:
            return (d["fwd"].astype(np.float64) + d["rev"].astype(np.float64))

    def _score_contig(
        self, true: np.ndarray, pred: np.ndarray
    ) -> dict:
        """Per-contig metric, ported exactly from the validated spec."""
        W = self.EVAL_WINDOW
        L = len(true)
        n = L // W
        t = true[: n * W].reshape(n, W)
        # Depth-normalise pred ONLY for the magnitude metric.
        scale = (true.sum() / pred.sum()) if pred.sum() > 0 else 1.0
        pdn = (pred * scale)[: n * W].reshape(n, W)
        p = pred[: n * W].reshape(n, W)
        keep = t.var(axis=1) > self.FLOOR_EPS
        idx = np.where(keep)[0]

        pear = np.full(len(idx), np.nan)
        jsd = np.full(len(idx), np.nan)
        for j, i in enumerate(idx):
            if p[i].sum() == 0:
                # Pearson undefined; JS uses the +1e-12 guard.
                jsd[j] = _js_divergence(_normalize(t[i]), _normalize(p[i]))
                continue
            pear[j] = float(pearsonr(t[i], p[i]).statistic)
            jsd[j] = _js_divergence(_normalize(t[i]), _normalize(p[i]))

        shape_pearson = (float(np.nanmedian(pear))
                         if len(idx) and np.any(np.isfinite(pear))
                         else float("nan"))
        shape_js = float(np.median(jsd)) if len(idx) else float("nan")
        if len(idx):
            fc = np.log2((pdn[idx].sum(axis=1) + 1) / (t[idx].sum(axis=1) + 1))
            mag_fc_mean = float(fc.mean())
            mag_fc_sd = float(fc.std())
        else:
            mag_fc_mean = mag_fc_sd = float("nan")
        return {
            "shape_pearson": shape_pearson,
            "shape_js": shape_js,
            "mag_fc_mean": mag_fc_mean,
            "mag_fc_sd": mag_fc_sd,
            "n_windows_kept": int(len(idx)),
            "n_windows_total": int(n),
        }

    def evaluate(self, adapter: TiledCoverageTrackPredictor) -> MeneuResults:
        assert adapter.seq_len == self.window_len, (
            f"adapter seq_len {adapter.seq_len} != distribution window_len "
            f"{self.window_len}; pick a TSV that matches the model "
            "(`meneu_foreign_dna_v1.tsv` for Yorzoi @ 4992, "
            "`meneu_foreign_dna_v1_w16384.tsv` for Shorkie @ 16384)."
        )
        crop = adapter.crop_bp_each_side
        out_len = adapter.seq_len - 2 * crop

        contigs = list(self.df.chrom.drop_duplicates())
        per_contig: dict[str, dict] = {}
        stitched_pred: dict[str, np.ndarray] = {}

        for contig in contigs:
            g = self.df[self.df.chrom == contig].sort_values("center_start")
            true = self._load_truth(contig)
            L = len(true)
            pred = np.zeros(L, dtype=np.float64)

            tile_pred = self._run_batched(
                adapter,
                seqs=g.seq.tolist(),
                strands=["+"] * len(g),
                strains=[None] * len(g),
                desc=f"{contig} (n={len(g)})",
            )
            assert tile_pred.shape == (len(g), out_len), (
                f"adapter returned {tile_pred.shape}, expected "
                f"({len(g)}, {out_len})"
            )

            for row_i, center_start in enumerate(g.center_start.to_numpy()):
                cs = int(center_start)
                ce = min(cs + out_len, L)
                pred[cs:ce] = tile_pred[row_i, : ce - cs]

            per_contig[contig] = self._score_contig(true, pred)
            stitched_pred[contig] = pred

        return MeneuResults(
            window_len=self.window_len,
            contigs=contigs,
            per_contig=per_contig,
            stitched_pred=stitched_pred,
        )

    def plot(self, results: MeneuResults, out_dir: Path) -> None:
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        title_model = out_dir.name.split("__")[0] if "__" in out_dir.name else ""
        W = self.EVAL_WINDOW

        for contig in results.contigs:
            pred = results.stitched_pred[contig]
            true = self._load_truth(contig)
            L = min(len(pred), len(true))
            pred = pred[:L]
            true = true[:L]
            pc = results.per_contig[contig]

            fig, axes = plt.subplots(2, 1, figsize=(10, 6))
            # (1) Downsampled per-base coverage, true vs pred (depth-scaled).
            n = L // W
            if n > 0:
                t_w = true[: n * W].reshape(n, W).sum(axis=1)
                scale = (true.sum() / pred.sum()) if pred.sum() > 0 else 1.0
                p_w = (pred * scale)[: n * W].reshape(n, W).sum(axis=1)
                x = np.arange(n) * W
                axes[0].plot(x, t_w, lw=0.6, color="#444444", label="true")
                axes[0].plot(x, p_w, lw=0.6, color="#d62728", alpha=0.8,
                             label="pred (depth-scaled)")
                axes[0].set_xlabel("contig position (bp)")
                axes[0].set_ylabel(f"coverage / {W} bp")
                axes[0].legend(loc="upper right", fontsize=8)
            axes[0].set_title(
                f"Meneu {contig}"
                + (f" — {title_model}" if title_model else "")
                + f"   shape r={pc['shape_pearson']:.3f}  "
                f"JS={pc['shape_js']:.3f}  "
                f"mag_fc={pc['mag_fc_mean']:+.3f}±{pc['mag_fc_sd']:.3f}  "
                f"(kept {pc['n_windows_kept']}/{pc['n_windows_total']})"
            )

            # (2) Per-window Pearson histogram over the kept windows.
            if n > 0:
                t = true[: n * W].reshape(n, W)
                scale = (true.sum() / pred.sum()) if pred.sum() > 0 else 1.0
                p = pred[: n * W].reshape(n, W)
                keep = t.var(axis=1) > self.FLOOR_EPS
                idx = np.where(keep)[0]
                rs = []
                for i in idx:
                    if p[i].sum() == 0:
                        continue
                    rs.append(float(pearsonr(t[i], p[i]).statistic))
                if rs:
                    axes[1].hist(rs, bins=30, color="#1f77b4", alpha=0.8)
                    axes[1].axvline(pc["shape_pearson"], color="#d62728",
                                    ls="--", lw=1.0,
                                    label=f"median {pc['shape_pearson']:.3f}")
                    axes[1].legend(loc="upper left", fontsize=8)
            axes[1].set_xlabel("per-window Pearson r (true vs pred)")
            axes[1].set_ylabel("count")

            fig.tight_layout()
            fig.savefig(out_dir / f"{contig}.png", dpi=120)
            plt.close(fig)

    def save_results(self, results: MeneuResults, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / "stitched_pred.npz",
            **{c: results.stitched_pred[c] for c in results.contigs},
        )
        (out_dir / "results.json").write_text(json.dumps({
            "window_len": results.window_len,
            "contigs": results.contigs,
            "per_contig": results.per_contig,
        }, indent=2))

    def load_results(self, out_dir: Path) -> MeneuResults:
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "results.json").read_text())
        contigs = list(meta["contigs"])
        with np.load(out_dir / "stitched_pred.npz") as d:
            stitched_pred = {c: d[c].astype(np.float64) for c in contigs}
        return MeneuResults(
            window_len=int(meta["window_len"]),
            contigs=contigs,
            per_contig={c: dict(meta["per_contig"][c]) for c in contigs},
            stitched_pred=stitched_pred,
        )

    def summary_dict(self, results: MeneuResults) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for c in results.contigs:
            pc = results.per_contig[c]
            out[f"{c}_shape_pearson"] = pc["shape_pearson"]
            out[f"{c}_shape_js"] = pc["shape_js"]
            out[f"{c}_mag_fc_mean"] = pc["mag_fc_mean"]
            out[f"{c}_mag_fc_sd"] = pc["mag_fc_sd"]
            out[f"{c}_n_windows_kept"] = pc["n_windows_kept"]
            out[f"{c}_n_windows_total"] = pc["n_windows_total"]
        return out

    def headline(self, results: MeneuResults) -> str:
        parts = [
            f"{c} r={results.per_contig[c]['shape_pearson']:.3f}"
            for c in results.contigs
        ]
        return " ".join(parts)

    def headline_metric_labels(self) -> dict[str, str]:
        labels: dict[str, str] = {}
        for c in self.df.chrom.drop_duplicates():
            labels[f"{c}_shape_pearson"] = f"{c} shape r"
            labels[f"{c}_shape_js"] = f"{c} shape JS"
        return labels

    @property
    def compare_task_name(self) -> str:
        """Meneu ships two registry entries (`meneu_foreign_dna` for
        Yorzoi @ 4992, `meneu_foreign_dna_shorkie` for Shorkie @ 16384)
        because the two models need differently-sized windows. Cross-model
        comparisons treat them as one task — the canonical name is
        `meneu_foreign_dna`."""
        return "meneu_foreign_dna"
