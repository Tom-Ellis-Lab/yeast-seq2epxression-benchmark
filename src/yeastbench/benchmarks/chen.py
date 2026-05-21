"""Chen et al. 2017 synonymous-mutation MPRA benchmark.

One benchmark class, parameterised by ``library`` — registered three
times in ``yeastbench.registry`` as ``chen_gfp_r1`` / ``chen_gfp_r2`` /
``chen_tdh3``. All three share ``compare_task_name = "chen_synonymous"``
so the cross-model compare runner groups them into one panel.

For libraries with two normalised mRNA columns (GFP r1, GFP r2) we
report **Pearson and Spearman separately for each replicate**, never a
pre-averaged label, so we can compare both numbers against the
published replicate-replicate ceiling.

See ``benchmarks/chen_synonymous.md`` for the spec.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from yeastbench.adapters.protocols import LocalCodingVariantPredictor
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo

VARIABLE_LEN = 36

TWO_REPLICATE_LIBS = {"gfp_r1", "gfp_r2"}


@dataclass(frozen=True)
class ChenResults:
    library_id: str
    variant_ids: np.ndarray              # (N,) object array of variant IDs
    scores: np.ndarray                   # (N,) predicted scalar per variant
    label_columns: tuple[str, ...]       # ("log2mRNA_rep1", "log2mRNA_rep2") or ("log2mRNA",)
    labels: np.ndarray                   # (N, len(label_columns)) — per-replicate normalised log2(mRNA)
    ceiling_pearson: float               # published replicate-replicate ceiling


class ChenSynonymousBenchmark(Benchmark[LocalCodingVariantPredictor, ChenResults]):
    adapter_protocol: ClassVar[type] = LocalCodingVariantPredictor

    def __init__(
        self,
        library: str,
        data_path: Path,
        fasta_path: Path,
        gtf_path: Path,
        library_loci_path: Path,
        replicate_ceiling_pearson: float,
        info: BenchmarkInfo,
    ) -> None:
        if library not in {"gfp_r1", "gfp_r2", "tdh3"}:
            raise ValueError(f"unknown Chen library: {library!r}")
        self.library = library
        self.data_path = Path(data_path)
        self._fasta_path = Path(fasta_path)
        self._gtf_path = Path(gtf_path)
        self._library_loci_path = Path(library_loci_path)
        self.ceiling_pearson = float(replicate_ceiling_pearson)
        self.info = info

        df = pd.read_csv(self.data_path, sep="\t")
        if not (df["variable_seq"].str.len() == VARIABLE_LEN).all():
            raise ValueError(
                f"{self.data_path}: not all variable_seq are {VARIABLE_LEN} nt"
            )
        self.variant_ids: np.ndarray = df["variant_id"].to_numpy()
        self.variant_seqs: list[str] = df["variable_seq"].astype(str).str.upper().tolist()

        if library in TWO_REPLICATE_LIBS:
            self.label_columns = ("log2mRNA_rep1", "log2mRNA_rep2")
        else:
            self.label_columns = ("log2mRNA",)
        self.labels: np.ndarray = df[list(self.label_columns)].to_numpy(dtype=float)

    @property
    def fasta_path(self) -> Path:
        return self._fasta_path

    @property
    def gtf_path(self) -> Path:
        return self._gtf_path

    @property
    def library_loci_path(self) -> Path:
        return self._library_loci_path

    # We intentionally do *not* override ``compare_task_name``: each Chen
    # library is its own compare group (chen_gfp_r1 / chen_gfp_r2 /
    # chen_tdh3). Grouping all three under a single ``chen_synonymous``
    # name confuses the cross-model compare runner because the same
    # model appears in three sub-tasks of one group, and the runner's
    # one-model-per-group assumption rejects all but the first. A single
    # 3-panel compare figure is a v2 enhancement (custom compare_plot
    # override).

    def evaluate(self, adapter: LocalCodingVariantPredictor) -> ChenResults:
        library_ids = [self.library] * len(self.variant_seqs)
        scores = np.asarray(
            adapter.predict_local_variants(library_ids, self.variant_seqs),
            dtype=float,
        )
        if len(scores) != len(self.variant_seqs):
            raise ValueError(
                f"adapter returned {len(scores)} scores for "
                f"{len(self.variant_seqs)} variants"
            )
        return ChenResults(
            library_id=self.library,
            variant_ids=self.variant_ids,
            scores=scores,
            label_columns=self.label_columns,
            labels=self.labels,
            ceiling_pearson=self.ceiling_pearson,
        )

    def _per_column_stats(self, results: ChenResults) -> list[dict[str, float]]:
        out = []
        for j, col in enumerate(results.label_columns):
            meas = results.labels[:, j]
            mask = np.isfinite(results.scores) & np.isfinite(meas)
            n = int(mask.sum())
            if n < 2:
                out.append({"column": col, "n": n,
                            "pearson": float("nan"), "spearman": float("nan")})
                continue
            p = results.scores[mask]
            m = meas[mask]
            out.append({
                "column": col,
                "n": n,
                "pearson": float(pearsonr(p, m).statistic),
                "spearman": float(spearmanr(p, m).statistic),
            })
        return out

    def plot(self, results: ChenResults, out_dir: Path) -> None:
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        title_model = out_dir.name.split("__")[0] if "__" in out_dir.name else ""

        stats = self._per_column_stats(results)
        n_panels = len(results.label_columns)
        fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.0), squeeze=False)

        for j, col in enumerate(results.label_columns):
            ax = axes[0, j]
            meas = results.labels[:, j]
            mask = np.isfinite(results.scores) & np.isfinite(meas)
            pred, m = results.scores[mask], meas[mask]
            ax.scatter(m, pred, s=4, alpha=0.4, rasterized=True)
            if len(m) > 1:
                slope, b = np.polyfit(m, pred, 1)
                xs = np.linspace(m.min(), m.max(), 50)
                ax.plot(xs, slope * xs + b, color="red", linewidth=1, alpha=0.8)
            ax.set_xlabel(f"measured {col}")
            ax.set_ylabel("predicted (adapter scalar)")
            s = stats[j]
            ax.set_title(
                f"{col}\n"
                f"n = {s['n']}  r = {s['pearson']:.3f}  "
                f"ρ = {s['spearman']:.3f}  "
                f"ceiling r ≤ {results.ceiling_pearson:.2f}",
                fontsize=10,
            )

        sup = f"Chen synonymous MPRA — {results.library_id}"
        if title_model:
            sup += f"  ({title_model})"
        fig.suptitle(sup, fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "scatter.png", dpi=150)
        plt.close(fig)

    def save_results(self, results: ChenResults, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "scores.npy", results.scores)
        np.save(out_dir / "labels.npy", results.labels)
        (out_dir / "results_meta.json").write_text(json.dumps({
            "library_id": results.library_id,
            "label_columns": list(results.label_columns),
            "ceiling_pearson": results.ceiling_pearson,
            "variant_ids": [str(v) for v in results.variant_ids],
        }))

    def load_results(self, out_dir: Path) -> ChenResults:
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "results_meta.json").read_text())
        return ChenResults(
            library_id=meta["library_id"],
            variant_ids=np.array(meta["variant_ids"], dtype=object),
            scores=np.load(out_dir / "scores.npy"),
            label_columns=tuple(meta["label_columns"]),
            labels=np.load(out_dir / "labels.npy"),
            ceiling_pearson=float(meta["ceiling_pearson"]),
        )

    def summary_dict(self, results: ChenResults) -> dict[str, Any]:
        stats = self._per_column_stats(results)
        summary: dict[str, Any] = {
            "library_id": results.library_id,
            "n_rows_total": int(len(results.scores)),
            "ceiling_pearson": results.ceiling_pearson,
        }
        if results.library_id in TWO_REPLICATE_LIBS:
            for s in stats:
                rep = s["column"].split("_")[-1]   # rep1 / rep2
                summary[f"pearson_{rep}"]  = s["pearson"]
                summary[f"spearman_{rep}"] = s["spearman"]
                summary[f"n_{rep}"]        = s["n"]
        else:
            s = stats[0]
            summary["pearson"]  = s["pearson"]
            summary["spearman"] = s["spearman"]
            summary["n_scored"] = s["n"]
        return summary

    def headline(self, results: ChenResults) -> str:
        stats = self._per_column_stats(results)
        if len(stats) == 1:
            s = stats[0]
            return (
                f"{results.library_id}: Pearson r = {s['pearson']:.4f}  "
                f"Spearman ρ = {s['spearman']:.4f}  "
                f"(n = {s['n']}, ceiling r ≤ {results.ceiling_pearson:.2f})"
            )
        parts = []
        for s in stats:
            rep = s["column"].split("_")[-1]
            parts.append(f"{rep} r={s['pearson']:.3f} ρ={s['spearman']:.3f}")
        return (
            f"{results.library_id}: "
            + "  ".join(parts)
            + f"  (n = {stats[0]['n']}, ceiling r ≤ {results.ceiling_pearson:.2f})"
        )

    def headline_metric_labels(self) -> dict[str, str]:
        if self.library in TWO_REPLICATE_LIBS:
            return {
                "pearson_rep1":  "Pearson r (rep1)",
                "pearson_rep2":  "Pearson r (rep2)",
                "spearman_rep1": "Spearman ρ (rep1)",
                "spearman_rep2": "Spearman ρ (rep2)",
            }
        return {
            "pearson":  "Pearson r",
            "spearman": "Spearman ρ",
        }

    def compare_plot_title(self) -> str:
        return f"Chen synonymous MPRA — {self.library}"
