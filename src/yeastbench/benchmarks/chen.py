"""Chen et al. 2017 synonymous-mutation MPRA benchmark.

One task, ``chen_synonymous``, that evaluates all three libraries (GFP r1,
GFP r2, TDH3) and reports results **stratified per library** plus a single
aggregate. Each library is scored by its own ``predict_local_variants``
call, so per-library numbers are identical to scoring the libraries
separately.

For libraries with two normalised mRNA columns (GFP r1, GFP r2) we report
**Pearson and Spearman separately for each replicate**, never a
pre-averaged label, so we can compare both numbers against the published
replicate-replicate ceiling. TDH3 ships a single merged ``log2mRNA``
column (and no Spearman ceiling).

See ``docs/benchmarks/chen_synonymous.md`` for the spec.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from yeastbench.adapters.protocols import LocalCodingVariantPredictor
from yeastbench.benchmarks.base import Benchmark, BenchmarkInfo

VARIABLE_LEN = 36

TWO_REPLICATE_LIBS = {"gfp_r1", "gfp_r2"}
_KNOWN_LIBS = {"gfp_r1", "gfp_r2", "tdh3"}
_DISPLAY = {"gfp_r1": "GFP r1", "gfp_r2": "GFP r2", "tdh3": "TDH3"}


@dataclass(frozen=True)
class ChenLibrary:
    """One library's measured data + comparison ceilings (loaded at init)."""
    library: str
    data_path: Path
    ceiling_pearson: float
    ceiling_spearman: float | None
    label_columns: tuple[str, ...]      # ("log2mRNA_rep1","log2mRNA_rep2") or ("log2mRNA",)
    variant_ids: np.ndarray             # (N,)
    variant_seqs: list[str]             # length N, 36-nt blocks
    labels: np.ndarray                  # (N, len(label_columns))


@dataclass(frozen=True)
class ChenLibraryResult:
    library_id: str
    variant_ids: np.ndarray             # (N,)
    scores: np.ndarray                  # (N,) predicted scalar per variant
    label_columns: tuple[str, ...]
    labels: np.ndarray                  # (N, len(label_columns))
    ceiling_pearson: float
    ceiling_spearman: float | None


@dataclass(frozen=True)
class ChenResults:
    libraries: tuple[str, ...]          # ordered library ids
    per_library: dict[str, ChenLibraryResult]


class ChenSynonymousBenchmark(Benchmark[LocalCodingVariantPredictor, ChenResults]):
    adapter_protocol: ClassVar[type] = LocalCodingVariantPredictor

    def __init__(
        self,
        libraries: Sequence[Mapping[str, Any]],
        fasta_path: Path,
        hosts_path: Path,
        data_dir: Path,
        info: BenchmarkInfo,
    ) -> None:
        if not libraries:
            raise ValueError("chen_synonymous needs at least one library")
        self._fasta_path = Path(fasta_path)
        self._hosts_path = Path(hosts_path)
        self._data_dir = Path(data_dir)
        self.info = info

        libs: list[ChenLibrary] = []
        for spec in libraries:
            name = spec["library"]
            if name not in _KNOWN_LIBS:
                raise ValueError(f"unknown Chen library: {name!r}")
            data_path = Path(spec["data_path"])
            ceiling_pearson = float(spec["replicate_ceiling_pearson"])
            ceiling_spearman = spec.get("replicate_ceiling_spearman")
            ceiling_spearman = (
                float(ceiling_spearman) if ceiling_spearman is not None else None
            )

            df = pd.read_csv(data_path, sep="\t")
            if not (df["variable_seq"].str.len() == VARIABLE_LEN).all():
                raise ValueError(
                    f"{data_path}: not all variable_seq are {VARIABLE_LEN} nt"
                )
            label_columns = (
                ("log2mRNA_rep1", "log2mRNA_rep2")
                if name in TWO_REPLICATE_LIBS else ("log2mRNA",)
            )
            libs.append(ChenLibrary(
                library=name,
                data_path=data_path,
                ceiling_pearson=ceiling_pearson,
                ceiling_spearman=ceiling_spearman,
                label_columns=label_columns,
                variant_ids=df["variant_id"].to_numpy(),
                variant_seqs=df["variable_seq"].astype(str).str.upper().tolist(),
                labels=df[list(label_columns)].to_numpy(dtype=float),
            ))
        self._libs = libs

    # Attributes read by the adapter dispatch (see registry.CHEN_FIELDS for the
    # genomic models; the baselines read ``libraries`` via from_task).
    @property
    def libraries(self) -> list[ChenLibrary]:
        return self._libs

    @property
    def fasta_path(self) -> Path:
        return self._fasta_path

    @property
    def hosts_path(self) -> Path:
        return self._hosts_path

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    def evaluate(self, adapter: LocalCodingVariantPredictor) -> ChenResults:
        per: dict[str, ChenLibraryResult] = {}
        for lib in self._libs:
            library_ids = [lib.library] * len(lib.variant_seqs)
            scores = np.asarray(
                adapter.predict_local_variants(library_ids, lib.variant_seqs),
                dtype=float,
            )
            if len(scores) != len(lib.variant_seqs):
                raise ValueError(
                    f"{lib.library}: adapter returned {len(scores)} scores for "
                    f"{len(lib.variant_seqs)} variants"
                )
            per[lib.library] = ChenLibraryResult(
                library_id=lib.library,
                variant_ids=lib.variant_ids,
                scores=scores,
                label_columns=lib.label_columns,
                labels=lib.labels,
                ceiling_pearson=lib.ceiling_pearson,
                ceiling_spearman=lib.ceiling_spearman,
            )
        return ChenResults(
            libraries=tuple(lib.library for lib in self._libs),
            per_library=per,
        )

    def _per_column_stats(self, result: ChenLibraryResult) -> list[dict[str, float]]:
        out = []
        for j, col in enumerate(result.label_columns):
            meas = result.labels[:, j]
            mask = np.isfinite(result.scores) & np.isfinite(meas)
            n = int(mask.sum())
            if n < 2:
                out.append({"column": col, "n": n,
                            "pearson": float("nan"), "spearman": float("nan")})
                continue
            p = result.scores[mask]
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

        # One panel per (library, replicate column).
        panels: list[tuple[str, int, str, ChenLibraryResult]] = []
        for lib in results.libraries:
            r = results.per_library[lib]
            for j, col in enumerate(r.label_columns):
                panels.append((lib, j, col, r))

        n_panels = len(panels)
        fig, axes = plt.subplots(
            1, n_panels, figsize=(5.0 * n_panels, 4.6), squeeze=False,
        )
        for k, (lib, j, col, r) in enumerate(panels):
            ax = axes[0, k]
            meas = r.labels[:, j]
            mask = np.isfinite(r.scores) & np.isfinite(meas)
            pred, m = r.scores[mask], meas[mask]
            ax.scatter(m, pred, s=4, alpha=0.4, rasterized=True)
            if len(m) > 1:
                slope, b = np.polyfit(m, pred, 1)
                xs = np.linspace(m.min(), m.max(), 50)
                ax.plot(xs, slope * xs + b, color="red", linewidth=1, alpha=0.8)
            s = self._per_column_stats(r)[j]
            ceiling_text = f"ceiling r ≤ {r.ceiling_pearson:.2f}"
            if r.ceiling_spearman is not None:
                ceiling_text += f", ρ ≤ {r.ceiling_spearman:.2f}"
            ax.set_xlabel(f"measured {col}")
            ax.set_ylabel("predicted (adapter scalar)")
            ax.set_title(
                f"{_DISPLAY.get(lib, lib)} · {col}\n"
                f"n = {s['n']}  r = {s['pearson']:.3f}  "
                f"ρ = {s['spearman']:.3f}  {ceiling_text}",
                fontsize=10,
            )

        sup = "Chen synonymous MPRA"
        if title_model:
            sup += f"  ({title_model})"
        fig.suptitle(sup, fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "scatter.png", dpi=150)
        plt.close(fig)

    def save_results(self, results: ChenResults, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {"libraries": list(results.libraries), "per_library": {}}
        for lib in results.libraries:
            r = results.per_library[lib]
            np.save(out_dir / f"scores_{lib}.npy", r.scores)
            np.save(out_dir / f"labels_{lib}.npy", r.labels)
            meta["per_library"][lib] = {
                "label_columns": list(r.label_columns),
                "ceiling_pearson": r.ceiling_pearson,
                "ceiling_spearman": r.ceiling_spearman,
                "variant_ids": [str(v) for v in r.variant_ids],
            }
        (out_dir / "results_meta.json").write_text(json.dumps(meta))

    def load_results(self, out_dir: Path) -> ChenResults:
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / "results_meta.json").read_text())
        per: dict[str, ChenLibraryResult] = {}
        for lib in meta["libraries"]:
            m = meta["per_library"][lib]
            ceiling_spearman = m.get("ceiling_spearman")
            per[lib] = ChenLibraryResult(
                library_id=lib,
                variant_ids=np.array(m["variant_ids"], dtype=object),
                scores=np.load(out_dir / f"scores_{lib}.npy"),
                label_columns=tuple(m["label_columns"]),
                labels=np.load(out_dir / f"labels_{lib}.npy"),
                ceiling_pearson=float(m["ceiling_pearson"]),
                ceiling_spearman=(
                    float(ceiling_spearman) if ceiling_spearman is not None else None
                ),
            )
        return ChenResults(libraries=tuple(meta["libraries"]), per_library=per)

    def summary_dict(self, results: ChenResults) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        rep_pearsons: list[float] = []
        rep_spearmans: list[float] = []
        n_rows_total = 0

        for lib in results.libraries:
            r = results.per_library[lib]
            stats = self._per_column_stats(r)
            n_rows_total += int(len(r.scores))
            summary[f"{lib}_ceiling_pearson"] = r.ceiling_pearson
            if r.ceiling_spearman is not None:
                summary[f"{lib}_ceiling_spearman"] = r.ceiling_spearman
            if lib in TWO_REPLICATE_LIBS:
                for s in stats:
                    rep = s["column"].split("_")[-1]   # rep1 / rep2
                    summary[f"{lib}_pearson_{rep}"] = s["pearson"]
                    summary[f"{lib}_spearman_{rep}"] = s["spearman"]
                    summary[f"{lib}_n_{rep}"] = s["n"]
                    rep_pearsons.append(s["pearson"])
                    rep_spearmans.append(s["spearman"])
            else:
                s = stats[0]
                summary[f"{lib}_pearson"] = s["pearson"]
                summary[f"{lib}_spearman"] = s["spearman"]
                summary[f"{lib}_n_scored"] = s["n"]
                rep_pearsons.append(s["pearson"])
                rep_spearmans.append(s["spearman"])

        # Aggregate over every replicate column (rep1+rep2 for the GFP libs, the
        # single column for TDH3). nan-aware so one missing replicate doesn't
        # poison the headline; suppress the all-NaN-slice RuntimeWarning.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            summary["pearson_mean"] = (
                float(np.nanmean(rep_pearsons)) if rep_pearsons else float("nan")
            )
            summary["spearman_mean"] = (
                float(np.nanmean(rep_spearmans)) if rep_spearmans else float("nan")
            )
        summary["n_rows_total"] = n_rows_total
        return summary

    def headline(self, results: ChenResults) -> str:
        s = self.summary_dict(results)
        parts = []
        for lib in results.libraries:
            disp = _DISPLAY.get(lib, lib)
            if lib in TWO_REPLICATE_LIBS:
                parts.append(
                    f"{disp} r={s[f'{lib}_pearson_rep1']:.3f}/{s[f'{lib}_pearson_rep2']:.3f}"
                )
            else:
                parts.append(f"{disp} r={s[f'{lib}_pearson']:.3f}")
        return (
            f"Chen synonymous: mean Pearson r = {s['pearson_mean']:.4f}  "
            f"(mean Spearman ρ = {s['spearman_mean']:.4f})  ·  "
            + "  ".join(parts)
        )

    def headline_metric_labels(self) -> dict[str, str]:
        labels: dict[str, str] = {}
        for lib in self._libs:
            disp = _DISPLAY.get(lib.library, lib.library)
            if lib.library in TWO_REPLICATE_LIBS:
                labels[f"{lib.library}_pearson_rep1"] = f"{disp} r (rep1)"
                labels[f"{lib.library}_pearson_rep2"] = f"{disp} r (rep2)"
                labels[f"{lib.library}_spearman_rep1"] = f"{disp} ρ (rep1)"
                labels[f"{lib.library}_spearman_rep2"] = f"{disp} ρ (rep2)"
            else:
                labels[f"{lib.library}_pearson"] = f"{disp} r"
                labels[f"{lib.library}_spearman"] = f"{disp} ρ"
        labels["pearson_mean"] = "mean Pearson r"
        labels["spearman_mean"] = "mean Spearman ρ"
        return labels

    def compare_plot_title(self) -> str:
        return "Chen synonymous MPRA"
