#!/usr/bin/env python3
"""Stratified AUROC/AUPRC for the Caudal eQTL benchmark: clean vs. not-clean.

Reuses a SINGLE benchmark run's per-pair scores (no per-stratum rescoring). The
benchmark stores, per negset, a scores array of length 2N ordered
[pos_0, neg_0, pos_1, neg_1, ...] matching the input negset row order, so we map
each row's *positive* variant to its ploidy/VAF stratum (from the audit) and
recompute AUROC/AUPRC over each stratum's positives + their paired negatives.

Strata (from variant_summary.parquet produced by extract_positive_carriers.py):

  clean              biallelic SNP in the panel, carriers not dominated by
                     heterozygous (≤30%) or non-diploid (≤30%) isolates
  not_clean          complement of clean (union of the categories below)

  diagnostic categories (overlapping):
    multiallelic_complex   allele not a clean biallelic SNP in the panel
    het_dominated          >30% of carriers heterozygous at the locus
    nondiploid_dominated   >30% of carriers non-diploid (1n / 3n+)
    low_carrier_vaf        mean carrier allele-balance VAF < 0.80 (dosage smear)

We report, per stratum: |score| AUROC/AUPRC (mean±SEM over the 4 negsets), the
number of pairs, the fraction of positive scores that are exactly 0 (variant
beyond the model window), the distance-to-TSS profile, and a distance-controlled
"close-only" (≤2 kb) AUROC -- because the Caudal signal is strongly
distance-dependent, a clean-vs-not gap could otherwise be a distance artefact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[3]
HET_THR = 0.30
NONDIP_THR = 0.30
VAF_THR = 0.80
CLOSE_BP = 2000


def assign_strata(v: pd.DataFrame) -> pd.DataFrame:
    found = v["allele_found"] == True  # noqa: E712
    v = v.copy()
    v["multiallelic_complex"] = ~found
    v["het_dominated"] = found & (v["frac_het_carriers"] > HET_THR)
    v["nondiploid_dominated"] = found & (v["frac_nondiploid_carriers"] > NONDIP_THR)
    v["low_carrier_vaf"] = found & (v["mean_carrier_vaf"] < VAF_THR)
    v["clean"] = (
        found
        & (v["frac_het_carriers"] <= HET_THR)
        & (v["frac_nondiploid_carriers"] <= NONDIP_THR)
    )
    v["not_clean"] = ~v["clean"]
    return v


def _auc_pair(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if len(set(labels.tolist())) < 2:
        return np.nan, np.nan
    s = np.abs(scores)
    return roc_auc_score(labels, s), average_precision_score(labels, s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-dir", type=Path,
                    default=REPO / "results/caudal_ploidy_audit")
    ap.add_argument("--negset-dir", type=Path,
                    default=Path("/home/tds122/yeast-seq2epxression-benchmark/data/tasks/caudal_eqtl"))
    ap.add_argument("--scores-dir", type=Path,
                    default=Path("/home/tds122/yeast-seq2epxression-benchmark/results/default/yorzoi__caudal_eqtl"))
    ap.add_argument("--out", type=Path,
                    default=REPO / "results/caudal_ploidy_audit/stratified_yorzoi.json")
    args = ap.parse_args()

    variants = assign_strata(pd.read_parquet(args.audit_dir / "variant_summary.parquet"))
    strat = variants.set_index("variant_id")
    categories = ["clean", "not_clean", "multiallelic_complex", "het_dominated",
                  "nondiploid_dominated", "low_carrier_vaf"]

    # Collect, per negset, every pair's (positive |score|, negative |score|,
    # positive variant_id, pos_distance_to_tss, pos_score_is_zero).
    negsets = sorted(args.negset_dir.glob("negset_*.tsv"))
    per_negset_rows: list[pd.DataFrame] = []
    for tsv in negsets:
        name = tsv.stem
        sp = args.scores_dir / f"{name}_scores.npy"
        if not sp.exists():
            raise SystemExit(f"missing scores: {sp} (run the benchmark first)")
        scores = np.load(sp)
        df = pd.read_csv(tsv, sep="\t", dtype={"pos_chrom": str, "neg_chrom": str})
        assert len(scores) == 2 * len(df), f"{name}: {len(scores)} != 2*{len(df)}"
        df["variant_id"] = (
            "chromosome" + df["pos_chrom"].astype(str) + ":"
            + df["pos_pos"].astype(str) + ":"
            + df["pos_ref"].astype(str) + ">" + df["pos_alt"].astype(str)
        )
        df["pos_score"] = scores[0::2]
        df["neg_score"] = scores[1::2]
        df["negset"] = name
        per_negset_rows.append(df[["negset", "variant_id", "pos_distance_to_tss",
                                   "pos_score", "neg_score"]])
    allrows = pd.concat(per_negset_rows, ignore_index=True)

    # join strata
    for cat in categories:
        allrows[cat] = allrows["variant_id"].map(strat[cat]).fillna(False).astype(bool)
    unmatched = int(allrows["variant_id"].map(lambda x: x not in strat.index).sum())

    def stratum_metrics(mask_col: str, close_only: bool = False) -> dict:
        per_neg = []
        for name, g in allrows[allrows[mask_col]].groupby("negset"):
            if close_only:
                g = g[g["pos_distance_to_tss"] <= CLOSE_BP]
            if len(g) < 5:
                continue
            scores = np.concatenate([g["pos_score"].to_numpy(), g["neg_score"].to_numpy()])
            labels = np.concatenate([np.ones(len(g)), np.zeros(len(g))])
            au, ap_ = _auc_pair(scores, labels)
            per_neg.append((au, ap_, len(g), float((g["pos_score"] == 0).mean())))
        if not per_neg:
            return {}
        arr = np.array([(a, b) for a, b, _, _ in per_neg], dtype=float)
        return dict(
            n_pairs=int(np.mean([n for _, _, n, _ in per_neg])),
            auroc_mean=float(np.nanmean(arr[:, 0])),
            auroc_sem=float(np.nanstd(arr[:, 0], ddof=1) / np.sqrt(len(per_neg))) if len(per_neg) > 1 else 0.0,
            auprc_mean=float(np.nanmean(arr[:, 1])),
            auprc_sem=float(np.nanstd(arr[:, 1], ddof=1) / np.sqrt(len(per_neg))) if len(per_neg) > 1 else 0.0,
            pos_zero_frac=float(np.mean([z for _, _, _, z in per_neg])),
        )

    out = {"thresholds": dict(het=HET_THR, nondiploid=NONDIP_THR, vaf=VAF_THR,
                              close_bp=CLOSE_BP),
           "unmatched_variant_rows": unmatched,
           "n_unique_variants_per_stratum": {
               c: int(variants[variants[c]].shape[0]) for c in categories},
           "full": {c: stratum_metrics(c) for c in categories},
           "close_only": {c: stratum_metrics(c, close_only=True) for c in categories}}

    # distance profile per stratum (pairs, across negsets pooled then /4)
    dist = {}
    for c in categories:
        d = allrows[allrows[c]]["pos_distance_to_tss"]
        dist[c] = dict(median_bp=float(d.median()),
                       frac_gt_8kb=float((d > 8000).mean()),
                       frac_gt_16kb=float((d > 16000).mean()))
    out["distance_profile"] = dist

    args.out.write_text(json.dumps(out, indent=2))

    # console table
    print(f"unmatched variant rows: {unmatched}")
    print("\nunique variants per stratum:")
    for c in categories:
        print(f"  {c:22s} {out['n_unique_variants_per_stratum'][c]:5d}")
    hdr = f"\n{'stratum':22s} {'n_pairs':>8s} {'AUROC':>16s} {'AUPRC':>16s} {'pos0%':>7s} {'>8kb%':>7s}"
    print(hdr); print("-" * len(hdr))
    for c in categories:
        f = out["full"][c]
        if not f:
            print(f"{c:22s}  (empty)"); continue
        print(f"{c:22s} {f['n_pairs']:8d} "
              f"{f['auroc_mean']:.3f}±{f['auroc_sem']:.3f}   "
              f"{f['auprc_mean']:.3f}±{f['auprc_sem']:.3f}   "
              f"{f['pos_zero_frac']*100:5.1f}  {dist[c]['frac_gt_8kb']*100:5.1f}")
    print("\nclose-only (≤2kb) AUROC/AUPRC:")
    for c in ["clean", "not_clean"]:
        co = out["close_only"][c]
        if co:
            print(f"  {c:22s} n={co['n_pairs']:4d}  AUROC {co['auroc_mean']:.3f}  AUPRC {co['auprc_mean']:.3f}")
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
