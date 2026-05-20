"""Dump Yorzoi per-bin predictions for the Wu RFP-insertion windows,
annotated with cassette sub-features and native gene bodies, for manual
inspection.  Focuses on a single track (default 0) plus the
strand-matched cross-track mean the benchmark actually scores.

Outputs (results/default/yorzoi_wu_track_dump/ by default):
  track_matrix.npy        (n_loci, OUTPUT_BINS)  chosen single track
  strandmean_matrix.npy   (n_loci, OUTPUT_BINS)  strand-matched mean
  loci_index.tsv          row → gene_id, chrom, strand, measured,
                           score, window placement offsets
  annotations.tsv         per detailed locus: feature/gene → bin span
  plots/<gene_id>.png     detailed subset (measured extremes)
  README.md

Run:
  uv run python scripts/wu/dump_yorzoi_tracks.py            # extremes only
  uv run python scripts/wu/dump_yorzoi_tracks.py --track 0 --n-extremes 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from yeastbench.adapters._genome import Gene, parse_gene_annotations
from yeastbench.adapters._wu_scaffold import (
    CASSETTE_FEATURES,
    PAYLOAD_LEN,
    WuInsertionContext,
    build_insertion_context,
    load_cassette_payload,
    payload_feature_window_span,
    resolve_loci,
    span_to_bins,
)
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
)
from yeastbench.adapters.yorzoi_wu import YorzoiWuPredictor

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data/tasks/wu_rfpins/table_s2_fluorescence_1044_loci.csv"
FASTA = ROOT / "data/tasks/R64-1-1.fa"
GTF = ROOT / "data/tasks/R64-1-1.115.gtf"
CASSETTE = ROOT / "data/tasks/wu_rfpins/expression_cassette.fasta"


def _genes_by_chrom(genes: dict[str, Gene]) -> dict[str, list[tuple]]:
    out: dict[str, list[tuple]] = {}
    for gid, g in genes.items():
        out.setdefault(g.chrom_roman, []).append(
            (gid, g.strand, g.gene_start, g.gene_end)
        )
    return out


def _annotate(ctx: WuInsertionContext, genes_by_chrom: dict) -> list[dict]:
    """Cassette features + native genes → output-bin spans for one window."""
    rows: list[dict] = []
    # cassette sub-features
    for name, kind, a, b in CASSETTE_FEATURES:
        lo, hi = payload_feature_window_span(
            a, b, ctx.payload_start_in_window, ctx.payload_rc
        )
        bins = span_to_bins(lo, hi, CROP_BP_EACH_SIDE, BIN_WIDTH, OUTPUT_BINS)
        if bins:
            rows.append(dict(
                gene_id=ctx.gene_id, feature=name, kind=kind,
                start_bin=bins[0], end_bin=bins[1],
            ))
    # native genes overlapping the two native flanks
    lc = ctx.locus
    ws, ua, Lp = ctx.window_start_in_spliced, ctx.up_avail, PAYLOAD_LEN
    for gid, strand, gs, ge in genes_by_chrom.get(lc.chrom, []):
        if gid == lc.gene_id:
            continue  # the deleted ORF — replaced by the cassette
        for region, g0_to_w in (
            ("native_up", lambda g: (g - (lc.gene_start - ua)) - ws),
            ("native_down", lambda g: (g - (lc.gene_end + 1)) + ua + Lp - ws),
        ):
            w_lo = g0_to_w(gs)
            w_hi = g0_to_w(ge) + 1
            if w_hi <= 0 or w_lo >= SEQ_LEN:
                continue
            bins = span_to_bins(
                max(0, w_lo), min(SEQ_LEN, w_hi),
                CROP_BP_EACH_SIDE, BIN_WIDTH, OUTPUT_BINS,
            )
            if bins:
                rows.append(dict(
                    gene_id=ctx.gene_id,
                    feature=f"{gid}({strand})",
                    kind=f"{region}_gene",
                    start_bin=bins[0], end_bin=bins[1],
                ))
    return rows


def _plot(out_png: Path, track: np.ndarray, ctx: WuInsertionContext,
          ann: list[dict], gene_id: str, strand: str,
          measured: float, score: float, track_id: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 4.5))
    x = np.arange(len(track))
    ax.fill_between(x, track, step="mid", alpha=0.5, color="steelblue")
    ax.plot(x, track, lw=0.6, color="steelblue")
    kind_color = {
        "reporter_cds": "crimson", "marker_cds": "darkorange",
        "promoter": "seagreen", "terminator": "slateblue",
        "barcode": "grey", "scar": "lightgrey",
        "native_up_gene": "saddlebrown", "native_down_gene": "saddlebrown",
    }
    ymax = float(track.max()) if track.size else 1.0
    for a in ann:
        c = kind_color.get(a["kind"], "black")
        is_gene = a["kind"].endswith("_gene")
        ax.axvspan(a["start_bin"], a["end_bin"],
                   ymin=0.0, ymax=0.12 if is_gene else 1.0,
                   color=c, alpha=0.5 if is_gene else 0.18, lw=0)
        mid = (a["start_bin"] + a["end_bin"]) / 2
        ax.text(mid, ymax * (0.05 if is_gene else 0.92), a["feature"],
                ha="center", va="center", fontsize=6, rotation=90,
                color=c)
    # mCherry readout bins (what the benchmark sums)
    rb = ctx.rfp_bins
    ax.axvspan(rb.min(), rb.max() + 1, color="crimson", alpha=0.10, lw=0)
    ax.set_xlabel(f"output bin (0..{OUTPUT_BINS}, {BIN_WIDTH} bp/bin)")
    ax.set_ylabel(f"Yorzoi track {track_id} predicted coverage")
    ax.set_title(
        f"{gene_id} ({strand})  measured={measured:.2f}  "
        f"score={score:.1f}  — mCherry readout shaded red"
    )
    ax.set_xlim(0, OUTPUT_BINS)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=int, default=0,
                    help="single track index to dump (default 0)")
    ap.add_argument("--n-extremes", type=int, default=6,
                    help="N lowest + N highest measured loci for detailed "
                         "plots/annotations (default 6)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/default/yorzoi_wu_track_dump")
    ap.add_argument("--hf-repo", default="tom-ellis-lab/yorzoi")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "plots").mkdir(exist_ok=True)

    df = pd.read_csv(LABELS)
    gene_ids = df["ORF_name"].astype(str).tolist()
    measured = df["Relative_Fluorescence_Average"].to_numpy(float)
    genes = parse_gene_annotations(GTF)
    gbc = _genes_by_chrom(genes)
    loci, _ = resolve_loci(gene_ids, genes)
    payload = load_cassette_payload(CASSETTE)

    adapter = YorzoiWuPredictor.from_pretrained(
        hf_repo=args.hf_repo, fasta_path=FASTA, gtf_path=GTF,
        device="cuda", batch_size=args.batch_size, use_rc=True,
    )
    import torch
    from yeastbench.adapters._genome import one_hot_encode_channels_first

    n = len(loci)
    track_mat = np.full((n, OUTPUT_BINS), np.nan, np.float32)
    smean_mat = np.full((n, OUTPUT_BINS), np.nan, np.float32)
    score = np.full(n, np.nan)
    contexts: dict[int, WuInsertionContext] = {}

    idx = [i for i, lc in enumerate(loci) if lc is not None]
    from tqdm import tqdm
    for bs in tqdm(range(0, len(idx), args.batch_size), desc="Yorzoi dump"):
        chunk = idx[bs : bs + args.batch_size]
        ctxs, rows = [], []
        for i in chunk:
            c = build_insertion_context(
                loci[i], payload, adapter.fasta,
                SEQ_LEN, CROP_BP_EACH_SIDE, BIN_WIDTH, OUTPUT_BINS,
            )
            if c is not None:
                ctxs.append((i, c))
                rows.append(one_hot_encode_channels_first(c.window_seq).T)
        if not rows:
            continue
        x = torch.from_numpy(np.stack(rows)).to(adapter.device)
        with torch.no_grad():
            pred = adapter._forward_full_tracks(x).float()  # (B,162,bins)
        for j, (i, c) in enumerate(ctxs):
            contexts[i] = c
            track_mat[i] = pred[j, args.track].cpu().numpy()
            ts, te = (0, 81) if loci[i].strand == "+" else (81, 162)
            smean_mat[i] = pred[j, ts:te].mean(0).cpu().numpy()
            rb = torch.as_tensor(c.rfp_bins, device=adapter.device)
            score[i] = float(pred[j, ts:te].mean(0).index_select(0, rb).sum())

    np.save(args.out / "track_matrix.npy", track_mat)
    np.save(args.out / "strandmean_matrix.npy", smean_mat)
    pd.DataFrame({
        "row": np.arange(n),
        "gene_id": gene_ids,
        "chrom": [loci[i].chrom if loci[i] else "" for i in range(n)],
        "strand": [loci[i].strand if loci[i] else "" for i in range(n)],
        "measured": measured,
        "score": score,
        "resolved": [loci[i] is not None for i in range(n)],
    }).to_csv(args.out / "loci_index.tsv", sep="\t", index=False)

    # detailed subset = measured extremes
    order = np.array(sorted(idx, key=lambda i: measured[i]))
    sel = list(order[: args.n_extremes]) + list(order[-args.n_extremes :])
    ann_rows: list[dict] = []
    for i in sel:
        c = contexts.get(i)
        if c is None:
            continue
        a = _annotate(c, gbc)
        ann_rows += a
        _plot(args.out / "plots" / f"{gene_ids[i]}.png",
              track_mat[i], c, a, gene_ids[i], loci[i].strand,
              measured[i], score[i], args.track)
    pd.DataFrame(ann_rows).to_csv(
        args.out / "annotations.tsv", sep="\t", index=False
    )

    (args.out / "README.md").write_text(
        f"# Yorzoi Wu track dump (track {args.track})\n\n"
        f"- `track_matrix.npy`  ({n}, {OUTPUT_BINS}) — raw track "
        f"{args.track} per-bin coverage; row order = `loci_index.tsv`.\n"
        f"- `strandmean_matrix.npy` — strand-matched cross-track mean "
        f"(what the benchmark sums over mCherry bins).\n"
        f"- `loci_index.tsv` — per-locus metadata + scalar score.\n"
        f"- `annotations.tsv` — cassette features & native genes as "
        f"output-bin spans `[start_bin, end_bin)`, detailed subset only.\n"
        f"- `plots/` — {len(sel)} measured-extreme loci, track profile "
        f"with cassette features (top) and native genes (bottom) shaded; "
        f"mCherry readout region in red.\n\n"
        f"Window: SEQ_LEN={SEQ_LEN}, {OUTPUT_BINS} bins @ {BIN_WIDTH} "
        f"bp/bin, crop {CROP_BP_EACH_SIDE} bp/side. mCherry stop codon "
        f"sits at the downstream crop edge (max upstream context).\n"
    )
    print(f"wrote {args.out}  ({len(idx)} loci, {len(sel)} detailed)")


if __name__ == "__main__":
    main()
