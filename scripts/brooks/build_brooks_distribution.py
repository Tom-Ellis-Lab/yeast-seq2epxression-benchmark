"""Build the single self-contained Brooks SCRaMBLE distribution file.

This is the ONLY component that touches `gs://brooks-nanopore`. It
resolves every (gene, strain, copy) sample under the locked rule
(benchmarks/brooks_scramble.md) and bakes everything the benchmark needs
at eval time into one TSV — so the benchmark depends on that file alone
(no GCS, no per-strain genomes/GFF/BED, no R64 reference).

Pipeline per SCRaMBLE strain S (control = JS94, 3 runs):
  1. per-strain genome FASTA + GFF + read BED from the bucket (cached).
  2. native-genome **median-of-ratios size factor** per strain/run
     (native chromosomes are byte-identical across strains, so this
     removes depth + sequencing-batch confounds).
  3. per-copy CDS coverage on the strain's synthetic contig `JS<S>_1`,
     gene-strand, normalised by the size factor.
  4. `true_lfc = log2( normcov(strain copy) / mean normcov(JS94 gene) )`;
     JS94 per-run normalised coverages kept so the reproducibility
     ceiling is derivable from the same file (no sidecar).
  5. gene-centred alt window (on `JS<S>_1`) and native window (on
     `JS94_1`), with the in-window CDS interval.
  6. locked sample rule: gene present in JS94; intact CDS; alt window
     differs from native within the receptive field (real cis change);
     dedup byte-identical copies; deletions excluded (no CDS → no row).

Output: `data/tasks/brooks_scramble/brooks_scramble_v1.tsv` — one row per
sample. Column schema in COLUMNS below.

Run:
  uv run python scripts/brooks/build_brooks_distribution.py            # 5 ROADMAP strains
  uv run python scripts/brooks/build_brooks_distribution.py --strains all
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BUCKET = "gs://brooks-nanopore"
CACHE = ROOT / "data" / "tasks" / "brooks_scramble" / "_cache"
OUT = ROOT / "data" / "tasks" / "brooks_scramble" / "brooks_scramble_v1.tsv"

CONTROL = "JS94"
ROADMAP_STRAINS = ["JS606", "JS707", "JS711", "JS731", "JS732"]
NATIVE_CONTIGS = (
    [f"chr{r}" for r in
     "I II III IV V VI VII VIII X XI XII XIII XIV XV XVI".split()]
    + ["chrIXL"]
)
WINDOW = 4992          # Yorzoi receptive field; gene-centred
PSEUDOCOUNT = 1.0
MIN_READS = 10         # strain-side raw CDS read floor → low_support flag
                       # on the sample. Same threshold is used per-JS94-run
                       # inside the benchmark to decide which JS94 runs
                       # contribute to the per-replicate true_lfc set.
MIN_RUN_READS = 50_000 # per-run native library-size floor; failed/ultra-
                       # shallow runs (e.g. JS94 20180607=651, 20181122=
                       # 3878 reads) are dropped before they corrupt the
                       # control denominator / size factor / ceiling
SGD_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](?:-[A-Z])?$")

COLUMNS = [
    "sample_id", "gene_id", "strain", "copy_idx", "n_copies", "strand",
    "rearr_class", "syn_contig", "cds_start", "cds_end",
    "window_len", "cds_start_in_window", "cds_end_in_window",
    "alt_seq", "native_seq",
    "true_cov_alt", "true_cov_native",   # comma-list of WINDOW int32 per-base counts
    "strain_reads", "js94_reads_runs",   # raw counts (strain sum, JS94 per-run)
    "size_factor_strain",
    "norm_cov_strain", "norm_cov_js94_mean",
    "norm_cov_js94_runs",  # comma-list of all WT-run values (ceiling derivable)
    "true_lfc",
    # `low_support` is strain-side only: drop a single (strain, gene, copy)
    # sample if the strain's raw CDS-overlap count < MIN_READS. The JS94
    # side is handled per-replicate inside the benchmark (a JS94 run with
    # < MIN_READS reads for this gene drops out of the per-replicate LFC
    # set; n_reps_supported per sample = 0..3 derivable from
    # `js94_reads_runs`).
    "low_support",
]


# ── bucket access (cached) ────────────────────────────────────


def _gs(*args: str) -> str:
    return subprocess.run(
        ["gcloud", "storage", *args], capture_output=True, text=True, check=True
    ).stdout


def _ls(prefix: str) -> list[str]:
    return [l for l in _gs("ls", f"{BUCKET}/{prefix}").splitlines() if l.strip()]


def fetch(remote: str) -> Path:
    """Download `BUCKET/remote` once into the cache, return local path."""
    local = CACHE / remote
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        _gs("cp", f"{BUCKET}/{remote}", str(local))
    return local


def bed_paths(strain: str) -> list[str]:
    """Plain WT direct-RNA runs only. EXCLUDES tagged libraries — JS94
    has rrp6Δ/xrn1Δ RNA-decay-mutant runs (`..._20191017rrp6`, `xrn1`,
    `xrn1nc`) that are NOT the −SCRaMBLE WT baseline; the strict pattern
    (date immediately followed by `_porechopped`) drops them."""
    pat = re.compile(
        rf"^{strain}_\d{{8}}_porechopped_filtered_canuCorrected"
        r"_distinguished\.bed$"
    )
    return sorted(
        f"alignment/{l.rsplit('/', 1)[1]}"
        for l in _ls("alignment/")
        if pat.match(l.rsplit("/", 1)[1])
    )


# ── parsing ───────────────────────────────────────────────────


def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    cur = None
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                cur = line[1:].split()[0]
                seqs[cur] = []
            elif cur is not None:
                seqs[cur].append(line.strip())
    return {k: "".join(v).upper() for k, v in seqs.items()}


def parse_gff_cds(path: Path) -> pd.DataFrame:
    """One row per (contig, gene_id) with merged CDS span + strand.

    Uses CDS features where present, else the gene feature. Keeps only
    real SGD ORF ids.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] not in ("CDS", "gene"):
                continue
            m = re.search(r"ID=([^;]+)", f[8])
            if not m:
                continue
            gid = m.group(1)
            if not SGD_RE.match(gid):
                continue
            rows.append((f[0], gid, f[2], int(f[3]), int(f[4]), f[6]))
    df = pd.DataFrame(
        rows, columns=["contig", "gene_id", "ftype", "start", "end", "strand"]
    )
    if df.empty:
        return df
    cds = df[df.ftype == "CDS"]
    use = cds if not cds.empty else df
    g = (
        use.groupby(["contig", "gene_id", "strand"])
        .agg(start=("start", "min"), end=("end", "max"))
        .reset_index()
    )
    return g


def load_bed(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, sep="\t", header=None,
        names=["chrom", "start", "end", "rid", "mapq", "strand"],
        usecols=[0, 1, 2, 4, 5],
        dtype={"chrom": str, "start": np.int64, "end": np.int64,
               "mapq": np.int16, "strand": str},
    )


# ── coverage helpers ──────────────────────────────────────────


def count_overlaps(bed: pd.DataFrame, contig: str, lo: int, hi: int,
                   strand: str) -> int:
    """# reads on `contig`/`strand` overlapping 1-based inclusive [lo,hi]."""
    b = bed[(bed.chrom == contig) & (bed.strand == strand)]
    if b.empty:
        return 0
    # BED is 0-based half-open; gene coords 1-based inclusive
    return int(((b.start < hi) & (b.end > lo - 1)).sum())


def per_base_cov(bed: pd.DataFrame, contig: str, strand: str,
                 contig_len: int) -> np.ndarray:
    """Per-base read-depth array for one (contig, strand), via diff+cumsum
    over the BED intervals — O(N reads + contig_len), then `[w0:w0+W]`
    slicing per sample is O(W). Returns int32 array of length `contig_len`."""
    b = bed[(bed.chrom == contig) & (bed.strand == strand)]
    if b.empty:
        return np.zeros(contig_len, dtype=np.int32)
    starts = np.clip(b.start.to_numpy(), 0, contig_len)
    ends = np.clip(b.end.to_numpy(), 0, contig_len)
    delta = np.zeros(contig_len + 1, dtype=np.int32)
    np.add.at(delta, starts, 1)
    np.add.at(delta, ends, -1)
    return np.cumsum(delta[:contig_len]).astype(np.int32)


def _comma_ints(arr: np.ndarray) -> str:
    return ",".join(map(str, arr.tolist()))


def native_total_reads(bed: pd.DataFrame) -> int:
    """Total reads on the native nuclear contigs (chrI–chrXVI minus the
    synthetic synIXR, plus chrIXL; chrMT and the synthetic contig
    excluded). The native genome is byte-identical across all strains, so
    this is a clean, simple library-size factor that removes the depth +
    sequencing-batch confound without per-gene modelling (median-of-ratios
    is a v2 refinement — see spec)."""
    return int(bed.chrom.isin(NATIVE_CONTIGS).sum())


def size_factor(strain_total: int, ref_total: float) -> float:
    """Strain/run library-size factor vs the JS94-mean native total."""
    return float(strain_total / max(ref_total, 1.0))


# ── window extraction ─────────────────────────────────────────


def gene_window(seq: str, start: int, end: int) -> tuple[str, int, int, int] | None:
    """Gene-centred WINDOW-bp slice (forward contig orientation).
    Returns (window_seq, cds_start_in_window, cds_end_in_window) or None
    if the contig can't fill a full window around the gene."""
    if len(seq) < WINDOW:
        return None
    centre = (start - 1 + end) // 2
    w0 = centre - WINDOW // 2
    w0 = max(0, min(w0, len(seq) - WINDOW))
    win = seq[w0:w0 + WINDOW]
    cs, ce = (start - 1) - w0, end - w0
    if cs < 0 or ce > WINDOW:
        return None
    return win, cs, ce, w0   # w0 = window genomic 0-based start (for coverage)


def cluster_copies(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse near-coincident annotation entries (overlap artifacts);
    distinct copies are far-separated on the contig."""
    spans = sorted(spans)
    out: list[tuple[int, int]] = []
    for s, e in spans:
        if out and s <= out[-1][1] + 1000:  # same locus → keep longest
            ps, pe = out[-1]
            out[-1] = (min(ps, s), max(pe, e))
        else:
            out.append((s, e))
    return out


# ── main ──────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strains", default="roadmap",
                    help="'roadmap' (5), 'all', or comma list")
    args = ap.parse_args()

    # Control = JS94 (−SCRaMBLE). The bucket has NO JS94 FASTA, but
    # JS96_1 is the parental synIXR sequence: identical 98,752 bp length
    # and byte-identical synIXR GFF coords to JS94_1 (a rearrangement
    # would change these) → JS96 supplies the parental *sequence*; JS94's
    # 3 runs supply the parental *expression* (shared coordinate system).
    par_fa = read_fasta(fetch("genomes/JS96_ERCC92.fasta"))
    par_syn = next(c for c in par_fa if c.startswith("JS96_"))
    js94_gff = parse_gff_cds(fetch("annotations/JS94.gff"))
    js94_read_contig = "JS94_1"          # JS94 BED chrom for control reads
    assert len(par_fa[par_syn]) == 98752, len(par_fa[par_syn])
    js94_genes = js94_gff[js94_gff.contig == js94_read_contig].set_index("gene_id")
    js94_bed_paths = bed_paths(CONTROL)
    kept = []
    for p in js94_bed_paths:
        b = load_bed(fetch(p))
        nt = native_total_reads(b)
        date = p.rsplit("/", 1)[1].split("_")[1]
        if nt >= MIN_RUN_READS:
            kept.append((date, b, nt))
        else:
            print(f"  JS94 run {date}: DROPPED (native reads {nt} "
                  f"< {MIN_RUN_READS})")
    if len(kept) < 2:
        raise RuntimeError("need ≥2 deep JS94 control runs for a ceiling")
    js94_beds = [b for _, b, _ in kept]
    js94_native_tot = [nt for _, _, nt in kept]
    ref_total = float(np.mean(js94_native_tot))
    js94_sf = [size_factor(t, ref_total) for t in js94_native_tot]
    print(f"JS94: {len(js94_genes)} synIXR genes (parental seq from "
          f"{par_syn}); {len(js94_beds)} deep WT runs "
          f"{[d for d, _, _ in kept]}")

    # Per-base native coverage on JS94_1 (= JS96_1 coord system),
    # summed across the deep JS94 runs — used for Tier-2 native-truth.
    js94_syn_len = len(par_fa[par_syn])
    js94_native_cov = {
        s: sum(per_base_cov(b, js94_read_contig, s, js94_syn_len)
               for b in js94_beds)
        for s in ("+", "-")
    }

    if args.strains == "roadmap":
        strains = ROADMAP_STRAINS
    elif args.strains == "all":
        strains = sorted({
            re.match(r"(JS\d+)_", l.rsplit("/", 1)[1]).group(1)
            for l in _ls("genomes/") if re.search(r"/JS\d+_ERCC92\.fasta$", l)
        } - {CONTROL})
    else:
        strains = args.strains.split(",")

    rows: list[dict] = []
    for S in strains:
        try:
            fa = read_fasta(fetch(f"genomes/{S}_ERCC92.fasta"))
            gff = parse_gff_cds(fetch(f"annotations/{S}.gff"))
            beds = [load_bed(fetch(p)) for p in bed_paths(S)]
        except Exception as e:  # noqa: BLE001
            print(f"  {S}: skip ({e})")
            continue
        beds = [b for b in beds if native_total_reads(b) >= MIN_RUN_READS]
        if not beds:
            print(f"  {S}: skip (no run ≥ {MIN_RUN_READS} native reads)")
            continue
        syn = next(c for c in fa if c.startswith(f"{S}_"))
        syn_len = len(fa[syn])
        bed = beds[0] if len(beds) == 1 else pd.concat(beds, ignore_index=True)
        s_sf = size_factor(native_total_reads(bed), ref_total)
        # Per-base alt coverage on JS<S>_1, both strands (one cumsum per
        # strand, then O(WINDOW) slicing per sample).
        strain_alt_cov = {s: per_base_cov(bed, syn, s, syn_len) for s in ("+", "-")}
        g = gff[gff.contig == syn]
        n_kept = 0
        for gid, grp in g.groupby("gene_id"):
            if gid not in js94_genes.index:
                continue  # not in parental → no native baseline
            jrow = js94_genes.loc[gid]
            if isinstance(jrow, pd.DataFrame):
                jrow = jrow.iloc[0]
            nat = gene_window(par_fa[par_syn], int(jrow.start), int(jrow.end))
            if nat is None:
                continue
            nat_seq, ncs, nce, nat_w0 = nat
            nat_cov_vec = js94_native_cov[jrow.strand][nat_w0:nat_w0 + WINDOW]
            # JS94 per-run normalised CDS coverage (gene strand), reads
            # on JS94_1 (same coord system as the parental sequence)
            j_raws = [
                count_overlaps(b, js94_read_contig, int(jrow.start),
                               int(jrow.end), jrow.strand)
                for b in js94_beds
            ]
            j_norm = [r / sf for r, sf in zip(j_raws, js94_sf)]
            j_mean = float(np.mean(j_norm))

            copies = cluster_copies(list(zip(grp.start, grp.end)))
            seen_windows: set[str] = set()
            for ci, (cs, ce) in enumerate(copies):
                aw = gene_window(fa[syn], cs, ce)
                if aw is None:
                    continue
                alt_seq, acs, ace, alt_w0 = aw
                if alt_seq == nat_seq:
                    continue  # no cis change in-window → not a sample
                if alt_seq in seen_windows:
                    continue  # byte-identical duplicate copy
                seen_windows.add(alt_seq)
                strand = grp.strand.iloc[0]
                s_raw = count_overlaps(bed, syn, cs, ce, strand)
                s_norm = s_raw / s_sf
                true_lfc = float(np.log2(
                    (s_norm + PSEUDOCOUNT) / (j_mean + PSEUDOCOUNT)
                ))
                alt_cov_vec = strain_alt_cov[strand][alt_w0:alt_w0 + WINDOW]
                rows.append({
                    "sample_id": f"{S}:{gid}:{ci}",
                    "gene_id": gid, "strain": S, "copy_idx": ci,
                    "n_copies": len(copies), "strand": strand,
                    "rearr_class": "duplication" if len(copies) > 1 else "context_change",
                    "syn_contig": syn, "cds_start": cs, "cds_end": ce,
                    "window_len": WINDOW,
                    "cds_start_in_window": acs, "cds_end_in_window": ace,
                    "alt_seq": alt_seq, "native_seq": nat_seq,
                    "true_cov_alt": _comma_ints(alt_cov_vec),
                    "true_cov_native": _comma_ints(nat_cov_vec),
                    "strain_reads": s_raw,
                    "js94_reads_runs": ",".join(str(int(v)) for v in j_raws),
                    "size_factor_strain": round(s_sf, 4),
                    "norm_cov_strain": round(s_norm, 3),
                    "norm_cov_js94_mean": round(j_mean, 3),
                    "norm_cov_js94_runs": ",".join(f"{v:.3f}" for v in j_norm),
                    "true_lfc": round(true_lfc, 4),
                    "low_support": bool(s_raw < MIN_READS),
                })
                n_kept += 1
        print(f"  {S}: {n_kept} samples (sf={s_sf:.3f}, {len(beds)} run(s))")

    df = pd.DataFrame(rows, columns=COLUMNS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, sep="\t", index=False)
    print(f"\nwrote {OUT}  ({len(df)} samples, "
          f"{df.low_support.sum() if len(df) else 0} low-support; "
          f"{OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
