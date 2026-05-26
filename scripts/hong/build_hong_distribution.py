"""Build the Hong et al. IGR-insertion benchmark distribution.

Parses ``data/tasks/hong/YeIP_supp_table_R2_20260411.xlsx`` (vendored
from the bioRxiv supplementary archive), derives every locus's
**experimental** cut site by matching the listed gRNA against
R64-5-1, and writes:

  - ``data/tasks/hong/hong_igr_v1.tsv`` — 150 rows (98 IntTrain + 52
    IntProp) with ``locus_id, set, chrom, integration_coord,
    fluorescence_norm_intrain92, gRNA``.
  - ``data/tasks/hong/expression_cassette.fasta`` — frozen 1595 bp
    ``TDH3p-mCherry-ADH1t`` cassette from Supp Table S3 row 1, with
    the mCherry CDS offset asserted in the FASTA header.

See ``benchmarks/hong_igr.md`` for full design notes (in particular,
why Table S5's coordinates are ignored).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pysam

REPO_ROOT = Path(__file__).resolve().parents[2]
XLSX = REPO_ROOT / "data" / "tasks" / "hong" / "YeIP_supp_table_R2_20260411.xlsx"
FASTA = REPO_ROOT / "data" / "tasks" / "R64-5-1.fa"
OUT_TSV = REPO_ROOT / "data" / "tasks" / "hong" / "hong_igr_v1.tsv"
OUT_FASTA = REPO_ROOT / "data" / "tasks" / "hong" / "expression_cassette.fasta"

CASSETTE_NAME = "TDH3p-mCherry-ADH1t"
MCHERRY_START_MOTIF = "ATGGTGAGCAAGGGCGAGGAGGATAAC"  # canonical mCherry first 27 nt
MCHERRY_LEN = 711  # incl. TAA stop codon

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def rc(s: str) -> str:
    return s.translate(_COMPLEMENT)[::-1]


def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def _all_hits(genome: dict[str, str], gRNA: str) -> list[tuple[str, int, str]]:
    """Find every occurrence of ``gRNA`` (and its reverse complement) in
    ``genome``. Returns ``(chrom, 0-based-start, strand)`` tuples."""
    hits: list[tuple[str, int, str]] = []
    rc_gRNA = rc(gRNA)
    for ch, seq in genome.items():
        for query, strand in ((gRNA, "+"), (rc_gRNA, "-")):
            p = 0
            while True:
                i = seq.find(query, p)
                if i < 0:
                    break
                hits.append((ch, i, strand))
                p = i + 1
    return hits


def _gRNA_to_cut_site(gRNA_start: int, strand: str) -> int:
    """SpCas9 cuts blunt between protospacer positions 17 and 18 (1-indexed
    from the 5' end of the protospacer; = 0-indexed positions 16 and 17).
    Returns the 0-based + strand position of the base immediately before
    the cut; caller adds 1 to convert to a 1-based integration coordinate.

    + strand match: gRNA at + strand [pos, pos+20), PAM at [pos+20, pos+23).
        Protospacer 0-indexed position 16 = + strand pos+16. Cut between
        + strand 0-based pos+16 and pos+17 → base before cut = pos+16.
    − strand match: rc_gRNA found at + strand [pos, pos+20). The protospacer
        runs in − strand 5'→3' (= + strand 3'→5'), so protospacer position 16
        = + strand pos+19-16 = pos+3, protospacer position 17 = pos+2. Cut
        between + strand 0-based pos+2 and pos+3 → base before cut = pos+2.
    """
    return gRNA_start + (16 if strand == "+" else 2)


def _resolve_locus(
    locus_id: str,
    gRNA: str | None,
    fallback_int_site: int | None,
    fallback_chrom: str | None,
    genome: dict[str, str],
) -> tuple[str, int, str | None]:
    """Return ``(chrom, integration_coord_1based, gRNA_or_None)``.

    Resolution order:
      1. gRNA given + uniquely matches the genome  →  use the match.
      2. gRNA given + matches multiple places       →  pick the hit
         closest to ``fallback_int_site`` on ``fallback_chrom``.
      3. gRNA missing/empty                          →  use the
         fallback chrom/int_site directly (must both be provided).
    Raises ``RuntimeError`` if none of the above resolve.
    """
    g = (gRNA or "").strip().upper()
    if len(g) == 20 and set(g) <= set("ACGT"):
        hits = _all_hits(genome, g)
        if len(hits) == 1:
            ch, pos, strand = hits[0]
            return ch, _gRNA_to_cut_site(pos, strand) + 1, g  # 0→1-based
        if len(hits) > 1:
            assert fallback_int_site is not None and fallback_chrom is not None, (
                f"{locus_id}: gRNA matches {len(hits)} sites and no fallback "
                f"int_site to disambiguate"
            )
            cuts = [
                (ch, _gRNA_to_cut_site(pos, strand) + 1, strand)
                for ch, pos, strand in hits
            ]
            # Prefer hits on the listed chromosome, then minimize |cut - fallback|
            on_chrom = [c for c in cuts if c[0] == fallback_chrom]
            search = on_chrom or cuts
            best = min(search, key=lambda c: abs(c[1] - fallback_int_site))
            return best[0], best[1], g
        # gRNA given but zero hits — should not happen on R64-5-1 per the
        # spec's audit. Surface loudly.
        raise RuntimeError(
            f"{locus_id}: gRNA {g!r} has zero matches in the reference genome"
        )

    # No usable gRNA — fall back.
    assert fallback_int_site is not None and fallback_chrom is not None, (
        f"{locus_id}: no gRNA and no fallback int_site"
    )
    return fallback_chrom, fallback_int_site, None


def build_locus_table(genome: dict[str, str]) -> pd.DataFrame:
    """Return the 150-row distribution as a DataFrame (no I/O)."""
    s1 = _strip_cols(pd.read_excel(XLSX, sheet_name="Table S1"))
    s2 = _strip_cols(pd.read_excel(XLSX, sheet_name="Table S2"))
    s4 = _strip_cols(pd.read_excel(XLSX, sheet_name="Table S4"))

    s1["IGRs"] = s1["IGRs"].astype(str).str.strip()
    s2["IGRs"] = s2["IGRs"].astype(str).str.strip()
    s4["IGRs"] = s4["IGRs"].astype(str).str.strip()

    # Dedupe Table S1 (a few IntTrain names are listed twice)
    s1_unique = s1.drop_duplicates(subset=["IGRs"], keep="first")

    rows: list[dict[str, object]] = []

    # ── IntTrain (98) ────────────────────────────────────────────────
    intrain = s4.merge(s1_unique[["IGRs", "gRNA"]], on="IGRs", how="left")
    assert len(intrain) == 98, f"expected 98 IntTrain rows, got {len(intrain)}"
    for _, r in intrain.iterrows():
        chrom_listed = str(r["chr"]).replace("chr", "")
        int_site_listed = int(round(float(r["int_site"])))
        chrom, integration_coord, gRNA = _resolve_locus(
            locus_id=str(r["IGRs"]),
            gRNA=r["gRNA"] if pd.notna(r["gRNA"]) else None,
            fallback_int_site=int_site_listed,
            fallback_chrom=chrom_listed,
            genome=genome,
        )
        rows.append(
            dict(
                locus_id=str(r["IGRs"]),
                set="IntTrain",
                chrom=chrom,
                integration_coord=integration_coord,
                fluorescence_norm_intrain92=float(r["fluo_intensity"]),
                gRNA=gRNA,
            )
        )

    # ── IntProp (52) ─────────────────────────────────────────────────
    intprop = s2[s2["IGRs"].str.startswith("IntProp")].copy()
    assert len(intprop) == 52, f"expected 52 IntProp rows, got {len(intprop)}"
    label_col = "Relative fluorescence intensity"
    for _, r in intprop.iterrows():
        chrom, integration_coord, gRNA = _resolve_locus(
            locus_id=str(r["IGRs"]),
            gRNA=r["gRNA"] if pd.notna(r["gRNA"]) else None,
            fallback_int_site=None,
            fallback_chrom=None,
            genome=genome,
        )
        rows.append(
            dict(
                locus_id=str(r["IGRs"]),
                set="IntProp",
                chrom=chrom,
                integration_coord=integration_coord,
                fluorescence_norm_intrain92=float(r[label_col]),
                gRNA=gRNA,
            )
        )

    df = pd.DataFrame(rows)
    assert len(df) == 150
    assert df["locus_id"].is_unique
    assert df["chrom"].isin(genome.keys()).all()
    return df


def build_cassette_fasta() -> tuple[str, int]:
    """Write the frozen TDH3p-mCherry-ADH1t cassette to FASTA, return
    ``(payload, mcherry_cds_offset)``."""
    s3 = _strip_cols(pd.read_excel(XLSX, sheet_name="Table S3"))
    # First two columns are name + sequence; column names vary across versions
    name_col, seq_col = s3.columns[0], s3.columns[1]
    rows = s3[s3[name_col].astype(str).str.strip() == CASSETTE_NAME]
    assert len(rows) == 1, (
        f"expected exactly one row for {CASSETTE_NAME} in Table S3, got {len(rows)}"
    )
    payload = "".join(str(rows.iloc[0][seq_col]).split()).upper()
    assert set(payload) <= set("ACGT"), (
        f"cassette has non-ACGT characters: {set(payload) - set('ACGT')}"
    )
    cds_offset = payload.find(MCHERRY_START_MOTIF)
    assert cds_offset >= 0, "mCherry CDS start motif not found in cassette"
    cds = payload[cds_offset : cds_offset + MCHERRY_LEN]
    assert len(cds) == MCHERRY_LEN, (
        f"mCherry CDS truncated at cassette end: {len(cds)} != {MCHERRY_LEN}"
    )
    assert cds[-3:] in ("TAA", "TAG", "TGA"), (
        f"mCherry CDS does not end with stop codon (ends with {cds[-3:]})"
    )

    header = (
        f">hong_{CASSETTE_NAME} | len={len(payload)} bp | "
        f"mCherry CDS offset={cds_offset} len={MCHERRY_LEN} (incl. stop) | "
        f"source: Hong et al. 2026 Supp Table S3 row 1"
    )
    OUT_FASTA.parent.mkdir(parents=True, exist_ok=True)
    OUT_FASTA.write_text(header + "\n" + payload + "\n")
    return payload, cds_offset


def main() -> None:
    print(f"Loading R64-5-1 from {FASTA}", file=sys.stderr)
    fa = pysam.FastaFile(str(FASTA))
    genome = {ch: fa.fetch(ch).upper() for ch in fa.references}

    df = build_locus_table(genome)
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    n_intrain = (df["set"] == "IntTrain").sum()
    n_intprop = (df["set"] == "IntProp").sum()
    n_gRNA = df["gRNA"].notna().sum()
    print(
        f"Wrote {OUT_TSV.relative_to(REPO_ROOT)}: "
        f"{len(df)} rows ({n_intrain} IntTrain + {n_intprop} IntProp); "
        f"{n_gRNA} have a verified gRNA, {len(df) - n_gRNA} fall back to Table S4 int_site"
    )

    payload, cds_offset = build_cassette_fasta()
    print(
        f"Wrote {OUT_FASTA.relative_to(REPO_ROOT)}: cassette {len(payload)} bp, "
        f"mCherry CDS at offset {cds_offset} (length {MCHERRY_LEN})"
    )


if __name__ == "__main__":
    main()
