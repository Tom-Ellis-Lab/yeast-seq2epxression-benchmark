"""Build the processed MYTK integration-site × promoter benchmark table.

Joins the two raw SI inputs into one tidy long-format table the
benchmark loads directly — one row per (integration locus, promoter):

  1. ``mytk_ints_promoters_expression.csv`` — wide, spreadsheet-style:
     one block of columns per promoter (``rep1, rep2, rep3, mean, SD``)
     separated by blank columns, promoter names in the header row. Rows
     are the 11 loci (``ura3`` control + ``Int.1`` … ``Int.10``).
     Measurement: mScarlet mean fluorescence, fold-over-background (no
     log), 3 biological replicates.
  2. ``mytk_int_coordinates.csv`` — per-locus integration coordinates
     (``int_locus_id, chromosome, integration_coord_5_prime,
     integration_coord_3_prime, …``), 1-based R64. The cassette is
     modelled as inserted at the **midpoint** of the 5′/3′ span, with
     native flanking sequence left intact (Hong-style point insertion).

Output TSV columns:
    locus_id, promoter, chrom, integration_coord, rep1, rep2, rep3, mean, sd

``chrom`` is the Roman-numeral contig name (``I`` … ``XVI``) matching the
R64 FASTA the adapters fetch from. ``integration_coord`` is 1-based.

Stdlib-only (``csv``) so it runs on any Python without the project env.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "tasks" / "mytk_ints_promoter"
DEFAULT_EXPR = DEFAULT_DIR / "mytk_ints_promoters_expression.csv"
DEFAULT_COORDS = DEFAULT_DIR / "mytk_int_coordinates.csv"
DEFAULT_OUT = DEFAULT_DIR / "mytk_ints_promoter.tsv"

N_REPS = 3  # replicate columns per promoter block, followed by mean + SD
_NON_PROMOTER = {"mean", "sd", "nan", ""}
OUT_COLUMNS = (
    "locus_id", "promoter", "chrom", "integration_coord",
    "rep1", "rep2", "rep3", "mean", "sd",
)

# Arabic → Roman contig names, matching yeastbench.adapters._genome.ARABIC_TO_ROMAN.
# Inlined (not imported) to keep this build script runnable without the
# project env (importing _genome pulls in numpy).
ARABIC_TO_ROMAN = {
    "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V",
    "6": "VI", "7": "VII", "8": "VIII", "9": "IX", "10": "X",
    "11": "XI", "12": "XII", "13": "XIII", "14": "XIV", "15": "XV",
    "16": "XVI",
}


def _promoter_blocks(header: list[str]) -> dict[str, int]:
    """Map promoter name → first replicate column index.

    A promoter name in the header row marks the start of its block; the
    three columns from that name onward are its replicates (mean/SD
    follow but are recomputed here, not trusted from the export).
    """
    blocks: dict[str, int] = {}
    for i, cell in enumerate(header):
        name = cell.strip()
        if name.lower() not in _NON_PROMOTER:
            blocks[name] = i
    return blocks


def load_labels(expr_path: Path) -> dict[tuple[str, str], dict]:
    """Reshape the wide expression CSV to {(locus_id, promoter): row}."""
    with open(expr_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    blocks = _promoter_blocks(header)
    log.info("found %d promoter blocks: %s", len(blocks), list(blocks))

    out: dict[tuple[str, str], dict] = {}
    for r in body:
        if not r or not r[0].strip() or r[0].strip().lower() == "nan":
            continue
        locus_id = r[0].strip()
        for promoter, start in blocks.items():
            reps = [float(r[start + k]) for k in range(N_REPS)]
            mean = sum(reps) / len(reps)
            # sample SD (÷ N-1) to reproduce the published table's values.
            # The benchmark uses only `mean` as the label; `sd` is context.
            sd = (sum((x - mean) ** 2 for x in reps) / (len(reps) - 1)) ** 0.5
            out[(locus_id, promoter)] = {
                "rep1": reps[0], "rep2": reps[1], "rep3": reps[2],
                "mean": mean, "sd": sd,
            }
    return out


def load_coords(coord_path: Path) -> dict[str, dict]:
    """Load per-locus {locus_id: {chrom (Roman), integration_coord (midpoint)}}."""
    out: dict[str, dict] = {}
    with open(coord_path, newline="", encoding="utf-8-sig") as fh:
        # Header has stray whitespace (e.g. " chromosome"); normalise keys.
        reader = csv.DictReader(fh)
        reader.fieldnames = [f.strip() for f in (reader.fieldnames or [])]
        for row in reader:
            row = {k.strip(): (v.strip() if v is not None else v) for k, v in row.items()}
            locus_id = row["int_locus_id"]
            arabic = row["chromosome"]
            chrom = ARABIC_TO_ROMAN.get(arabic)
            if chrom is None:
                raise ValueError(f"Unknown chromosome {arabic!r} for {locus_id}")
            five = int(row["integration_coord_5_prime"])
            three = int(row["integration_coord_3_prime"])
            out[locus_id] = {"chrom": chrom, "integration_coord": (five + three) // 2}
    return out


def build(expr_path: Path, coord_path: Path) -> list[dict]:
    labels = load_labels(expr_path)
    coords = load_coords(coord_path)

    label_loci = {locus for locus, _ in labels}
    missing = label_loci - set(coords)
    if missing:
        raise ValueError(f"loci in labels but missing coordinates: {sorted(missing)}")

    rows: list[dict] = []
    for (locus_id, promoter), lab in labels.items():
        c = coords[locus_id]
        rows.append({
            "locus_id": locus_id,
            "promoter": promoter,
            "chrom": c["chrom"],
            "integration_coord": c["integration_coord"],
            **lab,
        })
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expr", type=Path, default=DEFAULT_EXPR)
    ap.add_argument("--coords", type=Path, default=DEFAULT_COORDS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = build(args.expr, args.coords)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLUMNS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    n_loci = len({r["locus_id"] for r in rows})
    n_prom = len({r["promoter"] for r in rows})
    log.info("wrote %d rows (%d loci × %d promoters) → %s",
             len(rows), n_loci, n_prom, args.out)


if __name__ == "__main__":
    main()
