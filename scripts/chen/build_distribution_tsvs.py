"""Build the per-library TSVs for the Chen 2017 synonymous-mutation benchmark.

Reads the published supplementary workbook (Tables S7/S8/S9) and writes:

    data/tasks/chen_synonymous/gfp_r1.tsv      (1124 rows, S7)
    data/tasks/chen_synonymous/gfp_r2.tsv      (2432 rows, S8)
    data/tasks/chen_synonymous/tdh3.tsv        ( 523 rows, S9)
    data/tasks/chen_synonymous/replicate_ceilings.json

CAI / tAI / MFE / GC3 columns are carried through from the supp tables verbatim
(see benchmarks/chen_synonymous.md for why we don't recompute CAI).

Sanity checks (must pass at build time, else the script raises):
    - variable_seq is exactly 36 nt
    - variable_seq translates to the published 12-aa peptide for its library
    - row count matches the paper (1124 / 2432 / 523)
    - empirical replicate-replicate Pearson is within ±0.01 of the published
      ceiling (0.83 / 0.73 / 0.72)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_REPO = Path(__file__).resolve().parents[2]
DEFAULT_XLSX = DEFAULT_REPO / "archive" / "chen" / "msx229_Supptables.xlsx"
DEFAULT_OUT_DIR = DEFAULT_REPO / "data" / "tasks" / "chen_synonymous"

# (library_id, sheet_name, expected_rows, expected_peptide, published_ceiling)
LIBRARIES = [
    ("gfp_r1", "Table S7", 1124, "LTLKFICTTGKL", 0.83),
    ("gfp_r2", "Table S8", 2432, "QKNGIKVNFKIR", 0.73),
    ("tdh3",   "Table S9",  523, "EVSHDDKHIIVD", 0.72),
]

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Header row 2 (0-indexed) → canonical column name.
RENAME_S78 = {
    "Synonymous variants": "variable_seq",
    "CAI": "CAI",
    "tAI": "tAI",
    "MFE": "MFE",
    "GC3": "GC3",
    "Normalized log2(mRNA) of replicate 1": "log2mRNA_rep1",
    "Normalized log2(mRNA) of replicate 2": "log2mRNA_rep2",
    "log2(protein) of replicate 1": "log2protein_rep1",
    "log2(protein) of replicate 2": "log2protein_rep2",
    "degradation rate": "degradation_rate",
}

RENAME_S9 = {
    "Synonymous variants": "variable_seq",
    "CAI": "CAI",
    "MFE": "MFE",
    "GC3": "GC3",
    "Normalized log2(mRNA)": "log2mRNA",
}


def translate(seq: str) -> str:
    seq = seq.upper()
    return "".join(CODON_TABLE[seq[i : i + 3]] for i in range(0, len(seq), 3))


def read_sheet(xlsx: Path, sheet: str) -> pd.DataFrame:
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    # Row 0 = title, row 1 = blank, row 2 = header, row 3+ = data.
    header = list(rows[2])
    data = rows[3:]
    df = pd.DataFrame(data, columns=header)
    # Drop fully-empty trailing rows.
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def build_library(
    library_id: str,
    sheet: str,
    expected_rows: int,
    expected_peptide: str,
    published_ceiling: float,
    xlsx: Path,
    out_dir: Path,
) -> dict:
    df = read_sheet(xlsx, sheet)
    rename = RENAME_S9 if library_id == "tdh3" else RENAME_S78
    # Keep only columns we know about; drop the all-None tail column S8 has.
    keep_in = [c for c in df.columns if c in rename]
    df = df[keep_in].rename(columns=rename)

    if len(df) != expected_rows:
        raise ValueError(
            f"{library_id}: expected {expected_rows} rows, got {len(df)}"
        )

    df["variable_seq"] = df["variable_seq"].astype(str).str.upper()
    if not (df["variable_seq"].str.len() == 36).all():
        raise ValueError(f"{library_id}: not all variable_seq are 36 nt")

    peptides = df["variable_seq"].map(translate)
    bad = peptides[peptides != expected_peptide]
    if len(bad) > 0:
        sample = bad.head(3).tolist()
        raise ValueError(
            f"{library_id}: {len(bad)} variants don't translate to "
            f"{expected_peptide!r}; first bad peptides: {sample}"
        )

    if library_id == "tdh3":
        empirical_ceiling = float("nan")  # single column, no replicate comparison
    else:
        empirical_ceiling = float(
            df["log2mRNA_rep1"].astype(float).corr(df["log2mRNA_rep2"].astype(float))
        )
        if abs(empirical_ceiling - published_ceiling) > 0.01:
            raise ValueError(
                f"{library_id}: empirical replicate Pearson "
                f"{empirical_ceiling:.3f} differs from published "
                f"{published_ceiling} by >0.01"
            )

    df.insert(0, "variant_id", [f"{library_id}_{i:05d}" for i in range(len(df))])
    out_path = out_dir / f"{library_id}.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    log.info(
        "%s: %d rows → %s  (replicate r=%.3f, published=%.2f)",
        library_id, len(df), out_path,
        empirical_ceiling if not np.isnan(empirical_ceiling) else 0.0,
        published_ceiling,
    )
    return {
        "library_id": library_id,
        "n_rows": int(len(df)),
        "published_ceiling_pearson": float(published_ceiling),
        "empirical_ceiling_pearson": (
            None if np.isnan(empirical_ceiling) else float(empirical_ceiling)
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for library_id, sheet, expected_rows, peptide, ceiling in LIBRARIES:
        summaries.append(
            build_library(
                library_id, sheet, expected_rows, peptide, ceiling,
                args.xlsx, args.out_dir,
            )
        )

    ceilings = {s["library_id"]: s["published_ceiling_pearson"] for s in summaries}
    (args.out_dir / "replicate_ceilings.json").write_text(
        json.dumps(ceilings, indent=2) + "\n"
    )
    log.info("wrote replicate_ceilings.json: %s", ceilings)


if __name__ == "__main__":
    main()
