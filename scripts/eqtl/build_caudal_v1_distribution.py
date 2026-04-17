"""Build the v1 processed distribution for the Caudal eQTL benchmark.

Reads the raw per-iteration TSVs produced by
``scripts/eqtl/0_data_generation/1_generate_negs.py`` and converts them into
the processed schema documented in ``benchmarks/caudal_eqtl.md``:

- Chromosome names normalized to Arabic numerals (``"1"``..``"16"``).
- ``pair_id`` column added (0..N-1 after sorting by ``(pos_chrom, pos_pos)``).
- ``pos_gene_strand`` and ``neg_gene_strand`` looked up from the GTF.
- Columns reordered to the spec schema.

Also symlinks the reference FASTA and GTF under ``<out-dir>/reference/`` and
writes a README.

Usage::

    python scripts/eqtl/build_caudal_v1_distribution.py \\
        --raw-dir data/processed/caudal_eqtl_unchanged_run \\
        --gtf data/raw/Saccharomyces_cerevisiae.R64-1-1.115.gtf \\
        --fasta data/raw/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa \\
        --out-dir data/processed/caudal_eqtl_v1 \\
        --iterations 4
"""
from __future__ import annotations

import argparse
import re
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd


ROMAN_TO_ARABIC = {
    "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5",
    "VI": "6", "VII": "7", "VIII": "8", "IX": "9", "X": "10",
    "XI": "11", "XII": "12", "XIII": "13", "XIV": "14", "XV": "15",
    "XVI": "16",
}

_CHROM_WORD_RE = re.compile(r"^chromosome(\d+)$")
_GENE_ID_RE = re.compile(r'gene_id\s+"([^"]+)"')

SPEC_COLUMNS = [
    "pair_id",
    "pos_chrom", "pos_pos", "pos_ref", "pos_alt",
    "pos_gene", "pos_gene_strand", "pos_distance_to_tss",
    "neg_chrom", "neg_pos", "neg_ref", "neg_alt",
    "neg_gene", "neg_gene_strand", "neg_distance_to_tss",
]


def to_arabic_chrom(chrom: str) -> str:
    """Normalize 'chromosome7' / 'VII' / '7' to the canonical '7'."""
    m = _CHROM_WORD_RE.match(chrom)
    if m:
        n = m.group(1)
    elif chrom.isdigit():
        n = chrom
    elif chrom in ROMAN_TO_ARABIC:
        n = ROMAN_TO_ARABIC[chrom]
    else:
        raise ValueError(f"Unrecognized chromosome name: {chrom!r}")
    if not 1 <= int(n) <= 16:
        raise ValueError(f"Chromosome {chrom!r} out of yeast range 1..16")
    return n


def parse_gene_strand_map(gtf_path: Path) -> dict[str, str]:
    """Return {gene_id: '+' or '-'} from GTF 'gene' features."""
    strand_by_gene: dict[str, str] = {}
    with gtf_path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2].lower() != "gene":
                continue
            m = _GENE_ID_RE.search(fields[8])
            if not m:
                continue
            strand_by_gene[m.group(1)] = fields[6]
    return strand_by_gene


def build_iteration(raw_tsv: Path, strand_by_gene: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(raw_tsv, sep="\t")
    df["pos_chrom"] = df["pos_chrom"].map(to_arabic_chrom)
    df["neg_chrom"] = df["neg_chrom"].map(to_arabic_chrom)

    df["pos_gene_strand"] = df["pos_gene"].map(strand_by_gene)
    df["neg_gene_strand"] = df["neg_gene"].map(strand_by_gene)
    missing_pos = df.loc[df["pos_gene_strand"].isna(), "pos_gene"].unique().tolist()
    missing_neg = df.loc[df["neg_gene_strand"].isna(), "neg_gene"].unique().tolist()
    if missing_pos or missing_neg:
        raise ValueError(
            "GTF missing strand for gene_ids: "
            f"pos={missing_pos[:5]} (n={len(missing_pos)}), "
            f"neg={missing_neg[:5]} (n={len(missing_neg)})"
        )

    df = df.sort_values(
        by=["pos_chrom", "pos_pos"],
        key=lambda s: s.astype(int) if s.name == "pos_chrom" else s,
        kind="stable",
    ).reset_index(drop=True)
    df.insert(0, "pair_id", df.index)
    return df[SPEC_COLUMNS]


def link_reference(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    return start.resolve()


def write_readme(out_dir: Path, n_per_iter: int, iterations: int, commit: str) -> None:
    (out_dir / "README.md").write_text(
        f"""# caudal_eqtl_v1

Processed distribution for the Caudal et al. yeast cis-eQTL classification
benchmark. See `benchmarks/caudal_eqtl.md` for the full spec.

- **Version:** v1
- **Generated:** {date.today().isoformat()}
- **Source commit:** {commit}
- **Rows per iteration:** {n_per_iter}
- **Iterations:** {iterations}

## Files

- `negset_{{1..{iterations}}}.tsv` — one paired (positive, negative) row per
  line. Chromosome naming: Arabic numerals (`1`..`16`), no prefix.
- `reference/R64-1-1.fa` — reference FASTA (chromosomes inside the file are
  Roman numerals).
- `reference/R64-1-1.115.gtf` — Ensembl 115 annotation.

Reference files are symlinks into `data/raw/` for local development. A
publishable bundle copies them as real files.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing GTF for gene strands: {args.gtf}")
    strand_by_gene = parse_gene_strand_map(args.gtf)
    print(f"  Loaded {len(strand_by_gene)} gene strand entries.")

    rows_per_iter: list[int] = []
    for i in range(1, args.iterations + 1):
        raw_tsv = args.raw_dir / f"negset_set{i}.tsv"
        out_tsv = args.out_dir / f"negset_{i}.tsv"
        print(f"Building iteration {i}: {raw_tsv.name} -> {out_tsv.name}")
        df = build_iteration(raw_tsv, strand_by_gene)
        df.to_csv(out_tsv, sep="\t", index=False)
        rows_per_iter.append(len(df))
        print(f"  Wrote {len(df)} rows.")

    if len(set(rows_per_iter)) != 1:
        print(f"WARNING: inconsistent row counts across iterations: {rows_per_iter}")

    link_reference(args.fasta, args.out_dir / "reference" / "R64-1-1.fa")
    link_reference(args.gtf, args.out_dir / "reference" / "R64-1-1.115.gtf")
    fai = args.fasta.with_suffix(args.fasta.suffix + ".fai")
    if fai.exists():
        link_reference(fai, args.out_dir / "reference" / "R64-1-1.fa.fai")

    write_readme(
        args.out_dir,
        n_per_iter=rows_per_iter[0],
        iterations=args.iterations,
        commit=git_head(find_repo_root(args.out_dir)),
    )
    print("Done.")


if __name__ == "__main__":
    main()
