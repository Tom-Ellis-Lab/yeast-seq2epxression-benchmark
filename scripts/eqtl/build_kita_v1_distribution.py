"""Build the v1 processed distribution for the Kita eQTL benchmark.

Reads Kuanhao's emailed Kita negsets (`negative_eqtls_set{i}.tsv` under
``shorkie-paper/from_kuanhao/eQTL/data/kita_etal/negative/``) and converts them
into the same processed schema we use for Caudal:

- Chromosome names normalized to Arabic numerals (``"1"``..``"16"``).
- ``pair_id`` column added (0..N-1 after sorting by ``(pos_chrom, pos_pos)``).
- ``pos_gene_strand`` and ``neg_gene_strand`` looked up from the GTF.
- Columns reordered to match ``docs/benchmarks/caudal_eqtl.md`` schema.

Reference FASTA/GTF live at ``data/tasks/`` (shared with Caudal).

Usage::

    python scripts/eqtl/build_kita_v1_distribution.py \\
        --raw-dir shorkie-paper/from_kuanhao/eQTL/data/kita_etal/negative \\
        --gtf data/tasks/R64-1-1.115.gtf \\
        --out-dir data/tasks/kita_eqtl \\
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


def write_readme(out_dir: Path, n_per_iter: list[int], iterations: int, commit: str) -> None:
    rows_str = "/".join(str(n) for n in n_per_iter)
    (out_dir / "README.md").write_text(
        f"""# kita_eqtl_v1

Processed distribution for the Kita et al. yeast cis-eQTL classification
benchmark. See `docs/benchmarks/kita_eqtl.md` for the full spec.

- **Version:** v1
- **Generated:** {date.today().isoformat()}
- **Source commit:** {commit}
- **Source files:** `shorkie-paper/from_kuanhao/eQTL/data/kita_etal/negative/negative_eqtls_set{{1..{iterations}}}.tsv` (Kuanhao Chao, emailed 2026-04-30).
- **Rows per iteration:** {rows_str}
- **Iterations:** {iterations}

## Files

- `negset_{{1..{iterations}}}.tsv` — one paired (positive, negative) row per
  line. Chromosome naming: Arabic numerals (`1`..`16`), no prefix. Schema
  matches `caudal_eqtl_v1`.

Reference FASTA + GTF live at `data/tasks/R64-1-1.{{fa,115.gtf}}` (shared
across eQTL tasks).

## Provenance note

The raw negsets were generated by Kuanhao's pipeline against Ensembl Fungi
release 59. We re-annotate `pos_gene_strand` / `neg_gene_strand` using
Ensembl 115 (the release used by the rest of this benchmark) and keep
all other Kuanhao-generated columns as-is. All Kita gene_ids (positives
and negatives, including ncRNAs like `ICR1`, `snR3`, `tA(AGC)F`) resolve
in the E115 GTF.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir", type=Path, required=True,
        help="Directory containing negative_eqtls_set{i}.tsv from Kuanhao",
    )
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing GTF for gene strands: {args.gtf}")
    strand_by_gene = parse_gene_strand_map(args.gtf)
    print(f"  Loaded {len(strand_by_gene)} gene strand entries.")

    rows_per_iter: list[int] = []
    for i in range(1, args.iterations + 1):
        raw_tsv = args.raw_dir / f"negative_eqtls_set{i}.tsv"
        out_tsv = args.out_dir / f"negset_{i}.tsv"
        print(f"Building iteration {i}: {raw_tsv.name} -> {out_tsv.name}")
        df = build_iteration(raw_tsv, strand_by_gene)
        df.to_csv(out_tsv, sep="\t", index=False)
        rows_per_iter.append(len(df))
        print(f"  Wrote {len(df)} rows.")

    write_readme(
        args.out_dir,
        n_per_iter=rows_per_iter,
        iterations=args.iterations,
        commit=git_head(find_repo_root(args.out_dir)),
    )
    print("Done.")


if __name__ == "__main__":
    main()
