"""Build the Cuperus HIS3-reporter construct flanks from the R64-1-1 genome.

The assay's reporter is ``CYC1`` promoter - [50 bp UTR] - ``HIS3`` CDS -
``CYC1`` terminator on a p415-CYC1 plasmid (Mumberg 1995). We could not find a
clean machine-readable p415-CYC1 sequence, so the flanks are reconstructed from
the genome at fixed coordinates — and this is faithful: the genomic ``CYC1``
promoter comes out at exactly 298 bp (the length the paper states), and both
junctions match the paper's own cloning overhangs (Methods, native library):

  5' overhang overlapping the CYC1 promoter : acattaggacctttgcagc
  3' overhang overlapping HIS3              : ATGacagagcagaaagccct  (ATG = HIS3 start)

so the 50 bp insert sits directly between the promoter (…ACATTAGGACCTTTGCAGC)
and the HIS3 ATG, with no cloning scar. The promoter/terminator are constant
context across all test UTRs, so their exact boundaries shift only the baseline,
not the UTR-driven ranking; HIS3 (the readout) and the UTR (variable) are exact.

Writes ``data/tasks/cuperus_mpra_5utr/construct.json``: the three flank
sequences + coordinates. The adapter assembles ``promoter + UTR + HIS3 +
terminator`` per UTR and reads out the HIS3 ORF.

Usage:  uv run python scripts/cuperus/build_construct.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pysam

# Fixed R64-1-1 coordinates (1-based inclusive, all + strand).
HIS3 = ("XV", 721946, 722608)        # YOR202W ORF incl. stop (663 bp)
CYC1_PROMOTER = ("X", 525981, 526278)  # 298 bp, ends at the 5' cloning overhang
CYC1_TERMINATOR = ("X", 526665, 526914)  # 250 bp downstream of the CYC1 stop

# Paper Methods cloning overhangs that verify the junctions.
OVERHANG_5 = "ACATTAGGACCTTTGCAGC"      # promoter 3' end (abuts the insert)
HIS3_HEAD = "ATGACAGAGCAGAAAGCCCT"      # HIS3 ORF 5' end (ATG abuts the insert)
STOPS = {"TAA", "TAG", "TGA"}


def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "data" / "tasks" / "R64-1-1.fa").exists():
            return parent
    raise FileNotFoundError("R64-1-1.fa not found under data/tasks from %s" % p)


def _fetch(fa: pysam.FastaFile, region: tuple[str, int, int]) -> str:
    chrom, start1, end1 = region
    return fa.fetch(chrom, start1 - 1, end1).upper()


def build(root: Path) -> None:
    fa = pysam.FastaFile(str(root / "data" / "tasks" / "R64-1-1.fa"))
    promoter = _fetch(fa, CYC1_PROMOTER)
    his3 = _fetch(fa, HIS3)
    terminator = _fetch(fa, CYC1_TERMINATOR)

    # Verify against the paper (Methods) — these are the load-bearing checks.
    assert len(promoter) == 298, f"CYC1 promoter is {len(promoter)} bp, expected 298"
    assert promoter.endswith(OVERHANG_5), "promoter 3' end != paper 5' overhang"
    assert his3.startswith(HIS3_HEAD), "HIS3 ORF 5' end != paper 3' overhang"
    assert his3[:3] == "ATG" and his3[-3:] in STOPS and len(his3) % 3 == 0, "HIS3 not a clean ORF"

    out = root / "data" / "tasks" / "cuperus_mpra_5utr"
    out.mkdir(parents=True, exist_ok=True)
    construct = {
        "cyc1_promoter": promoter,
        "his3_orf": his3,
        "cyc1_terminator": terminator,
        "assembly": "cyc1_promoter + UTR + his3_orf + cyc1_terminator",
        "readout": "his3_orf (starts at len(cyc1_promoter)+len(UTR) in the assembled construct)",
        "coords_R64_1_1": {
            "his3_orf": list(HIS3),
            "cyc1_promoter": list(CYC1_PROMOTER),
            "cyc1_terminator": list(CYC1_TERMINATOR),
        },
        "source": "reconstructed from R64-1-1.fa; junctions verified vs Cuperus 2017 Methods overhangs",
    }
    (out / "construct.json").write_text(json.dumps(construct, indent=2) + "\n")
    print(f"CYC1 promoter   {len(promoter)} bp  …{promoter[-19:]}")
    print(f"HIS3 ORF        {len(his3)} bp  {his3[:20]}… {his3[-3:]}")
    print(f"CYC1 terminator {len(terminator)} bp")
    print(f"wrote {out / 'construct.json'}  (junctions verified)")


if __name__ == "__main__":
    build(_find_repo_root())
