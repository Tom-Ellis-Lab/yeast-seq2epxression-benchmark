"""Build the verified Wu et al. RFP cassette payload FASTA.

Freezes the *constant* scaffold that is identical across all 1044
integrant strains, following the Giaever & Nislow 2014 deletion-cassette
design (genomic homology arms OUTERMOST, universal priming sites + the
two strain-specific 20 bp barcodes INSIDE the arms):

    U1 - [UPTAG] - U2 - <RFP-TU core> - D2 - [DNTAG] - D1

- The RFP-TU core (GenBank 210..3619, 3410 bp;
  tCYC1-pURA3-mCherry-tADH1-pLEU2-LEU2-tLEU2) is sequence-verified by
  scripts/wu/verify_cassette.py.
- U1/U2/D1/D2 universal sites are constant and match Giaever 2014
  Fig. 1B exactly.
- UPTAG/DNTAG are strain-specific (one unique pair per deleted ORF) and
  are emitted as 20 x N placeholders; the adapter injects the real
  per-locus barcodes from the SGD deletion barcode table at splice time.
- The homology arms are NOT in the payload: they are the native genomic
  junction (the adapter splices this scaffold between native left/right
  flanks at the SGDP deletion boundary).

Run: uv run python scripts/wu/build_cassette_fasta.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GB = ROOT / "data/tasks/wu_rfpins/cassette_reconstruction.gb"
OUT = ROOT / "data/tasks/wu_rfpins/expression_cassette.fasta"

TAG = "N" * 20  # strain-specific barcode placeholder (injected per locus)


def gb_sequence(path: Path) -> str:
    txt = path.read_text()
    body = txt.split("ORIGIN", 1)[1].split("//")[0]
    return re.sub(r"[^acgtnACGTN]", "", body).upper()


def main() -> None:
    seq = gb_sequence(GB)
    assert len(seq) == 4210, f"unexpected GenBank length {len(seq)}"

    u1 = seq[0:18]                       # GATGTCCACGAGGTCTCT
    u2 = seq[38:56]                      # CGTACGCTGCAGGTCGAC
    core = seq[209:3619]                 # 210..3619, 3410 bp, verified
    comp = str.maketrans("ACGTN", "TGCAN")
    d2 = seq[4154:4173].translate(comp)[::-1]   # ATCGATGAATTCGAGCTCG
    d1 = seq[4193:4210].translate(comp)[::-1]   # CGGTGTCGGTCTCGTAG

    # sanity: universal sites must match Giaever 2014 Fig. 1B
    assert u1 == "GATGTCCACGAGGTCTCT", u1
    assert u2 == "CGTACGCTGCAGGTCGAC", u2
    assert d2 == "ATCGATGAATTCGAGCTCG", d2
    assert d1 == "CGGTGTCGGTCTCGTAG", d1
    assert len(core) == 3410, len(core)

    payload = u1 + TAG + u2 + core + d2 + TAG + d1
    header = (
        ">wu_rfpins_cassette_scaffold "
        "U1-[UPTAG]-U2-RFPTUcore(tCYC1-pURA3-mCherry-tADH1-pLEU2-LEU2-tLEU2)"
        "-D2-[DNTAG]-D1 | core=GenBank:210..3619(3410bp) verified | "
        "N20=strain-specific barcode placeholder (inject from SGD table) | "
        "homology arms NOT included (native genomic junction) | "
        f"len={len(payload)}"
    )
    wrapped = "\n".join(payload[i:i + 70] for i in range(0, len(payload), 70))
    OUT.write_text(f"{header}\n{wrapped}\n")
    print(f"wrote {OUT} ({len(payload)} bp)")
    print(f"  U1={u1}  U2={u2}")
    print(f"  D2={d2}  D1={d1}")
    print(f"  core={len(core)} bp  total={len(payload)} bp "
          f"(= 18+20+18+3410+19+20+17)")


if __name__ == "__main__":
    main()
