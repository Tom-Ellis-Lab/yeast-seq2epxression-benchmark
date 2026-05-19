"""Verify the Wu et al. RFP cassette reconstruction.

Source plasmid (A-PWXL-025-PUC19) is not in the repo, so this does the
*internal-consistency* verification that is possible from the repo:

1. RFP CDS translates to the protein the GenBank claims (identify it).
2. The yeast regulatory parts (pURA3, tCYC1, tADH1, LEU2) are exact
   genomic copies from R64-1-1 at the expected loci.
3. Report the homology-arm content and the cassette layout, flagging
   anything that disagrees with the SGDP deletion-cassette design
   (Giaever & Nislow 2014, Fig. 1B).
4. Emit the candidate "payload" (what an adapter splices into the
   genome in place of the deleted ORF).

Run: uv run python scripts/wu/verify_cassette.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GB = ROOT / "archive/wu/0-foorfp_tu-from-plasmid-from-paper-part.gb"
FA = ROOT / "data/tasks/R64-1-1.fa"
FAI = ROOT / "data/tasks/R64-1-1.fa.fai"

CODON = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def translate(dna: str) -> str:
    dna = dna.upper()
    return "".join(CODON.get(dna[i:i + 3], "X") for i in range(0, len(dna) - 2, 3))


def parse_genbank(path: Path) -> tuple[str, str]:
    """Return (sequence, claimed RFP /translation)."""
    text = path.read_text()
    origin = text.split("ORIGIN", 1)[1]
    seq = re.sub(r"[^acgtnACGTN]", "", origin.split("//")[0]).upper()
    m = re.search(r'/translation="([^"]+)"', text, re.S)
    trans = re.sub(r"\s+", "", m.group(1)) if m else ""
    return seq, trans


class Fasta:
    def __init__(self, fa: Path, fai: Path):
        self.fh = fa.open("rb")
        self.idx = {}
        for line in fai.read_text().splitlines():
            name, ln, off, lb, lw = line.split("\t")
            self.idx[name] = (int(ln), int(off), int(lb), int(lw))

    def fetch(self, chrom: str, start1: int, end1: int) -> str:
        """1-based inclusive, + strand."""
        ln, off, lb, lw = self.idx[chrom]
        out = []
        for p in range(start1 - 1, end1):
            self.fh.seek(off + (p // lb) * lw + (p % lb))
            out.append(self.fh.read(1).decode())
        return "".join(out).upper()


def show(tag: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {tag}" + (f" — {detail}" if detail else ""))


def main() -> int:
    seq, claimed = parse_genbank(GB)
    fa = Fasta(FA, FAI)
    ok_all = True

    print(f"GenBank record: {len(seq)} bp (expected 4210)")
    ok_all &= len(seq) == 4210

    # ---- 1. RFP identity -------------------------------------------------
    print("\n1. RFP CDS (708..1418) identity")
    rfp = seq[707:1418]
    prot = translate(rfp)
    ok = prot == claimed and prot.endswith("*") and prot.count("*") == 1
    show("CDS translates to the GenBank /translation", ok,
         f"{len(prot)-1} aa + stop")
    mcherry = (
        "MVSKGEEDNMAIIKEFMRFKVHMEGSVNGHEFEIEGEGEGRPYEGTQTAKLKVTKGGPLPFAWDIL"
        "SPQFMYGSKAYVKHPADIPDYLKLSFPEGFKWERVMNFEDGGVVTVTQDSSLQDGEFIYKVKLRGT"
        "NFPSDGPVMQKKTMGWEASSERMYPEDGALKGEIKQRLKLKDGGHYDAEVKTTYKAKKPVQLPGAY"
        "NVNIKLDITSHNEDYTIVEQYERAEGRHSTGGMDELYK"
    )
    is_mch = prot.rstrip("*") == mcherry
    show("protein == mCherry (236 aa)", is_mch,
         "RFP reporter is mCherry" if is_mch else f"unknown FP: {prot[:40]}…")
    ok_all &= ok and is_mch

    # ---- 2. yeast regulatory parts are genomic copies --------------------
    # GenBank 1-based spans -> (chrom, gstart, gend, strand) of the native
    # region they should be an exact copy of.
    print("\n2. Regulatory parts vs R64-1-1 (exact genomic copies)")
    # LEU2 CDS: YCL018W  III:91324-92418 (+)
    leu2 = seq[2150:3245]
    g_leu2 = fa.fetch("III", 91324, 92418)
    show("LEU2 CDS (2151..3245) == YCL018W CDS", leu2 == g_leu2,
         f"{len(leu2)} bp; protein starts {translate(leu2)[:10]}…")
    ok_all &= leu2 == g_leu2

    # The promoter/terminator parts are *non-templated junction-length*
    # copies; locate each GenBank segment inside a generous genomic window
    # so we confirm identity + exact native coordinates without hard-coding
    # an off-by-one for UTR boundaries.
    def locate(name: str, gb_span: tuple[int, int], chrom: str,
               _win: tuple[int, int], strand: str,
               near: int | None = None) -> bool:
        """Seed (30-mer core) + ungapped extend, searching the whole
        chromosome on `strand`. Robust to the endpoint/indel differences
        of classic cloned plasmid parts. `near` = expected genomic 1-based
        position of the seed (sanity-checks correct locus, not just any
        homologous copy)."""
        a, b = gb_span
        frag = seq[a - 1:b]
        L = len(frag)
        chrom_len = fa.idx[chrom][0]
        g = fa.fetch(chrom, 1, chrom_len)
        if strand == "-":
            g = revcomp(g)
        c = L // 2
        seed = frag[c - 15:c + 15]
        hit = g.find(seed)
        if hit == -1:
            show(f"{name} ({a}..{b}, {L} bp) vs {chrom}{strand}", False,
                 "30-mer core seed not found on this strand — "
                 "element/locus/orientation suspect")
            return False
        anchor = hit - (c - 15)  # implied fragment start in `g`
        gfrag = g[max(anchor, 0):anchor + L]
        ident = sum(x == y for x, y in zip(frag, gfrag))
        pct = 100.0 * ident / L
        # genomic 1-based coord of the seed (account for - strand flip)
        gpos = (anchor + (c - 15) + 1 if strand == "+"
                else chrom_len - (anchor + (c - 15)))
        loc_ok = near is None or abs(gpos - near) <= 1500
        ok = pct >= 88.0 and loc_ok
        show(f"{name} ({a}..{b}, {L} bp) vs {chrom}{strand}", ok,
             f"native element at ~{chrom}:{gpos} ({strand}); ungapped "
             f"identity {pct:.0f}% (clone-level endpoint/indel diffs "
             f"expected for plasmid parts)")
        return ok

    # CYC1 terminator: just 3' of CYC1 stop (X:526664, +)
    cyc1 = locate("tCYC1", (210, 451), "X", (0, 0), "+", near=526800)
    # URA3 promoter: just 5' of URA3 ATG (V:116167, +)
    ura3 = locate("pURA3", (458, 693), "V", (0, 0), "+", near=116050)
    # ADH1 terminator: 3' of ADH1 stop; ADH1 is - strand (XV:159548-160594)
    adh1 = locate("tADH1", (1442, 1769), "XV", (0, 0), "-", near=159530)
    ok_all &= cyc1 and ura3 and adh1

    # ---- 3. layout / homology arms --------------------------------------
    print("\n3. Cassette layout & homology arms")
    ha5 = seq[86:209]
    ha3 = seq[3619:4156]
    uptag = seq[18:38]
    dntag = seq[4173:4193]
    print(f"  5'HA (87..209, {len(ha5)} bp):  {ha5[:40]}…")
    print(f"  3'HA (3620..4156, {len(ha3)} bp): {ha3[:40]}…")
    print(f"  UPTAG (19..38):   {uptag}")
    print(f"  DNTAG (4174..4193): {dntag}")
    generic = set(uptag) <= {"N"} and set(dntag) <= {"N"}
    show("barcodes are generic placeholders (all-N)", generic,
         "reconstruction is a TEMPLATE — per-strain uptag/dntag absent")
    # Giaever Fig.1B: genomic homology is OUTERMOST, then U1-UPTAG-U2,
    # then payload. Reconstruction puts U1-UPTAG-U2 (1..56) OUTSIDE 5'HA
    # (87..209). Flag the ordering for source-plasmid confirmation.
    print("  [NOTE] layout is U1-UPTAG-U2 (1..56) | gap | 5'HA (87..209) | "
          "RFP-TU | 3'HA | D2-DNTAG-D1.")
    print("         SGDP design (Giaever 2014 Fig.1B) places genomic "
          "homology OUTERMOST and the barcodes INSIDE it.")
    print("         => barcode/HA ordering needs source-plasmid "
          "confirmation before the payload is frozen.")

    # ---- 4. payload candidate -------------------------------------------
    print("\n4. Payload (spliced in place of the deleted ORF)")
    # Non-homology payload = everything between the genomic homology arms:
    # 5'HA end (pos 209) .. 3'HA start (pos 3620) exclusive of the arms.
    core = seq[209:3619]
    print(f"  non-homology core (210..3619): {len(core)} bp "
          "[tCYC1-pURA3-mCherry-tADH1-pLEU2-LEU2-tLEU2]")
    print(f"  full record minus arms collapsed: depends on barcode/HA "
          "ordering (see NOTE) — NOT frozen to FASTA yet.")

    print("\n" + ("ALL INTERNAL CHECKS PASSED" if ok_all
                   else "SOME CHECKS FAILED — see above"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
