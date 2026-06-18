"""Build ``barcodes.tsv`` — per-ORF UPTAG/DNTAG for the Wu RFP cassette.

Each YKO deletion strain carries two unique 20 bp molecular barcodes
(UPTAG, DNTAG) that the Wu reporter cassette inherits at its junctions
(``U1-UPTAG-U2 … D2-DNTAG-D1``).  The frozen cassette scaffold
(``expression_cassette.fasta``) stores both as 20×N placeholders; the
track adapters inject the real per-locus barcodes at splice time.  A run
of N one-hot-encodes to all-zero columns — an out-of-distribution input
the models were never trained on — so we replace them with the real tags.

Source
------
The canonical Saccharomyces Genome Deletion Project (SGTC / Giaever)
barcode + primer master table, ``Deletion_primers_PCR_sizes.txt``:

    http://yeastdeletion-sgtc.yeastgenome.org/Deletion_primers_PCR_sizes.txt

(the original ``www-sequence.stanford.edu`` host is dead; the SGD-hosted
alias above serves the same file.  A Wayback copy of the Stanford
original works too.)  Vendored alongside the other Wu build inputs at
``data/tasks/wu_rfpins/Deletion_primers_PCR_sizes.txt`` (~4.6 MB).
These are the **as-designed** tags — the deep-sequenced/recharacterized
Smith 2009 set is not retrievable (host down, no archive snapshots, and
it is only a ~20 % correction delta anyway).  ~3 % of strains differ from
this design in reality; documented as a caveat in ``docs/benchmarks/wu_rfpins.md``.

Orientation
-----------
- ``UPTAG_sequence_20mer`` is on the cassette top strand (between U1 and
  U2) → used verbatim.
- ``DNTAG_sequence_20mer`` is written on the opposite strand (the column
  sits as ``D1 + dntag + D2``); the cassette slot is the top-strand
  ``D2-DNTAG-D1`` region, so the tag is **reverse-complemented** before
  use.  Cross-validated against the independent EricEdwardBryant/
  YeastBarcodes mirror: its ``dntag`` equals ``revcomp`` of this column
  for 4830/4832 shared ORFs (and 0/4832 forward), and its ``uptag``
  matches verbatim for 4988/4990.

Coverage (over the 1044 Wu loci)
--------------------------------
1013 carry both designed tags; 31 are UPTAG-only (no designed DNTAG; the
earliest-deleted ORFs).  For those 31 the missing DNTAG is filled with a
deterministic per-ORF synthetic ACGT 20-mer (``dntag_source=synthetic``)
— never left as N — since the real down tag is unknown for these strains
and is in any case an inert 20-mer ~2.2 kb from the mCherry readout.

Run: uv run python scripts/wu/build_barcodes.py
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SGTC = ROOT / "data/tasks/wu_rfpins/Deletion_primers_PCR_sizes.txt"
LABELS = ROOT / "data/tasks/wu_rfpins/table_s2_fluorescence_1044_loci.csv"
OUT = ROOT / "data/tasks/wu_rfpins/barcodes.tsv"

ORF_COL = "ORF_name"
UPTAG_COL = "UPTAG_sequence_20mer"
DNTAG_COL = "DNTAG_sequence_20mer"

_COMP = str.maketrans("ACGT", "TGCA")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def is_tag(s: str) -> bool:
    return len(s) == 20 and set(s) <= set("ACGT")


def synthetic_dntag(orf: str) -> str:
    """Deterministic per-ORF ACGT 20-mer (2 bits/base from a SHA-256 of the
    ORF name).  Reproducible, in-distribution, and unique per locus so the
    fill introduces no shared motif across the UPTAG-only strains."""
    digest = hashlib.sha256(f"wu_dntag_fill:{orf}".encode()).digest()
    bits = int.from_bytes(digest[:5], "big")  # 40 bits → 20 bases
    return "".join("ACGT"[(bits >> (2 * i)) & 3] for i in range(20))


def load_sgtc(path: Path) -> dict[str, tuple[str, str]]:
    """ORF_name → (uptag_20mer, dntag_20mer) from the SGTC master table.

    Tags are taken from the ``*_sequence_20mer`` columns (bare 20-mers, no
    flanking priming bases).  Empty/non-20bp tags are returned as ``""``.
    """
    out: dict[str, tuple[str, str]] = {}
    with open(path) as fh:
        header: list[str] | None = None
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if header is None:
                if ORF_COL in (c.strip() for c in fields):
                    header = [c.strip() for c in fields]
                continue
            if not fields or fields[0].startswith("==="):
                continue
            # pad short rows (trailing empty cells may be omitted)
            if len(fields) < len(header):
                fields = fields + [""] * (len(header) - len(fields))
            row = dict(zip(header, (c.strip() for c in fields)))
            orf = row.get(ORF_COL, "")
            if not orf:
                continue
            up = row.get(UPTAG_COL, "").upper()
            dn = row.get(DNTAG_COL, "").upper()
            out[orf] = (up if is_tag(up) else "", dn if is_tag(dn) else "")
    return out


def main() -> None:
    sgtc = load_sgtc(SGTC)

    with open(LABELS) as fh:
        orfs = [r[ORF_COL].strip() for r in csv.DictReader(fh)]

    rows: list[tuple[str, str, str, str, str]] = []
    n_up_designed = n_up_synth = n_dn_designed = n_dn_synth = n_absent = 0
    for orf in orfs:
        up_raw, dn_raw = sgtc.get(orf, ("", ""))
        if not up_raw and not dn_raw and orf not in sgtc:
            n_absent += 1

        if up_raw:
            uptag, up_src = up_raw, "designed"
            n_up_designed += 1
        else:  # no real uptag (does not occur for the 1044, kept for safety)
            uptag, up_src = synthetic_dntag("UP:" + orf), "synthetic"
            n_up_synth += 1

        if dn_raw:
            dntag, dn_src = revcomp(dn_raw), "designed"
            n_dn_designed += 1
        else:
            dntag, dn_src = synthetic_dntag(orf), "synthetic"
            n_dn_synth += 1

        assert is_tag(uptag) and is_tag(dntag), (orf, uptag, dntag)
        rows.append((orf, uptag, dntag, up_src, dn_src))

    assert len(rows) == len(orfs) == 1044, len(rows)
    assert n_absent == 0, f"{n_absent} Wu ORFs absent from the SGTC table"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["ORF_name", "uptag", "dntag", "uptag_source", "dntag_source"])
        w.writerows(rows)

    print(f"wrote {OUT} ({len(rows)} loci)")
    print(f"  UPTAG: {n_up_designed} designed, {n_up_synth} synthetic")
    print(f"  DNTAG: {n_dn_designed} designed, {n_dn_synth} synthetic")
    print(f"  source: {SGTC.name} (as-designed SGTC tags; DNTAG reverse-complemented)")


if __name__ == "__main__":
    main()
