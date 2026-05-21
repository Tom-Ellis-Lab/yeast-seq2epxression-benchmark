"""Build a synthetic chrII for the Chen 2017 synonymous-mutation benchmark.

Replaces the GAL1 CDS (chrII +strand 279021-280607 in R64-1-1) with the
benchmark's variant-gene CDS (GFP or TDH3, depending on library). All
other chrII coordinates outside the replaced span are unchanged; chrII
shrinks by ``len(GAL1) − len(variant gene)``.

Outputs **two** construct FASTA + GTF pairs (one per variant gene), plus
a single ``library_loci.json`` that maps each Chen library to the right
locus:

    data/tasks/chen_synonymous/
        construct_chrII_gfp.fa      construct_gfp.gtf      # for gfp_r1 + gfp_r2
        construct_chrII_tdh3.fa     construct_tdh3.gtf     # for tdh3
        library_loci.json

Trade-off vs the "minimum-modification chrII" the spec sketches: we are
**not** splicing in dTomato at GAL7 or the LEU2 / GAL1-terminator
cassette downstream of the variant gene. Those bits matter for the
*assay* (dTomato is the FACS normaliser, LEU2 is the selection marker)
but they sit thousands of bp away from the variant block and do not
affect a logSED computed over the variant-gene CDS bins. Keeping the
construct minimal also keeps the GFP encoding decision (WT vs S65T,
which DNA encoding) from accidentally biasing the model's score for the
*variable region*, which is the only thing we replace per variant.

The variant-gene CDS encodings:

- **GFP**: synthesised here. Every codon is set to S. cerevisiae's
  most-frequent codon for the corresponding amino acid (per Sharp & Li
  1987's highly-expressed reference set), *except* codons 41–52 and
  156–167 which are copied from the first row of each library's TSV so
  the construct's REF sequence is one valid library variant.
- **TDH3**: pulled native from R64-1-1 chrVII (YGR192C, 882815-883810
  on the minus strand). 996 nt incl. stop. Identical to the WT gene.

Sanity checks (must pass at build time, raise otherwise):

- The variant-gene CDS translates to the published protein.
- The variant-gene CDS slices at the variable-region offset translate
  to the 12-aa peptide for each library that uses the gene.
- The output FASTA's chrII length equals the original chrII length
  minus (GAL1 CDS length − variant-gene CDS length).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from pyfaidx import Fasta

log = logging.getLogger(__name__)

DEFAULT_REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = DEFAULT_REPO / "data" / "tasks" / "chen_synonymous"
DEFAULT_GENOME_FA = DEFAULT_REPO / "data" / "tasks" / "R64-1-1.fa"

# R64-1-1 GAL1 (YBR020W) on chrII +strand, 1-based inclusive coordinates.
GAL1_CHROM = "II"
GAL1_START = 279021
GAL1_END = 280607
GAL1_CDS_LEN = GAL1_END - GAL1_START + 1   # 1587 nt incl. stop

# R64-1-1 TDH3 (YGR192C) on chrVII −strand, 1-based inclusive.
TDH3_CHROM = "VII"
TDH3_START = 882815
TDH3_END = 883810
TDH3_CDS_LEN = TDH3_END - TDH3_START + 1   # 996 nt incl. stop
TDH3_STRAND = "-"

# WT A. victoria GFP protein (Prasher 1992; UniProt P42212), 238 aa.
GFP_PROTEIN = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGV"
    "QCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNI"
    "LGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQ"
    "SALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)

# Most-frequent S. cerevisiae codon per amino acid, from Sharp & Li
# 1987's 24-gene highly-expressed reference set. One canonical codon per
# aa — used to encode the GFP CDS so the construct is reproducible from
# the protein alone (no dependence on which GenBank entry we vendored).
PREFERRED_CODON = {
    "A": "GCT", "C": "TGT", "D": "GAC", "E": "GAA", "F": "TTC",
    "G": "GGT", "H": "CAC", "I": "ATC", "K": "AAG", "L": "TTG",
    "M": "ATG", "N": "AAC", "P": "CCA", "Q": "CAA", "R": "AGA",
    "S": "TCT", "T": "ACT", "V": "GTT", "W": "TGG", "Y": "TAC",
    "*": "TAA",
}

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


def revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def translate(dna: str) -> str:
    return "".join(CODON_TABLE[dna[i : i + 3]] for i in range(0, len(dna), 3))


def build_gfp_cds(
    ref_codons_r1: str,  # 36 nt
    ref_codons_r2: str,  # 36 nt
) -> str:
    """Encode WT GFP with preferred yeast codons everywhere, except
    codons 41-52 use ref_codons_r1 and codons 156-167 use ref_codons_r2.
    Both ref blocks must already translate to the library's peptide,
    which the TSV builder guarantees."""
    codons = [PREFERRED_CODON[aa] for aa in GFP_PROTEIN]
    # codons is 0-based; we want positions 41-52 and 156-167 in 0-based
    # indexing into the protein (which equals the codon index here).
    for j, off in enumerate(range(41, 53)):
        codons[off] = ref_codons_r1[3 * j : 3 * j + 3]
    for j, off in enumerate(range(156, 168)):
        codons[off] = ref_codons_r2[3 * j : 3 * j + 3]
    cds = "".join(codons) + PREFERRED_CODON["*"]    # append a TAA stop
    # Sanity check
    prot = translate(cds).rstrip("*")
    if prot != GFP_PROTEIN:
        raise ValueError(
            f"synthesised GFP CDS does not translate to WT GFP "
            f"(len {len(prot)} vs {len(GFP_PROTEIN)})"
        )
    return cds


def pull_native_cds(
    fasta: Fasta, chrom: str, start: int, end: int, strand: str,
) -> str:
    seq = str(fasta[chrom][start - 1 : end].seq).upper()
    if strand == "-":
        seq = revcomp(seq)
    return seq


def build_modified_chrii(
    fasta: Fasta,
    variant_cds: str,
) -> str:
    chrii = str(fasta[GAL1_CHROM][:].seq).upper()
    new_chrii = chrii[: GAL1_START - 1] + variant_cds + chrii[GAL1_END:]
    expected = len(chrii) - GAL1_CDS_LEN + len(variant_cds)
    if len(new_chrii) != expected:
        raise ValueError(f"new chrII length {len(new_chrii)} != expected {expected}")
    return new_chrii


def write_fasta(path: Path, name: str, seq: str, wrap: int = 80) -> None:
    with path.open("w") as fh:
        fh.write(f">{name}\n")
        for i in range(0, len(seq), wrap):
            fh.write(seq[i : i + wrap] + "\n")


def write_gtf(path: Path, gene_id: str, gene_name: str, cds_start: int, cds_end: int) -> None:
    """Write a minimal Ensembl-style GTF entry for one + strand CDS."""
    attrs_gene = (
        f'gene_id "{gene_id}"; gene_name "{gene_name}"; gene_source "chen2017"; '
        f'gene_biotype "protein_coding";'
    )
    transcript_id = f"{gene_id}_mRNA"
    attrs_tx = (
        f'gene_id "{gene_id}"; transcript_id "{transcript_id}"; '
        f'gene_name "{gene_name}"; gene_source "chen2017"; '
        f'gene_biotype "protein_coding"; '
        f'transcript_name "{gene_name}"; transcript_source "chen2017"; '
        f'transcript_biotype "protein_coding"; tag "Ensembl_canonical";'
    )
    attrs_exon = attrs_tx.replace(
        "tag \"Ensembl_canonical\";",
        'exon_number "1"; exon_id "' + transcript_id + '-E1"; tag "Ensembl_canonical";'
    )
    attrs_cds = attrs_tx.replace(
        "tag \"Ensembl_canonical\";",
        'exon_number "1"; protein_id "' + gene_id + '"; tag "Ensembl_canonical";'
    )
    attrs_start = attrs_tx.replace(
        "tag \"Ensembl_canonical\";",
        'exon_number "1"; tag "Ensembl_canonical";'
    )
    lines = [
        f"{GAL1_CHROM}\tchen2017\tgene\t{cds_start}\t{cds_end}\t.\t+\t.\t{attrs_gene}",
        f"{GAL1_CHROM}\tchen2017\ttranscript\t{cds_start}\t{cds_end}\t.\t+\t.\t{attrs_tx}",
        f"{GAL1_CHROM}\tchen2017\texon\t{cds_start}\t{cds_end}\t.\t+\t.\t{attrs_exon}",
        f"{GAL1_CHROM}\tchen2017\tCDS\t{cds_start}\t{cds_end - 3}\t.\t+\t0\t{attrs_cds}",
        f"{GAL1_CHROM}\tchen2017\tstart_codon\t{cds_start}\t{cds_start + 2}\t.\t+\t0\t{attrs_start}",
        f"{GAL1_CHROM}\tchen2017\tstop_codon\t{cds_end - 2}\t{cds_end}\t.\t+\t0\t{attrs_start}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genome-fa", type=Path, default=DEFAULT_GENOME_FA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fasta = Fasta(str(args.genome_fa))
    log.info("R64-1-1 loaded; chrII length = %d", len(fasta[GAL1_CHROM]))

    # ── Build GFP CDS using ref codons from row 0 of each library TSV ──
    gfp_r1_tsv = args.out_dir / "gfp_r1.tsv"
    gfp_r2_tsv = args.out_dir / "gfp_r2.tsv"
    tdh3_tsv = args.out_dir / "tdh3.tsv"
    ref_r1 = pd.read_csv(gfp_r1_tsv, sep="\t").iloc[0]["variable_seq"].upper()
    ref_r2 = pd.read_csv(gfp_r2_tsv, sep="\t").iloc[0]["variable_seq"].upper()
    gfp_cds = build_gfp_cds(ref_r1, ref_r2)
    log.info("GFP CDS built; len=%d, ref var blocks from row 0 of each TSV", len(gfp_cds))

    tdh3_cds = pull_native_cds(
        fasta, TDH3_CHROM, TDH3_START, TDH3_END, TDH3_STRAND,
    )
    log.info("TDH3 CDS pulled from R64-1-1; len=%d", len(tdh3_cds))
    if translate(tdh3_cds).rstrip("*") != (
        "MVRVAINGFGRIGRLVMRIALSRPNVEVVALNDPFITNDYAAYMFKYDSTHGRYAGEVSHDDKHIIVD"
        "GKKIATYQERDPANLPWGSSNVDIAIDSTGVFKELDTAQKHIDAGAKKVVITAPSSTAPMFVMGVNEE"
        "KYTSDLKIVSNASCTTNCLAPLAKVINDAFGIEEGLMTTVHSLTATQKTVDGPSHKDWRGGRTASGNI"
        "IPSSTGAAKAVGKVLPELQGKLTGMAFRVPTVDVSVVDLTVKLNKETTYDEIKKVVKAAAEGKLKGVL"
        "GYTEDAVVSSDFLGDSHSSIFDASAGIQLSPKFVKLVSWYDNEYGYSTRVVDLVEHVAKA"
    ):
        raise ValueError("pulled TDH3 CDS does not translate to expected protein")

    # ── Write the GFP construct ──
    gfp_chrii = build_modified_chrii(fasta, gfp_cds)
    gfp_cds_start = GAL1_START                       # 1-based
    gfp_cds_end = GAL1_START + len(gfp_cds) - 1      # 1-based inclusive
    write_fasta(
        args.out_dir / "construct_chrII_gfp.fa",
        name=GAL1_CHROM, seq=gfp_chrii,
    )
    write_gtf(
        args.out_dir / "construct_gfp.gtf",
        gene_id="GFP_variant", gene_name="GFP_variant",
        cds_start=gfp_cds_start, cds_end=gfp_cds_end,
    )
    log.info("wrote construct_chrII_gfp.fa (%d nt) + construct_gfp.gtf", len(gfp_chrii))

    # ── Write the TDH3 construct ──
    tdh3_chrii = build_modified_chrii(fasta, tdh3_cds)
    tdh3_cds_start = GAL1_START
    tdh3_cds_end = GAL1_START + len(tdh3_cds) - 1
    write_fasta(
        args.out_dir / "construct_chrII_tdh3.fa",
        name=GAL1_CHROM, seq=tdh3_chrii,
    )
    write_gtf(
        args.out_dir / "construct_tdh3.gtf",
        gene_id="TDH3_variant", gene_name="TDH3_variant",
        cds_start=tdh3_cds_start, cds_end=tdh3_cds_end,
    )
    log.info("wrote construct_chrII_tdh3.fa (%d nt) + construct_tdh3.gtf", len(tdh3_chrii))

    # ── Library loci ──
    # var_start / var_end are 1-based inclusive chrII coordinates of the
    # 36 nt variable block. cds_start_in_construct is 1-based chrII
    # position of the first CDS nucleotide.
    # var_start / var_end derived from the peptide's position in the
    # protein, not from the spec's "codon X" naming (which is ambiguous
    # across the GFP libraries). The 12-aa peptide is at 0-based
    # protein positions [p, p+12); the nt offset from CDS start is 3*p.
    loci = {
        "gfp_r1": {
            "fasta_name": "construct_chrII_gfp.fa",
            "gtf_name": "construct_gfp.gtf",
            "gene_id": "GFP_variant",
            "cds_start_in_construct": gfp_cds_start,
            "cds_end_in_construct": gfp_cds_end,
            "var_start": gfp_cds_start + 3 * 41,   # nt offset 123
            "var_end":   gfp_cds_start + 3 * 41 + 35,
            "protein_var_start_0based": 41,
            "expected_peptide": "LTLKFICTTGKL",
        },
        "gfp_r2": {
            "fasta_name": "construct_chrII_gfp.fa",
            "gtf_name": "construct_gfp.gtf",
            "gene_id": "GFP_variant",
            "cds_start_in_construct": gfp_cds_start,
            "cds_end_in_construct": gfp_cds_end,
            "var_start": gfp_cds_start + 3 * 156,  # nt offset 468
            "var_end":   gfp_cds_start + 3 * 156 + 35,
            "protein_var_start_0based": 156,
            "expected_peptide": "QKNGIKVNFKIR",
        },
        "tdh3": {
            "fasta_name": "construct_chrII_tdh3.fa",
            "gtf_name": "construct_tdh3.gtf",
            "gene_id": "TDH3_variant",
            "cds_start_in_construct": tdh3_cds_start,
            "cds_end_in_construct": tdh3_cds_end,
            "var_start": tdh3_cds_start + 3 * 56,  # nt offset 168
            "var_end":   tdh3_cds_start + 3 * 56 + 35,
            "protein_var_start_0based": 56,
            "expected_peptide": "EVSHDDKHIIVD",
        },
    }

    # Cross-check the variable-block slice translates to the published peptide.
    for lib_id, lib in loci.items():
        chrii_seq = gfp_chrii if lib_id.startswith("gfp") else tdh3_chrii
        var = chrii_seq[lib["var_start"] - 1 : lib["var_end"]]
        if len(var) != 36:
            raise ValueError(f"{lib_id}: variable block in construct is {len(var)} nt")
        pep = translate(var)
        if pep != lib["expected_peptide"]:
            raise ValueError(
                f"{lib_id}: variable block in construct translates to {pep!r}, "
                f"expected {lib['expected_peptide']!r}"
            )
        log.info("%s: chrII slice [%d:%d] → %r ✓",
                 lib_id, lib["var_start"], lib["var_end"], pep)

    (args.out_dir / "library_loci.json").write_text(json.dumps(loci, indent=2) + "\n")
    log.info("wrote library_loci.json with %d libraries", len(loci))


if __name__ == "__main__":
    main()
