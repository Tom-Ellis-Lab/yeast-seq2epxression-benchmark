"""Reference GFP protein and coding DNA for the Chen 2017 benchmark.

Single source of truth — imported by the runtime adapter
(``_chen_marginalized``), the distribution builder
(``scripts/chen/build_construct_reference.py``), and the CodonTransformer
baseline. Keeping one copy is deliberate: the same sequence used to live in
three places, which is how a wrong GFP encoding went unnoticed (see issue #8).

Provenance
----------
``GFP_PROTEIN`` is wild-type *Aequorea victoria* GFP, 238 aa (canonical avGFP,
UniProt P42212). Confirmed correct: it matches PDB **1EMA** everywhere except
the two engineered chromophore-region residues (S65T, Q80R), and it matches
Chen's own data at the two positions their supplement pins (Q157 from the
region-2 variant block; E172 from the region-2 construction primer).

``GFP_CDS_WT`` is the wild-type avGFP coding DNA from GenBank **L29345.1**
(Prasher 1992), with six single-base edits so it translates to ``GFP_PROTEIN``
(L29345's deposited sequence has six conflicts vs canonical avGFP). Residue 172
uses ``GAA``, which matches Chen's construction primer (supp Table S1)
base-for-base; the other five edits are minimal single-base changes (we have no
direct DNA evidence for those positions). Chen never published the full GFP
construct DNA, but every piece of it we *can* see — the supp Table S1 primer
flanks around both variable regions — matches this sequence exactly (region-1
flanks are identical to L29345; region-2 carries the E172 correction). The two
library variable regions (codons 41-52 and 156-167) are overwritten per-variant
from the distribution TSVs, so this backbone only sets the surrounding context.
"""

from __future__ import annotations

GFP_PROTEIN = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGV"
    "QCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNI"
    "LGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQ"
    "SALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)

# 717 nt incl. the TAA stop; translates to GFP_PROTEIN.
GFP_CDS_WT = (
    "ATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGATGGCGATGTTAAT"
    "GGGCACAAATTCTCTGTCAGTGGAGAGGGTGAAGGTGATGCAACATACGGAAAACTTACCCTTAAATTT"
    "ATTTGCACTACTGGGAAGCTACCTGTTCCATGGCCAACACTTGTCACTACTTTCTCTTATGGTGTTCAA"
    "TGCTTTTCAAGATACCCAGATCATATGAAACAGCATGACTTTTTCAAGAGTGCCATGCCCGAAGGTTAT"
    "GTACAGGAAAGAACTATATTTTTCAAAGATGACGGGAACTACAAGACACGTGCTGAAGTCAAGTTTGAA"
    "GGTGATACCCTTGTTAATAGAATCGAGTTAAAAGGTATTGATTTTAAAGAAGATGGAAACATTCTTGGA"
    "CACAAACTGGAATACAACTATAACTCACATAATGTATACATCATGGCAGACAAACAAAAGAATGGAATC"
    "AAAGTTAACTTCAAAATTAGACACAACATTGAAGATGGAAGCGTTCAATTAGCAGACCATTATCAACAA"
    "AATACTCCAATTGGCGATGGCCCTGTCCTTTTACCAGACAACCATTACCTGTCCACACAATCTGCCCTT"
    "TCCAAAGATCCCAACGAAAAGAGAGATCACATGGTCCTTCTTGAGTTTGTAACAGCTGCTGGGATTACA"
    "CATGGCATGGATGAACTATACAAATAA"
)
