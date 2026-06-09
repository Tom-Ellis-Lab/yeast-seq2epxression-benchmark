"""Chen GFP reference sequence (issue #8).

Locks the GFP backbone to wild-type avGFP matched against Chen's own
evidence, and guards against reverting to the old all-preferred-codon
back-translation that scored models on the wrong DNA.
"""

from __future__ import annotations

from yeastbench.adapters._chen_gfp_reference import GFP_CDS_WT, GFP_PROTEIN
from yeastbench.adapters._chen_marginalized import _translate

# Yeast most-preferred codon per aa (Sharp & Li 1987) — the old, wrong backbone.
_PREFERRED = {
    "A": "GCT", "C": "TGT", "D": "GAC", "E": "GAA", "F": "TTC", "G": "GGT",
    "H": "CAC", "I": "ATC", "K": "AAG", "L": "TTG", "M": "ATG", "N": "AAC",
    "P": "CCA", "Q": "CAA", "R": "AGA", "S": "TCT", "T": "ACT", "V": "GTT",
    "W": "TGG", "Y": "TAC",
}


def test_wt_cds_translates_to_gfp_protein():
    assert len(GFP_CDS_WT) == 717  # 238 aa + stop
    assert _translate(GFP_CDS_WT).rstrip("*") == GFP_PROTEIN


def test_matches_chen_construction_primer_flanks():
    # Constant flanks around both variable regions, from Chen supp Table S1.
    assert "GAAGGTGATGCAACATACGGAAAA" in GFP_CDS_WT  # region 1, EGDATYGK
    assert "CCTGTTCCATGGCCAACA" in GFP_CDS_WT        # region 1, PVPWPT
    assert "CACAACATTGAAGATGG" in GFP_CDS_WT         # region 2, ...HNIED


def test_residue_172_is_glutamate_not_lysine():
    # The WT deposit L29345 has K172; Chen's region-2 primer pins E172 (GAA).
    assert GFP_CDS_WT[171 * 3 : 171 * 3 + 3] == "GAA"


def test_is_wildtype_not_all_preferred_backtranslation():
    all_preferred = "".join(_PREFERRED[a] for a in GFP_PROTEIN) + "TAA"
    assert len(all_preferred) == len(GFP_CDS_WT)
    assert GFP_CDS_WT != all_preferred
    # Same protein, but real avGFP differs from all-preferred at many codons.
    diffs = sum(a != b for a, b in zip(GFP_CDS_WT, all_preferred))
    assert diffs > 100
