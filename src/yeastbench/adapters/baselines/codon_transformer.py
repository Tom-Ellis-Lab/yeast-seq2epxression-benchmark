"""CodonTransformer baseline for the Chen synonymous-mutation benchmark.

Loads Fallahpour et al. 2025's BigBird masked-LM (Nat Commun 16:3205;
https://github.com/Adibvafa/CodonTransformer), runs one forward pass per
library over an all-`*_unk` merged input (protein only, no codons), and
scores each variant by summing per-position log-probabilities over the
12 variable codons.

See ``benchmarks/chen_synonymous.md`` for the math and caveats.

The convenience ``predict_dna_sequence()`` in the upstream package throws
away the logits we need, so we call the underlying HuggingFace model
directly. No upstream patch required.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from yeastbench.adapters.protocols import LocalCodingVariantPredictor

log = logging.getLogger(__name__)

# WT A. victoria GFP (Prasher 1992; UniProt P42212), 238 aa. Used as the
# protein context for both GFP libraries. The S65T variant differs only at
# position 65 (1-based from M), well outside both libraries' variable
# regions (codons 41–52 and 156–167), so the per-position log-probs at
# those positions are robust to that single substitution.
GFP_PROTEIN = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGV"
    "QCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNI"
    "LGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQ"
    "SALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)
# TDH3 (YGR192C from S. cerevisiae R64-1-1), 332 aa.
TDH3_PROTEIN = (
    "MVRVAINGFGRIGRLVMRIALSRPNVEVVALNDPFITNDYAAYMFKYDSTHGRYAGEVSHDDKHIIVD"
    "GKKIATYQERDPANLPWGSSNVDIAIDSTGVFKELDTAQKHIDAGAKKVVITAPSSTAPMFVMGVNEE"
    "KYTSDLKIVSNASCTTNCLAPLAKVINDAFGIEEGLMTTVHSLTATQKTVDGPSHKDWRGGRTASGNI"
    "IPSSTGAAKAVGKVLPELQGKLTGMAFRVPTVDVSVVDLTVKLNKETTYDEIKKVVKAAAEGKLKGVL"
    "GYTEDAVVSSDFLGDSHSSIFDASAGIQLSPKFVKLVSWYDNEYGYSTRVVDLVEHVAKA"
)

# Per-library protein + 0-based slice [start, start+12) of the 12-codon
# variable region inside that protein, plus the published 12-aa peptide.
LIBRARY_CONTEXTS: dict[str, dict] = {
    "gfp_r1": dict(protein=GFP_PROTEIN, var_start=41, expected_peptide="LTLKFICTTGKL"),
    "gfp_r2": dict(protein=GFP_PROTEIN, var_start=156, expected_peptide="QKNGIKVNFKIR"),
    "tdh3":   dict(protein=TDH3_PROTEIN, var_start=56, expected_peptide="EVSHDDKHIIVD"),
}

ORGANISM = "Saccharomyces cerevisiae"


def _translate12(seq: str, codon_table: dict[str, str]) -> str:
    return "".join(codon_table[seq[i : i + 3]] for i in range(0, 36, 3))


CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "CAT": "H", "CAC": "H",
    "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E", "TGT": "C", "TGC": "C",
    "TGG": "W", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


class CodonTransformerBaselinePredictor(LocalCodingVariantPredictor):
    """Score Chen synonymous variants under CodonTransformer's BigBird MLM.

    All work is in ``from_task``: load model + tokenizer once, then run
    one forward pass over the library's all-``*_unk`` merged input and
    cache the per-position log-probabilities. ``predict_local_variants``
    just indexes into that cache for each variant's 12 codons.
    """

    def __init__(
        self,
        *,
        library_id: str,
        protein: str,
        var_start: int,
        log_p: np.ndarray,             # [L, vocab]
        token2index: dict[str, int],
    ) -> None:
        self.library_id = library_id
        self.protein = protein
        self.var_start = var_start
        self.log_p = log_p
        self.token2index = token2index

    @classmethod
    def from_task(
        cls,
        task,
        device: str | torch.device = "cuda",
        **_ignored,
    ) -> "CodonTransformerBaselinePredictor":
        from CodonTransformer.CodonData import get_merged_seq
        from CodonTransformer.CodonUtils import TOKEN2INDEX
        from transformers import AutoTokenizer, BigBirdForMaskedLM

        library_id = task.library
        if library_id not in LIBRARY_CONTEXTS:
            raise ValueError(f"unknown Chen library: {library_id!r}")
        ctx = LIBRARY_CONTEXTS[library_id]
        protein, var_start, expected_peptide = (
            ctx["protein"], ctx["var_start"], ctx["expected_peptide"],
        )
        if protein[var_start : var_start + 12] != expected_peptide:
            raise ValueError(
                f"{library_id}: hard-coded protein slice "
                f"{protein[var_start:var_start+12]!r} != {expected_peptide!r}"
            )

        log.info("loading CodonTransformer (HF: adibvafa/CodonTransformer)")
        tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
        model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer")
        model = model.eval().to(device)

        # Use the upstream helper to build the merged "protein + dna" string
        # (with dna="" → every codon collapses to "{aa}_unk"), so the token
        # format exactly matches what the tokenizer expects at inference.
        merged = get_merged_seq(protein=protein, dna="")
        inputs = tokenizer(
            merged, return_tensors="pt", padding=True, truncation=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs, return_dict=True).logits[0, 1:-1, :]
        log_p_all = torch.log_softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
        # The merged-seq helper appends a stop-codon token (`__UNK`) so we
        # see len(protein)+1 non-special positions. Slice to the protein
        # positions only; we never index into the stop position.
        if log_p_all.shape[0] not in (len(protein), len(protein) + 1):
            raise ValueError(
                f"{library_id}: tokenizer returned {log_p_all.shape[0]} non-special "
                f"positions; expected {len(protein)} or {len(protein) + 1}"
            )
        log_p = log_p_all[: len(protein)]

        return cls(
            library_id=library_id,
            protein=protein,
            var_start=var_start,
            log_p=log_p,
            token2index=dict(TOKEN2INDEX),
        )

    def predict_local_variants(
        self,
        library_ids: Sequence[str],
        variant_seqs: Sequence[str],
    ) -> np.ndarray:
        out = np.empty(len(library_ids), dtype=float)
        var_positions = list(range(self.var_start, self.var_start + 12))
        for i, (lib, seq) in enumerate(zip(library_ids, variant_seqs)):
            if lib != self.library_id:
                raise ValueError(
                    f"CodonTransformer baseline is bound to library "
                    f"{self.library_id!r}; got {lib!r}"
                )
            if len(seq) != 36:
                raise ValueError(
                    f"variant_seq has {len(seq)} nt, expected 36"
                )
            codons = [seq[j : j + 3].lower() for j in range(0, 36, 3)]
            try:
                token_ids = [
                    self.token2index[
                        f"{self.protein[var_positions[j]].lower()}_{codons[j]}"
                    ]
                    for j in range(12)
                ]
            except KeyError as e:
                raise ValueError(
                    f"CodonTransformer vocab missing token {e.args[0]!r} "
                    f"for variant {seq!r}"
                ) from e
            out[i] = float(sum(
                self.log_p[var_positions[j], token_ids[j]] for j in range(12)
            ))
        return out
