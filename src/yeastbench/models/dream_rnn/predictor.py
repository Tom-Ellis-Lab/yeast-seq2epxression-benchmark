"""Inference wrapper for DREAM-RNN on raw Rafi/deBoer MPRA inserts.

Reproduces the de-Boer-Lab reference inference path (``AutosomePredictor``)
exactly, but renamed and batched:

1. **Re-flank** the insert into the DREAM reporter plasmid context to 150 bp
   (strip the 17 bp constant left adapter, prepend the plasmid sequence
   immediately 5′ of the insert slot, keep the last 150 bp). This is the
   "crop to the variable sequence" step — it discards all genomic/host
   context, leaving the insert in the reporter the model was trained on.
2. **Encode** to 6 channels: 4 one-hot bases + an is_reverse flag channel +
   a zero is_singleton channel.
3. Average the forward pass and the reverse-complement pass.

The 18-bin soft-classification head returns one scalar (expected expression)
per sequence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

# Exact upstream nucleotide → index map (utils.n2id): A=0, G=1, C=2, T=3, N=4.
_CODES = {"A": 0, "T": 3, "G": 1, "C": 2, "N": 4}
_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}

LEFT_ADAPTER = "TGCATTTTTTTCACATC"  # 17 bp constant prefix of every Rafi sequence
SEQ_SIZE = 150  # model input length
INSERT_SLOT = "N" * 80  # marks the insert position in plasmid.json


def _reverse_complement(seq: str) -> str:
    return "".join(_COMPLEMENT[b] for b in reversed(seq))


class DreamRnnExpressionPredictor:
    """Run a single :class:`DreamRnn` on raw 110 bp MPRA strings."""

    def __init__(
        self,
        model: torch.nn.Module,
        plasmid_path: str | Path,
        device: str | torch.device = "cuda",
        batch_size: int = 1024,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size

        with open(plasmid_path) as fh:
            plasmid = json.load(fh)
        insert_start = plasmid.find(INSERT_SLOT)
        if insert_start < 0:
            raise ValueError(
                f"{plasmid_path}: no {len(INSERT_SLOT)}-base insert slot found"
            )
        # The 150 bp of plasmid immediately 5′ of the insert slot (ends with the
        # constant left adapter). Computed once, not re-read per sequence.
        upstream = plasmid[insert_start - SEQ_SIZE : insert_start]
        if len(upstream) != SEQ_SIZE:
            raise ValueError(
                f"{plasmid_path}: insert slot too close to the 5′ end "
                f"({insert_start} bp upstream, need {SEQ_SIZE})"
            )
        self._upstream = upstream

    def _reflank(self, seq: str) -> str:
        if seq[: len(LEFT_ADAPTER)] != LEFT_ADAPTER:
            raise ValueError(
                f"sequence must start with the {len(LEFT_ADAPTER)} bp adapter "
                f"{LEFT_ADAPTER!r}; got {seq[: len(LEFT_ADAPTER)]!r}"
            )
        return (self._upstream + seq[len(LEFT_ADAPTER) :])[-SEQ_SIZE:]

    @staticmethod
    def _encode(seq: str, rev_value: int) -> torch.Tensor:
        """One-hot (4 ch) + is_reverse + is_singleton(=0) → (6, L).

        Faithful to upstream: the one-hot is built as a *long* tensor and N
        positions are zeroed via a masked assignment that truncates to 0 in the
        integer dtype (the upstream ``= 0.25`` is a no-op on a long tensor).
        Rafi inserts + the plasmid flank contain no N, so this branch never
        fires here; it is kept identical for bit-faithfulness.
        """
        idx = torch.tensor([_CODES[b] for b in seq], dtype=torch.long)
        code = F.one_hot(idx, num_classes=5)
        code[code[:, 4] == 1] = 0.25  # long dtype → truncates to 0 (matches upstream)
        code = code[:, :4].float().transpose(0, 1)  # (4, L)
        length = code.shape[1]
        rev = torch.full((1, length), float(rev_value), dtype=torch.float32)
        single = torch.zeros((1, length), dtype=torch.float32)
        return torch.cat([code, rev, single], dim=0)  # (6, L)

    @torch.no_grad()
    def predict(self, seqs: Sequence[str]) -> np.ndarray:
        reflanked = [self._reflank(s.upper()) for s in seqs]
        fwd = torch.stack([self._encode(s, 0) for s in reflanked])
        rev = torch.stack([self._encode(_reverse_complement(s), 1) for s in reflanked])
        out = np.empty(len(seqs), dtype=float)
        for start in range(0, len(seqs), self.batch_size):
            sl = slice(start, start + self.batch_size)
            y_fwd = self.model(fwd[sl].to(self.device)).flatten().cpu().numpy()
            y_rev = self.model(rev[sl].to(self.device)).flatten().cpu().numpy()
            out[sl] = (y_fwd + y_rev) / 2.0
        return out
