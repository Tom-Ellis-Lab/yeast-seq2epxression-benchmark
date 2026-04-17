"""Shorkie marginalized / native-position adapter for Rafi MPRA.

Inserts the 110 bp MPRA sequence at native yeast genome positions upstream
of 22 host genes.  For each (gene, offset) context:
  REF = native genomic window (precomputed once)
  ALT = same window with 110 bp replaced at the insertion site
  logSED = log2(alt_exon_sum + 1) - log2(ref_exon_sum + 1)

Aggregation: cross-track mean (T0 RNA-seq tracks, unstranded in Shorkie's
targets sheet) → mean across offsets → mean across host genes.

REF one-hots are cached on GPU so only ALT forward passes run per
sequence.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from tqdm import tqdm

from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.adapters._marginalized_mpra import (
    INSERT_LEN,
    compute_insertion_contexts,
    extract_insert,
    reverse_complement,
)
from yeastbench.adapters.protocols import MarginalizedSequenceExpressionPredictor
from yeastbench.adapters.shorkie_eqtl import (
    BIN_WIDTH,
    CROP_BP_EACH_SIDE,
    OUTPUT_BINS,
    SEQ_LEN,
)

if TYPE_CHECKING:
    import torch
    from yeastbench.models.shorkie import Shorkie

log = logging.getLogger(__name__)

# 384 T0 RNA-seq track indices extracted from data/models/shorkie/targets.txt.
# These are the tracks whose identifier contains "_T0_".  Shorkie RNA-seq
# tracks are unstranded coverage bigwigs, so a single subset works for
# both positive- and negative-strand host genes.
SHORKIE_T0_RNA_SEQ_TRACK_IDS: list[int] = [
    1148, 1156, 1163, 1168, 1173, 1175, 1178, 1186, 1193, 1201,
    1208, 1212, 1219, 1227, 1234, 1240, 1248, 1256, 1263, 1271,
    1279, 1294, 1302, 1310, 1317, 1323, 1331, 1339, 1347, 1355,
    1363, 1371, 1377, 1385, 1393, 1400, 1408, 1416, 1424, 1432,
    1440, 1448, 1456, 1464, 1471, 1479, 1487, 1504, 1515, 1520,
    1525, 1536, 1549, 1576, 1586, 1594, 1599, 1607, 1615, 1631,
    1646, 1652, 1657, 1662, 1664, 1672, 1680, 1687, 1692, 1698,
    1711, 1720, 1728, 1729, 1739, 1740, 1759, 1760, 1770, 1799,
    1808, 1811, 1817, 1818, 1845, 1857, 1858, 1876, 1883, 1884,
    1894, 1900, 1908, 1909, 1918, 1919, 1929, 1930, 1938, 1939,
    1947, 1948, 1960, 1963, 1979, 1980, 1993, 1994, 2005, 2006,
    2020, 2028, 2035, 2036, 2049, 2050, 2062, 2066, 2067, 2075,
    2076, 2109, 2117, 2125, 2133, 2141, 2149, 2157, 2165, 2173,
    2181, 2189, 2197, 2205, 2213, 2221, 2229, 2237, 2245, 2253,
    2261, 2269, 2277, 2285, 2293, 2301, 2309, 2317, 2325, 2333,
    2341, 2349, 2357, 2365, 2373, 2381, 2389, 2397, 2405, 2413,
    2421, 2429, 2437, 2445, 2453, 2461, 2469, 2477, 2485, 2497,
    2504, 2516, 2523, 2530, 2538, 2550, 2558, 2566, 2574, 2581,
    2589, 2597, 2604, 2612, 2618, 2626, 2634, 2640, 2647, 2655,
    2661, 2673, 2685, 2693, 2701, 2706, 2713, 2720, 2726, 2731,
    2743, 2750, 2758, 2762, 2768, 2775, 2783, 2791, 2799, 2807,
    2815, 2823, 2831, 2839, 2847, 2854, 2861, 2868, 2876, 2884,
    2892, 2899, 2907, 2915, 2923, 2931, 2939, 2947, 2954, 2962,
    2970, 2977, 2985, 2993, 3001, 3009, 3017, 3025, 3033, 3041,
    3049, 3057, 3065, 3073, 3081, 3088, 3096, 3104, 3112, 3120,
    3128, 3136, 3144, 3152, 3160, 3168, 3176, 3184, 3192, 3200,
    3208, 3216, 3224, 3232, 3240, 3248, 3256, 3264, 3270, 3278,
    3285, 3292, 3300, 3307, 3314, 3322, 3330, 3338, 3346, 3354,
    3362, 3370, 3378, 3386, 3393, 3401, 3409, 3417, 3421, 3429,
    3437, 3445, 3453, 3461, 3469, 3475, 3482, 3490, 3498, 3506,
    3514, 3522, 3527, 3535, 3543, 3551, 3559, 3567, 3575, 3583,
    3591, 3599, 3607, 3615, 3623, 3637, 3645, 3653, 3661, 3669,
    3677, 3685, 3693, 3701, 3709, 3717, 3725, 3733, 3741, 3749,
    3757, 3765, 3773, 3781, 3788, 3796, 3804, 3812, 3820, 3827,
    3835, 3842, 3850, 3858, 3866, 3874, 3882, 3890, 3904, 3912,
    3919, 3926, 3934, 3942, 3949, 3957, 3965, 3978, 3984, 3994,
    4002, 4009, 4017, 4022, 4030, 4037, 4045, 4053, 4063, 4074,
    4082, 4087, 4095, 4103, 4111, 4119, 4130, 4138, 4146, 4154,
    4166, 4184, 4188, 4193,
]


class ShorkieMPRAMarginalizedPredictor(MarginalizedSequenceExpressionPredictor):
    def __init__(
        self,
        models: list["Shorkie"],
        fasta_path: str | Path,
        gtf_path: str | Path,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: "str | torch.device" = "cuda",
        batch_size: int = 32,
        use_rc: bool = True,
        n_sample: int | None = None,
        seed: int = 42,
    ) -> None:
        import pysam
        import torch as _torch

        from yeastbench.adapters._genome import parse_gene_annotations

        if not models:
            raise ValueError("Must provide at least one model fold")
        self.models = models
        self.fasta = pysam.FastaFile(str(fasta_path))
        self.track_subset = list(track_subset)
        self.device = _torch.device(device)
        self.batch_size = batch_size
        self.use_rc = use_rc
        self.n_sample = n_sample
        self.seed = seed
        for m in self.models:
            m.to(self.device).eval()

        self.genes = parse_gene_annotations(gtf_path)
        self.contexts = compute_insertion_contexts(
            gtf_path, self.fasta,
            seq_len=SEQ_LEN,
            crop_bp_each_side=CROP_BP_EACH_SIDE,
            bin_width=BIN_WIDTH,
            output_bins=OUTPUT_BINS,
        )
        log.info(
            "Marginalized MPRA (Shorkie): %d contexts across %d genes",
            len(self.contexts),
            len({c.gene_id for c in self.contexts}),
        )

        self._gene_ids = sorted({c.gene_id for c in self.contexts})
        self._gene_contexts: dict[str, list[int]] = {g: [] for g in self._gene_ids}
        for i, c in enumerate(self.contexts):
            self._gene_contexts[c.gene_id].append(i)

        # Cache REF one-hots as a single GPU tensor (n_ctx, 4, SEQ_LEN) — channels-first
        ref_np = np.zeros((len(self.contexts), 4, SEQ_LEN), dtype=np.float32)
        for i, ctx in enumerate(self.contexts):
            gene = self.genes[ctx.gene_id]
            seq = self.fasta.fetch(
                gene.chrom_roman, ctx.window_start, ctx.window_start + SEQ_LEN,
            ).upper()
            ref_np[i] = one_hot_encode_channels_first(seq)
        self._ref_ohs_gpu = _torch.from_numpy(ref_np).to(self.device)

        self._track_idx_t = _torch.tensor(
            self.track_subset, device=self.device, dtype=_torch.long
        )

        self._ref_exon_sums = self._precompute_baselines()  # GPU tensor (n_ctx, n_tracks)

    @classmethod
    def from_checkpoints(
        cls,
        params_path: str | Path,
        checkpoint_paths: Sequence[str | Path],
        fasta_path: str | Path,
        gtf_path: str | Path,
        track_subset: list[int] = SHORKIE_T0_RNA_SEQ_TRACK_IDS,
        device: str = "cuda",
        batch_size: int = 32,
        use_rc: bool = True,
        n_sample: int | None = None,
        seed: int = 42,
    ) -> "ShorkieMPRAMarginalizedPredictor":
        from yeastbench.models.shorkie import Shorkie

        with open(params_path) as f:
            config = json.load(f)
        models = [
            Shorkie.from_tf_checkpoint(config["model"], str(p))
            for p in checkpoint_paths
        ]
        return cls(
            models, fasta_path, gtf_path, list(track_subset),
            device, batch_size, use_rc=use_rc,
            n_sample=n_sample, seed=seed,
        )

    def _forward_avg(self, x: "torch.Tensor") -> "torch.Tensor":
        """Run all folds (+ optional RC) on *x* and average across folds and tracks.

        Returns (B, OUTPUT_BINS): the cross-track-mean prediction per bin.
        """
        import torch as _torch

        B = x.shape[0]
        acc = _torch.zeros(
            B, OUTPUT_BINS, device=self.device, dtype=_torch.float32,
        )
        x_rc = x.flip(dims=[1, 2]) if self.use_rc else None
        for m in self.models:
            out = m(x).index_select(2, self._track_idx_t)  # (B, OUTPUT_BINS, n_tracks)
            if self.use_rc:
                out_rc = m(x_rc).index_select(2, self._track_idx_t).flip(dims=[1])
                out = 0.5 * (out + out_rc)
            acc.add_(out.mean(dim=2))
        acc.div_(len(self.models))
        return acc

    def _precompute_baselines(self) -> "torch.Tensor":
        """Returns (n_ctx,) GPU tensor of REF exon-bin sums (cross-track averaged)."""
        import torch as _torch

        n_ctx = len(self.contexts)
        ref_sums = _torch.zeros(n_ctx, device=self.device, dtype=_torch.float32)

        for batch_start in tqdm(
            range(0, n_ctx, self.batch_size), desc="Shorkie REF baseline"
        ):
            batch_end = min(batch_start + self.batch_size, n_ctx)
            x = self._ref_ohs_gpu[batch_start:batch_end]

            with _torch.no_grad():
                cov = self._forward_avg(x)  # (B, OUTPUT_BINS)

            for i in range(batch_end - batch_start):
                ctx = self.contexts[batch_start + i]
                if ctx.exon_bins.size == 0:
                    continue
                bins_t = _torch.as_tensor(ctx.exon_bins, device=self.device, dtype=_torch.long)
                ref_sums[batch_start + i] = cov[i].index_select(0, bins_t).sum()

        return ref_sums

    def _build_alt_batch_gpu(
        self, insert_fwd_oh: "torch.Tensor", insert_rc_oh: "torch.Tensor",
        batch_start: int, batch_end: int,
    ) -> "torch.Tensor":
        """Construct ALT one-hots on GPU by copying REFs and splicing the insert.

        Channels-first layout: (B, 4, SEQ_LEN).
        """
        alt = self._ref_ohs_gpu[batch_start:batch_end].clone()
        for i in range(batch_end - batch_start):
            ctx = self.contexts[batch_start + i]
            s = ctx.insert_start_in_window
            oh = insert_rc_oh if ctx.gene_strand == "-" else insert_fwd_oh
            alt[i, :, s : s + INSERT_LEN] = oh
        return alt

    def _score_one_sequence(self, insert_seq: str) -> float:
        import torch as _torch

        insert_fwd_np = one_hot_encode_channels_first(insert_seq.upper())  # (4, L)
        insert_rc_np = one_hot_encode_channels_first(
            reverse_complement(insert_seq).upper()
        )
        insert_fwd_oh = _torch.from_numpy(insert_fwd_np).to(self.device)
        insert_rc_oh = _torch.from_numpy(insert_rc_np).to(self.device)

        n_ctx = len(self.contexts)
        alt_exon_sums = _torch.zeros(n_ctx, device=self.device, dtype=_torch.float32)

        for batch_start in range(0, n_ctx, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_ctx)
            x = self._build_alt_batch_gpu(insert_fwd_oh, insert_rc_oh, batch_start, batch_end)

            with _torch.no_grad():
                cov = self._forward_avg(x)

            for i in range(batch_end - batch_start):
                ctx = self.contexts[batch_start + i]
                if ctx.exon_bins.size == 0:
                    continue
                bins_t = _torch.as_tensor(ctx.exon_bins, device=self.device, dtype=_torch.long)
                alt_exon_sums[batch_start + i] = cov[i].index_select(0, bins_t).sum()

        # logSED per context (already cross-track-averaged by _forward_avg)
        logsed_per_ctx = (
            _torch.log2(alt_exon_sums + 1.0) - _torch.log2(self._ref_exon_sums + 1.0)
        )

        gene_means: list[float] = []
        for gene_id in self._gene_ids:
            idx = self._gene_contexts[gene_id]
            gene_means.append(float(logsed_per_ctx[idx].mean()))

        return float(np.mean(gene_means))

    def predict_marginalized_expressions(self, seqs: Sequence[str]) -> np.ndarray:
        n = len(seqs)
        scores = np.full(n, np.nan, dtype=np.float64)

        if self.n_sample is not None and self.n_sample < n:
            rng = np.random.default_rng(self.seed)
            sample_idx = rng.choice(n, size=self.n_sample, replace=False)
        else:
            sample_idx = np.arange(n)

        for idx in tqdm(sample_idx, desc="Shorkie marginalized"):
            insert = extract_insert(seqs[idx])
            scores[idx] = self._score_one_sequence(insert)

        return scores
