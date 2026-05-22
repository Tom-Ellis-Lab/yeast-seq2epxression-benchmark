"""Builder: emit two ipynbs investigating why Shorkie/Yorzoi predict signal at
PGAL1 — a promoter that should be repressed in every condition either model
saw during training. One notebook per model. Runs are cached to pickles so
re-executing the notebook doesn't re-predict.

Usage:
    uv run python scripts/chen/build_investigation_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/chen_yorzoi_investigation.ipynb \
        notebooks/chen_shorkie_investigation.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "notebooks"


SETUP_COMMON = '''\
import pickle
from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pysam
import torch
from scipy.stats import pearsonr, spearmanr

%config InlineBackend.figure_format = "retina"
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 150

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = ROOT / "data" / "tasks" / "chen_synonymous"
TSV_PATH = DATA_DIR / "gfp_r2.tsv"
CONSTRUCT_FA = DATA_DIR / "construct_chrII_gfp.fa"
LOCI_PATH = DATA_DIR / "library_loci.json"

# Native R64-1-1 chrVII around TDH3 — used as a "known highly-expressed locus"
# sanity check (PGAL1 is repressed in glucose; TDH3 is not).
NATIVE_FA = ROOT / "data" / "tasks" / "R64-1-1.fa"

loci = json.loads(LOCI_PATH.read_text())["gfp_r2"]
df = pd.read_csv(TSV_PATH, sep="\\t")

# Stratify 40 variants across the log2mRNA_rep1 dynamic range so plots see
# the full label range. Add the REF (TSV row 0 — what the construct already
# encodes at codons 156-167) as a separate entry.
N_VARIANTS = 40
sorted_idx = df["log2mRNA_rep1"].argsort().to_numpy()
sample_pos = np.linspace(0, len(df) - 1, N_VARIANTS).round().astype(int)
sample_idx = sorted_idx[sample_pos]
variant_seqs = df["variable_seq"].iloc[sample_idx].astype(str).str.upper().tolist()
variant_log2mrna_rep1 = df["log2mRNA_rep1"].iloc[sample_idx].to_numpy()
variant_log2mrna_rep2 = df["log2mRNA_rep2"].iloc[sample_idx].to_numpy()
ref_seq = df["variable_seq"].iloc[0].upper()       # construct REF == TSV row 0

print(f"selected {len(variant_seqs)} variants spanning log2mRNA_rep1 "
      f"[{variant_log2mrna_rep1.min():.2f}, {variant_log2mrna_rep1.max():.2f}]")
'''


YORZOI_SETUP = '''\
# ---- Yorzoi predict + cache ----
from yeastbench.adapters._genome import one_hot_encode_channels_first, parse_gene_annotations, place_window
from yeastbench.adapters._yorzoi_constants import (
    BIN_WIDTH, CROP_BP_EACH_SIDE, OUTPUT_BINS, SEQ_LEN, YORZOI_PLUS_TRACK_IDS,
)
from yeastbench.models.yorzoi import Yorzoi

CACHE = ROOT / "notebooks" / "_chen_yorzoi_cache.pkl"

if CACHE.exists():
    with open(CACHE, "rb") as fh:
        c = pickle.load(fh)
    ref_pred = c["ref_pred"]           # (162, OUTPUT_BINS)
    variant_preds = c["variant_preds"] # (N, 162, OUTPUT_BINS)
    native_tdh3_pred = c["native_tdh3_pred"]  # (162, OUTPUT_BINS)
    window_start = c["window_start"]
    var_start_in_window = c["var_start_in_window"]
    cds_start_in_window = c["cds_start_in_window"]
    cds_end_in_window = c["cds_end_in_window"]
    tdh3_window_start = c["tdh3_window_start"]
    tdh3_cds_start_in_window = c["tdh3_cds_start_in_window"]
    tdh3_cds_end_in_window = c["tdh3_cds_end_in_window"]
    print("loaded cache:", CACHE.name)
else:
    model = Yorzoi.from_pretrained("tom-ellis-lab/yorzoi", device="cuda", use_rc=True, autocast=True)
    fasta = pysam.FastaFile(str(CONSTRUCT_FA))
    genes = parse_gene_annotations(DATA_DIR / "construct_gfp.gtf")
    gene = genes[loci["gene_id"]]
    chrom_len = fasta.get_reference_length(gene.chrom_roman)
    window_start = place_window(
        var_pos=loci["var_start"], gene_center=gene.gene_center,
        chrom_length=chrom_len, seq_len=SEQ_LEN, crop_bp_each_side=CROP_BP_EACH_SIDE,
    )
    var_start_in_window = loci["var_start"] - 1 - window_start
    cds_start_in_window = loci["cds_start_in_construct"] - 1 - window_start
    cds_end_in_window = loci["cds_end_in_construct"] - 1 - window_start

    ref_seq_window = fasta.fetch(gene.chrom_roman, window_start, window_start + SEQ_LEN).upper()
    ref_oh = torch.from_numpy(one_hot_encode_channels_first(ref_seq_window).T).to(model.device)  # (L,4)

    def predict_one(alt_window_oh):
        with torch.no_grad():
            return model.forward_tracks_binned(alt_window_oh.unsqueeze(0)).float()[0].cpu().numpy()

    ref_pred = predict_one(ref_oh)

    variant_preds = np.zeros((len(variant_seqs), 162, OUTPUT_BINS), dtype=np.float32)
    for i, vseq in enumerate(variant_seqs):
        alt = ref_oh.clone()
        alt[var_start_in_window : var_start_in_window + 36, :] = torch.from_numpy(
            one_hot_encode_channels_first(vseq).T
        ).to(model.device)
        variant_preds[i] = predict_one(alt)

    # TDH3 native locus for absolute-level comparison.
    nfa = pysam.FastaFile(str(NATIVE_FA))
    tdh3_gene_center = (882815 + 883810) // 2   # YGR192C centre on chrVII
    chrvii_len = nfa.get_reference_length("VII")
    tdh3_window_start = place_window(
        var_pos=tdh3_gene_center, gene_center=tdh3_gene_center,
        chrom_length=chrvii_len, seq_len=SEQ_LEN, crop_bp_each_side=CROP_BP_EACH_SIDE,
    )
    tdh3_cds_start_in_window = 882815 - 1 - tdh3_window_start
    tdh3_cds_end_in_window   = 883810 - 1 - tdh3_window_start
    tdh3_seq = nfa.fetch("VII", tdh3_window_start, tdh3_window_start + SEQ_LEN).upper()
    tdh3_oh = torch.from_numpy(one_hot_encode_channels_first(tdh3_seq).T).to(model.device)
    native_tdh3_pred = predict_one(tdh3_oh)

    with open(CACHE, "wb") as fh:
        pickle.dump({
            "ref_pred": ref_pred,
            "variant_preds": variant_preds,
            "native_tdh3_pred": native_tdh3_pred,
            "window_start": window_start,
            "var_start_in_window": var_start_in_window,
            "cds_start_in_window": cds_start_in_window,
            "cds_end_in_window": cds_end_in_window,
            "tdh3_window_start": tdh3_window_start,
            "tdh3_cds_start_in_window": tdh3_cds_start_in_window,
            "tdh3_cds_end_in_window": tdh3_cds_end_in_window,
        }, fh)

print("ref_pred:", ref_pred.shape, "(162 tracks × OUTPUT_BINS)")
print("variant_preds:", variant_preds.shape)
print("native TDH3 pred:", native_tdh3_pred.shape)

# Strand-aware track subsets (variant gene is + strand)
PLUS = list(range(0, 81))
MINUS = list(range(81, 162))

# Per-bin → per-bp position (for x-axis labels). Output bins cover bp
# [window_start + CROP, window_start + CROP + OUTPUT_BINS * BIN_WIDTH).
BIN_BP = BIN_WIDTH
'''


SHORKIE_SETUP = '''\
# ---- Shorkie predict + cache ----
# Single fold (f0) + no RC for investigation speed; the production benchmark
# uses 8-fold ensemble + RC. Note the difference when comparing magnitudes.
from yeastbench.adapters._genome import one_hot_encode_channels_first, parse_gene_annotations, place_window
from yeastbench.adapters._shorkie_constants import (
    BIN_WIDTH, CROP_BP_EACH_SIDE, OUTPUT_BINS, SEQ_LEN, SHORKIE_T0_RNA_SEQ_TRACK_IDS,
)
from yeastbench.models.shorkie import Shorkie

CACHE = ROOT / "notebooks" / "_chen_shorkie_cache.pkl"

if CACHE.exists():
    with open(CACHE, "rb") as fh:
        c = pickle.load(fh)
    ref_pred = c["ref_pred"]
    variant_preds = c["variant_preds"]
    native_tdh3_pred = c["native_tdh3_pred"]
    track_ids = c["track_ids"]
    window_start = c["window_start"]
    var_start_in_window = c["var_start_in_window"]
    cds_start_in_window = c["cds_start_in_window"]
    cds_end_in_window = c["cds_end_in_window"]
    tdh3_window_start = c["tdh3_window_start"]
    tdh3_cds_start_in_window = c["tdh3_cds_start_in_window"]
    tdh3_cds_end_in_window = c["tdh3_cds_end_in_window"]
    print("loaded cache:", CACHE.name)
else:
    model = Shorkie.from_checkpoints(
        params_path=ROOT / "data/models/shorkie/params.json",
        checkpoint_paths=[ROOT / "data/models/shorkie/checkpoints/f0.h5"],
        device="cuda", use_rc=False,
    )
    fasta = pysam.FastaFile(str(CONSTRUCT_FA))
    genes = parse_gene_annotations(DATA_DIR / "construct_gfp.gtf")
    gene = genes[loci["gene_id"]]
    chrom_len = fasta.get_reference_length(gene.chrom_roman)
    window_start = place_window(
        var_pos=loci["var_start"], gene_center=gene.gene_center,
        chrom_length=chrom_len, seq_len=SEQ_LEN, crop_bp_each_side=CROP_BP_EACH_SIDE,
    )
    var_start_in_window = loci["var_start"] - 1 - window_start
    cds_start_in_window = loci["cds_start_in_construct"] - 1 - window_start
    cds_end_in_window = loci["cds_end_in_construct"] - 1 - window_start

    track_ids = SHORKIE_T0_RNA_SEQ_TRACK_IDS
    track_idx_t = torch.tensor(track_ids, device=model.device, dtype=torch.long)

    ref_seq_window = fasta.fetch(gene.chrom_roman, window_start, window_start + SEQ_LEN).upper()
    ref_oh = torch.from_numpy(one_hot_encode_channels_first(ref_seq_window)).to(model.device)  # (4,L)

    def predict_one(alt_window_oh):
        # Forward through track-subset path: (1, OUTPUT_BINS, n_tracks)
        with torch.no_grad():
            return model.forward_tracks_binned(alt_window_oh.unsqueeze(0), track_idx_t).float()[0].cpu().numpy()

    ref_pred = predict_one(ref_oh)            # (OUTPUT_BINS, n_tracks)
    variant_preds = np.zeros((len(variant_seqs), OUTPUT_BINS, len(track_ids)), dtype=np.float32)
    for i, vseq in enumerate(variant_seqs):
        alt = ref_oh.clone()
        alt[:, var_start_in_window : var_start_in_window + 36] = torch.from_numpy(
            one_hot_encode_channels_first(vseq)
        ).to(model.device)
        variant_preds[i] = predict_one(alt)

    # Native TDH3 sanity reference
    nfa = pysam.FastaFile(str(NATIVE_FA))
    tdh3_gene_center = (882815 + 883810) // 2
    chrvii_len = nfa.get_reference_length("VII")
    tdh3_window_start = place_window(
        var_pos=tdh3_gene_center, gene_center=tdh3_gene_center,
        chrom_length=chrvii_len, seq_len=SEQ_LEN, crop_bp_each_side=CROP_BP_EACH_SIDE,
    )
    tdh3_cds_start_in_window = 882815 - 1 - tdh3_window_start
    tdh3_cds_end_in_window   = 883810 - 1 - tdh3_window_start
    tdh3_seq = nfa.fetch("VII", tdh3_window_start, tdh3_window_start + SEQ_LEN).upper()
    tdh3_oh = torch.from_numpy(one_hot_encode_channels_first(tdh3_seq)).to(model.device)
    native_tdh3_pred = predict_one(tdh3_oh)

    with open(CACHE, "wb") as fh:
        pickle.dump({
            "ref_pred": ref_pred,
            "variant_preds": variant_preds,
            "native_tdh3_pred": native_tdh3_pred,
            "track_ids": track_ids,
            "window_start": window_start,
            "var_start_in_window": var_start_in_window,
            "cds_start_in_window": cds_start_in_window,
            "cds_end_in_window": cds_end_in_window,
            "tdh3_window_start": tdh3_window_start,
            "tdh3_cds_start_in_window": tdh3_cds_start_in_window,
            "tdh3_cds_end_in_window": tdh3_cds_end_in_window,
        }, fh)

# Transpose so the per-track axis comes first, matching Yorzoi notebook
ref_pred = ref_pred.T                                 # (n_tracks, OUTPUT_BINS)
variant_preds = variant_preds.transpose(0, 2, 1)      # (N, n_tracks, OUTPUT_BINS)
native_tdh3_pred = native_tdh3_pred.T                 # (n_tracks, OUTPUT_BINS)

print("ref_pred:", ref_pred.shape, f"({len(track_ids)} T0 RNA-seq tracks × {OUTPUT_BINS} bins)")
print("variant_preds:", variant_preds.shape)
print("native TDH3 pred:", native_tdh3_pred.shape)

# Shorkie's T0 RNA-seq tracks are unstranded — no plus/minus split.
# Define PLUS = all tracks so the shared plot code below "just works".
PLUS = list(range(ref_pred.shape[0]))
BIN_BP = BIN_WIDTH
'''


# Plot cells (model-agnostic — use the variables set up above).

PLOT1_MD = '''\
## Plot 1 — one variant (REF), all forward-strand tracks

x = output-bin index → bp position inside the window. y = predicted
coverage. One line per track. The variant gene's CDS is shaded grey;
the 36 nt variable block has a vertical dashed line.

If the model thinks PGAL1 is repressed, every track should sit near
zero across the gene body. The fact that we get *any* signal-vs-noise
ratio in the benchmark means tracks must somehow be modulated by the
variant codons — this plot is the per-track view of what those
modulations look like in absolute terms.
'''

PLOT1_CODE = '''\
fig, ax = plt.subplots(figsize=(11, 4.2))
x = np.arange(ref_pred.shape[1])           # bin indices
for t in PLUS if "PLUS" in dir() else range(ref_pred.shape[0]):
    ax.plot(x, ref_pred[t], lw=0.4, alpha=0.35)

cds_lo = max(0, cds_start_in_window // BIN_BP - CROP_BP_EACH_SIDE // BIN_BP)
cds_hi = max(0, (cds_end_in_window) // BIN_BP - CROP_BP_EACH_SIDE // BIN_BP)
var_bin = (var_start_in_window - CROP_BP_EACH_SIDE) // BIN_BP

ax.axvspan(cds_lo, cds_hi, color="grey", alpha=0.10, label="variant-gene CDS")
ax.axvline(var_bin, color="crimson", lw=1, ls="--", alpha=0.6, label="36 nt variable block")
ax.set_xlabel("output bin index")
ax.set_ylabel("predicted coverage (per-track)")
ax.set_title("All forward-strand tracks, REF variant (gfp_r2)")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
'''


PLOT2_MD = '''\
## Plot 2 — N variants, cross-track mean ± std, colored by measured log2mRNA

For each variant we take the mean across forward-strand tracks per
bin and plot the resulting profile. Lines are colored by the
variant's measured `log2mRNA_rep1` from low (blue) to high (red);
black bold = REF. Shaded band = ±1 std across variants per bin.

If the model is genuinely picking up the codon signal, the high-log2mRNA
variants should peak higher in the CDS region than the low-log2mRNA
ones — a colored ordering visible by eye.
'''

PLOT2_CODE = '''\
fig, ax = plt.subplots(figsize=(11, 4.4))
x = np.arange(ref_pred.shape[1])

cross_track_mean_variants = variant_preds[:, PLUS, :].mean(axis=1)   # (N, BINS)
mean = cross_track_mean_variants.mean(axis=0)
std  = cross_track_mean_variants.std(axis=0)
ax.fill_between(x, mean - std, mean + std, color="grey", alpha=0.2, label="±1 std across variants")

norm = plt.Normalize(variant_log2mrna_rep1.min(), variant_log2mrna_rep1.max())
cmap = plt.cm.coolwarm
for i in range(len(variant_seqs)):
    ax.plot(x, cross_track_mean_variants[i], color=cmap(norm(variant_log2mrna_rep1[i])),
            lw=0.7, alpha=0.6)

ref_mean = ref_pred[PLUS].mean(axis=0)
ax.plot(x, ref_mean, color="black", lw=1.4, label="REF (cross-track mean)")
ax.axvspan(cds_lo, cds_hi, color="grey", alpha=0.10)
ax.axvline(var_bin, color="crimson", lw=1, ls="--", alpha=0.6)
ax.set_xlabel("output bin index")
ax.set_ylabel("cross-track mean coverage")
ax.set_title("N variants, cross-track mean (colored by measured log2mRNA_rep1)")

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="measured log2mRNA_rep1", pad=0.01)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
'''


PLOT3_MD = '''\
## Plot 3 — ALT − REF residual profile

The benchmark logSED reduces to a single number per variant; this
plot shows the spatial structure of the prediction *change* per
position. Plotting `(variant − REF)` cross-track-mean for each
variant. Anything outside the variable block (red dashed) is purely
the model's distal-context "reaction" to the 36 nt swap.
'''

PLOT3_CODE = '''\
residuals = (variant_preds[:, PLUS, :] - ref_pred[PLUS][None, :, :]).mean(axis=1)
fig, ax = plt.subplots(figsize=(11, 4.4))
x = np.arange(residuals.shape[1])
for i in range(residuals.shape[0]):
    ax.plot(x, residuals[i], color=cmap(norm(variant_log2mrna_rep1[i])), lw=0.7, alpha=0.6)
ax.axhline(0, color="black", lw=0.8)
ax.axvspan(cds_lo, cds_hi, color="grey", alpha=0.10)
ax.axvline(var_bin, color="crimson", lw=1, ls="--", alpha=0.6)
ax.set_xlabel("output bin index")
ax.set_ylabel("ALT − REF cross-track mean")
ax.set_title("Per-variant residual profile (where the model 'sees' the swap)")
fig.tight_layout()
'''


PLOT4_MD = '''\
## Plot 4 — per-track variance: which tracks discriminate variants?

For each forward-strand track, compute the std of its CDS-bin sum
across the N variants. Tracks with high std are the ones whose
predictions move when the variant changes — they're the tracks
carrying the variant-effect signal.

If only a small minority of tracks discriminate variants, the
"all-tracks-mean" logSED is averaging real signal with mostly-flat
noise, which is a marginal/dilution effect we'd want to know about.
'''

PLOT4_CODE = '''\
# CDS-bin sum per (variant, track): (N, n_plus_tracks)
cds_lo_bin = max(0, cds_lo)
cds_hi_bin = min(variant_preds.shape[2], cds_hi)
cds_sum_per_track = variant_preds[:, PLUS, cds_lo_bin:cds_hi_bin].sum(axis=2)
per_track_std = cds_sum_per_track.std(axis=0)
per_track_mean = cds_sum_per_track.mean(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
order = np.argsort(per_track_std)[::-1]
axes[0].bar(range(len(per_track_std)), per_track_std[order], color="steelblue")
axes[0].set_xlabel("forward-strand track rank")
axes[0].set_ylabel("std of CDS-bin sum across variants")
axes[0].set_title("Per-track variant discrimination (higher = informative)")

axes[1].scatter(per_track_mean, per_track_std, s=10, alpha=0.6)
axes[1].set_xlabel("mean CDS-bin sum (across variants)")
axes[1].set_ylabel("std of CDS-bin sum (across variants)")
axes[1].set_title("Variance vs mean per track")
fig.tight_layout()
print(f"top-5 most discriminating tracks (forward-strand indices): {order[:5].tolist()}")
print(f"std range: {per_track_std.min():.3f} – {per_track_std.max():.3f}")
print(f"fraction of tracks within 10% of max std: "
      f"{(per_track_std >= 0.9 * per_track_std.max()).mean():.2f}")
'''


PLOT5_MD = '''\
## Plot 5 — per-track Pearson against measured log2mRNA_rep1

For each forward-strand track, compute Pearson r of its variant-wise
CDS-bin sum against the measured `log2mRNA_rep1`. Tracks with high
|r| are the ones whose predictions correlate with the experimental
measurement — these are the tracks doing the work in our logSED
average.

If a small handful of tracks dominate, the cross-track mean is
hiding the real signal. If most tracks correlate similarly, the
mean is a fair aggregator.
'''

PLOT5_CODE = '''\
per_track_r = np.zeros(len(PLUS))
for j, t in enumerate(PLUS):
    per_track_r[j] = pearsonr(cds_sum_per_track[:, j], variant_log2mrna_rep1).statistic if cds_sum_per_track[:, j].std() > 0 else np.nan

fig, ax = plt.subplots(figsize=(10, 3.6))
order = np.argsort(per_track_r)[::-1]
ax.bar(range(len(per_track_r)), per_track_r[order],
       color=["crimson" if v > 0 else "navy" for v in per_track_r[order]])
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("forward-strand track (sorted by r)")
ax.set_ylabel("Pearson r vs measured log2mRNA_rep1")
ax.set_title("Per-track correlation with the experimental label")
fig.tight_layout()
print(f"per-track |r|: median {np.nanmedian(np.abs(per_track_r)):.3f}, "
      f"max {np.nanmax(np.abs(per_track_r)):.3f}")
print(f"fraction of tracks with |r| > 0.3: {(np.abs(per_track_r) > 0.3).mean():.2f}")
'''


PLOT6_MD = '''\
## Plot 6 — predicted CDS-sum (cross-track mean) vs measured log2mRNA

The "marginal" view of how well the model's mean over forward
tracks does on this 40-variant sample. Both replicates shown.
'''

PLOT6_CODE = '''\
pred_cds_sum_mean_tracks = cds_sum_per_track.mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, lbl, ymeas in zip(axes, ["rep1", "rep2"], [variant_log2mrna_rep1, variant_log2mrna_rep2]):
    r = pearsonr(pred_cds_sum_mean_tracks, ymeas).statistic
    rho = spearmanr(pred_cds_sum_mean_tracks, ymeas).statistic
    ax.scatter(ymeas, pred_cds_sum_mean_tracks, s=18, alpha=0.7,
               c=variant_log2mrna_rep1, cmap=cmap, norm=norm)
    ax.set_xlabel(f"measured log2mRNA_{lbl}")
    ax.set_title(f"{lbl}: r={r:.3f}  ρ={rho:.3f}")
axes[0].set_ylabel("predicted CDS-sum (cross-track mean)")
fig.tight_layout()
'''


PLOT7_MD = '''\
## Plot 7 — absolute coverage: the construct vs native TDH3 locus

PGAL1 should be glucose-repressed; TDH3 is one of the most strongly
transcribed yeast genes (its promoter is constitutive). If the model
correctly thinks PGAL1 is off, the construct's predicted coverage at
the GFP_variant gene body should sit far below the predicted coverage
at the native TDH3 gene body.

If they sit at similar levels, the model is *not* respecting the
condition mismatch — and whatever variant-discriminating signal we
get is something other than "actual transcription of the gene".
'''

PLOT7_CODE = '''\
fig, ax = plt.subplots(figsize=(11, 4.4))
x = np.arange(ref_pred.shape[1])

construct_profile = ref_pred[PLUS].mean(axis=0)
tdh3_profile = native_tdh3_pred[PLUS].mean(axis=0)

ax.plot(x, construct_profile, color="steelblue", lw=1.0,
        label="construct REF (cross-track mean) — PGAL1 locus")
ax.plot(x, tdh3_profile, color="darkorange", lw=1.0,
        label="native TDH3 locus (cross-track mean) — strongly expressed")

ax.axvspan(cds_lo, cds_hi, color="steelblue", alpha=0.08,
           label="construct CDS region")
tdh3_cds_lo = max(0, tdh3_cds_start_in_window // BIN_BP - CROP_BP_EACH_SIDE // BIN_BP)
tdh3_cds_hi = max(0, tdh3_cds_end_in_window // BIN_BP - CROP_BP_EACH_SIDE // BIN_BP)
ax.axvspan(tdh3_cds_lo, tdh3_cds_hi, color="darkorange", alpha=0.10,
           label="TDH3 CDS region")
ax.set_xlabel("output bin index")
ax.set_ylabel("predicted coverage (cross-track mean)")
ax.set_title("Absolute coverage — construct PGAL1 vs native TDH3")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()

print(f"construct CDS-bin sum (cross-track mean): {construct_profile[cds_lo:cds_hi].sum():.2f}")
print(f"native TDH3 CDS-bin sum (cross-track mean): {tdh3_profile[tdh3_cds_lo:tdh3_cds_hi].sum():.2f}")
print(f"ratio (TDH3 / construct): "
      f"{tdh3_profile[tdh3_cds_lo:tdh3_cds_hi].sum() / max(1e-6, construct_profile[cds_lo:cds_hi].sum()):.2f}")
'''


PLOTS = [
    ("plot1", PLOT1_MD, PLOT1_CODE),
    ("plot2", PLOT2_MD, PLOT2_CODE),
    ("plot3", PLOT3_MD, PLOT3_CODE),
    ("plot4", PLOT4_MD, PLOT4_CODE),
    ("plot5", PLOT5_MD, PLOT5_CODE),
    ("plot6", PLOT6_MD, PLOT6_CODE),
    ("plot7", PLOT7_MD, PLOT7_CODE),
]


def build_notebook(model_name: str, model_setup_code: str, intro_md: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(intro_md),
        nbf.v4.new_code_cell(SETUP_COMMON),
        nbf.v4.new_code_cell(model_setup_code),
    ]
    for _, md, code in PLOTS:
        cells.append(nbf.v4.new_markdown_cell(md))
        cells.append(nbf.v4.new_code_cell(code))
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


YORZOI_INTRO = '''\
# Chen / Yorzoi — investigation: why does the model predict signal at a repressed promoter?

The Chen 2017 experiment runs in 2 % galactose, which is *required*
for the PGAL1 promoter at the construct's chrII GAL1 locus to be
active. Yorzoi's training data has **zero galactose-condition tracks**
(checked against `yorzoi/track_annotation.json`) — so the model is
being asked to score a locus that's supposed to be silent in every
condition it ever observed.

And yet we get Pearson r ≈ 0.5–0.6 on GFP r2. This notebook digs into
where that signal comes from: per-track behaviour, profile shape,
and whether the absolute coverage levels are consistent with an
"off" PGAL1 promoter.

(Library: GFP r2, 40 variants stratified across the log2mRNA_rep1
range, plus the construct's REF variant.)
'''

SHORKIE_INTRO = '''\
# Chen / Shorkie — investigation: why does the model predict signal at a repressed promoter?

Same question as the Yorzoi notebook, asked of Shorkie. Shorkie's
`targets.txt` has **zero galactose RNA-seq tracks** — the 5 lines
matching "gal" are all ChIP-exo of TFs that happen to be named GAL
(GAL3 / GAL4 / GAL11). The `_T0_` RNA-seq subset our adapter
averages over is all from glucose rich-media TF-knockout time
courses.

For speed, this notebook runs **single-fold (f0), no reverse-complement
averaging** — the production benchmark uses an 8-fold ensemble × RC.
Magnitudes will be lower-variance than the ensemble; structure
should be the same.

(Library: GFP r2, 40 variants stratified across the log2mRNA_rep1
range, plus the construct's REF variant.)
'''


def main():
    yorzoi_nb = build_notebook("yorzoi", YORZOI_SETUP, YORZOI_INTRO)
    shorkie_nb = build_notebook("shorkie", SHORKIE_SETUP, SHORKIE_INTRO)
    nbf.write(yorzoi_nb,  OUT_DIR / "chen_yorzoi_investigation.ipynb")
    nbf.write(shorkie_nb, OUT_DIR / "chen_shorkie_investigation.ipynb")
    print("wrote:")
    print(" ", OUT_DIR / "chen_yorzoi_investigation.ipynb")
    print(" ", OUT_DIR / "chen_shorkie_investigation.ipynb")


if __name__ == "__main__":
    main()
