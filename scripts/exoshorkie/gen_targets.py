"""ExoShorkie distillation — Step 1: per-genome soft-target generation (pure PyTorch).

Reproduces the authors' `scripts/distill_synthetic.py` recipe (ExoShorkie repo,
docs/Distillation.md) without TensorFlow: sample genomic windows, build
AlphaGenome-style synthetic windows, run the genome's 40 teachers through our
verified PyTorch port, and average to soft targets `(n_synth, 896)`.

Two deliberate fidelity choices vs the authors' code:
  * Inputs use OUR `_expand_species_encoding` (S. cerevisiae at global channel
    114), NOT their HEAD `build_shorkie_features` — the latter drifted to channel
    119 and no longer matches the published weights (confirmed by the conv_dna
    per-channel weight norm). A guard asserts each teacher's max-norm species
    channel is 114 so a wrong checkpoint/encoding fails loudly.
  * They run float32 (`mixed_precision float32`); so do we. Targets are the plain
    mean over the 40 teachers, single forward each (no test-time RC) — exactly
    their `mean_preds = sum_preds / n_total_models`.

The augmentation block below is vendored verbatim (modulo a local reverse-
complement to avoid a biopython dep) from the authors' distill_synthetic.py.

Usage (full run for one genome):
    uv run --extra shorkie python scripts/exoshorkie/gen_targets.py \
        --genome M_pneumoniae \
        --fasta /path/to/Mpneumo.fa \
        --teachers-root /home/tds122/exoshorkie-weights \
        --out /home/tds122/exoshorkie-weights/targets/M_pneumoniae.npz
Smoke test (1 teacher, tiny):
    ... --n-synth 200 --target-windows 100 --max-teachers 1 --fasta <any>.fa
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from yeastbench.adapters._genome import one_hot_encode_channels_first
from yeastbench.models.shorkie.nn import ShorkieModule

WINDOW_BP = 16384
OUTPUT_BINS = 896
SEED = 42
BASES = ["A", "C", "G", "T"]
_COMP = str.maketrans("ACGTN", "TGCAN")
SHORKIE_PARAMS = "data/models/shorkie/params.json"
# ExoShorkie feeds the S. cerevisiae host one-hot at global input column 119
# (build_shorkie_features: 5-wide DNA incl N + species[114]) in BOTH training and
# inference, so we must feed 119 to run its published weights faithfully. (The
# Shorkie trunk was pretrained at 114; the shift is the authors', baked into the
# weights — see the project memory / the filed upstream issue.)
EXOSHORKIE_SPECIES_CHANNEL = 119


def rc(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


# ───────────── AlphaGenome augmentation (vendored from authors) ─────────────
def random_base(exclude: str | None = None) -> str:
    if exclude is None:
        return random.choice(BASES)
    b = random.choice(BASES)
    while b == exclude:
        b = random.choice(BASES)
    return b


def mutate_bases(seq: str, frac: float = 0.04) -> str:
    L = len(seq)
    if L == 0:
        return seq
    n_mut = max(1, int(round(frac * L)))
    positions = np.random.choice(L, size=n_mut, replace=False)
    seq_list = list(seq)
    for pos in positions:
        seq_list[pos] = random_base(exclude=seq_list[pos])
    return "".join(seq_list)


def apply_structural_variants(seq: str, lam: float = 1.0, max_len: int = 20) -> str:
    k = np.random.poisson(lam)
    for _ in range(k):
        if len(seq) <= 1:
            break
        vtype = random.choice(["ins", "del", "inv"])
        L = random.randint(1, max_len)
        if vtype == "ins":
            pos = random.randint(0, len(seq))
            insert = "".join(random_base() for _ in range(L))
            seq = seq[:pos] + insert + seq[pos:]
        elif vtype == "del":
            if len(seq) <= L:
                continue
            pos = random.randint(0, len(seq) - L)
            seq = seq[:pos] + seq[pos + L:]
        elif vtype == "inv":
            if len(seq) <= L:
                continue
            pos = random.randint(0, len(seq) - L)
            seq = seq[:pos] + rc(seq[pos:pos + L]) + seq[pos + L:]
    return seq


def normalize_length(seq: str, target_len: int) -> str:
    L = len(seq)
    if L == target_len:
        return seq
    if L > target_len:
        start = random.randint(0, L - target_len)
        return seq[start:start + target_len]
    pad = target_len - L
    left = pad // 2
    return "N" * left + seq + "N" * (pad - left)


def alphagenome_augment_window(seq_win: str) -> str:
    if random.random() < 0.5:
        seq_win = rc(seq_win)
    seq_win = mutate_bases(seq_win, frac=0.04)
    seq_win = apply_structural_variants(seq_win, lam=1.0, max_len=20)
    return normalize_length(seq_win, WINDOW_BP)


# ───────────── windowing (ports data_loader.make_windows / build_windows) ─────────────
def make_windows(L: int, win: int, stride: int) -> list[tuple[int, int]]:
    return [(p, p + win) for p in range(0, L - win + 1, stride)]


def load_sources(fasta_path: str) -> list[tuple[str, str]]:
    import pysam

    fa = pysam.FastaFile(fasta_path)
    return [(name, fa.fetch(name).upper()) for name in fa.references]


def build_all_windows(
    sources: list[tuple[str, str]], target_windows: int
) -> list[tuple[int, int, int]]:
    total = sum(max(0, len(seq) - WINDOW_BP) for _, seq in sources)
    if total <= 0:
        raise RuntimeError("All sequences shorter than the window size.")
    stride = max(1, int(total / target_windows))
    out: list[tuple[int, int, int]] = []
    for si, (_, seq) in enumerate(sources):
        for a, b in make_windows(len(seq), WINDOW_BP, stride):
            out.append((si, a, b))
    return out


def generate_synthetic(
    sources: list[tuple[str, str]], windows: list[tuple[int, int, int]], n_synth: int
) -> list[str]:
    n = len(windows)
    seqs = []
    for i in range(n_synth):
        si, a, b = windows[np.random.randint(n)]
        seqs.append(alphagenome_augment_window(sources[si][1][a:b]))
        if (i + 1) % 5000 == 0:
            print(f"  generated {i + 1}/{n_synth}")
    return seqs


# ───────────── teachers ─────────────
def find_teacher_paths(teachers_root: str, genome: str) -> list[Path]:
    root = Path(teachers_root)
    paths: list[Path] = []
    for pattern in (
        f"Models/{genome}/cv*/f*/model_finetune.h5",
        f"**/Models/{genome}/cv*/f*/model_finetune.h5",
        "cv*/f*/model_finetune.h5",  # root already points at the genome dir
    ):
        hits = sorted(root.glob(pattern))
        if hits:
            paths = hits
            break
    # Dedupe by (cv, fold) so a stray probe copy can't be counted twice;
    # prefer the canonical (non-'probe') path.
    by_id: dict[tuple[str, str], Path] = {}
    for p in paths:
        key = (p.parent.parent.name, p.parent.name)  # (cvX, fY)
        if key not in by_id or "probe" in str(by_id[key]):
            by_id[key] = p
    return [by_id[k] for k in sorted(by_id)]


def assert_exoshorkie_encoding(model: ShorkieModule) -> None:
    """Guard that the model feeds ExoShorkie's host channel (119) + N channel,
    not Shorkie's defaults. conv_dna weight norms are NOT a reliable channel
    detector (the fine-tuned ensemble keeps residual signal on both 114 and
    119), so we assert the configured encoding instead."""
    if model._species_channel != EXOSHORKIE_SPECIES_CHANNEL:
        raise RuntimeError(
            f"model._species_channel={model._species_channel}, expected "
            f"{EXOSHORKIE_SPECIES_CHANNEL}; build with ShorkieModule.exoshorkie_config(...)."
        )
    if not model._encode_n_channel:
        raise RuntimeError("ExoShorkie model must encode the N channel (col 4).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--teachers-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-synth", type=int, default=50_000)
    ap.add_argument("--target-windows", type=int, default=10_000)
    ap.add_argument("--max-teachers", type=int, default=0, help="0 = all (smoke: 1)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(SHORKIE_PARAMS) as fh:
        exo_cfg = ShorkieModule.exoshorkie_config(json.load(fh)["model"])

    print(f"Loading FASTA {args.fasta} …")
    sources = load_sources(args.fasta)
    print(f"  {len(sources)} record(s), total {sum(len(s) for _, s in sources):,} bp")
    windows = build_all_windows(sources, args.target_windows)
    print(f"  {len(windows)} base windows")

    print(f"Generating {args.n_synth} synthetic windows …")
    synth = generate_synthetic(sources, windows, args.n_synth)

    # Encode once to a uint8 (N,4,WINDOW_BP) buffer reused across teachers.
    print("Encoding synthetic windows …")
    X = np.stack([one_hot_encode_channels_first(s) for s in synth]).astype(np.uint8)
    print(f"  X {X.shape} ({X.nbytes / 1e9:.2f} GB)")

    teachers = find_teacher_paths(args.teachers_root, args.genome)
    if args.max_teachers:
        teachers = teachers[: args.max_teachers]
    if not teachers:
        raise FileNotFoundError(
            f"No teachers under {args.teachers_root} for genome {args.genome}"
        )
    print(f"{len(teachers)} teacher(s):")
    for p in teachers:
        print("   ", p)

    device = torch.device(args.device)
    sum_preds = np.zeros((len(synth), OUTPUT_BINS), dtype=np.float64)
    t0 = time.time()
    for ti, path in enumerate(teachers):
        model = ShorkieModule.from_tf_checkpoint(exo_cfg, str(path)).to(device).eval()
        assert_exoshorkie_encoding(model)
        for bs in range(0, len(synth), args.batch_size):
            xb = torch.from_numpy(X[bs : bs + args.batch_size]).float().to(device)
            with torch.no_grad():
                out = model(xb)  # (B, 896)
            sum_preds[bs : bs + args.batch_size] += out.double().cpu().numpy()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  [{ti + 1}/{len(teachers)}] {path.parent.name} done "
              f"({time.time() - t0:.1f}s elapsed)")

    mean_preds = (sum_preds / len(teachers)).astype(np.float32)
    print(f"mean_preds {mean_preds.shape}  range [{mean_preds.min():.3f}, "
          f"{mean_preds.max():.3f}]  in {time.time() - t0:.1f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        train_sequences=np.array(synth, dtype=object),
        train_mean_preds=mean_preds,
        meta=np.array(
            json.dumps({
                "genome": args.genome, "seed": args.seed,
                "n_teachers": len(teachers), "n_synth": len(synth),
                "target_windows": args.target_windows,
            })
        ),
    )
    print(f"saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
