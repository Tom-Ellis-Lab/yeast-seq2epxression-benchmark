"""Shared ExoShorkie architecture + per-genome normalization constants.

ExoShorkie (Mandl & Orenstein 2026) = the Shorkie trunk fine-tuned with a
linear ``Dense(1)`` per-bin head to predict RNA-seq coverage of exogenous DNA
in yeast. We distilled each of the 6 published per-genome 40-model ensembles
into one PyTorch student (``data/models/exoshorkie/students/<genome>.pt``); the
benchmark model is the mean over the 6 students (see
``yeastbench.models.exoshorkie.ExoShorkie``).

Geometry is identical to Shorkie: 16,384 bp in → 896 bins × 16 bp covering the
central 14,336 bp.

The students predict in the authors' log-z space; count space requires the
inverse ``count = expm1(z*sigma + mu)`` (their ``ISM_student.py``). ``mu``/``sigma``
are NOT shipped with the weights and not published — they are the global
mean/std of ``log1p`` of the 16 bp-summed CPM coverage of each exogenous genome
(their ``compute_logz_stats_multi``). We recompute them from the public figshare
coverage NPZ (DOI 10.6084/m9.figshare.31075375) via
``scripts/exoshorkie/compute_logz_stats.py`` — that script is the reproducible
source of truth; the values below are its output, embedded here because ``data/``
is not version-controlled.

Validated end-to-end (``scripts/exoshorkie/validate_denorm.py``): running each
genome's student on its real windows and denormalizing with these constants
reproduces the figshare truth coverage at forward-strand Spearman 0.80–0.97 and
a stable per-genome magnitude ratio (sum(pred)/sum(truth)) of 0.74–0.95 — the
mild <1 bias is distillation shrinkage through the convex ``expm1`` and cancels
in shape/ratio metrics.
"""
from __future__ import annotations

# Architecture (identical to Shorkie's; see data/models/shorkie/params.json).
SEQ_LEN: int = 16384
OUTPUT_BINS: int = 896
BIN_WIDTH: int = 16
CROP_BP_EACH_SIDE: int = 1024  # 64 bins × 16 bp/bin

# The 6 exogenous genomes (HF dir name == student filename stem). Order is the
# ensemble order; fixed so the student↔(mu,sigma) pairing is unambiguous.
EXOSHORKIE_GENOMES: list[str] = [
    "M_pneumoniae",
    "M_mycoides",
    "Data_storage_chr",
    "HPRT1",
    "HPRT1R",
    "Human_chr_7",
]

# genome -> (mu, sigma) for the log-z inverse, reproduced by
# scripts/exoshorkie/compute_logz_stats.py over the figshare coverage NPZ.
EXOSHORKIE_LOGZ_STATS: dict[str, tuple[float, float]] = {
    "M_pneumoniae": (3.5227722545285824, 1.721217520084453),
    "M_mycoides": (1.3004100315469918, 1.517358675702041),
    "Data_storage_chr": (4.087326421867158, 1.9734211309165708),
    "HPRT1": (4.4911224030322, 1.8123546620137028),
    "HPRT1R": (4.555503016345565, 1.7732991131140663),
    "Human_chr_7": (2.0888971418147566, 1.5283684969481497),
}

# Default on-disk location of the distilled students (gitignored, under data/).
DEFAULT_STUDENTS_DIR: str = "data/models/exoshorkie/students"
