"""Shared Yorzoi architecture + track constants.

Lives here (rather than inside any task-specific adapter) so other
adapters don't have to import from a sibling — removing one
task-adapter must not break another. See PR #1 review.
"""
from __future__ import annotations

# Architecture (verified by loading the HF checkpoint):
# input 4,992 bp → output (B, 162 tracks, 300 bins), 10 bp/bin,
# post-crop output covers the central 3,000 bp of input.
SEQ_LEN: int = 4992
OUTPUT_BINS: int = 300
BIN_WIDTH: int = 10
CROP_BP_EACH_SIDE: int = 996  # (4992 - 3000) // 2

# Track layout from yorzoi/track_annotation.json:
#   indices 0..80   → '+' (forward) strand, 81 tracks
#   indices 81..161 → '-' (reverse) strand, 81 tracks (same samples)
YORZOI_PLUS_TRACK_IDS: list[int] = list(range(0, 81))
YORZOI_MINUS_TRACK_IDS: list[int] = list(range(81, 162))

# Brooks Nanopore direct-RNA subset of Yorzoi's plus-strand tracks. These
# 63 indices are all the strict-WT Brooks BEDs in the manifest (pattern
# `JS\d+_\d{8}_porechopped_filtered_canuCorrected_distinguished.plus.bw`,
# JS94 restricted to the 3 deep-WT runs we use as truth: 20180214,
# 20180628, 20181203). Minus-strand indices are derived by +81.
#
# Source of truth: `yorzoi/track_annotation.json` (regenerate this map
# if that file changes — script in chat 2026-05-20).
BROOKS_NANOPORE_TRACK_IDS_PLUS_ALL: list[int] = [
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 77,
]

# Per-strain Nanopore tracks for `track_mode="matched"`. Most strains
# have a single WT BED in the manifest; JS710/JS731/JS94 have 3–4.
# JS94 here is the 3 deep-WT runs only (matching the truth-side filter).
BROOKS_NANOPORE_TRACKS_BY_STRAIN: dict[str, list[int]] = {
    "JS274": [11], "JS571": [12], "JS599": [13], "JS601": [14],
    "JS602": [15], "JS603": [16], "JS604": [17], "JS605": [18],
    "JS606": [19], "JS607": [20], "JS608": [21], "JS609": [22],
    "JS610": [23], "JS611": [24], "JS612": [25], "JS613": [26],
    "JS614": [27], "JS615": [28], "JS622": [29], "JS623": [30],
    "JS624": [31], "JS625": [32], "JS626": [33], "JS627": [34],
    "JS628": [35], "JS629": [36], "JS705": [37], "JS706": [38],
    "JS707": [39], "JS708": [40], "JS709": [41], "JS710": [42, 43, 44, 45],
    "JS711": [48], "JS712": [49], "JS713": [50], "JS714": [51],
    "JS715": [52], "JS716": [53], "JS717": [54], "JS718": [55],
    "JS719": [56], "JS720": [57], "JS721": [58], "JS722": [59],
    "JS723": [60], "JS724": [61], "JS725": [62], "JS726": [63],
    "JS727": [64], "JS728": [65], "JS729": [66], "JS730": [67],
    "JS731": [68, 69, 70], "JS732": [71], "JS733": [72],
    "JS94": [73, 75, 77],
}
