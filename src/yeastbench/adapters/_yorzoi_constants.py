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
