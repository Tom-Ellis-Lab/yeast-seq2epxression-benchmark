#!/usr/bin/env bash
# Set up a sibling git worktree, sharing data/ from the main checkout and
# installing a fresh uv-managed venv.
#
# Usage:
#   scripts/setup_worktree.sh <branch> [<path>]
#
# Examples:
#   scripts/setup_worktree.sh feature/foo
#       -> ../yeast-bench-feature-foo, on branch feature/foo (created if missing)
#   scripts/setup_worktree.sh feature/foo ~/wt/foo
#       -> ~/wt/foo, on branch feature/foo
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <branch> [<path>]" >&2
  exit 1
fi

BRANCH="$1"
DEFAULT_PATH="../yeast-bench-${BRANCH//\//-}"
WT_PATH="${2:-$DEFAULT_PATH}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
DATA_SRC="$REPO_ROOT/data"

if [[ ! -d "$DATA_SRC" ]]; then
  echo "error: $DATA_SRC does not exist; aborting" >&2
  exit 1
fi

if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git -C "$REPO_ROOT" worktree add "$WT_PATH" "$BRANCH"
else
  git -C "$REPO_ROOT" worktree add -b "$BRANCH" "$WT_PATH"
fi

# Resolve absolute path of the new worktree
WT_ABS="$(cd "$WT_PATH" && pwd)"

# Share data/ rather than copying 9+ GB
ln -s "$DATA_SRC" "$WT_ABS/data"

# Fresh venv (uv reuses the global package cache, so this is fast)
( cd "$WT_ABS" && uv sync )

cat <<EOF

Worktree ready:
  path:   $WT_ABS
  branch: $BRANCH
  data/:  symlinked -> $DATA_SRC
  .venv/: fresh (uv sync)

Next:
  cd $WT_ABS
  source .venv/bin/activate
  claude
EOF
