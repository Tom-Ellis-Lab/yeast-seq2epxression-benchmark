#!/usr/bin/env bash
# Unattended per-genome ExoShorkie distillation driver.
#
# For each genome, serially: download its 40 teachers + genome FASTA from
# figshare, generate soft targets, train the student, and ONLY if the student
# clears the Pearson r>0.98 gate, delete the teachers and move on. Any failure
# (download, target-gen, OOM, or a failed gate) HALTS the driver and leaves that
# genome's teachers in place for inspection.
#
# Usage:  bash scripts/exoshorkie/distill_all.sh <gpu_index>
# Run it detached:  tmux new-session -d -s exo_distill -c <worktree> \
#     "bash scripts/exoshorkie/distill_all.sh 2 > logs/distill_all.log 2>&1"
set -uo pipefail

WT=/home/tds122/yeast-exoshorkie
WEIGHTS=/home/tds122/exoshorkie-weights
GPU=${1:?usage: distill_all.sh <gpu_index>}
cd "$WT" || exit 1

# HF genome dir -> figshare FASTA file id(s) (space-separated => multi-part, concatenated)
declare -A FASTA_IDS=(
  [M_mycoides]="61042561"          # Mmmyco.fa
  [HPRT1]="61042522"               # HPRT1.fa
  [HPRT1R]="61042531"              # HPRT1R.fa
  [Data_storage_chr]="61042513"    # dChr.fa
  [Human_chr_7]="61042546 61042555"  # chr7_part1.fa + chr7_part2.fa
)
ORDER=(M_mycoides HPRT1 HPRT1R Data_storage_chr Human_chr_7)

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "$WEIGHTS/genomes" "$WEIGHTS/targets" "$WEIGHTS/students"

for G in "${ORDER[@]}"; do
  echo "[$(ts)] ===== GENOME $G ====="
  FA="$WEIGHTS/genomes/$G.fa"

  echo "[$(ts)] $G: downloading 40 teachers ..."
  uv run --with huggingface_hub python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Jonathan-Mandl/ExoShorkie-models', allow_patterns='Models/$G/*', local_dir='$WEIGHTS')
" || { echo "[$(ts)] FAIL: teacher download for $G"; exit 1; }
  n=$(ls "$WEIGHTS"/Models/"$G"/cv*/f*/model_finetune.h5 2>/dev/null | wc -l)
  echo "[$(ts)] $G: $n teacher files"
  [ "$n" -eq 40 ] || { echo "[$(ts)] FAIL: $G expected 40 teachers, got $n"; exit 1; }

  echo "[$(ts)] $G: downloading FASTA ..."
  : > "$FA"
  for fid in ${FASTA_IDS[$G]}; do
    curl -fsSL "https://ndownloader.figshare.com/files/$fid" >> "$FA" \
      || { echo "[$(ts)] FAIL: FASTA $fid for $G"; exit 1; }
  done
  echo "[$(ts)] $G: FASTA $(grep -c '^>' "$FA") record(s), $(wc -c < "$FA") bytes"

  echo "[$(ts)] $G: target-gen ..."
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 uv run --extra shorkie python scripts/exoshorkie/gen_targets.py \
    --genome "$G" --fasta "$FA" --teachers-root "$WEIGHTS/Models/$G" \
    --out "$WEIGHTS/targets/$G.npz" --n-synth 50000 --target-windows 10000 \
    --batch-size 32 --device cuda:0 \
    || { echo "[$(ts)] FAIL: target-gen for $G"; exit 1; }

  echo "[$(ts)] $G: training student (gate r>0.98) ..."
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 uv run --extra shorkie python scripts/exoshorkie/train_student.py \
    --targets "$WEIGHTS/targets/$G.npz" --trunk-ckpt data/models/shorkie/checkpoints/f0.h5 \
    --out "$WEIGHTS/students/$G.pt" --device cuda:0
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$(ts)] GATE FAILED (or error rc=$rc) for $G — keeping teachers, HALTING."
    exit 2
  fi

  echo "[$(ts)] $G: PASSED gate, deleting teachers."
  rm -rf "$WEIGHTS/Models/$G"
  echo "[$(ts)] ===== $G COMPLETE ====="
done

echo "[$(ts)] ALL GENOMES COMPLETE"
