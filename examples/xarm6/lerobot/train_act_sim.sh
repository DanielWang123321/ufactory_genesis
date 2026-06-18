#!/usr/bin/env bash
# Phase 3: ACT pretraining on sim dataset (requires: pip install -r requirements-lerobot.txt)
set -euo pipefail
cd "$(dirname "$0")/../../.."
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$HOME/.cache/numba}"

REPO_ID="${REPO_ID:-local/xarm6_g2_sim_pickplace}"
ROOT="${ROOT:-data/lerobot_datasets/local_xarm6_g2_sim_pickplace}"
OUT="${OUT:-outputs/act_sim_pretrain}"

lerobot-train \
  --policy.type=act \
  --policy.repo_id="${REPO_ID}" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$ROOT" \
  --output_dir="$OUT" \
  --policy.push_to_hub=false \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --batch_size=8 \
  --steps=5000

# On 8GB GPUs use: BATCH_SIZE=1 CHUNK=50 STEPS=100 bash train_act_sim.sh

echo "Training complete. Checkpoints: $OUT/checkpoints"
