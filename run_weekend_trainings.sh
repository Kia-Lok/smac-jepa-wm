#!/usr/bin/env bash

# Do NOT use `set -e`.
# We want later commands to continue even if an earlier training run fails.
set -u
set -o pipefail

mkdir -p logs/weekend_trainings

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

run_cmd() {
  local name="$1"
  shift

  local log_file="logs/weekend_trainings/${name}.log"

  echo "============================================================"
  echo "[$(timestamp)] START: ${name}"
  echo "Log file: ${log_file}"
  echo "============================================================"

  "$@" 2>&1 | tee "${log_file}"
  local exit_code=${PIPESTATUS[0]}

  echo "============================================================"
  echo "[$(timestamp)] END: ${name}"
  echo "Exit code: ${exit_code}"
  echo "============================================================"

  echo "${name},${exit_code},$(timestamp)" >> logs/weekend_trainings/summary.csv

  return 0
}

echo "run_name,exit_code,finished_at" > logs/weekend_trainings/summary.csv

run_cmd "nstep_uniform_category_seq10" \
  env LD_LIBRARY_PATH="" python -m smac_jepa.train_nstep_temporal \
    --manifest splits/generated_seed4.json \
    --split train \
    --out-dir runs/generated_seed4_nstep_uniform_category_seq10 \
    --model-size default \
    --epochs 10 \
    --batch-size 32 \
    --num-workers 4 \
    --context-len 8 \
    --window-len 8 \
    --temporal-loss uniform \
    --category-prefixes smac_jepa/category_prefixes.json \
    --priority-alpha 1.0 \
    --priority-ema 0.7 \
    --priority-min-prob 0.005 \
    --priority-metric pred_loss \
    --device cuda \
    --amp \
    --wandb \
    --wandb-project SMAC-JEPA-losses \
    --wandb-name nstep_uniform_category_seq10

run_cmd "enemy_visibility_mask_seq10" \
  env LD_LIBRARY_PATH="" python -m smac_jepa.train_enemy_visibility_mask \
    --data-dir data/generated_mapdims_only \
    --split train \
    --out-dir runs/generated_seed4_enemy_visibility_mask_seq10 \
    --model-size default \
    --epochs 10 \
    --batch-size 64 \
    --num-workers 4 \
    --context-len 4 \
    --window-mode sequential \
    --enemy-visibility-mask \
    --enemy-sight-range 9.0 \
    --device cuda \
    --amp \
    --wandb \
    --wandb-project SMAC-JEPA-losses \
    --wandb-name enemy_visibility_mask_seq10

run_cmd "sample_prio_seq10" \
  env LD_LIBRARY_PATH="" python -m smac_jepa.train_sample_prio \
    --manifest splits/generated_seed4_mapdims_only.json \
    --split train \
    --out-dir runs/generated_seed4_sample_prio_seq30 \
    --model-size default \
    --epochs 10 \
    --batch-size 64 \
    --num-workers 4 \
    --context-len 4 \
    --window-mode sequential \
    --device cuda \
    --amp \
    --sample-prioritized \
    --priority-alpha 0.6 \
    --priority-uniform-mix 0.5 \
    --priority-ema-beta 0.9 \
    --priority-warmup-epochs 1 \
    --priority-score pred_loss \
    --wandb \
    --wandb-project SMAC-JEPA-losses \
    --wandb-name sample-prio-seq10

echo "============================================================"
echo "[$(timestamp)] ALL RUNS FINISHED"
echo "Summary:"
cat logs/weekend_trainings/summary.csv
echo "============================================================"
