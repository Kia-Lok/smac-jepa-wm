#!/usr/bin/env bash
# Three-day JEPA world-model experiment runner.
# Goal: find a JEPA seqmem checkpoint that is a stronger functional RSSM alternative:
#   - good decoded world-state prediction
#   - better h3-h5/h10 latent rollout stability
#   - visibility/full-target belief modeling
#   - optional PER/sample-priority for faster improvement
#
# Error behavior:
#   - no set -e
#   - each command logs to its own file and status
#   - a failed run does not stop later runs

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR" || { echo "ERROR: cannot cd to $ROOT_DIR"; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/three_day_jepa_logs_${STAMP}"
mkdir -p "$LOG_DIR"

run_cmd() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  local status_file="${LOG_DIR}/${name}.status"
  echo "============================================================"
  echo "START: $name"
  echo "TIME:  $(date)"
  echo "LOG:   $log_file"
  echo "============================================================"
  "$@" 2>&1 | tee "$log_file"
  local code=${PIPESTATUS[0]}
  echo "$code" > "$status_file"
  if [[ "$code" -eq 0 ]]; then
    echo "SUCCESS: $name"
  else
    echo "FAILED: $name with exit code $code"
    echo "Continuing..."
  fi
  echo "END:   $name at $(date)"
  echo
  return 0
}

run_python_module() {
  # Helper only for readability; pass: name module args...
  local name="$1"
  local module="$2"
  shift 2
  run_cmd "$name" env LD_LIBRARY_PATH="" python -m "$module" "$@"
}

run_eval_if_any() {
  local name="$1"
  local out_dir="$2"
  shift 2
  local ckpts=()
  for ckpt in "$@"; do
    if [[ -f "$ckpt" ]]; then
      ckpts+=(--checkpoint "$ckpt")
    else
      echo "SKIP missing checkpoint for eval $name: $ckpt"
    fi
  done
  if [[ ${#ckpts[@]} -eq 0 ]]; then
    echo "SKIP eval $name: no checkpoints exist"
    return 0
  fi
  run_cmd "$name" env LD_LIBRARY_PATH="" python eval_rnn_seqmem_combined_metrics.py \
    --manifest splits/generated_seed4_mapdims_only.json \
    --split eval \
    "${ckpts[@]}" \
    --out-dir "$out_dir" \
    --batch-size 16 \
    --num-workers 4 \
    --max-batches 500 \
    --window-mode sequential \
    --device cuda \
    --amp
}

echo "============================================================"
echo "Three-day JEPA experiment runner"
echo "Root:       $ROOT_DIR"
echo "Logs:       $LOG_DIR"
echo "Start time: $(date)"
echo "============================================================"
echo

# -------------------------------------------------------------------
# Install/copy required scripts if available from /mnt/data.
# -------------------------------------------------------------------
if [[ -f /mnt/data/three_day_jepa/train_markov_rollout_rnn_visibility_seqmem_per.py ]]; then
  cp /mnt/data/three_day_jepa/train_markov_rollout_rnn_visibility_seqmem_per.py \
    smac_jepa/train_markov_rollout_rnn_visibility_seqmem_per.py
  echo "Installed smac_jepa/train_markov_rollout_rnn_visibility_seqmem_per.py"
else
  echo "WARNING: /mnt/data/three_day_jepa/train_markov_rollout_rnn_visibility_seqmem_per.py not found"
fi

if [[ -f /mnt/data/seqmem_combined_eval_fixed/eval_rnn_seqmem_combined_metrics.py ]]; then
  cp /mnt/data/seqmem_combined_eval_fixed/eval_rnn_seqmem_combined_metrics.py \
    eval_rnn_seqmem_combined_metrics.py
  echo "Installed fixed eval_rnn_seqmem_combined_metrics.py"
elif [[ -f /mnt/data/seqmem_combined_eval/eval_rnn_seqmem_combined_metrics.py ]]; then
  cp /mnt/data/seqmem_combined_eval/eval_rnn_seqmem_combined_metrics.py \
    eval_rnn_seqmem_combined_metrics.py
  echo "Installed eval_rnn_seqmem_combined_metrics.py"
fi

if [[ -f /mnt/data/rnn_weekend_experiments/train_markov_rollout_rnn_visibility_seqmem_experiments.py ]]; then
  cp /mnt/data/rnn_weekend_experiments/train_markov_rollout_rnn_visibility_seqmem_experiments.py \
    smac_jepa/train_markov_rollout_rnn_visibility_seqmem_experiments.py
  echo "Installed smac_jepa/train_markov_rollout_rnn_visibility_seqmem_experiments.py"
fi

# -------------------------------------------------------------------
# Phase 0: evaluate existing action-memory runs if available.
# This completes the Exp01/03 comparisons with Exp02/04.
# -------------------------------------------------------------------
run_eval_if_any "00_eval_existing_exp02_exp04_first8000" \
  runs/eval_existing_exp02_exp04_first8000 \
  runs/rnn_seqmem_exp02_action_memory_full/checkpoint_epoch_010.pt \
  runs/rnn_seqmem_exp04_action_memory_onestep_full/checkpoint_epoch_010.pt \
  runs/rnn_seqmem_exp04_action_memory_onestep_full/checkpoint.pt

# -------------------------------------------------------------------
# Phase 1: finish/resume Exp07, the H10 version of Exp03.
# Relevance: tests Dreamer-style longer imagination horizon.
# -------------------------------------------------------------------
EXP07_RESUME=()
if [[ -f runs/rnn_seqmem_exp07_onestep_full_h10/checkpoint.pt ]]; then
  EXP07_RESUME=(--resume runs/rnn_seqmem_exp07_onestep_full_h10/checkpoint.pt)
fi
run_python_module "01_train_or_resume_exp07_onestep_full_h10" \
  smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
  --manifest splits/generated_seed4_mapdims_only.json \
  --split train \
  --out-dir runs/rnn_seqmem_exp07_onestep_full_h10 \
  --model-size default \
  --epochs 10 \
  --batch-size 8 \
  --num-workers 4 \
  --rollout-window 20 \
  --rollout-horizon 10 \
  --window-mode random \
  --samples-per-epoch 50000 \
  --enemy-visibility-mask \
  --enemy-sight-range 9.0 \
  --temporal-loss lambda \
  --td-lambda 0.9 \
  --sigreg-weight 0.01 \
  --decoder-weight 0.01 \
  --presence-weight 0.01 \
  --rollout-memory-dim 128 \
  --target-mode full \
  --one-step-weight 0.5 \
  "${EXP07_RESUME[@]}" \
  --device cuda \
  --amp \
  --wandb \
  --wandb-project SMAC-JEPA-losses \
  --wandb-name rnn-seqmem-exp07-onestep-full-h10

# -------------------------------------------------------------------
# Phase 2: Exp09, weaker one-step anchor.
# Relevance: tries to keep Exp03 decoded improvement while reducing h5 latent drift.
# -------------------------------------------------------------------
run_python_module "02_train_exp09_onestep025_full_h5" \
  smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
  --manifest splits/generated_seed4_mapdims_only.json \
  --split train \
  --out-dir runs/rnn_seqmem_exp09_onestep025_full_h5 \
  --model-size default \
  --epochs 10 \
  --batch-size 16 \
  --num-workers 4 \
  --rollout-window 20 \
  --rollout-horizon 5 \
  --window-mode random \
  --samples-per-epoch 50000 \
  --enemy-visibility-mask \
  --enemy-sight-range 9.0 \
  --temporal-loss lambda \
  --td-lambda 0.9 \
  --sigreg-weight 0.01 \
  --decoder-weight 0.01 \
  --presence-weight 0.01 \
  --rollout-memory-dim 128 \
  --target-mode full \
  --one-step-weight 0.25 \
  --device cuda \
  --amp \
  --wandb \
  --wandb-project SMAC-JEPA-losses \
  --wandb-name rnn-seqmem-exp09-onestep025-full-h5

# -------------------------------------------------------------------
# Phase 3: Exp10, uniform horizon weighting.
# Relevance: keeps one-step anchor but puts stronger pressure on h4/h5.
# -------------------------------------------------------------------
run_python_module "03_train_exp10_onestep_uniform_full_h5" \
  smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
  --manifest splits/generated_seed4_mapdims_only.json \
  --split train \
  --out-dir runs/rnn_seqmem_exp10_onestep_uniform_full_h5 \
  --model-size default \
  --epochs 10 \
  --batch-size 16 \
  --num-workers 4 \
  --rollout-window 20 \
  --rollout-horizon 5 \
  --window-mode random \
  --samples-per-epoch 50000 \
  --enemy-visibility-mask \
  --enemy-sight-range 9.0 \
  --temporal-loss uniform \
  --td-lambda 0.9 \
  --sigreg-weight 0.01 \
  --decoder-weight 0.01 \
  --presence-weight 0.01 \
  --rollout-memory-dim 128 \
  --target-mode full \
  --one-step-weight 0.5 \
  --device cuda \
  --amp \
  --wandb \
  --wandb-project SMAC-JEPA-losses \
  --wandb-name rnn-seqmem-exp10-onestep-uniform-full-h5

# -------------------------------------------------------------------
# Phase 4: Exp11, PER/sample-priority version of Exp03.
# Relevance: tests whether PER gets the useful Exp03 behavior faster/better.
# Priority settings are deliberately soft to avoid hard-sample overfitting.
# -------------------------------------------------------------------
run_python_module "04_train_exp11_per_onestep_full_h5" \
  smac_jepa.train_markov_rollout_rnn_visibility_seqmem_per \
  --manifest splits/generated_seed4_mapdims_only.json \
  --split train \
  --out-dir runs/rnn_seqmem_exp11_per_onestep_full_h5 \
  --model-size default \
  --epochs 10 \
  --batch-size 16 \
  --num-workers 4 \
  --rollout-window 20 \
  --rollout-horizon 5 \
  --window-mode random \
  --samples-per-epoch 50000 \
  --enemy-visibility-mask \
  --enemy-sight-range 9.0 \
  --temporal-loss lambda \
  --td-lambda 0.9 \
  --sigreg-weight 0.01 \
  --decoder-weight 0.01 \
  --presence-weight 0.01 \
  --rollout-memory-dim 128 \
  --target-mode full \
  --one-step-weight 0.5 \
  --sample-prioritized \
  --priority-alpha 0.4 \
  --priority-uniform-mix 0.7 \
  --priority-ema-beta 0.95 \
  --priority-warmup-epochs 2 \
  --priority-score pred_loss \
  --device cuda \
  --amp \
  --wandb \
  --wandb-project SMAC-JEPA-losses \
  --wandb-name rnn-seqmem-exp11-per-onestep-full-h5

# -------------------------------------------------------------------
# Phase 5: Exp08 curriculum, H3 -> H5 -> H10.
# Relevance: teaches stable local rollout first, then stretches imagination.
# This may fail if resume config forbids horizon changes; script will continue.
# -------------------------------------------------------------------
run_python_module "05a_train_exp08_curriculum_h3" \
  smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
  --manifest splits/generated_seed4_mapdims_only.json \
  --split train \
  --out-dir runs/rnn_seqmem_exp08_curriculum_h3 \
  --model-size default \
  --epochs 4 \
  --batch-size 16 \
  --num-workers 4 \
  --rollout-window 20 \
  --rollout-horizon 3 \
  --window-mode random \
  --samples-per-epoch 50000 \
  --enemy-visibility-mask \
  --enemy-sight-range 9.0 \
  --temporal-loss lambda \
  --td-lambda 0.9 \
  --sigreg-weight 0.01 \
  --decoder-weight 0.01 \
  --presence-weight 0.01 \
  --rollout-memory-dim 128 \
  --target-mode full \
  --one-step-weight 0.5 \
  --device cuda \
  --amp \
  --wandb \
  --wandb-project SMAC-JEPA-losses \
  --wandb-name rnn-seqmem-exp08-curriculum-h3

run_python_module "05b_train_exp08_curriculum_h5" \
  smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
  --manifest splits/generated_seed4_mapdims_only.json \
  --split train \
  --out-dir runs/rnn_seqmem_exp08_curriculum_h5 \
  --model-size default \
  --epochs 7 \
  --batch-size 16 \
  --num-workers 4 \
  --rollout-window 20 \
  --rollout-horizon 5 \
  --window-mode random \
  --samples-per-epoch 50000 \
  --enemy-visibility-mask \
  --enemy-sight-range 9.0 \
  --temporal-loss lambda \
  --td-lambda 0.9 \
  --sigreg-weight 0.01 \
  --decoder-weight 0.01 \
  --presence-weight 0.01 \
  --rollout-memory-dim 128 \
  --target-mode full \
  --one-step-weight 0.5 \
  --resume runs/rnn_seqmem_exp08_curriculum_h3/checkpoint.pt \
  --device cuda \
  --amp \
  --wandb \
  --wandb-project SMAC-JEPA-losses \
  --wandb-name rnn-seqmem-exp08-curriculum-h5

run_python_module "05c_train_exp08_curriculum_h10" \
  smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
  --manifest splits/generated_seed4_mapdims_only.json \
  --split train \
  --out-dir runs/rnn_seqmem_exp08_curriculum_h10 \
  --model-size default \
  --epochs 10 \
  --batch-size 8 \
  --num-workers 4 \
  --rollout-window 20 \
  --rollout-horizon 10 \
  --window-mode random \
  --samples-per-epoch 50000 \
  --enemy-visibility-mask \
  --enemy-sight-range 9.0 \
  --temporal-loss lambda \
  --td-lambda 0.9 \
  --sigreg-weight 0.01 \
  --decoder-weight 0.01 \
  --presence-weight 0.01 \
  --rollout-memory-dim 128 \
  --target-mode full \
  --one-step-weight 0.5 \
  --resume runs/rnn_seqmem_exp08_curriculum_h5/checkpoint.pt \
  --device cuda \
  --amp \
  --wandb \
  --wandb-project SMAC-JEPA-losses \
  --wandb-name rnn-seqmem-exp08-curriculum-h10

# -------------------------------------------------------------------
# Phase 6: reproducible combined eval for all available new candidates.
# Uses deterministic first 8000 eval windows: batch 16 * max-batches 500.
# -------------------------------------------------------------------
run_eval_if_any "06_eval_new_candidates_first8000" \
  runs/eval_new_candidates_first8000 \
  runs/rnn_seqmem_exp07_onestep_full_h10/checkpoint_epoch_010.pt \
  runs/rnn_seqmem_exp09_onestep025_full_h5/checkpoint_epoch_010.pt \
  runs/rnn_seqmem_exp10_onestep_uniform_full_h5/checkpoint_epoch_010.pt \
  runs/rnn_seqmem_exp11_per_onestep_full_h5/checkpoint_epoch_010.pt \
  runs/rnn_seqmem_exp08_curriculum_h10/checkpoint_epoch_010.pt \
  runs/rnn_seqmem_exp08_curriculum_h10/checkpoint.pt

echo "============================================================"
echo "Three-day JEPA runner completed attempts at $(date)"
echo "Logs: $LOG_DIR"
echo "Status summary:"
for f in "$LOG_DIR"/*.status; do
  [[ -e "$f" ]] || continue
  echo "  $(basename "$f" .status): $(cat "$f")"
done
echo "============================================================"
