#!/usr/bin/env bash
# Continue the remaining non-Barlow JEPA experiments.
#
# Order:
#   1. Exp10: uniform temporal weighting
#   2. Exp11: PER
#   3. Exp12: action-conditioned memory + PER
#   4. Exp08 curriculum: H3 -> H5 -> H10
#
# Behaviour:
#   - skips a stage when its target checkpoint already exists
#   - resumes from checkpoint.pt, or the latest checkpoint_epoch_*.pt
#   - logs every experiment separately
#   - continues to the next independent experiment after a failure
#   - only advances Exp08 when the preceding curriculum stage produced a checkpoint

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR" || {
  echo "ERROR: could not enter $ROOT_DIR"
  exit 1
}

STANDARD_MODULE="smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments"
PER_MODULE="smac_jepa.train_markov_rollout_rnn_visibility_seqmem_per"

# Fail early when the required trainers are not installed.
LD_LIBRARY_PATH="" python -m "$STANDARD_MODULE" --help >/dev/null || {
  echo "ERROR: $STANDARD_MODULE is unavailable"
  exit 1
}
LD_LIBRARY_PATH="" python -m "$PER_MODULE" --help >/dev/null || {
  echo "ERROR: $PER_MODULE is unavailable"
  exit 1
}

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/unfinished_exp10_exp11_exp12_exp08_logs_${RUN_STAMP}"
mkdir -p "$LOG_DIR"

target_checkpoint() {
  local run_dir="$1"
  local epoch="$2"
  printf "%s/checkpoint_epoch_%03d.pt" "$run_dir" "$epoch"
}

is_complete() {
  local run_dir="$1"
  local epoch="$2"
  [[ -f "$(target_checkpoint "$run_dir" "$epoch")" ]]
}

latest_checkpoint() {
  local run_dir="$1"

  if [[ -f "${run_dir}/checkpoint.pt" ]]; then
    printf "%s\n" "${run_dir}/checkpoint.pt"
    return 0
  fi

  local latest=""
  latest="$(
    find "$run_dir" -maxdepth 1 -type f -name 'checkpoint_epoch_*.pt' \
      2>/dev/null | sort | tail -n 1
  )"

  if [[ -n "$latest" ]]; then
    printf "%s\n" "$latest"
    return 0
  fi

  return 1
}

run_logged() {
  local name="$1"
  shift

  local log_file="${LOG_DIR}/${name}.log"
  local status_file="${LOG_DIR}/${name}.status"

  echo
  echo "============================================================"
  echo "START: $name"
  echo "TIME:  $(date)"
  echo "LOG:   $log_file"
  echo "============================================================"

  "$@" 2>&1 | tee "$log_file"
  local code=${PIPESTATUS[0]}
  printf "%s\n" "$code" > "$status_file"

  echo "============================================================"
  echo "END:   $name"
  echo "CODE:  $code"
  echo "TIME:  $(date)"
  echo "============================================================"

  # Do not terminate the whole queue when one independent run fails.
  return 0
}

echo "Root: $ROOT_DIR"
echo "Logs: $LOG_DIR"
echo "Started: $(date)"

# ============================================================================
# Exp10: one-step=0.5, uniform temporal weighting, H5
# ============================================================================
EXP10_DIR="runs/rnn_seqmem_exp10_onestep_uniform_full_h5"

if is_complete "$EXP10_DIR" 10; then
  echo "SKIP Exp10: $(target_checkpoint "$EXP10_DIR" 10) exists"
else
  EXP10_RESUME=()
  if ckpt="$(latest_checkpoint "$EXP10_DIR")"; then
    EXP10_RESUME=(--resume "$ckpt")
    echo "Exp10 will resume from $ckpt"
  fi

  run_logged "01_exp10_uniform_h5" \
    env LD_LIBRARY_PATH="" python -m "$STANDARD_MODULE" \
      --manifest splits/generated_seed4_mapdims_only.json \
      --split train \
      --out-dir "$EXP10_DIR" \
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
      "${EXP10_RESUME[@]}" \
      --device cuda \
      --amp \
      --wandb \
      --wandb-project SMAC-JEPA-losses \
      --wandb-name rnn-seqmem-exp10-onestep-uniform-full-h5
fi

# ============================================================================
# Exp11: PER + one-step=0.5, H5
# ============================================================================
EXP11_DIR="runs/rnn_seqmem_exp11_per_onestep_full_h5"

if is_complete "$EXP11_DIR" 10; then
  echo "SKIP Exp11: $(target_checkpoint "$EXP11_DIR" 10) exists"
else
  EXP11_RESUME=()
  if ckpt="$(latest_checkpoint "$EXP11_DIR")"; then
    EXP11_RESUME=(--resume "$ckpt")
    echo "Exp11 will resume from $ckpt"
  fi

  run_logged "02_exp11_per_h5" \
    env LD_LIBRARY_PATH="" python -m "$PER_MODULE" \
      --manifest splits/generated_seed4_mapdims_only.json \
      --split train \
      --out-dir "$EXP11_DIR" \
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
      "${EXP11_RESUME[@]}" \
      --device cuda \
      --amp \
      --wandb \
      --wandb-project SMAC-JEPA-losses \
      --wandb-name rnn-seqmem-exp11-per-onestep-full-h5
fi

# ============================================================================
# Exp12: action-conditioned memory + PER + one-step=0.5, H5
# ============================================================================
EXP12_DIR="runs/rnn_seqmem_exp12_action_memory_per_onestep_full_h5"

if is_complete "$EXP12_DIR" 10; then
  echo "SKIP Exp12: $(target_checkpoint "$EXP12_DIR" 10) exists"
else
  EXP12_RESUME=()
  if ckpt="$(latest_checkpoint "$EXP12_DIR")"; then
    EXP12_RESUME=(--resume "$ckpt")
    echo "Exp12 will resume from $ckpt"
  fi

  run_logged "03_exp12_action_memory_per_h5" \
    env LD_LIBRARY_PATH="" python -m "$PER_MODULE" \
      --manifest splits/generated_seed4_mapdims_only.json \
      --split train \
      --out-dir "$EXP12_DIR" \
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
      --action-conditioned-memory \
      --one-step-weight 0.5 \
      --sample-prioritized \
      --priority-alpha 0.4 \
      --priority-uniform-mix 0.7 \
      --priority-ema-beta 0.95 \
      --priority-warmup-epochs 2 \
      --priority-score pred_loss \
      "${EXP12_RESUME[@]}" \
      --device cuda \
      --amp \
      --wandb \
      --wandb-project SMAC-JEPA-losses \
      --wandb-name rnn-seqmem-exp12-action-memory-per-onestep-full-h5
fi

# ============================================================================
# Exp08 curriculum stage 1: H3 through epoch 4
# ============================================================================
EXP08_H3_DIR="runs/rnn_seqmem_exp08_curriculum_h3"
EXP08_H5_DIR="runs/rnn_seqmem_exp08_curriculum_h5"
EXP08_H10_DIR="runs/rnn_seqmem_exp08_curriculum_h10"

if is_complete "$EXP08_H3_DIR" 4; then
  echo "SKIP Exp08 H3: $(target_checkpoint "$EXP08_H3_DIR" 4) exists"
else
  H3_RESUME=()
  if ckpt="$(latest_checkpoint "$EXP08_H3_DIR")"; then
    H3_RESUME=(--resume "$ckpt")
    echo "Exp08 H3 will resume from $ckpt"
  fi

  run_logged "04_exp08_curriculum_h3" \
    env LD_LIBRARY_PATH="" python -m "$STANDARD_MODULE" \
      --manifest splits/generated_seed4_mapdims_only.json \
      --split train \
      --out-dir "$EXP08_H3_DIR" \
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
      "${H3_RESUME[@]}" \
      --device cuda \
      --amp \
      --wandb \
      --wandb-project SMAC-JEPA-losses \
      --wandb-name rnn-seqmem-exp08-curriculum-h3
fi

# ============================================================================
# Exp08 curriculum stage 2: resume H3, train H5 through epoch 7
# ============================================================================
if is_complete "$EXP08_H5_DIR" 7; then
  echo "SKIP Exp08 H5: $(target_checkpoint "$EXP08_H5_DIR" 7) exists"
else
  H5_SOURCE=""
  if H5_SOURCE="$(latest_checkpoint "$EXP08_H5_DIR")"; then
    :
  elif is_complete "$EXP08_H3_DIR" 4; then
    H5_SOURCE="$(target_checkpoint "$EXP08_H3_DIR" 4)"
  elif H5_SOURCE="$(latest_checkpoint "$EXP08_H3_DIR")"; then
    :
  else
    H5_SOURCE=""
  fi

  if [[ -z "$H5_SOURCE" ]]; then
    echo "SKIP Exp08 H5: no H3 checkpoint is available"
  else
    echo "Exp08 H5 will resume from $H5_SOURCE"
    run_logged "05_exp08_curriculum_h5" \
      env LD_LIBRARY_PATH="" python -m "$STANDARD_MODULE" \
        --manifest splits/generated_seed4_mapdims_only.json \
        --split train \
        --out-dir "$EXP08_H5_DIR" \
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
        --resume "$H5_SOURCE" \
        --device cuda \
        --amp \
        --wandb \
        --wandb-project SMAC-JEPA-losses \
        --wandb-name rnn-seqmem-exp08-curriculum-h5
  fi
fi

# ============================================================================
# Exp08 curriculum stage 3: resume H5, train H10 through epoch 10
# ============================================================================
if is_complete "$EXP08_H10_DIR" 10; then
  echo "SKIP Exp08 H10: $(target_checkpoint "$EXP08_H10_DIR" 10) exists"
else
  H10_SOURCE=""
  if H10_SOURCE="$(latest_checkpoint "$EXP08_H10_DIR")"; then
    :
  elif is_complete "$EXP08_H5_DIR" 7; then
    H10_SOURCE="$(target_checkpoint "$EXP08_H5_DIR" 7)"
  elif H10_SOURCE="$(latest_checkpoint "$EXP08_H5_DIR")"; then
    :
  else
    H10_SOURCE=""
  fi

  if [[ -z "$H10_SOURCE" ]]; then
    echo "SKIP Exp08 H10: no H5 checkpoint is available"
  else
    echo "Exp08 H10 will resume from $H10_SOURCE"
    run_logged "06_exp08_curriculum_h10" \
      env LD_LIBRARY_PATH="" python -m "$STANDARD_MODULE" \
        --manifest splits/generated_seed4_mapdims_only.json \
        --split train \
        --out-dir "$EXP08_H10_DIR" \
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
        --resume "$H10_SOURCE" \
        --device cuda \
        --amp \
        --wandb \
        --wandb-project SMAC-JEPA-losses \
        --wandb-name rnn-seqmem-exp08-curriculum-h10
  fi
fi

echo
echo "============================================================"
echo "QUEUE FINISHED: $(date)"
echo "Logs: $LOG_DIR"
echo "Status summary:"
for status_file in "$LOG_DIR"/*.status; do
  [[ -e "$status_file" ]] || continue
  printf "  %-45s %s\n" \
    "$(basename "$status_file" .status)" \
    "$(cat "$status_file")"
done
echo "============================================================"
