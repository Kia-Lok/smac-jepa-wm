#!/usr/bin/env bash
# Confirmed weekend RNN seqmem experiment matrix.
#
# Runs all 6 commands exactly as confirmed:
#   1. Normal seqmem, full target
#   2. Action-conditioned memory, full target
#   3. One-step auxiliary loss, full target
#   4. Action-conditioned memory + one-step loss, full target
#   5. Observed target
#   6. Action-conditioned memory + one-step loss + observed target
#
# Error behavior:
#   - Does NOT use `set -e`
#   - If one experiment crashes, logs the error and continues to the next
#   - Writes per-run logs and exit-code status files

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR" || {
  echo "ERROR: could not cd into ROOT_DIR=$ROOT_DIR"
  exit 1
}

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/weekend_rnn_seqmem_experiment_logs_${RUN_STAMP}"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting confirmed weekend RNN seqmem experiments"
echo "Root:       $ROOT_DIR"
echo "Log dir:    $LOG_DIR"
echo "Start time: $(date)"
echo "============================================================"
echo

run_experiment() {
  local name="$1"
  shift

  local log_file="${LOG_DIR}/${name}.log"
  local status_file="${LOG_DIR}/${name}.status"

  echo "============================================================"
  echo "START: ${name}"
  echo "TIME:  $(date)"
  echo "LOG:   ${log_file}"
  echo "============================================================"

  "$@" 2>&1 | tee "$log_file"
  local code=${PIPESTATUS[0]}

  echo "$code" > "$status_file"

  if [[ "$code" -eq 0 ]]; then
    echo "SUCCESS: ${name}"
  else
    echo "FAILED: ${name} with exit code ${code}"
    echo "Continuing to next experiment..."
  fi

  echo "END:   ${name}"
  echo "TIME:  $(date)"
  echo
  return 0
}

# -------------------------------------------------------------------
# 1. Normal seqmem, full target
# -------------------------------------------------------------------
run_experiment "01_normal_seqmem_full" \
  env LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
    --manifest splits/generated_seed4_mapdims_only.json \
    --split train \
    --out-dir runs/rnn_seqmem_exp01_normal_full \
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
    --one-step-weight 0.0 \
    --device cuda \
    --amp \
    --wandb \
    --wandb-project SMAC-JEPA-losses \
    --wandb-name rnn-seqmem-exp01-normal-full

# -------------------------------------------------------------------
# 2. Idea 1: action-conditioned memory, full target
# -------------------------------------------------------------------
run_experiment "02_action_memory_full" \
  env LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
    --manifest splits/generated_seed4_mapdims_only.json \
    --split train \
    --out-dir runs/rnn_seqmem_exp02_action_memory_full \
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
    --one-step-weight 0.0 \
    --device cuda \
    --amp \
    --wandb \
    --wandb-project SMAC-JEPA-losses \
    --wandb-name rnn-seqmem-exp02-action-memory-full

# -------------------------------------------------------------------
# 3. Idea 2: one-step auxiliary loss, full target
# -------------------------------------------------------------------
run_experiment "03_onestep_full" \
  env LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
    --manifest splits/generated_seed4_mapdims_only.json \
    --split train \
    --out-dir runs/rnn_seqmem_exp03_onestep_full \
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
    --device cuda \
    --amp \
    --wandb \
    --wandb-project SMAC-JEPA-losses \
    --wandb-name rnn-seqmem-exp03-onestep-full

# -------------------------------------------------------------------
# 4. Idea 1 + 2: action-conditioned memory + one-step loss, full target
# -------------------------------------------------------------------
run_experiment "04_action_memory_onestep_full" \
  env LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
    --manifest splits/generated_seed4_mapdims_only.json \
    --split train \
    --out-dir runs/rnn_seqmem_exp04_action_memory_onestep_full \
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
    --device cuda \
    --amp \
    --wandb \
    --wandb-project SMAC-JEPA-losses \
    --wandb-name rnn-seqmem-exp04-action-memory-onestep-full

# -------------------------------------------------------------------
# 5. Idea 4: observed target
# -------------------------------------------------------------------
run_experiment "05_observed_target" \
  env LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
    --manifest splits/generated_seed4_mapdims_only.json \
    --split train \
    --out-dir runs/rnn_seqmem_exp05_observed_target \
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
    --target-mode observed \
    --one-step-weight 0.0 \
    --device cuda \
    --amp \
    --wandb \
    --wandb-project SMAC-JEPA-losses \
    --wandb-name rnn-seqmem-exp05-observed-target

# -------------------------------------------------------------------
# 6. Idea 1 + 2 + 4: action-conditioned memory + one-step loss + observed target
# -------------------------------------------------------------------
run_experiment "06_action_memory_onestep_observed" \
  env LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
    --manifest splits/generated_seed4_mapdims_only.json \
    --split train \
    --out-dir runs/rnn_seqmem_exp06_action_memory_onestep_observed \
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
    --target-mode observed \
    --action-conditioned-memory \
    --one-step-weight 0.5 \
    --device cuda \
    --amp \
    --wandb \
    --wandb-project SMAC-JEPA-losses \
    --wandb-name rnn-seqmem-exp06-action-memory-onestep-observed

echo "============================================================"
echo "All confirmed weekend RNN seqmem experiments attempted."
echo "Finish time: $(date)"
echo
echo "Status files:"
ls -lh "${LOG_DIR}"/*.status 2>/dev/null || true
echo
echo "Summary:"
for status in "${LOG_DIR}"/*.status; do
  [[ -e "$status" ]] || continue
  name="$(basename "$status" .status)"
  code="$(cat "$status")"
  echo "  ${name}: exit_code=${code}"
done
echo
echo "Logs are in: ${LOG_DIR}"
echo "============================================================"
