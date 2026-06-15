#!/usr/bin/env bash
# Overnight plan:
#   1) Evaluate Exp01/Exp02/Exp03 checkpoint_epoch_010 on eval split with batch size 16.
#   2) Resume Exp04 to epoch 10.
#   3) Train Exp07: Exp03-style one-step model but rollout_horizon=10.
#
# Error behavior:
#   - Does NOT use set -e.
#   - If one command fails, it logs the error and continues.
#   - Each command writes a .log and .status file.

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR" || {
  echo "ERROR: Could not cd into ROOT_DIR=$ROOT_DIR"
  exit 1
}

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/overnight_eval_exp04_exp07_logs_${RUN_STAMP}"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Starting overnight eval + Exp04 resume + Exp07 H10"
echo "Root:       $ROOT_DIR"
echo "Log dir:    $LOG_DIR"
echo "Start time: $(date)"
echo "============================================================"
echo

run_command() {
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
    echo "Continuing to next command..."
  fi

  echo "END:   ${name}"
  echo "TIME:  $(date)"
  echo
  return 0
}

if [[ ! -f eval_rnn_seqmem_checkpoints.py ]]; then
  echo "WARNING: eval_rnn_seqmem_checkpoints.py not found in repo root."
  echo "Copy it first:"
  echo "  cp /mnt/data/seqmem_eval/eval_rnn_seqmem_checkpoints.py ."
fi

if [[ ! -f smac_jepa/train_markov_rollout_rnn_visibility_seqmem_experiments.py ]]; then
  echo "WARNING: smac_jepa/train_markov_rollout_rnn_visibility_seqmem_experiments.py not found."
  echo "Copy it first:"
  echo "  cp /mnt/data/rnn_weekend_experiments/train_markov_rollout_rnn_visibility_seqmem_experiments.py smac_jepa/"
fi

run_command "01_eval_exp01_exp02_exp03_epoch10_bs16" \
  env LD_LIBRARY_PATH="" python eval_rnn_seqmem_checkpoints.py \
    --manifest splits/generated_seed4_mapdims_only.json \
    --split eval \
    --checkpoint runs/rnn_seqmem_exp01_normal_full/checkpoint_epoch_010.pt \
    --checkpoint runs/rnn_seqmem_exp02_action_memory_full/checkpoint_epoch_010.pt \
    --checkpoint runs/rnn_seqmem_exp03_onestep_full/checkpoint_epoch_010.pt \
    --out-dir runs/rnn_seqmem_eval_exp01_exp02_exp03_epoch10_bs16 \
    --batch-size 16 \
    --num-workers 4 \
    --device cuda \
    --amp

run_command "02_resume_exp04_action_memory_onestep_full" \
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
    --resume runs/rnn_seqmem_exp04_action_memory_onestep_full/checkpoint.pt \
    --device cuda \
    --amp \
    --wandb \
    --wandb-project SMAC-JEPA-losses \
    --wandb-name rnn-seqmem-exp04-action-memory-onestep-full-resume

run_command "03_train_exp07_onestep_full_h10" \
  env LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
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
    --device cuda \
    --amp \
    --wandb \
    --wandb-project SMAC-JEPA-losses \
    --wandb-name rnn-seqmem-exp07-onestep-full-h10

echo "============================================================"
echo "All requested commands attempted."
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
