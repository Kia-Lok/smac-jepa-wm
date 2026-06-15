#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR" || {
  echo "ERROR: Could not cd into ROOT_DIR=$ROOT_DIR"
  exit 1
}

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/overnight_sigreg_warmup_logs_${RUN_STAMP}"
mkdir -p "$LOG_DIR"

echo "Starting overnight SIGReg-warmup experiments"
echo "Root: $ROOT_DIR"
echo "Logs: $LOG_DIR"
echo "Start time: $(date)"
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
  local exit_code=${PIPESTATUS[0]}

  echo "$exit_code" > "$status_file"
  if [[ "$exit_code" -eq 0 ]]; then
    echo "SUCCESS: ${name}"
  else
    echo "FAILED: ${name} with exit code ${exit_code}"
    echo "Continuing to next experiment..."
  fi
  echo
  return 0
}

run_experiment "markov_rollout_sample_prio_sigreg_warmup" \
  env LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_sample_prio \
    --manifest splits/generated_seed4_mapdims_only.json \
    --split train \
    --out-dir runs/generated_seed4_markov_rollout_sample_prio_p20_n5_sigreg_warmup \
    --model-size default \
    --epochs 10 \
    --batch-size 16 \
    --num-workers 4 \
    --rollout-window 20 \
    --rollout-horizon 5 \
    --window-mode random \
    --samples-per-epoch 50000 \
    --temporal-loss lambda \
    --td-lambda 0.9 \
    --sigreg-weight 0.09 \
    --sigreg-weight-start 0.02 \
    --sigreg-weight-end 0.09 \
    --sigreg-warmup-epochs 8 \
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
    --wandb-name markov-rollout-sample-prio-p20-n5-sigreg-warmup

run_experiment "markov_rollout_rnn_enemyvis_sigreg_warmup" \
  env LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_rnn_visibility_mask \
    --manifest splits/generated_seed4_mapdims_only.json \
    --split train \
    --out-dir runs/generated_seed4_markov_rollout_rnn_enemyvis_p20_n5_sigreg_warmup \
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
    --sigreg-weight 0.09 \
    --sigreg-weight-start 0.02 \
    --sigreg-weight-end 0.09 \
    --sigreg-warmup-epochs 8 \
    --rollout-memory-dim 128 \
    --device cuda \
    --amp \
    --wandb \
    --wandb-project SMAC-JEPA-losses \
    --wandb-name markov-rollout-rnn-enemyvis-p20-n5-sigreg-warmup

echo "============================================================"
echo "All requested SIGReg-warmup experiments attempted."
echo "Finish time: $(date)"
echo
echo "Exit status files:"
ls -lh "${LOG_DIR}"/*.status 2>/dev/null || true
echo
echo "Summary:"
for status in "${LOG_DIR}"/*.status; do
  [[ -e "$status" ]] || continue
  name="$(basename "$status" .status)"
  code="$(cat "$status")"
  echo "  ${name}: exit_code=${code}"
done
echo "============================================================"
