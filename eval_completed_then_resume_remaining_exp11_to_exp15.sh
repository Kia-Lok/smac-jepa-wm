#!/usr/bin/env bash
# Evaluate completed Exp11–15 checkpoints first, then resume unfinished runs,
# then run the final combined evaluation once all five checkpoints exist.

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR" || {
  echo "ERROR: cannot enter $ROOT_DIR"
  exit 1
}

PER_MODULE="smac_jepa.train_markov_rollout_rnn_visibility_seqmem_per"
R2_MODULE="smac_jepa.train_markov_rollout_rnn_visibility_seqmem_r2offline"
EVALUATOR="./eval_rnn_seqmem_dreamer_probe_r2aware.py"

# The previous all-in-one script should already have installed these.
LD_LIBRARY_PATH="" python -m "$PER_MODULE" --help >/dev/null || {
  echo "ERROR: missing $PER_MODULE"
  exit 1
}
LD_LIBRARY_PATH="" python -m "$R2_MODULE" --help >/dev/null || {
  echo "ERROR: missing $R2_MODULE"
  exit 1
}
LD_LIBRARY_PATH="" python "$EVALUATOR" --help >/dev/null || {
  echo "ERROR: missing $EVALUATOR"
  exit 1
}

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/eval_then_resume_exp11_to_exp15_logs_${STAMP}"
mkdir -p "$LOG_DIR"

EXP11_DIR="runs/rnn_seqmem_exp11_per_onestep_full_h5"
EXP12_DIR="runs/rnn_seqmem_exp12_action_memory_per_onestep_full_h5"
EXP13_DIR="runs/rnn_seqmem_exp13_action_memory_per_uniform_full_h5"
EXP15_DIR="runs/rnn_seqmem_exp15_r2offline_sigreg_full_h5"
EXP14_DIR="runs/rnn_seqmem_exp14_r2offline_full_h5"

EXP11_FINAL="${EXP11_DIR}/checkpoint_epoch_006.pt"
EXP12_FINAL="${EXP12_DIR}/checkpoint_epoch_006.pt"
EXP13_FINAL="${EXP13_DIR}/checkpoint_epoch_006.pt"
EXP15_FINAL="${EXP15_DIR}/checkpoint_epoch_010.pt"
EXP14_FINAL="${EXP14_DIR}/checkpoint_epoch_010.pt"

EVAL_OUT_DIR="${EVAL_OUT_DIR:-runs/eval_exp11_to_exp15_r2aware_h5}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-500}"
DIAGNOSTIC_MAX_BATCHES="${DIAGNOSTIC_MAX_BATCHES:-100}"
PROBE_EPOCHS="${PROBE_EPOCHS:-20}"
PROBE_MAX_BATCHES="${PROBE_MAX_BATCHES:-300}"
PROBE_SAMPLES_PER_EPOCH="${PROBE_SAMPLES_PER_EPOCH:-20000}"

mkdir -p "$EVAL_OUT_DIR"

target_checkpoint() {
  local run_dir="$1"
  local epoch="$2"
  printf "%s/checkpoint_epoch_%03d.pt" "$run_dir" "$epoch"
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

  return 0
}

evaluate_checkpoints() {
  local label="$1"
  shift
  local checkpoints=("$@")

  if [[ ${#checkpoints[@]} -eq 0 ]]; then
    echo "No completed checkpoints available for $label"
    return 0
  fi

  local args=()
  local checkpoint=""
  for checkpoint in "${checkpoints[@]}"; do
    args+=(--checkpoint "$checkpoint")
  done

  run_logged "$label" \
    env LD_LIBRARY_PATH="" python "$EVALUATOR" \
      --manifest splits/generated_seed4_mapdims_only.json \
      --split eval \
      "${args[@]}" \
      --out-dir "$EVAL_OUT_DIR" \
      --batch-size 16 \
      --num-workers 4 \
      --max-batches "$EVAL_MAX_BATCHES" \
      --diagnostics \
      --diagnostic-max-batches "$DIAGNOSTIC_MAX_BATCHES" \
      --eval-rollout-horizon 5 \
      --target-mode full \
      --window-mode sequential \
      --probe-decoder \
      --probe-train-split train \
      --probe-epochs "$PROBE_EPOCHS" \
      --probe-max-batches-per-epoch "$PROBE_MAX_BATCHES" \
      --probe-samples-per-epoch "$PROBE_SAMPLES_PER_EPOCH" \
      --probe-lr 0.001 \
      --probe-weight-decay 0.00001 \
      --probe-seed 123 \
      --device cuda \
      --amp
}

echo "############################################################"
echo "PHASE 1: EVALUATE CURRENTLY COMPLETED FINAL CHECKPOINTS"
echo "############################################################"

completed=()
for checkpoint in \
  "$EXP11_FINAL" \
  "$EXP12_FINAL" \
  "$EXP13_FINAL" \
  "$EXP15_FINAL" \
  "$EXP14_FINAL"
do
  if [[ -f "$checkpoint" ]]; then
    completed+=("$checkpoint")
    echo "Including: $checkpoint"
  else
    echo "Not final yet: $checkpoint"
  fi
done

# Partial Exp14 checkpoints are deliberately not treated as a final result.
evaluate_checkpoints "01_pre_resume_evaluation" "${completed[@]}"

echo
echo "############################################################"
echo "PHASE 2: RESUME ANY UNFINISHED EXPERIMENTS"
echo "############################################################"

run_per_experiment() {
  local label="$1"
  local out_dir="$2"
  local wandb_name="$3"
  local temporal_loss="$4"
  local action_conditioned="$5"

  if [[ -f "$(target_checkpoint "$out_dir" 6)" ]]; then
    echo "SKIP $label: epoch 6 final checkpoint exists"
    return 0
  fi

  local resume_args=()
  local checkpoint=""
  if checkpoint="$(latest_checkpoint "$out_dir")"; then
    resume_args=(--resume "$checkpoint")
    echo "$label resumes from $checkpoint"
  fi

  local action_args=()
  if [[ "$action_conditioned" == "true" ]]; then
    action_args=(--action-conditioned-memory)
  fi

  run_logged "$label" \
    env LD_LIBRARY_PATH="" python -m "$PER_MODULE" \
      --manifest splits/generated_seed4_mapdims_only.json \
      --split train \
      --out-dir "$out_dir" \
      --model-size default \
      --epochs 6 \
      --batch-size 16 \
      --num-workers 4 \
      --rollout-window 20 \
      --rollout-horizon 5 \
      --window-mode random \
      --samples-per-epoch 50000 \
      --enemy-visibility-mask \
      --enemy-sight-range 9.0 \
      --temporal-loss "$temporal_loss" \
      --td-lambda 0.9 \
      --sigreg-weight 0.01 \
      --decoder-weight 0.01 \
      --presence-weight 0.01 \
      --rollout-memory-dim 128 \
      --target-mode full \
      "${action_args[@]}" \
      --one-step-weight 0.5 \
      --sample-prioritized \
      --priority-alpha 0.4 \
      --priority-uniform-mix 0.7 \
      --priority-ema-beta 0.95 \
      --priority-warmup-epochs 2 \
      --priority-score pred_loss \
      "${resume_args[@]}" \
      --grad-clip 1.0 \
      --device cuda \
      --amp \
      --wandb \
      --wandb-project SMAC-JEPA-losses \
      --wandb-name "$wandb_name"
}

run_r2_experiment() {
  local label="$1"
  local out_dir="$2"
  local wandb_name="$3"
  local use_sigreg="$4"

  if [[ -f "$(target_checkpoint "$out_dir" 10)" ]]; then
    echo "SKIP $label: epoch 10 final checkpoint exists"
    return 0
  fi

  local resume_args=()
  local checkpoint=""
  if checkpoint="$(latest_checkpoint "$out_dir")"; then
    resume_args=(--resume "$checkpoint")
    echo "$label resumes from $checkpoint"
  fi

  local sigreg_args=(--sigreg-weight 0.0)
  if [[ "$use_sigreg" == "true" ]]; then
    sigreg_args=(
      --sigreg-weight 0.01
      --r2-sigreg-divide-by-dim
      --r2-sigreg-knots 17
      --r2-sigreg-num-proj 1024
      --r2-sigreg-proj-chunk 64
      --r2-sigreg-max-samples 0
    )
  fi

  run_logged "$label" \
    env LD_LIBRARY_PATH="" python -m "$R2_MODULE" \
      --manifest splits/generated_seed4_mapdims_only.json \
      --split train \
      --out-dir "$out_dir" \
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
      --r2-dyn-scale 1.0 \
      --r2-rep-scale 0.1 \
      --r2-barlow-scale 0.05 \
      --r2-barlow-lambda 0.0005 \
      --r2-latent-normalize \
      --lr-warmup-steps 1000 \
      "${sigreg_args[@]}" \
      --decoder-weight 0.01 \
      --presence-weight 0.01 \
      --rollout-memory-dim 128 \
      --target-mode full \
      --action-conditioned-memory \
      --one-step-weight 0.5 \
      "${resume_args[@]}" \
      --grad-clip 1.0 \
      --device cuda \
      --amp \
      --wandb \
      --wandb-project SMAC-JEPA-losses \
      --wandb-name "$wandb_name"
}

# Completed runs are skipped automatically. Normally only Exp14 resumes.
run_per_experiment \
  "02_resume_exp11" \
  "$EXP11_DIR" \
  "rnn-seqmem-exp11-per-onestep-full-h5-e6-resume" \
  "lambda" \
  "false"

run_per_experiment \
  "03_resume_exp12" \
  "$EXP12_DIR" \
  "rnn-seqmem-exp12-action-memory-per-onestep-full-h5-e6-resume" \
  "lambda" \
  "true"

run_per_experiment \
  "04_resume_exp13" \
  "$EXP13_DIR" \
  "rnn-seqmem-exp13-action-memory-per-uniform-full-h5-e6-resume" \
  "uniform" \
  "true"

run_r2_experiment \
  "05_resume_exp15" \
  "$EXP15_DIR" \
  "rnn-seqmem-exp15-r2offline-sigreg-full-h5-resume" \
  "true"

run_r2_experiment \
  "06_resume_exp14" \
  "$EXP14_DIR" \
  "rnn-seqmem-exp14-r2offline-full-h5-resume" \
  "false"

echo
echo "############################################################"
echo "PHASE 3: FINAL COMBINED EVALUATION"
echo "############################################################"

missing=()
for checkpoint in \
  "$EXP11_FINAL" \
  "$EXP12_FINAL" \
  "$EXP13_FINAL" \
  "$EXP15_FINAL" \
  "$EXP14_FINAL"
do
  if [[ ! -f "$checkpoint" ]]; then
    missing+=("$checkpoint")
  fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Final evaluation cannot start. Missing checkpoints:"
  printf "  %s\n" "${missing[@]}"
  echo
  echo "Inspect logs in: $LOG_DIR"
  echo "Rerun this same script after fixing the failed run."
  exit 1
fi

all_final=(
  "$EXP11_FINAL"
  "$EXP12_FINAL"
  "$EXP13_FINAL"
  "$EXP15_FINAL"
  "$EXP14_FINAL"
)

# Same output directory: existing probe decoders are reused when compatible.
evaluate_checkpoints "07_final_combined_evaluation" "${all_final[@]}"

echo
echo "============================================================"
echo "PIPELINE FINISHED: $(date)"
echo "Logs:"
echo "  $LOG_DIR"
echo "Final evaluation:"
echo "  $EVAL_OUT_DIR"
echo "Final summary:"
echo "  $EVAL_OUT_DIR/eval_seqmem_combined_summary.csv"
echo "============================================================"
