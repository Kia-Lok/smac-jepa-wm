#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR" || exit 1

# fresh: start Exp17, Exp18, and Exp19 from epoch 1.
# resume: intentionally resume checkpoints created by this corrected suite.
RUN_MODE="${RUN_MODE:-fresh}"
EPOCHS="${EPOCHS:-5}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"

EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-500}"
DIAGNOSTIC_MAX_BATCHES="${DIAGNOSTIC_MAX_BATCHES:-100}"
PROBE_EPOCHS="${PROBE_EPOCHS:-20}"
PROBE_MAX_BATCHES="${PROBE_MAX_BATCHES:-300}"
PROBE_SAMPLES_PER_EPOCH="${PROBE_SAMPLES_PER_EPOCH:-20000}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/exp17_exp19_final_fixed_logs_${STAMP}"
EVAL_ROOT="runs/eval_exp17_exp19_final_fixed"
PROBE_DIR="${EVAL_ROOT}/probe_decoders"

DATASET_FILE="smac_jepa/data/markov_rollout_visibility_dataset.py"
TRAINER_FILE="smac_jepa/train_jepa_exp17_exp19_fixed.py"
EVALUATOR_FILE="eval_jepa_exp17_exp19_fixed.py"
RUNNER_FILE="run_exp17_exp19_fresh_fixed.sh"

TRAIN_MODULE="smac_jepa.train_jepa_exp17_exp19_fixed"
EVALUATOR="./${EVALUATOR_FILE}"

# New v3 folders guarantee that the old accidental Exp17 checkpoint is not
# reused. Fresh mode also backs up an existing v3 folder before restarting.
EXP16_DIR="runs/rnn_seqmem_exp16_blocker_fixed_full_h5"
EXP17_DIR="runs/rnn_seqmem_exp17_ema_explicit_visibility_v3"
EXP18_DIR="runs/rnn_seqmem_exp18_ema_memory_barlow_explicit_visibility_v3"
EXP19_DIR="runs/rnn_seqmem_exp19_ema_memory_barlow_events_explicit_visibility_v3"

mkdir -p "$LOG_DIR" "$EVAL_ROOT" "$PROBE_DIR"

required_files=(
  "$DATASET_FILE"
  "$TRAINER_FILE"
  "$EVALUATOR_FILE"
  "$RUNNER_FILE"
)

for required_file in "${required_files[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    echo "ERROR: missing required file: $ROOT_DIR/$required_file"
    exit 2
  fi
done

# Catch syntax, imports, and parser-construction errors before starting GPUs.
python -m py_compile \
  "$DATASET_FILE" \
  "$TRAINER_FILE" \
  "$EVALUATOR_FILE" || exit 1

python -m "$TRAIN_MODULE" --help >/dev/null || {
  echo "ERROR: trainer import/argument parser preflight failed."
  exit 1
}

python "$EVALUATOR" --help >/dev/null || {
  echo "ERROR: evaluator import/argument parser preflight failed."
  exit 1
}

backup_directory() {
  local directory="$1"
  if [[ -d "$directory" ]]; then
    local backup="${directory}_backup_${STAMP}"
    echo "Backing up existing run:"
    echo "  $directory"
    echo "  -> $backup"
    mv "$directory" "$backup"
  fi
}

if [[ "$RUN_MODE" == "fresh" ]]; then
  backup_directory "$EXP17_DIR"
  backup_directory "$EXP18_DIR"
  backup_directory "$EXP19_DIR"
elif [[ "$RUN_MODE" != "resume" ]]; then
  echo "ERROR: RUN_MODE must be fresh or resume."
  exit 2
fi

run_logged() {
  local label="$1"
  shift

  local log_file="$LOG_DIR/${label}.log"

  echo
  echo "============================================================"
  echo "START: $label"
  echo "TIME:  $(date)"
  echo "LOG:   $log_file"
  echo "COMMAND:"
  printf ' %q' "$@"
  echo
  echo "============================================================"

  "$@" 2>&1 | tee "$log_file"
  return "${PIPESTATUS[0]}"
}

latest_checkpoint() {
  local run_dir="$1"

  python - "$run_dir" <<'PY'
from pathlib import Path
import sys
import torch

run_dir = Path(sys.argv[1])
candidates = []

for filename in ("checkpoint_recovery.pt", "checkpoint.pt"):
    path = run_dir / filename
    if path.is_file():
        candidates.append(path)

candidates.extend(run_dir.glob("checkpoint_epoch_*.pt"))

valid = []
for path in candidates:
    try:
        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")

        valid.append(
            (
                int(checkpoint.get("global_step", -1)),
                int(checkpoint.get("epoch", -1)),
                path.stat().st_mtime_ns,
                path,
            )
        )
    except Exception:
        pass

if valid:
    print(max(valid, key=lambda item: item[:3])[3])
PY
}

checkpoint_completed_epochs() {
  local checkpoint_path="$1"

  python - "$checkpoint_path" <<'PY'
import sys
import torch

path = sys.argv[1]
try:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")

epoch = int(checkpoint.get("epoch", 0))
epoch_complete = bool(checkpoint.get("epoch_complete", True))
print(epoch if epoch_complete else 0)
PY
}

COMMON_TRAIN_ARGS=(
  --manifest splits/generated_seed4_mapdims_only.json
  --split train
  --model-size default
  --epochs "$EPOCHS"
  --batch-size 16
  --num-workers 4
  --rollout-window 20
  --rollout-horizon 5
  --window-mode random
  --samples-per-epoch 50000
  --enemy-visibility-mask
  --enemy-sight-range 9.0
  --temporal-loss lambda
  --td-lambda 0.9
  --r2-dyn-scale 1.0
  --r2-rep-scale 0.1
  --r2-barlow-scale 0.05
  --r2-barlow-lambda 0.0005
  --r2-latent-normalize
  --sigreg-weight 0.01
  --r2-sigreg-divide-by-dim
  --r2-sigreg-knots 17
  --r2-sigreg-num-proj 1024
  --r2-sigreg-proj-chunk 64
  --r2-sigreg-max-samples 0
  --decoder-weight 0.01
  --presence-weight 0.01
  --rollout-memory-dim 128
  --target-mode full
  --action-conditioned-memory
  --one-step-weight 0.5
  --lr-warmup-steps 2000
  --aux-loss-warmup-steps 2000
  --grad-clip 1.0
  --amp-dtype auto
  --amp-init-scale 1024
  --amp-growth-interval 10000
  --amp-fallback-after-nonfinite 3
  --nonfinite-lr-backoff 0.5
  --min-learning-rate 0.000001
  --adam-eps 0.000001
  --checkpoint-every-steps 250
  --max-consecutive-nonfinite-grad-steps 0
  --max-total-nonfinite-grad-steps 0
  --device cuda
  --amp
)

echo
echo "============================================================"
echo "EXP17–EXP19 FINAL CORRECTED SUITE"
echo "RUN_MODE: $RUN_MODE"
echo "Exp17 directory: $EXP17_DIR"
echo "Exp18 directory: $EXP18_DIR"
echo "Exp19 directory: $EXP19_DIR"
echo "============================================================"

# This is a real parser/data/forward/backward/checkpoint/evaluator check.
# --no-wandb is now a valid trainer flag.
if [[ "$RUN_SMOKE_TEST" == "1" ]]; then
  SMOKE_TRAIN_DIR="runs/_smoke_exp17_final_${STAMP}"
  SMOKE_EVAL_DIR="runs/_smoke_eval17_final_${STAMP}"

  if ! run_logged \
    smoke_train_exp17 \
    env LD_LIBRARY_PATH="" \
    python -m "$TRAIN_MODULE" \
      "${COMMON_TRAIN_ARGS[@]}" \
      --epochs 1 \
      --batch-size 8 \
      --num-workers 0 \
      --samples-per-epoch 32 \
      --no-wandb \
      --ema-target-encoder \
      --ema-momentum 0.996 \
      --memory-barlow-scale 0.0 \
      --no-event-balanced-sampling \
      --out-dir "$SMOKE_TRAIN_DIR"
  then
    echo "ERROR: smoke training failed. Full experiments were not started."
    exit 1
  fi

  if ! run_logged \
    smoke_eval_exp17 \
    env LD_LIBRARY_PATH="" \
    python "$EVALUATOR" \
      --manifest splits/generated_seed4_mapdims_only.json \
      --split eval \
      --checkpoint "$SMOKE_TRAIN_DIR/checkpoint.pt" \
      --out-dir "$SMOKE_EVAL_DIR" \
      --batch-size 8 \
      --num-workers 0 \
      --max-batches 50 \
      --diagnostics \
      --diagnostic-max-batches 10 \
      --eval-rollout-horizon 5 \
      --target-mode full \
      --window-mode sequential \
      --thresholds 0.01 0.05 0.1 \
      --no-probe-decoder \
      --no-require-hidden-eval \
      --device cuda \
      --amp
  then
    echo "ERROR: smoke evaluation failed. Full experiments were not started."
    exit 1
  fi

  echo "Smoke train and evaluation passed."
fi

declare -a SUCCESSFUL_CHECKPOINTS=()
declare -a FAILED_STAGES=()

train_one() {
  local label="$1"
  local output_directory="$2"
  shift 2

  local experiment_args=("$@")
  local resume_args=()

  if [[ "$RUN_MODE" == "resume" ]]; then
    local checkpoint=""
    checkpoint="$(latest_checkpoint "$output_directory" || true)"
    if [[ -n "$checkpoint" ]]; then
      echo "$label explicitly resuming from: $checkpoint"
      resume_args=(--resume "$checkpoint")
    fi
  fi

  if ! run_logged \
    "train_${label}" \
    env LD_LIBRARY_PATH="" \
    python -m "$TRAIN_MODULE" \
      "${COMMON_TRAIN_ARGS[@]}" \
      "${experiment_args[@]}" \
      "${resume_args[@]}" \
      --out-dir "$output_directory" \
      --wandb \
      --wandb-project SMAC-JEPA-losses \
      --wandb-name "$label"
  then
    return 1
  fi

  local final_checkpoint="$output_directory/checkpoint.pt"
  if [[ ! -f "$final_checkpoint" ]]; then
    echo "ERROR: completed checkpoint missing: $final_checkpoint"
    return 1
  fi

  local completed_epochs
  completed_epochs="$(checkpoint_completed_epochs "$final_checkpoint")"
  if [[ "$completed_epochs" -lt "$EPOCHS" ]]; then
    echo "ERROR: checkpoint completed only $completed_epochs/$EPOCHS epochs."
    return 1
  fi

  return 0
}

evaluate_one() {
  local label="$1"
  local checkpoint="$2"
  local output_directory="$EVAL_ROOT/$label"

  rm -rf "$output_directory"
  mkdir -p "$output_directory"

  run_logged \
    "eval_${label}" \
    env LD_LIBRARY_PATH="" \
    python "$EVALUATOR" \
      --manifest splits/generated_seed4_mapdims_only.json \
      --split eval \
      --checkpoint "$checkpoint" \
      --out-dir "$output_directory" \
      --probe-dir "$PROBE_DIR" \
      --batch-size 16 \
      --num-workers 4 \
      --max-batches "$EVAL_MAX_BATCHES" \
      --diagnostics \
      --diagnostic-max-batches "$DIAGNOSTIC_MAX_BATCHES" \
      --eval-rollout-horizon 5 \
      --target-mode full \
      --window-mode sequential \
      --thresholds 0.01 0.05 0.1 \
      --probe-decoder \
      --probe-train-split train \
      --probe-epochs "$PROBE_EPOCHS" \
      --probe-max-batches-per-epoch "$PROBE_MAX_BATCHES" \
      --probe-samples-per-epoch "$PROBE_SAMPLES_PER_EPOCH" \
      --probe-lr 0.001 \
      --probe-weight-decay 0.00001 \
      --probe-seed 123 \
      --require-hidden-eval \
      --device cuda \
      --amp
}

run_train_then_evaluate() {
  local experiment_name="$1"
  local wandb_name="$2"
  local output_directory="$3"
  shift 3

  if train_one \
    "$wandb_name" \
    "$output_directory" \
    "$@"
  then
    local checkpoint="$output_directory/checkpoint.pt"

    if evaluate_one "$experiment_name" "$checkpoint"; then
      SUCCESSFUL_CHECKPOINTS+=("$checkpoint")
    else
      FAILED_STAGES+=("${experiment_name}:evaluation")
    fi
  else
    FAILED_STAGES+=("${experiment_name}:training")
    echo "$experiment_name failed; continuing to the next experiment."
  fi
}

# Exp17 begins from epoch 1 in the default fresh mode.
run_train_then_evaluate \
  exp17 \
  rnn-seqmem-exp17-ema-explicit-visibility-v3 \
  "$EXP17_DIR" \
  --ema-target-encoder \
  --ema-momentum 0.996 \
  --memory-barlow-scale 0.0 \
  --no-event-balanced-sampling

run_train_then_evaluate \
  exp18 \
  rnn-seqmem-exp18-ema-memory-barlow-explicit-visibility-v3 \
  "$EXP18_DIR" \
  --ema-target-encoder \
  --ema-momentum 0.996 \
  --memory-barlow-scale 0.01 \
  --no-event-balanced-sampling

run_train_then_evaluate \
  exp19 \
  rnn-seqmem-exp19-ema-memory-barlow-events-explicit-visibility-v3 \
  "$EXP19_DIR" \
  --ema-target-encoder \
  --ema-momentum 0.996 \
  --memory-barlow-scale 0.01 \
  --event-balanced-sampling \
  --event-fraction 0.30 \
  --event-pool-fraction 0.20 \
  --event-movement-threshold 0.01 \
  --event-state-threshold 0.001 \
  --event-attack-action-min 6 \
  --event-min-transitions 1

# Existing Exp16 is not retrained. It is only re-evaluated under the corrected
# explicit-visibility evaluator.
if [[ -f "$EXP16_DIR/checkpoint.pt" ]]; then
  if evaluate_one \
    exp16_reeval \
    "$EXP16_DIR/checkpoint.pt"
  then
    SUCCESSFUL_CHECKPOINTS=(
      "$EXP16_DIR/checkpoint.pt"
      "${SUCCESSFUL_CHECKPOINTS[@]}"
    )
  else
    FAILED_STAGES+=("exp16_reeval:evaluation")
  fi
else
  echo "Existing Exp16 checkpoint not found; Exp16 re-evaluation skipped."
fi

# Final combined comparison uses only checkpoints whose individual evaluation
# completed successfully.
if [[ "${#SUCCESSFUL_CHECKPOINTS[@]}" -gt 0 ]]; then
  COMBINED_OUTPUT="$EVAL_ROOT/combined"
  rm -rf "$COMBINED_OUTPUT"
  mkdir -p "$COMBINED_OUTPUT"

  checkpoint_arguments=()
  for checkpoint in "${SUCCESSFUL_CHECKPOINTS[@]}"; do
    checkpoint_arguments+=(--checkpoint "$checkpoint")
  done

  if ! run_logged \
    eval_combined \
    env LD_LIBRARY_PATH="" \
    python "$EVALUATOR" \
      --manifest splits/generated_seed4_mapdims_only.json \
      --split eval \
      "${checkpoint_arguments[@]}" \
      --out-dir "$COMBINED_OUTPUT" \
      --probe-dir "$PROBE_DIR" \
      --batch-size 16 \
      --num-workers 4 \
      --max-batches "$EVAL_MAX_BATCHES" \
      --diagnostics \
      --diagnostic-max-batches "$DIAGNOSTIC_MAX_BATCHES" \
      --eval-rollout-horizon 5 \
      --target-mode full \
      --window-mode sequential \
      --thresholds 0.01 0.05 0.1 \
      --probe-decoder \
      --probe-train-split train \
      --probe-epochs "$PROBE_EPOCHS" \
      --probe-max-batches-per-epoch "$PROBE_MAX_BATCHES" \
      --probe-samples-per-epoch "$PROBE_SAMPLES_PER_EPOCH" \
      --probe-lr 0.001 \
      --probe-weight-decay 0.00001 \
      --probe-seed 123 \
      --require-hidden-eval \
      --device cuda \
      --amp
  then
    FAILED_STAGES+=("combined:evaluation")
  fi
fi

echo
echo "============================================================"
echo "SUITE FINISHED"
echo "Successful evaluated checkpoints: ${#SUCCESSFUL_CHECKPOINTS[@]}"
echo "Logs:        $LOG_DIR"
echo "Evaluations: $EVAL_ROOT"

if [[ "${#FAILED_STAGES[@]}" -gt 0 ]]; then
  echo "Failed stages:"
  printf '  %s\n' "${FAILED_STAGES[@]}"
  exit 1
fi

echo "All requested stages completed successfully."
echo "============================================================"
