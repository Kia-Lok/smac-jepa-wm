#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR" || exit 1

EPOCHS="${EPOCHS:-5}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-500}"
DIAGNOSTIC_MAX_BATCHES="${DIAGNOSTIC_MAX_BATCHES:-100}"
PROBE_EPOCHS="${PROBE_EPOCHS:-20}"
PROBE_MAX_BATCHES="${PROBE_MAX_BATCHES:-300}"
PROBE_SAMPLES_PER_EPOCH="${PROBE_SAMPLES_PER_EPOCH:-20000}"

TRAIN_MODULE="smac_jepa.train_jepa_exp17_exp19_fixed"
EVALUATOR="./eval_jepa_exp17_exp19_recovery.py"

EXP17_DIR="runs/rnn_seqmem_exp17_ema_explicit_visibility_v3"
EXP18_DIR="runs/rnn_seqmem_exp18_ema_memory_barlow_explicit_visibility_v3"
EXP19_DIR="runs/rnn_seqmem_exp19_ema_memory_barlow_events_explicit_visibility_v3"

EVAL_ROOT="runs/eval_exp17_exp19_recovered"
# Reuse the probes already trained before the old evaluator crashed.
PROBE_DIR="runs/eval_exp17_exp19_final_fixed/probe_decoders"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/exp17_exp19_recovery_logs_${STAMP}"
mkdir -p "$EVAL_ROOT" "$PROBE_DIR" "$LOG_DIR"

if [[ ! -f "$EVALUATOR" ]]; then
  echo "ERROR: missing evaluator: $ROOT_DIR/$EVALUATOR"
  exit 2
fi

if [[ ! -f "smac_jepa/train_jepa_exp17_exp19_fixed.py" ]]; then
  echo "ERROR: missing trainer module."
  exit 2
fi

python -m py_compile \
  smac_jepa/train_jepa_exp17_exp19_fixed.py \
  "$EVALUATOR" || exit 1

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
    except Exception as exc:
        print(
            f"warning: ignored unreadable checkpoint {path}: {exc}",
            file=sys.stderr,
        )

if valid:
    print(max(valid, key=lambda item: item[:3])[3])
PY
}

completed_epoch() {
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
complete = bool(checkpoint.get("epoch_complete", True))
print(epoch if complete else 0)
PY
}

show_checkpoint() {
  local label="$1"
  local checkpoint_path="$2"

  python - "$label" "$checkpoint_path" <<'PY'
import sys
import torch

label, path = sys.argv[1], sys.argv[2]
try:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")

print(
    f"{label}: path={path} "
    f"epoch={checkpoint.get('epoch')} "
    f"epoch_complete={checkpoint.get('epoch_complete', True)} "
    f"global_step={checkpoint.get('global_step')} "
    f"batches_completed={checkpoint.get('batches_completed_in_epoch')}"
)
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

declare -a COMPLETED_CHECKPOINTS=()
declare -a FAILED_STAGES=()

# ---------------------------------------------------------------------------
# Record completed Exp17 and Exp18. They are never retrained here.
# ---------------------------------------------------------------------------
for entry in \
  "exp17:$EXP17_DIR" \
  "exp18:$EXP18_DIR"
do
  label="${entry%%:*}"
  directory="${entry#*:}"
  checkpoint="$directory/checkpoint.pt"

  if [[ -f "$checkpoint" ]] && \
     [[ "$(completed_epoch "$checkpoint")" -ge "$EPOCHS" ]]
  then
    show_checkpoint "$label" "$checkpoint"
    COMPLETED_CHECKPOINTS+=("$label:$checkpoint")
  else
    echo "ERROR: completed $label checkpoint not found: $checkpoint"
    FAILED_STAGES+=("$label:missing_completed_checkpoint")
  fi
done

# ---------------------------------------------------------------------------
# Resume only Exp19, using the newest recovery/epoch/final checkpoint.
# ---------------------------------------------------------------------------
EXP19_FINAL="$EXP19_DIR/checkpoint.pt"
exp19_is_complete=0

if [[ -f "$EXP19_FINAL" ]] && \
   [[ "$(completed_epoch "$EXP19_FINAL")" -ge "$EPOCHS" ]]
then
  exp19_is_complete=1
  echo "Exp19 is already complete; training resume skipped."
else
  EXP19_RESUME="$(latest_checkpoint "$EXP19_DIR" || true)"

  if [[ -z "$EXP19_RESUME" ]]; then
    echo "ERROR: no Exp19 checkpoint exists to resume."
    FAILED_STAGES+=("exp19:no_resume_checkpoint")
  else
    show_checkpoint "exp19_resume" "$EXP19_RESUME"

    if run_logged \
      resume_exp19 \
      env LD_LIBRARY_PATH="" \
      python -m "$TRAIN_MODULE" \
        "${COMMON_TRAIN_ARGS[@]}" \
        --ema-target-encoder \
        --ema-momentum 0.996 \
        --memory-barlow-scale 0.01 \
        --event-balanced-sampling \
        --event-fraction 0.30 \
        --event-pool-fraction 0.20 \
        --event-movement-threshold 0.01 \
        --event-state-threshold 0.001 \
        --event-attack-action-min 6 \
        --event-min-transitions 1 \
        --resume "$EXP19_RESUME" \
        --out-dir "$EXP19_DIR" \
        --wandb \
        --wandb-project SMAC-JEPA-losses \
        --wandb-name rnn-seqmem-exp19-ema-memory-barlow-events-explicit-visibility-v3-resumed
    then
      if [[ -f "$EXP19_FINAL" ]] && \
         [[ "$(completed_epoch "$EXP19_FINAL")" -ge "$EPOCHS" ]]
      then
        exp19_is_complete=1
      else
        FAILED_STAGES+=("exp19:resume_did_not_finish")
      fi
    else
      FAILED_STAGES+=("exp19:training")
    fi
  fi
fi

if [[ "$exp19_is_complete" == "1" ]]; then
  show_checkpoint "exp19" "$EXP19_FINAL"
  COMPLETED_CHECKPOINTS+=("exp19:$EXP19_FINAL")
fi

# ---------------------------------------------------------------------------
# Evaluate every completed checkpoint. Zero hidden examples no longer aborts.
# ---------------------------------------------------------------------------
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
      --no-require-hidden-eval \
      --device cuda \
      --amp
}

declare -a EVALUATED_CHECKPOINTS=()

for entry in "${COMPLETED_CHECKPOINTS[@]}"; do
  label="${entry%%:*}"
  checkpoint="${entry#*:}"

  if evaluate_one "$label" "$checkpoint"; then
    EVALUATED_CHECKPOINTS+=("$checkpoint")
  else
    FAILED_STAGES+=("$label:evaluation")
  fi
done

# Combined comparison of every successfully evaluated checkpoint.
if [[ "${#EVALUATED_CHECKPOINTS[@]}" -gt 0 ]]; then
  COMBINED_OUTPUT="$EVAL_ROOT/combined"
  rm -rf "$COMBINED_OUTPUT"
  mkdir -p "$COMBINED_OUTPUT"

  checkpoint_args=()
  for checkpoint in "${EVALUATED_CHECKPOINTS[@]}"; do
    checkpoint_args+=(--checkpoint "$checkpoint")
  done

  if ! run_logged \
    eval_combined \
    env LD_LIBRARY_PATH="" \
    python "$EVALUATOR" \
      --manifest splits/generated_seed4_mapdims_only.json \
      --split eval \
      "${checkpoint_args[@]}" \
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
      --no-require-hidden-eval \
      --device cuda \
      --amp
  then
    FAILED_STAGES+=("combined:evaluation")
  fi
fi

echo
echo "============================================================"
echo "RECOVERY FINISHED"
echo "Completed checkpoints: ${#COMPLETED_CHECKPOINTS[@]}"
echo "Successful evaluations: ${#EVALUATED_CHECKPOINTS[@]}"
echo "Evaluation root: $EVAL_ROOT"
echo "Logs: $LOG_DIR"

if [[ "${#FAILED_STAGES[@]}" -gt 0 ]]; then
  echo "Failed stages:"
  printf '  %s\n' "${FAILED_STAGES[@]}"
  exit 1
fi

echo "Exp19 resumed and Exp17–Exp19 evaluations completed."
echo "============================================================"
