#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR"

EVALUATOR="./eval_jepa_hidden_belief_v3.py"
BASE_EVALUATOR="./eval_jepa_weekend_structural_v2.py"
MANIFEST="splits/generated_seed4_mapdims_only.json"
OUT_DIR="runs/hidden_belief_eval_v3"
PROBE_DIR="runs/hidden_belief_probe_decoders_v3"
LOG_DIR="runs/hidden_belief_eval_v3_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/hidden_belief_${STAMP}.log"

mkdir -p "$OUT_DIR" "$PROBE_DIR" "$LOG_DIR"

if [[ ! -f "$EVALUATOR" ]]; then
  echo "ERROR: missing $EVALUATOR"
  exit 2
fi
if [[ ! -f "$BASE_EVALUATOR" ]]; then
  echo "ERROR: missing $BASE_EVALUATOR"
  exit 2
fi

# Reuse all already-trained independent probes. No probe training is allowed
# in the V3 evaluator: a missing probe causes an immediate error.
for source in \
  "runs/eval_exp17_exp19_final_fixed/probe_decoders" \
  "runs/weekend_structural_probe_decoders_v2"
do
  if [[ -d "$source" ]]; then
    cp -n "$source"/* "$PROBE_DIR"/ 2>/dev/null || true
  fi
done

CHECKPOINTS=(
  "runs/rnn_seqmem_exp17_ema_explicit_visibility_v3/checkpoint.pt"
  "runs/rnn_seqmem_exp22_delta_loss_v2/checkpoint.pt"
  "runs/rnn_seqmem_exp24_observation_dropout_v2/checkpoint.pt"
  "runs/rnn_seqmem_exp25_hidden_reconstruction_v2/checkpoint.pt"
)

for checkpoint in "${CHECKPOINTS[@]}"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "ERROR: missing checkpoint $checkpoint"
    exit 2
  fi

  expected_probe="$PROBE_DIR/$(basename "$(dirname "$checkpoint")")_checkpoint_meaningful_features_v2_probe_decoder.pt"
  if [[ ! -f "$expected_probe" ]]; then
    echo "ERROR: missing existing probe $expected_probe"
    exit 2
  fi
done

python -m py_compile "$BASE_EVALUATOR" "$EVALUATOR"
python "$EVALUATOR" --help >/dev/null

CHECKPOINT_ARGS=()
for checkpoint in "${CHECKPOINTS[@]}"; do
  CHECKPOINT_ARGS+=(--checkpoint "$checkpoint")
done

echo "Starting hidden-belief evaluation"
echo "Output: $OUT_DIR"
echo "Log:    $LOG_FILE"

env LD_LIBRARY_PATH="" \
python "$EVALUATOR" \
  --manifest "$MANIFEST" \
  --split eval \
  "${CHECKPOINT_ARGS[@]}" \
  --out-dir "$OUT_DIR" \
  --probe-dir "$PROBE_DIR" \
  --batch-size 16 \
  --num-workers 4 \
  --eval-rollout-horizon 5 \
  --target-mode full \
  --thresholds 0.01 0.05 0.1 \
  --presence-threshold 0.5 \
  --natural-hidden-eval \
  --natural-hidden-target-entity-times 3000 \
  --natural-hidden-max-scan-batches 5000 \
  --controlled-occlusion-eval \
  --controlled-occlusion-max-batches 150 \
  --controlled-occlusion-target-entity-times 10000 \
  --controlled-occlusion-spans 1 3 5 \
  --controlled-occlusion-seed 123 \
  --controlled-prefer-reappearance \
  --device cuda \
  --amp \
  2>&1 | tee "$LOG_FILE"

echo
echo "Hidden-belief evaluation completed."
echo "Results:"
echo "  $OUT_DIR/hidden_belief_summary.csv"
echo "  $OUT_DIR/hidden_belief_summary.jsonl"
echo "  $OUT_DIR/hidden_belief_*.json"
