#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR"

EVALUATOR="./eval_jepa_event_conditioned_v2.py"
MANIFEST="splits/generated_seed4_mapdims_only.json"
OUT_ROOT="runs/eval_exp17_exp19_event_conditioned_v2"
PROBE_DIR="runs/eval_exp17_exp19_final_fixed/probe_decoders"

EXP17="runs/rnn_seqmem_exp17_ema_explicit_visibility_v3/checkpoint.pt"
EXP18="runs/rnn_seqmem_exp18_ema_memory_barlow_explicit_visibility_v3/checkpoint.pt"
EXP19="runs/rnn_seqmem_exp19_ema_memory_barlow_events_explicit_visibility_v3/checkpoint.pt"

mkdir -p "$OUT_ROOT" "$PROBE_DIR"

python -m py_compile "$EVALUATOR"

for checkpoint in "$EXP17" "$EXP18" "$EXP19"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "ERROR: missing checkpoint: $checkpoint"
    exit 1
  fi
done

run_eval() {
  local label="$1"
  local checkpoint="$2"
  local output="$OUT_ROOT/$label"

  echo
  echo "============================================================"
  echo "Evaluating $label"
  echo "Checkpoint: $checkpoint"
  echo "Output:     $output"
  echo "Probe dir:  $PROBE_DIR"
  echo "============================================================"

  rm -rf "$output"
  mkdir -p "$output"

  env LD_LIBRARY_PATH="" python "$EVALUATOR" \
    --manifest "$MANIFEST" \
    --split eval \
    --checkpoint "$checkpoint" \
    --out-dir "$output" \
    --probe-dir "$PROBE_DIR" \
    --batch-size 16 \
    --num-workers 4 \
    --max-batches 500 \
    --diagnostics \
    --diagnostic-max-batches 100 \
    --eval-rollout-horizon 5 \
    --target-mode full \
    --window-mode sequential \
    --thresholds 0.01 0.05 0.1 \
    --change-threshold 0.01 \
    --event-threshold 0.01 \
    --attack-action-min 6 \
    --probe-decoder \
    --probe-train-split train \
    --probe-epochs 20 \
    --probe-max-batches-per-epoch 300 \
    --probe-samples-per-epoch 20000 \
    --probe-lr 0.001 \
    --probe-weight-decay 0.00001 \
    --probe-seed 123 \
    --no-require-hidden-eval \
    --device cuda \
    --amp
}

run_eval exp17 "$EXP17"
run_eval exp18 "$EXP18"
run_eval exp19 "$EXP19"

echo
echo "All event-conditioned evaluations completed."
echo "Results: $OUT_ROOT"
