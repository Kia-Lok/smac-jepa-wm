#!/usr/bin/env bash
# Evaluate all available JEPA seqmem experiments using a common H5 protocol.
# Baseline metrics: 500 deterministic batches = 8000 windows.
# Extra Dreamer diagnostics: first 100 deterministic batches = 1600 windows.
#
# Missing checkpoints are skipped automatically.

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR" || exit 1

EVAL_SCRIPT="${EVAL_SCRIPT:-eval_rnn_seqmem_dreamer_diagnostics.py}"
OUT_DIR="${OUT_DIR:-runs/dreamer_diagnostics_all_experiments_h5_first8000}"
mkdir -p "$OUT_DIR"

checkpoints=()

add_checkpoint() {
  local path="$1"
  if [[ -f "$path" ]]; then
    checkpoints+=(--checkpoint "$path")
    echo "including $path"
  else
    echo "missing, skipping $path"
  fi
}

add_checkpoint runs/rnn_seqmem_exp01_normal_full/checkpoint_epoch_010.pt
add_checkpoint runs/rnn_seqmem_exp02_action_memory_full/checkpoint_epoch_010.pt
add_checkpoint runs/rnn_seqmem_exp03_onestep_full/checkpoint_epoch_010.pt
add_checkpoint runs/rnn_seqmem_exp04_action_memory_onestep_full/checkpoint_epoch_010.pt
add_checkpoint runs/rnn_seqmem_exp05_observed_target/checkpoint_epoch_010.pt
add_checkpoint runs/rnn_seqmem_exp06_action_memory_onestep_observed/checkpoint_epoch_010.pt
add_checkpoint runs/rnn_seqmem_exp07_onestep_full_h10/checkpoint_epoch_010.pt
add_checkpoint runs/rnn_seqmem_exp08_curriculum_h10/checkpoint_epoch_010.pt
add_checkpoint runs/rnn_seqmem_exp09_onestep025_full_h5/checkpoint_epoch_010.pt
add_checkpoint runs/rnn_seqmem_exp10_onestep_uniform_full_h5/checkpoint_epoch_010.pt
add_checkpoint runs/rnn_seqmem_exp11_per_onestep_full_h5/checkpoint_epoch_010.pt
add_checkpoint runs/rnn_seqmem_exp12_action_memory_per_onestep_full_h5/checkpoint_epoch_010.pt

if [[ ${#checkpoints[@]} -eq 0 ]]; then
  echo "No checkpoints found."
  exit 1
fi

LD_LIBRARY_PATH="" python "$EVAL_SCRIPT" \
  --manifest splits/generated_seed4_mapdims_only.json \
  --split eval \
  "${checkpoints[@]}" \
  --out-dir "$OUT_DIR" \
  --batch-size 16 \
  --num-workers 4 \
  --max-batches 500 \
  --diagnostics \
  --diagnostic-max-batches 100 \
  --eval-rollout-horizon 5 \
  --window-mode sequential \
  --device cuda \
  --amp
