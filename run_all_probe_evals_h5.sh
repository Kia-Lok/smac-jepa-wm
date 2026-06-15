#!/usr/bin/env bash
# Train/load one independent decoder probe per checkpoint and evaluate all available
# seqmem experiments with a common H5 protocol.

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR" || exit 1

EVAL_SCRIPT="${EVAL_SCRIPT:-eval_rnn_seqmem_dreamer_probe.py}"
OUT_DIR="${OUT_DIR:-runs/dreamer_probe_eval_all_h5_first8000}"
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
  --target-mode full \
  --window-mode sequential \
  --probe-decoder \
  --probe-train-split train \
  --probe-epochs 20 \
  --probe-max-batches-per-epoch 300 \
  --probe-samples-per-epoch 20000 \
  --probe-lr 0.001 \
  --probe-weight-decay 0.00001 \
  --probe-seed 123 \
  --device cuda \
  --amp
