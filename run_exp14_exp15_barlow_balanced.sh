#!/usr/bin/env bash
# Balanced Exp14 / Exp15.
# Prediction loss remains the primary objective.
#
# Exp14 regularization budget:
#   Barlow <= 30% of pred_loss
#
# Exp15 regularization budget:
#   Barlow <= 25% of pred_loss
#   SIGReg <= 5% of pred_loss
#   Combined <= 30% of pred_loss
#
# --barlow-weight and --sigreg-weight are coefficient ceilings, not fixed
# coefficients. Per-batch effective coefficients are chosen below those ceilings
# to satisfy the contribution budgets.

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR" || exit 1

run_exp() {
  local name="$1"
  shift
  echo "============================================================"
  echo "Starting ${name}"
  echo "============================================================"
  LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_barlow_balanced "$@"
  local status=$?
  echo "${name} exited with status ${status}"
  return 0
}

COMMON_ARGS=(
  --manifest splits/generated_seed4_mapdims_only.json
  --split train
  --model-size default
  --epochs 10
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
  --barlow-weight 0.05
  --barlow-lambda 0.0005
  --barlow-mode global
  --regularizer-ratio-warmup-epochs 2
  --decoder-weight 0.01
  --presence-weight 0.01
  --rollout-memory-dim 128
  --target-mode full
  --action-conditioned-memory
  --one-step-weight 0.5
  --grad-clip 1.0
  --device cuda
  --amp
  --wandb
  --wandb-project SMAC-JEPA-losses
)

run_exp "Exp14 balanced Barlow only" \
  "${COMMON_ARGS[@]}" \
  --out-dir runs/rnn_seqmem_exp14_barlow_balanced_action_memory_onestep_full_h5 \
  --sigreg-weight 0.0 \
  --barlow-target-ratio 0.30 \
  --sigreg-target-ratio 0.0 \
  --wandb-name rnn-seqmem-exp14-barlow-balanced-action-memory-onestep-full-h5

run_exp "Exp15 balanced Barlow + weak SIGReg" \
  "${COMMON_ARGS[@]}" \
  --out-dir runs/rnn_seqmem_exp15_barlow_weak_sigreg_balanced_action_memory_onestep_full_h5 \
  --sigreg-weight 0.001 \
  --barlow-target-ratio 0.25 \
  --sigreg-target-ratio 0.05 \
  --wandb-name rnn-seqmem-exp15-barlow-weak-sigreg-balanced-action-memory-onestep-full-h5
