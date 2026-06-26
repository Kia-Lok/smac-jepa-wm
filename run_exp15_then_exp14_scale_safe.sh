#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR"

MODULE="smac_jepa.train_markov_rollout_rnn_visibility_seqmem_barlow_scale_safe"

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
  --barlow-target-ratio 0.20
  --barlow-lambda 0.0005
  --barlow-mode global
  --regularizer-reference-floor 0.05
  --regularizer-ratio-warmup-epochs 2
  --latent-std-floor 1.0
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

echo "============================================================"
echo "Starting Exp15 FIRST: Barlow + weak SIGReg, scale-safe"
echo "============================================================"
LD_LIBRARY_PATH="" python -m "$MODULE" \
  "${COMMON_ARGS[@]}" \
  --out-dir runs/rnn_seqmem_exp15_barlow_sigreg_scale_safe_action_memory_onestep_full_h5 \
  --sigreg-weight 0.001 \
  --sigreg-target-ratio 0.10 \
  --variance-floor-weight 0.0 \
  --variance-floor-target-ratio 0.0 \
  --wandb-name rnn-seqmem-exp15-barlow-sigreg-scale-safe-action-memory-onestep-full-h5

echo "============================================================"
echo "Exp15 completed. Starting Exp14: Barlow + variance floor"
echo "============================================================"
LD_LIBRARY_PATH="" python -m "$MODULE" \
  "${COMMON_ARGS[@]}" \
  --out-dir runs/rnn_seqmem_exp14_barlow_variance_scale_safe_action_memory_onestep_full_h5 \
  --sigreg-weight 0.0 \
  --sigreg-target-ratio 0.0 \
  --variance-floor-weight 1.0 \
  --variance-floor-target-ratio 0.10 \
  --wandb-name rnn-seqmem-exp14-barlow-variance-scale-safe-action-memory-onestep-full-h5

echo "Both scale-safe experiments completed."
