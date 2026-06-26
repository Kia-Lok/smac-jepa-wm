#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR"

MODULE="smac_jepa.train_markov_rollout_rnn_visibility_seqmem_barlow_plausible_v2"
EPOCHS="${EPOCHS:-10}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-50000}"
RUN_TAG="${RUN_TAG:-full}"

COMMON_ARGS=(
  --manifest splits/generated_seed4_mapdims_only.json
  --split train
  --model-size default
  --epochs "$EPOCHS"
  --batch-size 16
  --num-workers 4
  --rollout-window 20
  --rollout-horizon 5
  --window-mode random
  --samples-per-epoch "$SAMPLES_PER_EPOCH"
  --enemy-visibility-mask
  --enemy-sight-range 9.0
  --temporal-loss lambda
  --td-lambda 0.9
  --barlow-weight 0.05
  --barlow-lambda 0.0005
  --barlow-mode per-horizon
  --regularizer-reference-floor 0.05
  --regularizer-ratio-warmup-epochs 2
  --latent-std-floor-ratio 0.80
  --scale-calibration-batches 20
  --sigreg-knots 17
  --sigreg-num-proj 1024
  --sigreg-proj-chunk 64
  --sigreg-max-samples 0
  --health-check-after-steps 100
  --abort-target-std-ratio 0.60
  --abort-pred-target-std-ratio 0.05
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
echo "Starting Exp15 V2 FIRST"
echo "Barlow + calibrated variance floor + stable float32 SIGReg"
echo "============================================================"
LD_LIBRARY_PATH="" python -m "$MODULE" \
  "${COMMON_ARGS[@]}" \
  --out-dir "runs/rnn_seqmem_exp15_barlow_sigreg_variance_v2_${RUN_TAG}_h5" \
  --barlow-target-ratio 0.15 \
  --sigreg-weight 0.001 \
  --sigreg-target-ratio 0.05 \
  --variance-floor-weight 1.0 \
  --variance-floor-target-ratio 0.10 \
  --wandb-name "rnn-seqmem-exp15-barlow-sigreg-variance-v2-${RUN_TAG}-h5"

echo "============================================================"
echo "Exp15 V2 completed successfully. Starting Exp14 V2"
echo "Barlow + calibrated variance floor, no SIGReg"
echo "============================================================"
LD_LIBRARY_PATH="" python -m "$MODULE" \
  "${COMMON_ARGS[@]}" \
  --out-dir "runs/rnn_seqmem_exp14_barlow_variance_v2_${RUN_TAG}_h5" \
  --barlow-target-ratio 0.15 \
  --sigreg-weight 0.0 \
  --sigreg-target-ratio 0.0 \
  --variance-floor-weight 1.0 \
  --variance-floor-target-ratio 0.15 \
  --wandb-name "rnn-seqmem-exp14-barlow-variance-v2-${RUN_TAG}-h5"

echo "Both V2 Barlow experiments completed successfully."
