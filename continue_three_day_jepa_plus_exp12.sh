#!/usr/bin/env bash
# Continue / run 3-day JEPA experiments plus Exp12 after server crash.
#
# This script is restart-safe:
#   - if target epoch checkpoint exists, it skips that experiment
#   - if checkpoint.pt exists, it resumes
#   - otherwise it starts fresh
#   - it does NOT stop the whole script if one command fails
#
# Experiments included:
#   existing eval: Exp02/Exp04 if checkpoints exist
#   Exp07: H10 one-step full
#   Exp09: one-step 0.25 H5
#   Exp10: uniform temporal loss H5
#   Exp11: PER + one-step H5
#   Exp12: action-conditioned memory + PER + one-step H5
#   Exp08: curriculum H3 -> H5 -> H10
#   final eval: all completed new candidates, including Exp12

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR" || {
  echo "ERROR: cannot cd to ROOT_DIR=$ROOT_DIR"
  exit 1
}

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/continue_three_day_jepa_plus_exp12_logs_${RUN_STAMP}"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Continue JEPA experiments + Exp12"
echo "Root:    $ROOT_DIR"
echo "Logs:    $LOG_DIR"
echo "Started: $(date)"
echo "============================================================"

run_cmd() {
  local name="$1"; shift
  local log="${LOG_DIR}/${name}.log"
  local status="${LOG_DIR}/${name}.status"

  echo
  echo "============================================================"
  echo "START ${name} at $(date)"
  echo "LOG   ${log}"
  echo "============================================================"

  "$@" 2>&1 | tee "$log"
  local code=${PIPESTATUS[0]}
  echo "$code" > "$status"

  echo "============================================================"
  echo "END ${name} exit_code=${code} at $(date)"
  echo "============================================================"
  echo

  return 0
}

done_epoch() {
  local dir="$1"
  local ep="$2"
  [[ -f "${dir}/checkpoint_epoch_$(printf "%03d" "$ep").pt" ]]
}

resume_array_for() {
  local dir="$1"
  if [[ -f "${dir}/checkpoint.pt" ]]; then
    echo "--resume ${dir}/checkpoint.pt"
  fi
}

ensure_file() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "$dst" && -f "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    echo "copied $src -> $dst"
  fi
}

# -------------------------------------------------------------------
# Install / ensure required scripts.
# Prefer files copied from this package, fallback to /mnt/data paths.
# -------------------------------------------------------------------
ensure_file ./train_markov_rollout_rnn_visibility_seqmem_per.py smac_jepa/train_markov_rollout_rnn_visibility_seqmem_per.py
ensure_file /mnt/data/three_day_jepa_plus_exp12/train_markov_rollout_rnn_visibility_seqmem_per.py smac_jepa/train_markov_rollout_rnn_visibility_seqmem_per.py
ensure_file /mnt/data/three_day_jepa/train_markov_rollout_rnn_visibility_seqmem_per.py smac_jepa/train_markov_rollout_rnn_visibility_seqmem_per.py

ensure_file /mnt/data/rnn_weekend_experiments/train_markov_rollout_rnn_visibility_seqmem_experiments.py smac_jepa/train_markov_rollout_rnn_visibility_seqmem_experiments.py

ensure_file ./eval_rnn_seqmem_combined_metrics.py eval_rnn_seqmem_combined_metrics.py
ensure_file /mnt/data/three_day_jepa_plus_exp12/eval_rnn_seqmem_combined_metrics.py eval_rnn_seqmem_combined_metrics.py
ensure_file /mnt/data/seqmem_combined_eval_fixed/eval_rnn_seqmem_combined_metrics.py eval_rnn_seqmem_combined_metrics.py

# Quick argument sanity checks. These should be cheap.
echo
echo "Argument sanity checks:"
python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments --help | grep -E "one-step-weight|target-mode|presence-weight|action-conditioned-memory" || true
python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_per --help | grep -E "sample-prioritized|priority-alpha|one-step-weight|target-mode|action-conditioned-memory" || true
echo

# -------------------------------------------------------------------
# Existing Exp02/Exp04 deterministic eval.
# -------------------------------------------------------------------
if [[ -f runs/eval_existing_exp02_exp04_first8000/eval_seqmem_combined_summary.csv ]]; then
  echo "SKIP existing Exp02/Exp04 eval: summary already exists"
else
  ckpts=()
  [[ -f runs/rnn_seqmem_exp02_action_memory_full/checkpoint_epoch_010.pt ]] && ckpts+=(--checkpoint runs/rnn_seqmem_exp02_action_memory_full/checkpoint_epoch_010.pt)
  [[ -f runs/rnn_seqmem_exp04_action_memory_onestep_full/checkpoint_epoch_010.pt ]] && ckpts+=(--checkpoint runs/rnn_seqmem_exp04_action_memory_onestep_full/checkpoint_epoch_010.pt)
  if [[ ! -f runs/rnn_seqmem_exp04_action_memory_onestep_full/checkpoint_epoch_010.pt && -f runs/rnn_seqmem_exp04_action_memory_onestep_full/checkpoint.pt ]]; then
    ckpts+=(--checkpoint runs/rnn_seqmem_exp04_action_memory_onestep_full/checkpoint.pt)
  fi

  if [[ ${#ckpts[@]} -gt 0 ]]; then
    run_cmd 00_eval_existing_exp02_exp04_first8000 \
      env LD_LIBRARY_PATH="" python eval_rnn_seqmem_combined_metrics.py \
        --manifest splits/generated_seed4_mapdims_only.json \
        --split eval \
        "${ckpts[@]}" \
        --out-dir runs/eval_existing_exp02_exp04_first8000 \
        --batch-size 16 \
        --num-workers 4 \
        --max-batches 500 \
        --window-mode sequential \
        --device cuda \
        --amp
  else
    echo "SKIP existing Exp02/Exp04 eval: no checkpoints found"
  fi
fi

# -------------------------------------------------------------------
# Exp07: H10 one-step full.
# -------------------------------------------------------------------
if done_epoch runs/rnn_seqmem_exp07_onestep_full_h10 10; then
  echo "SKIP Exp07: checkpoint_epoch_010.pt exists"
else
  RESUME=($(resume_array_for runs/rnn_seqmem_exp07_onestep_full_h10))
  run_cmd 01_exp07_onestep_full_h10_continue \
    env LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
      --manifest splits/generated_seed4_mapdims_only.json \
      --split train \
      --out-dir runs/rnn_seqmem_exp07_onestep_full_h10 \
      --model-size default \
      --epochs 10 \
      --batch-size 8 \
      --num-workers 4 \
      --rollout-window 20 \
      --rollout-horizon 10 \
      --window-mode random \
      --samples-per-epoch 50000 \
      --enemy-visibility-mask \
      --enemy-sight-range 9.0 \
      --temporal-loss lambda \
      --td-lambda 0.9 \
      --sigreg-weight 0.01 \
      --decoder-weight 0.01 \
      --presence-weight 0.01 \
      --rollout-memory-dim 128 \
      --target-mode full \
      --one-step-weight 0.5 \
      "${RESUME[@]}" \
      --device cuda \
      --amp \
      --wandb \
      --wandb-project SMAC-JEPA-losses \
      --wandb-name rnn-seqmem-exp07-onestep-full-h10-continue
fi

# -------------------------------------------------------------------
# Exp09: one-step 0.25 H5.
# -------------------------------------------------------------------
if done_epoch runs/rnn_seqmem_exp09_onestep025_full_h5 10; then
  echo "SKIP Exp09: checkpoint_epoch_010.pt exists"
else
  RESUME=($(resume_array_for runs/rnn_seqmem_exp09_onestep025_full_h5))
  run_cmd 02_exp09_onestep025_full_h5_continue \
    env LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
      --manifest splits/generated_seed4_mapdims_only.json \
      --split train \
      --out-dir runs/rnn_seqmem_exp09_onestep025_full_h5 \
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
      --sigreg-weight 0.01 \
      --decoder-weight 0.01 \
      --presence-weight 0.01 \
      --rollout-memory-dim 128 \
      --target-mode full \
      --one-step-weight 0.25 \
      "${RESUME[@]}" \
      --device cuda \
      --amp \
      --wandb \
      --wandb-project SMAC-JEPA-losses \
      --wandb-name rnn-seqmem-exp09-onestep025-full-h5-continue
fi

# -------------------------------------------------------------------
# Exp10: uniform temporal loss H5.
# -------------------------------------------------------------------
if done_epoch runs/rnn_seqmem_exp10_onestep_uniform_full_h5 10; then
  echo "SKIP Exp10: checkpoint_epoch_010.pt exists"
else
  RESUME=($(resume_array_for runs/rnn_seqmem_exp10_onestep_uniform_full_h5))
  run_cmd 03_exp10_onestep_uniform_full_h5_continue \
    env LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
      --manifest splits/generated_seed4_mapdims_only.json \
      --split train \
      --out-dir runs/rnn_seqmem_exp10_onestep_uniform_full_h5 \
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
      --temporal-loss uniform \
      --td-lambda 0.9 \
      --sigreg-weight 0.01 \
      --decoder-weight 0.01 \
      --presence-weight 0.01 \
      --rollout-memory-dim 128 \
      --target-mode full \
      --one-step-weight 0.5 \
      "${RESUME[@]}" \
      --device cuda \
      --amp \
      --wandb \
      --wandb-project SMAC-JEPA-losses \
      --wandb-name rnn-seqmem-exp10-onestep-uniform-full-h5-continue
fi

# -------------------------------------------------------------------
# Exp11: PER + one-step H5.
# -------------------------------------------------------------------
if done_epoch runs/rnn_seqmem_exp11_per_onestep_full_h5 10; then
  echo "SKIP Exp11: checkpoint_epoch_010.pt exists"
else
  RESUME=($(resume_array_for runs/rnn_seqmem_exp11_per_onestep_full_h5))
  run_cmd 04_exp11_per_onestep_full_h5_continue \
    env LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_per \
      --manifest splits/generated_seed4_mapdims_only.json \
      --split train \
      --out-dir runs/rnn_seqmem_exp11_per_onestep_full_h5 \
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
      --sigreg-weight 0.01 \
      --decoder-weight 0.01 \
      --presence-weight 0.01 \
      --rollout-memory-dim 128 \
      --target-mode full \
      --one-step-weight 0.5 \
      --sample-prioritized \
      --priority-alpha 0.4 \
      --priority-uniform-mix 0.7 \
      --priority-ema-beta 0.95 \
      --priority-warmup-epochs 2 \
      --priority-score pred_loss \
      "${RESUME[@]}" \
      --device cuda \
      --amp \
      --wandb \
      --wandb-project SMAC-JEPA-losses \
      --wandb-name rnn-seqmem-exp11-per-onestep-full-h5-continue
fi

# -------------------------------------------------------------------
# Exp12: action-conditioned memory + PER + one-step H5.
# This is the extra experiment.
# -------------------------------------------------------------------
if done_epoch runs/rnn_seqmem_exp12_action_memory_per_onestep_full_h5 10; then
  echo "SKIP Exp12: checkpoint_epoch_010.pt exists"
else
  RESUME=($(resume_array_for runs/rnn_seqmem_exp12_action_memory_per_onestep_full_h5))
  run_cmd 05_exp12_action_memory_per_onestep_full_h5_continue \
    env LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_per \
      --manifest splits/generated_seed4_mapdims_only.json \
      --split train \
      --out-dir runs/rnn_seqmem_exp12_action_memory_per_onestep_full_h5 \
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
      --sigreg-weight 0.01 \
      --decoder-weight 0.01 \
      --presence-weight 0.01 \
      --rollout-memory-dim 128 \
      --target-mode full \
      --action-conditioned-memory \
      --one-step-weight 0.5 \
      --sample-prioritized \
      --priority-alpha 0.4 \
      --priority-uniform-mix 0.7 \
      --priority-ema-beta 0.95 \
      --priority-warmup-epochs 2 \
      --priority-score pred_loss \
      "${RESUME[@]}" \
      --device cuda \
      --amp \
      --wandb \
      --wandb-project SMAC-JEPA-losses \
      --wandb-name rnn-seqmem-exp12-action-memory-per-onestep-full-h5-continue
fi

# -------------------------------------------------------------------
# Exp08 curriculum H3 -> H5 -> H10.
# -------------------------------------------------------------------
if done_epoch runs/rnn_seqmem_exp08_curriculum_h3 4; then
  echo "SKIP Exp08 H3: checkpoint_epoch_004.pt exists"
else
  RESUME=($(resume_array_for runs/rnn_seqmem_exp08_curriculum_h3))
  run_cmd 06_exp08_curriculum_h3_continue \
    env LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
      --manifest splits/generated_seed4_mapdims_only.json \
      --split train \
      --out-dir runs/rnn_seqmem_exp08_curriculum_h3 \
      --model-size default \
      --epochs 4 \
      --batch-size 16 \
      --num-workers 4 \
      --rollout-window 20 \
      --rollout-horizon 3 \
      --window-mode random \
      --samples-per-epoch 50000 \
      --enemy-visibility-mask \
      --enemy-sight-range 9.0 \
      --temporal-loss lambda \
      --td-lambda 0.9 \
      --sigreg-weight 0.01 \
      --decoder-weight 0.01 \
      --presence-weight 0.01 \
      --rollout-memory-dim 128 \
      --target-mode full \
      --one-step-weight 0.5 \
      "${RESUME[@]}" \
      --device cuda \
      --amp \
      --wandb \
      --wandb-project SMAC-JEPA-losses \
      --wandb-name rnn-seqmem-exp08-curriculum-h3-continue
fi

if [[ -f runs/rnn_seqmem_exp08_curriculum_h3/checkpoint.pt ]]; then
  if done_epoch runs/rnn_seqmem_exp08_curriculum_h5 7; then
    echo "SKIP Exp08 H5: checkpoint_epoch_007.pt exists"
  else
    if [[ -f runs/rnn_seqmem_exp08_curriculum_h5/checkpoint.pt ]]; then
      RESUME=(--resume runs/rnn_seqmem_exp08_curriculum_h5/checkpoint.pt)
    else
      RESUME=(--resume runs/rnn_seqmem_exp08_curriculum_h3/checkpoint.pt)
    fi

    run_cmd 07_exp08_curriculum_h5_continue \
      env LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
        --manifest splits/generated_seed4_mapdims_only.json \
        --split train \
        --out-dir runs/rnn_seqmem_exp08_curriculum_h5 \
        --model-size default \
        --epochs 7 \
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
        --sigreg-weight 0.01 \
        --decoder-weight 0.01 \
        --presence-weight 0.01 \
        --rollout-memory-dim 128 \
        --target-mode full \
        --one-step-weight 0.5 \
        "${RESUME[@]}" \
        --device cuda \
        --amp \
        --wandb \
        --wandb-project SMAC-JEPA-losses \
        --wandb-name rnn-seqmem-exp08-curriculum-h5-continue
  fi
else
  echo "SKIP Exp08 H5: missing runs/rnn_seqmem_exp08_curriculum_h3/checkpoint.pt"
fi

if [[ -f runs/rnn_seqmem_exp08_curriculum_h5/checkpoint.pt ]]; then
  if done_epoch runs/rnn_seqmem_exp08_curriculum_h10 10; then
    echo "SKIP Exp08 H10: checkpoint_epoch_010.pt exists"
  else
    if [[ -f runs/rnn_seqmem_exp08_curriculum_h10/checkpoint.pt ]]; then
      RESUME=(--resume runs/rnn_seqmem_exp08_curriculum_h10/checkpoint.pt)
    else
      RESUME=(--resume runs/rnn_seqmem_exp08_curriculum_h5/checkpoint.pt)
    fi

    run_cmd 08_exp08_curriculum_h10_continue \
      env LD_LIBRARY_PATH="" python -m smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments \
        --manifest splits/generated_seed4_mapdims_only.json \
        --split train \
        --out-dir runs/rnn_seqmem_exp08_curriculum_h10 \
        --model-size default \
        --epochs 10 \
        --batch-size 8 \
        --num-workers 4 \
        --rollout-window 20 \
        --rollout-horizon 10 \
        --window-mode random \
        --samples-per-epoch 50000 \
        --enemy-visibility-mask \
        --enemy-sight-range 9.0 \
        --temporal-loss lambda \
        --td-lambda 0.9 \
        --sigreg-weight 0.01 \
        --decoder-weight 0.01 \
        --presence-weight 0.01 \
        --rollout-memory-dim 128 \
        --target-mode full \
        --one-step-weight 0.5 \
        "${RESUME[@]}" \
        --device cuda \
        --amp \
        --wandb \
        --wandb-project SMAC-JEPA-losses \
        --wandb-name rnn-seqmem-exp08-curriculum-h10-continue
  fi
else
  echo "SKIP Exp08 H10: missing runs/rnn_seqmem_exp08_curriculum_h5/checkpoint.pt"
fi

# -------------------------------------------------------------------
# Final eval. Only include completed epoch-10 checkpoints.
# -------------------------------------------------------------------
if [[ -f runs/eval_new_candidates_first8000/eval_seqmem_combined_summary.csv ]]; then
  echo "SKIP final eval: summary already exists"
else
  final=()
  [[ -f runs/rnn_seqmem_exp07_onestep_full_h10/checkpoint_epoch_010.pt ]] && final+=(--checkpoint runs/rnn_seqmem_exp07_onestep_full_h10/checkpoint_epoch_010.pt)
  [[ -f runs/rnn_seqmem_exp09_onestep025_full_h5/checkpoint_epoch_010.pt ]] && final+=(--checkpoint runs/rnn_seqmem_exp09_onestep025_full_h5/checkpoint_epoch_010.pt)
  [[ -f runs/rnn_seqmem_exp10_onestep_uniform_full_h5/checkpoint_epoch_010.pt ]] && final+=(--checkpoint runs/rnn_seqmem_exp10_onestep_uniform_full_h5/checkpoint_epoch_010.pt)
  [[ -f runs/rnn_seqmem_exp11_per_onestep_full_h5/checkpoint_epoch_010.pt ]] && final+=(--checkpoint runs/rnn_seqmem_exp11_per_onestep_full_h5/checkpoint_epoch_010.pt)
  [[ -f runs/rnn_seqmem_exp12_action_memory_per_onestep_full_h5/checkpoint_epoch_010.pt ]] && final+=(--checkpoint runs/rnn_seqmem_exp12_action_memory_per_onestep_full_h5/checkpoint_epoch_010.pt)
  [[ -f runs/rnn_seqmem_exp08_curriculum_h10/checkpoint_epoch_010.pt ]] && final+=(--checkpoint runs/rnn_seqmem_exp08_curriculum_h10/checkpoint_epoch_010.pt)

  if [[ ${#final[@]} -gt 0 ]]; then
    run_cmd 09_eval_new_candidates_plus_exp12_first8000 \
      env LD_LIBRARY_PATH="" python eval_rnn_seqmem_combined_metrics.py \
        --manifest splits/generated_seed4_mapdims_only.json \
        --split eval \
        "${final[@]}" \
        --out-dir runs/eval_new_candidates_plus_exp12_first8000 \
        --batch-size 16 \
        --num-workers 4 \
        --max-batches 500 \
        --window-mode sequential \
        --device cuda \
        --amp
  else
    echo "SKIP final eval: no completed candidate epoch-10 checkpoints found"
  fi
fi

echo
echo "============================================================"
echo "Run complete at $(date)"
echo "Logs: ${LOG_DIR}"
echo "Status summary:"
for f in "${LOG_DIR}"/*.status; do
  [[ -e "$f" ]] || continue
  echo "  $(basename "$f" .status): $(cat "$f")"
done
echo "============================================================"
