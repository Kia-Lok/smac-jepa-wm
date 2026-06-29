#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR" || exit 1

TRAIN_MODULE="smac_jepa.train_jepa_weekend_structural"
EVALUATOR="./eval_jepa_weekend_structural.py"
MANIFEST="splits/generated_seed4_mapdims_only.json"
EPOCHS="${EPOCHS:-5}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-50000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-500}"
DIAGNOSTIC_MAX_BATCHES="${DIAGNOSTIC_MAX_BATCHES:-100}"
PROBE_DIR="runs/weekend_structural_probe_decoders"
EVAL_ROOT="runs/weekend_structural_eval"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="runs/weekend_structural_logs_${STAMP}"
STATUS_FILE="$LOG_ROOT/status.tsv"

mkdir -p "$PROBE_DIR" "$EVAL_ROOT" "$LOG_ROOT"
printf "stage\tstatus\ttime\tdetail\n" > "$STATUS_FILE"

record_status() {
  printf "%s\t%s\t%s\t%s\n" \
    "$1" "$2" "$(date --iso-8601=seconds)" "${3:-}" \
    >> "$STATUS_FILE"
}

run_logged() {
  local label="$1"
  shift
  local log="$LOG_ROOT/${label}.log"

  echo
  echo "============================================================"
  echo "START: $label"
  echo "TIME:  $(date)"
  echo "LOG:   $log"
  printf ' %q' "$@"
  echo
  echo "============================================================"

  "$@" 2>&1 | tee "$log"
  local rc="${PIPESTATUS[0]}"
  if [[ "$rc" -eq 0 ]]; then
    record_status "$label" "ok" "$log"
  else
    record_status "$label" "failed:$rc" "$log"
  fi
  return "$rc"
}

checkpoint_completed() {
  local path="$1"
  python - "$path" "$EPOCHS" <<'PY'
import sys, torch
path, required = sys.argv[1], int(sys.argv[2])
try:
    c = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    c = torch.load(path, map_location="cpu")
print(int(bool(c.get("epoch_complete", True)) and int(c.get("epoch", 0)) >= required))
PY
}

latest_checkpoint() {
  local dir="$1"
  python - "$dir" <<'PY'
from pathlib import Path
import sys, torch
d = Path(sys.argv[1])
paths = []
for name in ("checkpoint_recovery.pt", "checkpoint.pt"):
    p = d / name
    if p.is_file():
        paths.append(p)
paths += list(d.glob("checkpoint_epoch_*.pt"))
valid = []
for p in paths:
    try:
        try:
            c = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            c = torch.load(p, map_location="cpu")
        valid.append((
            int(c.get("global_step", -1)),
            int(c.get("epoch", -1)),
            int(bool(c.get("epoch_complete", True))),
            p.stat().st_mtime_ns,
            p,
        ))
    except Exception as exc:
        print(f"ignored unreadable checkpoint {p}: {exc}", file=sys.stderr)
if valid:
    print(max(valid, key=lambda x: x[:-1])[-1])
PY
}

COMMON=(
  --manifest "$MANIFEST"
  --split train
  --model-size default
  --epochs "$EPOCHS"
  --batch-size 16
  --num-workers 4
  --rollout-window 20
  --rollout-horizon 5
  --window-mode random
  --samples-per-epoch "$TRAIN_SAMPLES"
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
  --ema-target-encoder
  --ema-momentum 0.996
  --memory-barlow-scale 0
  --device cuda
  --amp
  --wandb
  --wandb-project SMAC-JEPA-losses
)

eval_checkpoint() {
  local label="$1"
  local checkpoint="$2"
  local out="$EVAL_ROOT/$label"
  rm -rf "$out"
  mkdir -p "$out"

  run_logged "eval_${label}" \
    env LD_LIBRARY_PATH="" \
    python "$EVALUATOR" \
      --manifest "$MANIFEST" \
      --split eval \
      --checkpoint "$checkpoint" \
      --out-dir "$out" \
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

run_experiment() {
  local label="$1"
  local out_dir="$2"
  local seed="$3"
  shift 3
  local extra=("$@")
  local final="$out_dir/checkpoint.pt"

  mkdir -p "$out_dir"

  if [[ -f "$final" ]] && \
     [[ "$(checkpoint_completed "$final")" == "1" ]]; then
    echo "$label already complete; training skipped."
    record_status "train_${label}" "skipped_complete" "$final"
  else
    local resume
    resume="$(latest_checkpoint "$out_dir" || true)"
    local resume_args=()
    if [[ -n "$resume" ]]; then
      echo "$label resuming from $resume"
      resume_args=(--resume "$resume")
    fi

    if ! run_logged "train_${label}" \
      env LD_LIBRARY_PATH="" \
      python -m "$TRAIN_MODULE" \
        "${COMMON[@]}" \
        --seed "$seed" \
        --out-dir "$out_dir" \
        --wandb-name "$label" \
        "${resume_args[@]}" \
        "${extra[@]}"
    then
      echo "WARNING: $label training failed; suite will continue."
      return 1
    fi
  fi

  if [[ ! -f "$final" ]] || \
     [[ "$(checkpoint_completed "$final")" != "1" ]]; then
    record_status "eval_${label}" "skipped_no_final_checkpoint" "$final"
    return 1
  fi

  if ! eval_checkpoint "$label" "$final"; then
    echo "WARNING: $label evaluation failed; suite will continue."
    return 1
  fi
}

# Preflight.
python -m py_compile \
  smac_jepa/train_jepa_weekend_structural.py \
  "$EVALUATOR" || exit 2
python -m "$TRAIN_MODULE" --help >/dev/null || exit 2
python "$EVALUATOR" --help >/dev/null || exit 2

# Tiny smoke covers every new structural path.
SMOKE_DIR="runs/weekend_structural_smoke"
rm -rf "$SMOKE_DIR"
if ! run_logged smoke_structural \
  env LD_LIBRARY_PATH="" \
  python -m "$TRAIN_MODULE" \
    "${COMMON[@]}" \
    --epochs 1 \
    --samples-per-epoch 32 \
    --num-workers 0 \
    --out-dir "$SMOKE_DIR" \
    --no-wandb \
    --residual-state-decoder \
    --delta-loss-weight 0.05 \
    --event-dynamics-weight 1.0 \
    --direct-action-fusion \
    --enemy-observation-dropout 0.20 \
    --hidden-reconstruction-weight 0.05
then
  echo "FATAL: structural smoke training failed. Long runs were not started."
  exit 3
fi

if ! run_logged smoke_eval \
  env LD_LIBRARY_PATH="" \
  python "$EVALUATOR" \
    --manifest "$MANIFEST" \
    --split eval \
    --checkpoint "$SMOKE_DIR/checkpoint.pt" \
    --out-dir "$SMOKE_DIR/eval" \
    --probe-dir "$SMOKE_DIR/probes" \
    --batch-size 2 \
    --num-workers 0 \
    --max-batches 2 \
    --diagnostics \
    --diagnostic-max-batches 1 \
    --eval-rollout-horizon 5 \
    --target-mode full \
    --window-mode sequential \
    --thresholds 0.01 0.05 0.1 \
    --change-threshold 0.01 \
    --event-threshold 0.01 \
    --attack-action-min 6 \
    --no-probe-decoder \
    --no-require-hidden-eval \
    --device cuda \
    --amp
then
  echo "FATAL: structural smoke evaluation failed. Long runs were not started."
  exit 3
fi

# Reuse existing Exp18/19 probes and evaluate those checkpoints first.
OLD_PROBES="runs/eval_exp17_exp19_final_fixed/probe_decoders"
if [[ -d "$OLD_PROBES" ]]; then
  cp -n "$OLD_PROBES"/* "$PROBE_DIR"/ 2>/dev/null || true
fi

for entry in \
  "exp18_existing:runs/rnn_seqmem_exp18_ema_memory_barlow_explicit_visibility_v3/checkpoint.pt" \
  "exp19_existing:runs/rnn_seqmem_exp19_ema_memory_barlow_events_explicit_visibility_v3/checkpoint.pt"
do
  label="${entry%%:*}"
  checkpoint="${entry#*:}"
  if [[ -f "$checkpoint" ]]; then
    eval_checkpoint "$label" "$checkpoint" || true
  else
    record_status "eval_${label}" "missing_checkpoint" "$checkpoint"
  fi
done

# Structural queue.
run_experiment exp20_clean_event \
  runs/rnn_seqmem_exp20_clean_event_no_memory_barlow 1 \
  --event-balanced-sampling \
  --event-fraction 0.30 \
  --event-pool-fraction 0.20 \
  --event-movement-threshold 0.01 \
  --event-state-threshold 0.001 \
  --event-attack-action-min 6 \
  --event-min-transitions 1 || true

run_experiment exp21_residual_decoder \
  runs/rnn_seqmem_exp21_residual_decoder 1 \
  --residual-state-decoder || true

run_experiment exp22_delta_loss \
  runs/rnn_seqmem_exp22_delta_loss 1 \
  --delta-loss-weight 0.05 || true

run_experiment exp23_residual_delta \
  runs/rnn_seqmem_exp23_residual_delta 1 \
  --residual-state-decoder \
  --delta-loss-weight 0.05 || true

run_experiment exp26_event_weighted_dynamics \
  runs/rnn_seqmem_exp26_event_weighted_dynamics 1 \
  --event-dynamics-weight 1.0 \
  --event-dynamics-threshold 0.01 || true

run_experiment exp27_residual_delta_event \
  runs/rnn_seqmem_exp27_residual_delta_event 1 \
  --residual-state-decoder \
  --delta-loss-weight 0.05 \
  --event-dynamics-weight 1.0 \
  --event-dynamics-threshold 0.01 || true

run_experiment exp28_direct_action_fusion \
  runs/rnn_seqmem_exp28_direct_action_fusion 1 \
  --direct-action-fusion \
  --direct-action-hidden-dim 256 || true

run_experiment exp29_combined_transition \
  runs/rnn_seqmem_exp29_combined_transition 1 \
  --residual-state-decoder \
  --delta-loss-weight 0.05 \
  --event-dynamics-weight 1.0 \
  --event-dynamics-threshold 0.01 \
  --direct-action-fusion \
  --direct-action-hidden-dim 256 || true

run_experiment exp24_observation_dropout \
  runs/rnn_seqmem_exp24_observation_dropout 1 \
  --enemy-observation-dropout 0.20 || true

run_experiment exp25_hidden_reconstruction \
  runs/rnn_seqmem_exp25_hidden_reconstruction 1 \
  --enemy-observation-dropout 0.20 \
  --hidden-reconstruction-weight 0.05 || true

# Fixed seed-2 replication of the planned combined transition model.
run_experiment exp30_combined_transition_seed2 \
  runs/rnn_seqmem_exp30_combined_transition_seed2 2 \
  --residual-state-decoder \
  --delta-loss-weight 0.05 \
  --event-dynamics-weight 1.0 \
  --event-dynamics-threshold 0.01 \
  --direct-action-fusion \
  --direct-action-hidden-dim 256 || true

echo
echo "============================================================"
echo "WEEKEND SUITE FINISHED"
echo "Status:      $STATUS_FILE"
echo "Logs:        $LOG_ROOT"
echo "Evaluations: $EVAL_ROOT"
echo "Probes:      $PROBE_DIR"
echo "============================================================"
