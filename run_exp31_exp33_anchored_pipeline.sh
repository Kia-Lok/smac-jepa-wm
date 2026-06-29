#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR" || exit 1

TRAIN_MODULE="smac_jepa.train_jepa_exp31_exp33_anchored"
STANDARD_EVALUATOR="./eval_jepa_exp31_exp33_anchored.py"
HIDDEN_EVALUATOR="./eval_jepa_hidden_belief_exp31_exp33.py"
MANIFEST="${MANIFEST:-splits/generated_seed4_mapdims_only.json}"

BASE_EPOCHS="${BASE_EPOCHS:-5}"
BASE_TRAIN_SAMPLES="${BASE_TRAIN_SAMPLES:-50000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-500}"
DIAGNOSTIC_MAX_BATCHES="${DIAGNOSTIC_MAX_BATCHES:-100}"
HIDDEN_MAX_BATCHES="${HIDDEN_MAX_BATCHES:-150}"
HIDDEN_TARGET_ENTITY_TIMES="${HIDDEN_TARGET_ENTITY_TIMES:-10000}"
SKIP_SMOKE_TRAIN="${SKIP_SMOKE_TRAIN:-0}"

EXP31_DIR="${EXP31_DIR:-runs/rnn_seqmem_exp31_delta_hidden_combo_v2}"
EXP32_DIR="${EXP32_DIR:-runs/rnn_seqmem_exp32_contiguous_belief_v2}"
EXP33_DIR="${EXP33_DIR:-runs/rnn_seqmem_exp33_anchored_belief_v1}"
EVAL_ROOT="${EVAL_ROOT:-runs/exp31_exp33_anchored_eval_v1}"
HIDDEN_ROOT="${HIDDEN_ROOT:-runs/exp31_exp33_anchored_hidden_eval_v1}"
PROBE_DIR="${PROBE_DIR:-runs/exp31_exp33_anchored_probe_decoders_v1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="runs/exp31_exp33_anchored_logs_${STAMP}"
STATUS_FILE="$LOG_ROOT/status.tsv"

mkdir -p "$EVAL_ROOT" "$HIDDEN_ROOT" "$PROBE_DIR" "$LOG_ROOT"
printf "stage\tstatus\ttime\tdetail\n" > "$STATUS_FILE"

record_status() {
  printf "%s\t%s\t%s\t%s\n" \
    "$1" "$2" "$(date --iso-8601=seconds)" "${3:-}" >> "$STATUS_FILE"
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
  local required="$2"
  python - "$path" "$required" <<'PY'
import sys, torch
path, required = sys.argv[1], int(sys.argv[2])
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")
print(int(
    bool(checkpoint.get("epoch_complete", True))
    and int(checkpoint.get("epoch", 0)) >= required
))
PY
}

latest_checkpoint() {
  local directory="$1"
  python - "$directory" <<'PY'
from pathlib import Path
import sys, torch
root = Path(sys.argv[1])
best = None
for path in root.glob("checkpoint*.pt"):
    try:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        if "model_state" not in ckpt or "memory_module_state" not in ckpt:
            continue
        score = (
            int(ckpt.get("global_step", -1)),
            int(ckpt.get("epoch", -1)),
            int(ckpt.get("batches_completed_in_epoch", -1)),
            path.stat().st_mtime_ns,
        )
        if best is None or score > best[0]:
            best = (score, path)
    except Exception:
        continue
if best is not None:
    print(best[1])
PY
}

COMMON=(
  --manifest "$MANIFEST"
  --split train
  --model-size default
  --batch-size 16
  --rollout-window 20
  --window-mode random
  --enemy-visibility-mask
  --enemy-sight-range 9.0
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

EXP31_FLAGS=(
  --delta-loss-weight 0.05
  --occlusion-mode independent
  --enemy-observation-dropout 0.20
  --hidden-reconstruction-weight 0.05
)

EXP32_FLAGS=(
  --delta-loss-weight 0.05
  --occlusion-mode contiguous
  --contiguous-occlusion-spans 1 3 5
  --occlusion-spans-per-sample 2
  --hidden-reconstruction-weight 0.05
  --last-seen-anchor-weight 0.05
  --last-seen-change-threshold 0.01
  --hidden-presence-weight 0.02
  --reappearance-consistency-weight 0.02
)

run_training() {
  local label="$1"
  local out_dir="$2"
  local memory_dim="$3"
  local anchored="$4"
  shift 4
  local extra=("$@")
  local final="$out_dir/checkpoint.pt"

  mkdir -p "$out_dir"
  if [[ -f "$final" ]] && \
     [[ "$(checkpoint_completed "$final" "$BASE_EPOCHS")" == "1" ]]; then
    echo "$label already complete; training skipped."
    record_status "train_${label}" "skipped_complete" "$final"
    return 0
  fi

  local latest
  latest="$(latest_checkpoint "$out_dir" || true)"
  local resume_args=()
  if [[ -n "$latest" ]]; then
    echo "$label resuming from $latest"
    resume_args=(--resume "$latest")
  fi

  local env_args=(env LD_LIBRARY_PATH="")
  if [[ "$anchored" == "1" ]]; then
    env_args+=(
      SMAC_JEPA_ANCHORED_MEMORY=1
      SMAC_JEPA_ANCHOR_GATE_INIT=-3.0
      SMAC_JEPA_ANCHOR_DELTA_SCALE=0.25
      SMAC_JEPA_ANCHOR_HIDDEN_CORRECTION_SCALE=0.10
      SMAC_JEPA_ANCHOR_GATE_SPARSITY_WEIGHT=0.002
    )
  fi

  run_logged "train_${label}" \
    "${env_args[@]}" \
    python -m "$TRAIN_MODULE" \
      "${COMMON[@]}" \
      --epochs "$BASE_EPOCHS" \
      --samples-per-epoch "$BASE_TRAIN_SAMPLES" \
      --rollout-horizon 5 \
      --rollout-memory-dim "$memory_dim" \
      --seed 1 \
      --out-dir "$out_dir" \
      --wandb-name "$label" \
      "${resume_args[@]}" \
      "${extra[@]}"
}

eval_standard() {
  local label="$1"
  local checkpoint="$2"
  local out="$EVAL_ROOT/$label"
  rm -rf "$out"
  mkdir -p "$out"
  run_logged "eval_standard_${label}" \
    env LD_LIBRARY_PATH="" \
    python "$STANDARD_EVALUATOR" \
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

eval_hidden() {
  local label="$1"
  local checkpoint="$2"
  local gate_zero="${3:-0}"
  local out="$HIDDEN_ROOT/$label"
  rm -rf "$out"
  mkdir -p "$out"
  local env_args=(env LD_LIBRARY_PATH="")
  if [[ "$gate_zero" == "1" ]]; then
    env_args+=(SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO=1)
  fi
  run_logged "eval_hidden_${label}" \
    "${env_args[@]}" \
    python "$HIDDEN_EVALUATOR" \
      --manifest "$MANIFEST" \
      --split eval \
      --checkpoint "$checkpoint" \
      --out-dir "$out" \
      --probe-dir "$PROBE_DIR" \
      --batch-size 16 \
      --num-workers 4 \
      --eval-rollout-horizon 5 \
      --target-mode full \
      --thresholds 0.01 0.05 0.1 \
      --presence-threshold 0.5 \
      --no-natural-hidden-eval \
      --controlled-occlusion-eval \
      --controlled-occlusion-max-batches "$HIDDEN_MAX_BATCHES" \
      --controlled-occlusion-target-entity-times "$HIDDEN_TARGET_ENTITY_TIMES" \
      --controlled-occlusion-spans 1 3 5 \
      --controlled-occlusion-seed 123 \
      --controlled-prefer-reappearance \
      --device cuda \
      --amp
}

for required in \
  "smac_jepa/train_jepa_exp31_exp33.py" \
  "smac_jepa/train_jepa_exp31_exp33_anchored.py" \
  "smac_jepa/anchored_belief_memory.py" \
  "./eval_jepa_exp31_exp33.py" \
  "$STANDARD_EVALUATOR" \
  "$HIDDEN_EVALUATOR" \
  "./self_test_exp31_exp33_anchored.py"
do
  if [[ ! -f "$required" ]]; then
    echo "FATAL: missing $required"
    exit 2
  fi
done

python -m py_compile \
  smac_jepa/anchored_belief_memory.py \
  smac_jepa/train_jepa_exp31_exp33.py \
  smac_jepa/train_jepa_exp31_exp33_anchored.py \
  ./eval_jepa_exp31_exp33.py \
  "$STANDARD_EVALUATOR" \
  "$HIDDEN_EVALUATOR" \
  ./self_test_exp31_exp33_anchored.py || exit 2
python -m "$TRAIN_MODULE" --help >/dev/null || exit 2
python "$STANDARD_EVALUATOR" --help >/dev/null || exit 2
python "$HIDDEN_EVALUATOR" --help >/dev/null || exit 2
python ./self_test_exp31_exp33_anchored.py || exit 2

if [[ "$SKIP_SMOKE_TRAIN" != "1" ]]; then
  SMOKE="runs/exp33_anchored_smoke_v1"
  rm -rf "$SMOKE"
  if ! run_logged smoke_exp33_anchored \
    env LD_LIBRARY_PATH="" \
      SMAC_JEPA_ANCHORED_MEMORY=1 \
      SMAC_JEPA_ANCHOR_GATE_INIT=-3.0 \
      SMAC_JEPA_ANCHOR_DELTA_SCALE=0.25 \
      SMAC_JEPA_ANCHOR_HIDDEN_CORRECTION_SCALE=0.10 \
      SMAC_JEPA_ANCHOR_GATE_SPARSITY_WEIGHT=0.002 \
    python -m "$TRAIN_MODULE" \
      "${COMMON[@]}" \
      --epochs 1 \
      --samples-per-epoch 64 \
      --num-workers 0 \
      --rollout-horizon 5 \
      --rollout-memory-dim 322 \
      --seed 123 \
      --out-dir "$SMOKE" \
      --wandb-name smoke_exp33_anchored \
      --no-wandb \
      "${EXP32_FLAGS[@]}"
  then
    echo "FATAL: Exp33 anchored smoke training failed."
    exit 3
  fi

  if ! run_logged smoke_exp33_eval \
    env LD_LIBRARY_PATH="" \
    python "$STANDARD_EVALUATOR" \
      --manifest "$MANIFEST" \
      --split eval \
      --checkpoint "$SMOKE/checkpoint.pt" \
      --out-dir "$SMOKE/eval" \
      --probe-dir "$SMOKE/probes" \
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
    echo "FATAL: Exp33 anchored checkpoint/evaluator smoke failed."
    exit 3
  fi
fi

run_training exp31_delta_hidden_combo "$EXP31_DIR" 128 0 \
  --temporal-loss lambda --td-lambda 0.9 \
  "${EXP31_FLAGS[@]}" || exit 4
eval_standard exp31 "$EXP31_DIR/checkpoint.pt" || exit 4
eval_hidden exp31 "$EXP31_DIR/checkpoint.pt" || exit 4

run_training exp32_contiguous_belief "$EXP32_DIR" 128 0 \
  --temporal-loss lambda --td-lambda 0.9 \
  "${EXP32_FLAGS[@]}" || exit 5
eval_standard exp32 "$EXP32_DIR/checkpoint.pt" || exit 5
eval_hidden exp32 "$EXP32_DIR/checkpoint.pt" || exit 5

# Exp33 is a standalone world-model experiment. It starts from scratch because
# the memory state layout is deliberately different. No reward/continuation,
# inverse-dynamics, or H10/H15 curriculum is used.
run_training exp33_anchored_belief "$EXP33_DIR" 322 1 \
  --temporal-loss lambda --td-lambda 0.9 \
  "${EXP32_FLAGS[@]}" || exit 6
eval_standard exp33 "$EXP33_DIR/checkpoint.pt" || exit 6
eval_hidden exp33 "$EXP33_DIR/checkpoint.pt" 0 || exit 6

# Evaluation-only ablation: force the hidden anchor gate to zero and suppress
# the hidden recurrent correction. This isolates the learned hidden-anchor update. The hidden evaluator
# still reports its separate pure last-seen persistence baseline.
eval_hidden exp33_gate_zero "$EXP33_DIR/checkpoint.pt" 1 || exit 6

record_status pipeline complete "$EXP33_DIR/checkpoint.pt"
echo
echo "============================================================"
echo "EXP31--EXP33 ANCHORED PIPELINE COMPLETE"
echo "Status:      $STATUS_FILE"
echo "Exp31:       $EXP31_DIR/checkpoint.pt"
echo "Exp32:       $EXP32_DIR/checkpoint.pt"
echo "Exp33:       $EXP33_DIR/checkpoint.pt"
echo "Evaluations: $EVAL_ROOT and $HIDDEN_ROOT"
echo "============================================================"
