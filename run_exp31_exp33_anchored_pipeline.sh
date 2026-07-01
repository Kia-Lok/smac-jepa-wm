#!/usr/bin/env bash
# Exp31--Exp33 training and evaluation pipeline with full branch preflight,
# checkpoint-kind validation, safe evaluation retries, and non-destructive
# continuation after evaluation-only failures.
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jovyan/smac-jepa-wm}"
cd "$ROOT_DIR" || { echo "FATAL: cannot enter $ROOT_DIR"; exit 1; }

TRAIN_MODULE="smac_jepa.train_jepa_exp31_exp33_anchored"
BASE_STANDARD_EVALUATOR="./eval_jepa_exp31_exp33.py"
ANCHORED_STANDARD_EVALUATOR="./eval_jepa_exp31_exp33_anchored.py"
HIDDEN_EVALUATOR="./eval_jepa_hidden_belief_exp31_exp33.py"
MANIFEST="${MANIFEST:-splits/generated_seed4_mapdims_only.json}"

BASE_EPOCHS="${BASE_EPOCHS:-5}"
BASE_TRAIN_SAMPLES="${BASE_TRAIN_SAMPLES:-50000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-500}"
DIAGNOSTIC_MAX_BATCHES="${DIAGNOSTIC_MAX_BATCHES:-100}"
HIDDEN_MAX_BATCHES="${HIDDEN_MAX_BATCHES:-150}"
HIDDEN_TARGET_ENTITY_TIMES="${HIDDEN_TARGET_ENTITY_TIMES:-10000}"
SKIP_SMOKE_TRAIN="${SKIP_SMOKE_TRAIN:-0}"
RUN_PREFLIGHT_ONLY="${RUN_PREFLIGHT_ONLY:-0}"

EXP31_DIR="${EXP31_DIR:-runs/rnn_seqmem_exp31_delta_hidden_combo_v2}"
EXP32_DIR="${EXP32_DIR:-runs/rnn_seqmem_exp32_contiguous_belief_v2}"
EXP33_DIR="${EXP33_DIR:-runs/rnn_seqmem_exp33_anchored_belief_v1}"
EVAL_ROOT="${EVAL_ROOT:-runs/exp31_exp33_anchored_eval_v1}"
HIDDEN_ROOT="${HIDDEN_ROOT:-runs/exp31_exp33_anchored_hidden_eval_v1}"
PROBE_DIR="${PROBE_DIR:-runs/exp31_exp33_anchored_probe_decoders_v1}"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-runs/exp31_exp33_verified_preflight_v2}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="runs/exp31_exp33_anchored_logs_${STAMP}"
STATUS_FILE="$LOG_ROOT/status.tsv"
PIPELINE_WARNINGS=0

mkdir -p "$EVAL_ROOT" "$HIDDEN_ROOT" "$PROBE_DIR" "$LOG_ROOT" "$PREFLIGHT_ROOT"
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

checkpoint_kind() {
  local path="$1"
  python - "$path" <<'PY'
import sys, torch
path = sys.argv[1]
try:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("resolved_config", ckpt.get("config", {}))
    state = ckpt.get("memory_module_state", {})
    anchored = bool(cfg.get("anchored_belief_memory", False)) or any(
        str(key).startswith("hidden_gate_net.") for key in state
    )
    print("anchored" if anchored else "normal")
except Exception as exc:
    print(f"invalid:{type(exc).__name__}:{exc}")
PY
}

checkpoint_completed() {
  local path="$1"
  local required="$2"
  local expected_kind="$3"
  python - "$path" "$required" "$expected_kind" <<'PY'
import sys, torch
path, required, expected = sys.argv[1], int(sys.argv[2]), sys.argv[3]
try:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("resolved_config", ckpt.get("config", {}))
    state = ckpt.get("memory_module_state", {})
    anchored = bool(cfg.get("anchored_belief_memory", False)) or any(
        str(key).startswith("hidden_gate_net.") for key in state
    )
    kind = "anchored" if anchored else "normal"
    complete = (
        kind == expected
        and bool(ckpt.get("epoch_complete", True))
        and int(ckpt.get("epoch", 0)) >= required
        and "model_state" in ckpt
        and "memory_module_state" in ckpt
    )
    print(int(complete))
except Exception:
    print(0)
PY
}

checkpoint_directory_kind_audit() {
  local directory="$1"
  local expected_kind="$2"
  python - "$directory" "$expected_kind" <<'PY'
from pathlib import Path
import sys, torch
root, expected = Path(sys.argv[1]), sys.argv[2]
wrong = []
for path in sorted(root.glob("checkpoint*.pt")):
    try:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        if "model_state" not in ckpt or "memory_module_state" not in ckpt:
            continue
        cfg = ckpt.get("resolved_config", ckpt.get("config", {}))
        state = ckpt.get("memory_module_state", {})
        anchored = bool(cfg.get("anchored_belief_memory", False)) or any(
            str(key).startswith("hidden_gate_net.") for key in state
        )
        kind = "anchored" if anchored else "normal"
        if kind != expected:
            wrong.append(f"{path}:{kind}")
    except Exception:
        # Corrupt/partial files are ignored by resume selection and do not
        # define the directory architecture.
        pass
if wrong:
    print("\n".join(wrong))
    raise SystemExit(1)
PY
}

latest_checkpoint() {
  local directory="$1"
  local expected_kind="$2"
  python - "$directory" "$expected_kind" <<'PY'
from pathlib import Path
import sys, torch
root, expected = Path(sys.argv[1]), sys.argv[2]
best = None
for path in root.glob("checkpoint*.pt"):
    try:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        if "model_state" not in ckpt or "memory_module_state" not in ckpt:
            continue
        cfg = ckpt.get("resolved_config", ckpt.get("config", {}))
        state = ckpt.get("memory_module_state", {})
        anchored = bool(cfg.get("anchored_belief_memory", False)) or any(
            str(key).startswith("hidden_gate_net.") for key in state
        )
        kind = "anchored" if anchored else "normal"
        if kind != expected:
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

probe_path_for_checkpoint() {
  local checkpoint="$1"
  local parent stem
  parent="$(basename "$(dirname "$checkpoint")")"
  stem="$(basename "$checkpoint" .pt)"
  printf "%s/%s_%s_meaningful_features_v2_probe_decoder.pt" \
    "$PROBE_DIR" "$parent" "$stem"
}

verify_standard_output() {
  local out="$1"
  local checkpoint="$2"
  local probe json_file
  probe="$(probe_path_for_checkpoint "$checkpoint")"
  json_file="$(find "$out" -maxdepth 1 -type f -name 'eval_*checkpoint.json' -size +0c -print -quit)"
  [[ -n "$json_file" ]] || {
    echo "ERROR: standard evaluator returned success but wrote no checkpoint JSON in $out"
    return 90
  }
  [[ -s "$probe" ]] || {
    echo "ERROR: standard evaluator returned success but probe is missing: $probe"
    return 91
  }
}

verify_hidden_output() {
  local out="$1"
  local require_scored="${2:-0}"
  local json_file
  json_file="$(find "$out" -maxdepth 1 -type f -name 'hidden_belief_*checkpoint.json' -size +0c -print -quit)"
  [[ -n "$json_file" ]] || {
    echo "ERROR: hidden evaluator returned success but wrote no checkpoint JSON in $out"
    return 92
  }
  [[ -s "$out/hidden_belief_summary.csv" ]] || {
    echo "ERROR: hidden evaluator summary CSV is missing in $out"
    return 93
  }
  if [[ "$require_scored" == "1" ]]; then
    python - "$json_file" <<'PYJSON' || return 94
import json, sys
row = json.load(open(sys.argv[1]))
if int(row.get("controlled_evaluated_batches", 0)) < 1:
    raise SystemExit("controlled hidden smoke did not evaluate any batch")
if int(row.get("controlled_selected_spans", 0)) < 1:
    raise SystemExit("controlled hidden smoke selected no occlusion span")
if int(row.get("controlled_scored_entity_times", 0)) < 1:
    raise SystemExit("controlled hidden smoke scored no hidden entity-time")
PYJSON
  fi
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
  local expected_kind="normal"
  [[ "$anchored" == "1" ]] && expected_kind="anchored"
  local final="$out_dir/checkpoint.pt"

  mkdir -p "$out_dir"
  if ! checkpoint_directory_kind_audit "$out_dir" "$expected_kind"; then
    echo "FATAL: $out_dir contains checkpoints from the wrong memory architecture."
    echo "Move or rename that directory before continuing. No files were overwritten."
    return 12
  fi

  if [[ -f "$final" ]] && \
     [[ "$(checkpoint_completed "$final" "$BASE_EPOCHS" "$expected_kind")" == "1" ]]; then
    echo "$label already complete and architecture-verified; training skipped."
    record_status "train_${label}" "skipped_complete" "$final"
    return 0
  fi

  if [[ -f "$final" ]]; then
    local kind
    kind="$(checkpoint_kind "$final")"
    if [[ "$kind" != "$expected_kind" && "$kind" != invalid:* ]]; then
      echo "FATAL: $final is $kind but $expected_kind was expected."
      return 13
    fi
  fi

  local latest
  latest="$(latest_checkpoint "$out_dir" "$expected_kind" || true)"
  local resume_args=()
  if [[ -n "$latest" ]]; then
    echo "$label resuming from architecture-verified checkpoint $latest"
    resume_args=(--resume "$latest")
  fi

  local env_args=(env LD_LIBRARY_PATH="" SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO=0)
  if [[ "$anchored" == "1" ]]; then
    env_args+=(
      SMAC_JEPA_ANCHORED_MEMORY=1
      SMAC_JEPA_ANCHOR_GATE_INIT=-3.0
      SMAC_JEPA_ANCHOR_DELTA_SCALE=0.25
      SMAC_JEPA_ANCHOR_HIDDEN_CORRECTION_SCALE=0.10
      SMAC_JEPA_ANCHOR_GATE_SPARSITY_WEIGHT=0.002
    )
  else
    env_args+=(SMAC_JEPA_ANCHORED_MEMORY=0)
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
      "${extra[@]}" || return $?

  [[ -f "$final" ]] || {
    echo "FATAL: training command succeeded but did not write $final"
    return 14
  }
  local final_kind
  final_kind="$(checkpoint_kind "$final")"
  [[ "$final_kind" == "$expected_kind" ]] || {
    echo "FATAL: produced checkpoint kind=$final_kind expected=$expected_kind"
    return 15
  }
}

eval_standard_once() {
  local label="$1"
  local checkpoint="$2"
  local anchored="$3"
  local safe_mode="$4"
  local evaluator="$BASE_STANDARD_EVALUATOR"
  [[ "$anchored" == "1" ]] && evaluator="$ANCHORED_STANDARD_EVALUATOR"
  local out="$EVAL_ROOT/$label"
  local workers=4
  local amp_flag="--amp"
  local force_probe=()
  local suffix="primary"
  if [[ "$safe_mode" == "1" ]]; then
    workers=0
    amp_flag="--no-amp"
    force_probe=(--force-retrain-probe)
    suffix="safe_retry"
  fi
  rm -rf "$out"
  mkdir -p "$out"
  run_logged "eval_standard_${label}_${suffix}" \
    env LD_LIBRARY_PATH="" SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO=0 \
    python "$evaluator" \
      --manifest "$MANIFEST" \
      --split eval \
      --checkpoint "$checkpoint" \
      --out-dir "$out" \
      --probe-dir "$PROBE_DIR" \
      --batch-size 16 \
      --num-workers "$workers" \
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
      "${force_probe[@]}" \
      --no-require-hidden-eval \
      --device cuda \
      "$amp_flag" || return $?
  verify_standard_output "$out" "$checkpoint"
}

eval_standard() {
  local label="$1"
  local checkpoint="$2"
  local anchored="${3:-0}"
  if eval_standard_once "$label" "$checkpoint" "$anchored" 0; then
    return 0
  fi
  echo "WARNING: primary standard evaluation failed for $label."
  echo "Retrying once with num_workers=0, FP32, and a fresh probe."
  record_status "eval_standard_${label}" "retrying_safe_mode" "workers=0,no_amp,force_probe"
  eval_standard_once "$label" "$checkpoint" "$anchored" 1
}

eval_hidden_once() {
  local label="$1"
  local checkpoint="$2"
  local gate_zero="$3"
  local safe_mode="$4"
  local out="$HIDDEN_ROOT/$label"
  local workers=4
  local amp_flag="--amp"
  local suffix="primary"
  [[ "$safe_mode" == "1" ]] && { workers=0; amp_flag="--no-amp"; suffix="safe_retry"; }
  rm -rf "$out"
  mkdir -p "$out"
  local env_args=(env LD_LIBRARY_PATH="" SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO=0)
  [[ "$gate_zero" == "1" ]] && env_args=(env LD_LIBRARY_PATH="" SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO=1)
  run_logged "eval_hidden_${label}_${suffix}" \
    "${env_args[@]}" \
    python "$HIDDEN_EVALUATOR" \
      --manifest "$MANIFEST" \
      --split eval \
      --checkpoint "$checkpoint" \
      --out-dir "$out" \
      --probe-dir "$PROBE_DIR" \
      --batch-size 16 \
      --num-workers "$workers" \
      --eval-rollout-horizon 5 \
      --target-mode full \
      --thresholds 0.01 0.05 0.1 \
      --presence-threshold 0.5 \
      --change-threshold 0.01 \
      --no-natural-hidden-eval \
      --controlled-occlusion-eval \
      --controlled-occlusion-max-batches "$HIDDEN_MAX_BATCHES" \
      --controlled-occlusion-target-entity-times "$HIDDEN_TARGET_ENTITY_TIMES" \
      --controlled-occlusion-spans 1 3 5 \
      --controlled-occlusion-seed 123 \
      --controlled-prefer-reappearance \
      --device cuda \
      "$amp_flag" || return $?
  verify_hidden_output "$out"
}

eval_hidden() {
  local label="$1"
  local checkpoint="$2"
  local gate_zero="${3:-0}"
  if eval_hidden_once "$label" "$checkpoint" "$gate_zero" 0; then
    return 0
  fi
  echo "WARNING: primary hidden evaluation failed for $label."
  echo "Retrying once with num_workers=0 and FP32."
  record_status "eval_hidden_${label}" "retrying_safe_mode" "workers=0,no_amp"
  eval_hidden_once "$label" "$checkpoint" "$gate_zero" 1
}

required_files=(
  "smac_jepa/train_jepa_exp31_exp33.py"
  "smac_jepa/train_jepa_exp31_exp33_anchored.py"
  "smac_jepa/anchored_belief_memory.py"
  "$BASE_STANDARD_EVALUATOR"
  "$ANCHORED_STANDARD_EVALUATOR"
  "$HIDDEN_EVALUATOR"
  "./self_test_exp31_exp33_anchored.py"
  "$MANIFEST"
)
for required in "${required_files[@]}"; do
  if [[ ! -f "$required" ]]; then
    echo "FATAL: missing $required"
    exit 2
  fi
done

# Static syntax and import/API checks. These fail before any long training.
python -m py_compile \
  smac_jepa/anchored_belief_memory.py \
  smac_jepa/train_jepa_exp31_exp33.py \
  smac_jepa/train_jepa_exp31_exp33_anchored.py \
  "$BASE_STANDARD_EVALUATOR" \
  "$ANCHORED_STANDARD_EVALUATOR" \
  "$HIDDEN_EVALUATOR" \
  ./self_test_exp31_exp33_anchored.py || exit 2

python - <<'PY' || exit 2
import torch
import smac_jepa.train_jepa_exp31_exp33 as train_base
import eval_jepa_exp31_exp33 as eval_base

if "Compatibility wrapper for Exp31/32" in (train_base.__doc__ or ""):
    raise SystemExit("Base trainer is still the old wrapper; restore the full trainer.")
if "Compatibility wrapper for Exp31/32" in (eval_base.__doc__ or ""):
    raise SystemExit("Base evaluator is still the old wrapper; restore the full evaluator.")
required_train = ["main", "parse_args", "markov_rollout_rnn_losses", "ActionConditionedEntityRolloutGRUMemory"]
required_eval = ["main", "build_memory_module", "rollout_outputs", "decode_with_probe", "build_rollout_feature_masks"]
missing = [f"trainer.{x}" for x in required_train if not hasattr(train_base, x)]
missing += [f"evaluator.{x}" for x in required_eval if not hasattr(eval_base, x)]
if missing:
    raise SystemExit("Missing base APIs: " + ", ".join(missing))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available, but this pipeline is configured for --device cuda.")
print("base_module_and_cuda_preflight_passed")
PY

python -m "$TRAIN_MODULE" --help >/dev/null || exit 2
python "$BASE_STANDARD_EVALUATOR" --help >/dev/null || exit 2
python "$ANCHORED_STANDARD_EVALUATOR" --help >/dev/null || exit 2
python "$HIDDEN_EVALUATOR" --help >/dev/null || exit 2
python ./self_test_exp31_exp33_anchored.py || exit 2
record_status preflight_static ok "syntax, imports, APIs, recursion regression"

# Fingerprint all executable code plus the two full base scripts. A successful
# smoke marker is reused only while every relevant file is byte-identical.
PREFLIGHT_FINGERPRINT="$({ sha256sum \
  smac_jepa/train_jepa_exp31_exp33.py \
  smac_jepa/train_jepa_exp31_exp33_anchored.py \
  smac_jepa/anchored_belief_memory.py \
  "$BASE_STANDARD_EVALUATOR" \
  "$ANCHORED_STANDARD_EVALUATOR" \
  "$HIDDEN_EVALUATOR" \
  ./self_test_exp31_exp33_anchored.py; } | sha256sum | awk '{print $1}')"
PREFLIGHT_MARKER="$PREFLIGHT_ROOT/verified.sha256"

run_full_smoke_preflight() {
  local smoke_normal="$PREFLIGHT_ROOT/normal"
  local smoke_anchor="$PREFLIGHT_ROOT/anchored"
  local smoke_probe="$PREFLIGHT_ROOT/probes"
  rm -rf "$smoke_normal" "$smoke_anchor" "$smoke_probe"
  mkdir -p "$smoke_probe"

  # Normal branch: training wrapper disabled -> base trainer -> base evaluator
  # -> probe creation -> hidden evaluator ordinary dispatch.
  run_logged smoke_normal_train \
    env LD_LIBRARY_PATH="" SMAC_JEPA_ANCHORED_MEMORY=0 SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO=0 \
    python -m "$TRAIN_MODULE" \
      "${COMMON[@]}" \
      --epochs 1 --samples-per-epoch 64 --num-workers 0 \
      --rollout-horizon 5 --rollout-memory-dim 128 --seed 121 \
      --out-dir "$smoke_normal" --wandb-name smoke_normal --no-wandb \
      --temporal-loss lambda --td-lambda 0.9 \
      "${EXP31_FLAGS[@]}" || return 1
  [[ "$(checkpoint_kind "$smoke_normal/checkpoint.pt")" == "normal" ]] || {
    echo "FATAL: normal smoke produced the wrong checkpoint architecture"; return 1;
  }

  run_logged smoke_normal_standard_probe \
    env LD_LIBRARY_PATH="" SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO=0 \
    python "$BASE_STANDARD_EVALUATOR" \
      --manifest "$MANIFEST" --split eval \
      --checkpoint "$smoke_normal/checkpoint.pt" \
      --out-dir "$smoke_normal/eval" --probe-dir "$smoke_probe" \
      --batch-size 2 --num-workers 0 --max-batches 2 \
      --diagnostics --diagnostic-max-batches 1 \
      --eval-rollout-horizon 5 --target-mode full --window-mode sequential \
      --thresholds 0.01 0.05 0.1 --change-threshold 0.01 \
      --event-threshold 0.01 --attack-action-min 6 \
      --probe-decoder --probe-train-split train --probe-epochs 1 \
      --probe-max-batches-per-epoch 2 --probe-samples-per-epoch 64 \
      --probe-lr 0.001 --probe-weight-decay 0.00001 --probe-seed 123 \
      --force-retrain-probe --no-require-hidden-eval --device cuda --no-amp \
      || return 1
  PROBE_DIR="$smoke_probe" verify_standard_output "$smoke_normal/eval" "$smoke_normal/checkpoint.pt" || return 1

  run_logged smoke_normal_hidden \
    env LD_LIBRARY_PATH="" SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO=0 \
    python "$HIDDEN_EVALUATOR" \
      --manifest "$MANIFEST" --split eval \
      --checkpoint "$smoke_normal/checkpoint.pt" \
      --out-dir "$smoke_normal/hidden" --probe-dir "$smoke_probe" \
      --batch-size 2 --num-workers 0 --eval-rollout-horizon 5 \
      --target-mode full --thresholds 0.01 0.05 0.1 \
      --presence-threshold 0.5 --change-threshold 0.01 \
      --no-natural-hidden-eval --controlled-occlusion-eval \
      --controlled-occlusion-max-batches 10 \
      --controlled-occlusion-target-entity-times 1 \
      --controlled-occlusion-spans 1 3 5 --controlled-occlusion-seed 123 \
      --controlled-prefer-reappearance --device cuda --no-amp || return 1
  verify_hidden_output "$smoke_normal/hidden" 1 || return 1

  # Anchored branch: architecture construction, strict checkpoint reload,
  # anchored standard evaluator, hidden evaluator, and gate-zero ablation.
  run_logged smoke_anchored_train \
    env LD_LIBRARY_PATH="" SMAC_JEPA_ANCHORED_MEMORY=1 \
      SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO=0 \
      SMAC_JEPA_ANCHOR_GATE_INIT=-3.0 \
      SMAC_JEPA_ANCHOR_DELTA_SCALE=0.25 \
      SMAC_JEPA_ANCHOR_HIDDEN_CORRECTION_SCALE=0.10 \
      SMAC_JEPA_ANCHOR_GATE_SPARSITY_WEIGHT=0.002 \
    python -m "$TRAIN_MODULE" \
      "${COMMON[@]}" \
      --epochs 1 --samples-per-epoch 64 --num-workers 0 \
      --rollout-horizon 5 --rollout-memory-dim 322 --seed 123 \
      --out-dir "$smoke_anchor" --wandb-name smoke_anchored --no-wandb \
      --temporal-loss lambda --td-lambda 0.9 \
      "${EXP32_FLAGS[@]}" || return 1
  [[ "$(checkpoint_kind "$smoke_anchor/checkpoint.pt")" == "anchored" ]] || {
    echo "FATAL: anchored smoke produced the wrong checkpoint architecture"; return 1;
  }

  run_logged smoke_anchored_standard_probe \
    env LD_LIBRARY_PATH="" SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO=0 \
    python "$ANCHORED_STANDARD_EVALUATOR" \
      --manifest "$MANIFEST" --split eval \
      --checkpoint "$smoke_anchor/checkpoint.pt" \
      --out-dir "$smoke_anchor/eval" --probe-dir "$smoke_probe" \
      --batch-size 2 --num-workers 0 --max-batches 2 \
      --diagnostics --diagnostic-max-batches 1 \
      --eval-rollout-horizon 5 --target-mode full --window-mode sequential \
      --thresholds 0.01 0.05 0.1 --change-threshold 0.01 \
      --event-threshold 0.01 --attack-action-min 6 \
      --probe-decoder --probe-train-split train --probe-epochs 1 \
      --probe-max-batches-per-epoch 2 --probe-samples-per-epoch 64 \
      --probe-lr 0.001 --probe-weight-decay 0.00001 --probe-seed 123 \
      --force-retrain-probe --no-require-hidden-eval --device cuda --no-amp \
      || return 1
  PROBE_DIR="$smoke_probe" verify_standard_output "$smoke_anchor/eval" "$smoke_anchor/checkpoint.pt" || return 1

  for gate_mode in 0 1; do
    local hidden_name="hidden"
    local gate_env=0
    [[ "$gate_mode" == "1" ]] && { hidden_name="hidden_gate_zero"; gate_env=1; }
    run_logged "smoke_anchored_${hidden_name}" \
      env LD_LIBRARY_PATH="" SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO="$gate_env" \
      python "$HIDDEN_EVALUATOR" \
        --manifest "$MANIFEST" --split eval \
        --checkpoint "$smoke_anchor/checkpoint.pt" \
        --out-dir "$smoke_anchor/$hidden_name" --probe-dir "$smoke_probe" \
        --batch-size 2 --num-workers 0 --eval-rollout-horizon 5 \
        --target-mode full --thresholds 0.01 0.05 0.1 \
        --presence-threshold 0.5 --change-threshold 0.01 \
        --no-natural-hidden-eval --controlled-occlusion-eval \
        --controlled-occlusion-max-batches 10 \
        --controlled-occlusion-target-entity-times 1 \
        --controlled-occlusion-spans 1 3 5 --controlled-occlusion-seed 123 \
        --controlled-prefer-reappearance --device cuda --no-amp || return 1
    verify_hidden_output "$smoke_anchor/$hidden_name" 1 || return 1
  done
  return 0
}

if [[ "$SKIP_SMOKE_TRAIN" == "1" ]]; then
  echo "WARNING: SKIP_SMOKE_TRAIN=1 bypasses runtime branch tests."
  record_status preflight_runtime skipped_by_user "$PREFLIGHT_FINGERPRINT"
elif [[ -s "$PREFLIGHT_MARKER" ]] && [[ "$(cat "$PREFLIGHT_MARKER")" == "$PREFLIGHT_FINGERPRINT" ]]; then
  echo "Runtime preflight already passed for the exact current file fingerprint."
  record_status preflight_runtime reused_verified_marker "$PREFLIGHT_FINGERPRINT"
else
  if ! run_full_smoke_preflight; then
    echo "FATAL: runtime preflight failed. Long experiments have not been started."
    echo "Inspect logs under $LOG_ROOT."
    exit 3
  fi
  printf "%s" "$PREFLIGHT_FINGERPRINT" > "$PREFLIGHT_MARKER"
  record_status preflight_runtime ok "$PREFLIGHT_FINGERPRINT"
  echo "All normal, anchored, probe, hidden, and gate-zero smoke paths passed."
fi

if [[ "$RUN_PREFLIGHT_ONLY" == "1" ]]; then
  record_status pipeline preflight_only_complete "$PREFLIGHT_FINGERPRINT"
  echo "PREFLIGHT-ONLY RUN COMPLETE. No long experiment was started."
  echo "Status: $STATUS_FILE"
  exit 0
fi

run_training exp31_delta_hidden_combo "$EXP31_DIR" 128 0 \
  --temporal-loss lambda --td-lambda 0.9 \
  "${EXP31_FLAGS[@]}" || exit 4
if eval_standard exp31 "$EXP31_DIR/checkpoint.pt" 0; then
  if ! eval_hidden exp31 "$EXP31_DIR/checkpoint.pt" 0; then
    PIPELINE_WARNINGS=1
    record_status eval_hidden_exp31 warning "primary and safe retry failed; pipeline continued"
  fi
else
  PIPELINE_WARNINGS=1
  record_status eval_hidden_exp31 skipped "standard evaluation failed twice"
  echo "WARNING: Exp31 evaluation failed twice; continuing to Exp32."
fi

run_training exp32_contiguous_belief "$EXP32_DIR" 128 0 \
  --temporal-loss lambda --td-lambda 0.9 \
  "${EXP32_FLAGS[@]}" || exit 5
if eval_standard exp32 "$EXP32_DIR/checkpoint.pt" 0; then
  if ! eval_hidden exp32 "$EXP32_DIR/checkpoint.pt" 0; then
    PIPELINE_WARNINGS=1
    record_status eval_hidden_exp32 warning "primary and safe retry failed; pipeline continued"
  fi
else
  PIPELINE_WARNINGS=1
  record_status eval_hidden_exp32 skipped "standard evaluation failed twice"
  echo "WARNING: Exp32 evaluation failed twice; continuing to Exp33."
fi

# Exp33 starts from scratch because its recurrent-state layout differs.
run_training exp33_anchored_belief "$EXP33_DIR" 322 1 \
  --temporal-loss lambda --td-lambda 0.9 \
  "${EXP32_FLAGS[@]}" || exit 6
if eval_standard exp33 "$EXP33_DIR/checkpoint.pt" 1; then
  if ! eval_hidden exp33 "$EXP33_DIR/checkpoint.pt" 0; then
    PIPELINE_WARNINGS=1
    record_status eval_hidden_exp33 warning "primary and safe retry failed; pipeline continued"
  fi
  if ! eval_hidden exp33_gate_zero "$EXP33_DIR/checkpoint.pt" 1; then
    PIPELINE_WARNINGS=1
    record_status eval_hidden_exp33_gate_zero warning "primary and safe retry failed; pipeline continued"
  fi
else
  PIPELINE_WARNINGS=1
  record_status eval_hidden_exp33 skipped "standard evaluation failed twice"
  record_status eval_hidden_exp33_gate_zero skipped "standard evaluation failed twice"
  echo "WARNING: Exp33 evaluation failed twice; checkpoint remains available."
fi

if [[ "$PIPELINE_WARNINGS" -eq 0 ]]; then
  record_status pipeline complete "$EXP33_DIR/checkpoint.pt"
else
  record_status pipeline complete_with_warnings "$EXP33_DIR/checkpoint.pt"
fi

echo
echo "============================================================"
echo "EXP31--EXP33 ANCHORED PIPELINE COMPLETE"
echo "Status:      $STATUS_FILE"
echo "Exp31:       $EXP31_DIR/checkpoint.pt"
echo "Exp32:       $EXP32_DIR/checkpoint.pt"
echo "Exp33:       $EXP33_DIR/checkpoint.pt"
echo "Evaluations: $EVAL_ROOT and $HIDDEN_ROOT"
echo "============================================================"
