#!/usr/bin/env bash
set -euo pipefail

episodes="${EPISODES:-100}"
max_steps="${MAX_STEPS:-120}"
seed="${SEED:-1}"
out_dir="${OUT_DIR:-data/original}"
python_cmd="${PYTHON:-python3}"

mkdir -p "$out_dir"

scenarios=(
  10m_vs_11m
  27m_vs_30m
  2c_vs_64zg
  2s3z
  2s_vs_1sc
  3s5z
  3s5z_vs_3s6z
  3s_vs_5z
  bane_vs_bane
  corridor
  mmm
  mmm2
)

for scenario in "${scenarios[@]}"; do
  env_scenario="$scenario"
  if [[ "$scenario" == "mmm" ]]; then
    env_scenario="MMM"
  elif [[ "$scenario" == "mmm2" ]]; then
    env_scenario="MMM2"
  fi
  $python_cmd simulator/collect_smaclite_data.py \
    --env-key "smaclite:smaclite/${env_scenario}-v0" \
    --scenario-name "$scenario" \
    --episodes "$episodes" \
    --max-steps "$max_steps" \
    --out "$out_dir/${scenario}.npz" \
    --seed "$seed"
done

$python_cmd -m smac_jepa.splits \
  --preset original \
  --data-dir "$out_dir" \
  --out splits/original_seed${seed}.json \
  --seed "$seed"
