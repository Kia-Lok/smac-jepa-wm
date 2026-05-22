#!/usr/bin/env bash
set -euo pipefail

python_cmd="${PYTHON:-uv run python}"

$python_cmd -m smac_jepa.evaluate \
  --manifest splits/original_seed1.json \
  --split eval \
  --checkpoint runs/original_entity_smoke_cpu/checkpoint.pt \
  --out runs/original_entity_smoke_cpu/eval_metrics.json \
  --device cpu
