#!/usr/bin/env bash
set -euo pipefail

python -m smac_jepa.train \
  --manifest splits/original_seed1.json \
  --out-dir runs/original_entity_smoke_cpu \
  --model-size smoke \
  --epochs 5 \
  --device cpu \
  --no-amp \
  --context-len 4
