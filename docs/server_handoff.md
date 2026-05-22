# Server Handoff Runbook

This runbook is for moving the repo to a fresh single-GPU Linux server and
running it without relying on local generated artifacts.

The repository should contain source code, tests, docs, split templates, and
generated JSON SMACLite configs. It should not depend on checked-in `.npz`
datasets, checkpoints, logs, reports, or Python bytecode. Those are generated
on the server.

## 1. Environment

Install `uv` on the server, then create the project environment:

```bash
uv venv
uv pip install -e ".[dev]"
uv pip install git+https://github.com/uoe-agents/smaclite.git
```

If the server needs a CUDA-specific PyTorch wheel, install the matching PyTorch
build first, then run `uv pip install -e ".[dev]"`.

Check the environment:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run python -m smac_jepa.train --help
uv run python -m smac_jepa.evaluate --help
uv run python -m smac_jepa.predict_next --help
uv run pytest tests -q
```

## 2. Smoke Test Without SMACLite

Use this when checking the repo before installing SMACLite:

```bash
mkdir -p data/synthetic_entity
uv run python scripts/smoke_synthetic.py --entity-layout --out data/synthetic_entity/a.npz
uv run python scripts/smoke_synthetic.py --entity-layout --out data/synthetic_entity/b.npz --seed 2
uv run python -m smac_jepa.splits \
  --preset generated \
  --data-dir data/synthetic_entity \
  --out splits/synthetic_entity.json
uv run python -m smac_jepa.train \
  --manifest splits/synthetic_entity.json \
  --model-size smoke \
  --out-dir runs/synthetic_entity_cpu \
  --epochs 1 \
  --device cpu \
  --no-amp
uv run python -m smac_jepa.evaluate \
  --manifest splits/synthetic_entity.json \
  --split eval \
  --checkpoint runs/synthetic_entity_cpu/checkpoint.pt \
  --out runs/synthetic_entity_cpu/eval_metrics.json \
  --device cpu
```

This confirms the dataset, model, train, eval, and artifact-writing paths.

## 3. Generated SMACLite Config Run

The repo currently includes 640 generated JSON configs under
`configs/generated`. They are intentionally uneven across families, so prefer
the generalization probe for stratified sampling instead of taking the first N
files by name.

Quick CPU probe:

```bash
uv run python -m smac_jepa.run_generalization_probe \
  --config-dir configs/generated \
  --out-dir runs/generated_probe_cpu \
  --train-configs 8 \
  --eval-configs 2 \
  --episodes 2 \
  --max-steps 30 \
  --epochs 1 \
  --samples-per-epoch 32 \
  --batch-size 4 \
  --device cpu
```

Single-GPU starter run:

```bash
uv run python -m smac_jepa.run_generalization_probe \
  --config-dir configs/generated \
  --out-dir runs/generated_probe_cuda \
  --train-configs 50 \
  --eval-configs 10 \
  --episodes 8 \
  --max-steps 80 \
  --epochs 3 \
  --samples-per-epoch 2000 \
  --batch-size 64 \
  --context-len 4 \
  --device cuda
```

For larger runs, increase breadth before depth. A practical first target is
32-64 episodes per config across as many config families as possible, then add
episodes to families with high held-out rollout error.

## 4. Manual Collection, Training, And Evaluation

Collect generated configs directly:

```bash
uv run python simulator/collect_generated_configs.py \
  --config-dir configs/generated \
  --out-dir data/generated \
  --manifest-out splits/generated_seed1.json \
  --episodes 64 \
  --max-steps 120 \
  --seed 1
```

Audit before training:

```bash
uv run python -m smac_jepa.audit_dataset \
  --manifest splits/generated_seed1.json \
  --out reports/generated_audit.json
```

Train on one GPU:

```bash
uv run python -m smac_jepa.train \
  --manifest splits/generated_seed1.json \
  --out-dir runs/generated_entity \
  --model-size default \
  --epochs 20 \
  --batch-size 128 \
  --device cuda \
  --amp \
  --window-mode random \
  --window-len 8 \
  --samples-per-epoch 20000 \
  --num-workers 4
```

Evaluate:

```bash
uv run python -m smac_jepa.evaluate \
  --manifest splits/generated_seed1.json \
  --split eval \
  --checkpoint runs/generated_entity/checkpoint.pt \
  --out runs/generated_entity/eval_metrics.json \
  --decode-sample-out runs/generated_entity/decoded_samples.json \
  --per-config-out runs/generated_entity/per_config.json \
  --rollout-horizons 1,2,4,8 \
  --device cuda
```

## 5. Human-Readable Prediction

`predict_next` expects one raw SMACLite global state and one joint action. For a
known collected transition, use `states[episode, step]` and
`actions[episode, step]`; the actual answer is `states[episode, step + 1]`.

```bash
uv run python -m smac_jepa.predict_next \
  --checkpoint runs/generated_entity/checkpoint.pt \
  --state-npy /path/to/state_t.npy \
  --actions-json /path/to/actions_t.json \
  --metadata-npz data/generated/example_config.npz \
  --out runs/generated_entity/pred_next.json \
  --device cuda
```

The output JSON includes decoded `prediction`, decoded `current_state`,
`input_action_ids`, and model metadata. Use eval decoded samples when you want
automatic prediction-vs-target comparisons without preparing separate input
files.

## 6. Artifact Policy

Generated files belong in ignored paths:

- `data/`: collected `.npz` trajectories
- `runs/`: checkpoints, logs, plots, metrics
- `reports/`: audits and analysis outputs
- `checkpoints/`: manually saved model weights

Before pushing a server run, check:

```bash
git status --short
git ls-files '*.html'
git ls-files | rg '(__pycache__|\.pyc|^data/|^runs/|^reports/)'
```

The last two commands should print nothing.
