# SMAC-JEPA

Entity-token JEPA world model scaffold for SMACLite scenario generalization.

## Install

Install the local dependencies, then install SMACLite from its repository or package source.

```bash
pip install -r requirements.txt
pip install git+https://github.com/uoe-agents/smaclite.git
```

## Collect Data

The collector runs a random valid-action policy and writes padded `.npz` trajectories
with SMACLite state-layout metadata.

```bash
python simulator/collect_smaclite_data.py \
  --env-key smaclite:smaclite/2s3z-v0 \
  --episodes 100 \
  --max-steps 120 \
  --out data/2s3z_random.npz \
  --seed 1
```

The saved file contains `states`, `actions`, `action_onehot`, `rewards`, `dones`,
`valid`, `avail_actions`, scenario metadata, and SMACLite state-layout metadata
used by the entity-token encoder.

To collect all original SMACLite scenarios and write the default 80/20
scenario-level split, run:

```bash
EPISODES=100 MAX_STEPS=120 SEED=1 scripts/collect_original_split.sh
```

The default split trains on 10 scenarios and evaluates on held-out `corridor`
and `mmm2`.

To collect many generated JSON configurations, use one `.npz` per configuration
and build an 80/20 config-level split:

```bash
python simulator/collect_generated_configs.py \
  --config-dir path/to/generated_json \
  --out-dir data/generated \
  --manifest-out splits/generated_seed1.json \
  --episodes 64 \
  --max-steps 120 \
  --seed 1
```

Start with 32-64 episodes per generated configuration. For thousands of
configurations, prioritize breadth first and add more episodes adaptively to
config families with high held-out error.

## Train

The public training path is manifest/entity only. `--model-size default` is a
single-GPU-oriented preset; use `--model-size smoke` for quick CPU checks.

```bash
python -m smac_jepa.train \
  --manifest splits/original_seed1.json \
  --out-dir runs/original_entity \
  --model-size default \
  --epochs 5 \
  --context-len 4
```

Useful presets:

- `smoke`: small fast test model.
- `default`: larger single-GPU default.
- `large`: heavier model for larger GPU runs.

Losses and artifacts are written to the selected run directory:

- `loss_log.csv` / `loss_log.jsonl`
- `epoch_loss.csv` / `epoch_loss.jsonl`
- `checkpoint.pt`
- `config.json`
- SVG loss plots

## Evaluate

```bash
python -m smac_jepa.evaluate \
  --manifest splits/original_seed1.json \
  --split eval \
  --checkpoint runs/original_entity/checkpoint.pt \
  --out runs/original_entity/eval_metrics.json
```

Evaluation reports next-state embedding MSE plus decoded next-state MAE, MSE,
R2, and tolerance accuracies at 0.01, 0.05, and 0.10.

A useful first signal is finite training loss that trends downward over a few
epochs on collected trajectories.

## Synthetic Smoke Test

If SMACLite is not installed yet, create tiny entity-layout synthetic datasets,
write a manifest, and run the smoke preset:

```bash
mkdir -p data/synthetic_entity
python scripts/smoke_synthetic.py --entity-layout --out data/synthetic_entity/a.npz
python scripts/smoke_synthetic.py --entity-layout --out data/synthetic_entity/b.npz --seed 2
python -m smac_jepa.splits \
  --preset generated \
  --data-dir data/synthetic_entity \
  --out splits/synthetic_entity.json
python -m smac_jepa.train \
  --manifest splits/synthetic_entity.json \
  --model-size smoke \
  --out-dir runs/synthetic_entity_cpu \
  --epochs 2
```
