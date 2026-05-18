# SMAC-JEPA

Entity-token JEPA world model scaffold for SMACLite scenario generalization.

## Server Handoff

For a fresh server setup, follow `docs/server_handoff.md`. It covers `uv`
environment setup, smoke tests, generated-config collection, GPU training,
evaluation, and human-readable prediction.

Generated datasets, checkpoints, logs, reports, HTML, and Python bytecode are
not source files. They are ignored and should be regenerated on the target
server.

## Install

The recommended setup uses `uv`:

```bash
uv venv
uv pip install -e ".[dev]"
uv pip install git+https://github.com/uoe-agents/smaclite.git
```

If you prefer plain `pip`, install the local dependencies, then install SMACLite
from its repository or package source:

```bash
pip install -r requirements.txt
pip install git+https://github.com/uoe-agents/smaclite.git
```

## Collect Data

The collector runs a random valid-action policy and writes padded `.npz` trajectories
with SMACLite state-layout metadata.

```bash
uv run python simulator/collect_smaclite_data.py \
  --env-key smaclite:smaclite/2s3z-v0 \
  --episodes 100 \
  --max-steps 120 \
  --out data/2s3z_random.npz \
  --seed 1
```

The saved file contains `states`, `actions`, `action_onehot`, `rewards`, `dones`,
`valid`, `avail_actions`, scenario metadata, and SMACLite state-layout metadata
used by the entity-token encoder.

For prediction later, `states[episode, step]` is the raw SMACLite global state
from `env.unwrapped.get_state()`, and `actions[episode, step]` is the joint
action that produced `states[episode, step + 1]`.

To collect all original SMACLite scenarios and write the default 80/20
scenario-level split, run:

```bash
PYTHON="uv run python" EPISODES=100 MAX_STEPS=120 SEED=1 scripts/collect_original_split.sh
```

The default split trains on 10 scenarios and evaluates on held-out `corridor`
and `mmm2`.

To collect many generated JSON configurations, use one `.npz` per configuration
and build an 80/20 config-level split:

```bash
uv run python simulator/collect_generated_configs.py \
  --config-dir path/to/generated_json \
  --out-dir data/generated \
  --manifest-out splits/generated_seed1.json \
  --episodes 64 \
  --max-steps 120 \
  --seed 1
```

Newly collected datasets include static map/config conditioning in addition to
state/action trajectories: terrain, map size, attack point, shield/unit-type
settings, and per-slot unit stat features. This matters for generated configs
because the same state/action can evolve differently under different terrain or
unit stats.

Before training across generated configs, audit the manifest:

```bash
uv run python -m smac_jepa.audit_dataset \
  --manifest splits/generated_seed1.json \
  --out reports/generated_audit.json
```

The audit reports manifest-wide caps for allies, enemies, actions, feature
widths, static metadata coverage, terrain distribution, and step errors.

Start with 32-64 episodes per generated configuration. For thousands of
configurations, prioritize breadth first and add more episodes adaptively to
config families with high held-out error.

## Train

The public training path is manifest/entity only. `--model-size default` is a
single-GPU-oriented preset; use `--model-size smoke` for quick CPU checks.
The clean repository does not include `.npz` datasets, so run a collection step
before using the checked-in split template.

```bash
uv run python -m smac_jepa.train \
  --manifest splits/original_seed1.json \
  --out-dir runs/original_entity \
  --model-size default \
  --epochs 5 \
  --context-len 4
```

By default, the dataset iterates over all valid sequential windows. To train on
random starts within each episode, use `--window-mode random`. `--window-len`
sets how many state/action steps are sampled after the random start, and
`--samples-per-epoch` controls how many random windows make up one epoch.
If a sampled window reaches the end of an episode, it is padded and the padded
steps are masked out of the losses.

When training from a manifest, model padding caps are computed from all manifest
splits, not just the active train split. This prevents held-out generated
configs from exceeding checkpoint capacity during eval or prediction.

```bash
uv run python -m smac_jepa.train \
  --manifest splits/original_seed1.json \
  --out-dir runs/original_entity_random \
  --model-size smoke \
  --epochs 5 \
  --device cpu \
  --no-amp \
  --window-mode random \
  --window-len 8 \
  --samples-per-epoch 2000
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
uv run python -m smac_jepa.evaluate \
  --manifest splits/original_seed1.json \
  --split eval \
  --checkpoint runs/original_entity/checkpoint.pt \
  --out runs/original_entity/eval_metrics.json
```

Evaluation reports next-state embedding MSE plus decoded next-state MAE, MSE,
R2, and tolerance accuracies at 0.01, 0.05, and 0.10.
It can also write human-readable decoded prediction samples:

```bash
uv run python -m smac_jepa.evaluate \
  --manifest splits/original_seed1.json \
  --split eval \
  --checkpoint runs/original_entity/checkpoint.pt \
  --out runs/original_entity/eval_metrics.json \
  --num-decode-samples 16 \
  --decode-sample-out runs/original_entity/decoded_samples.json
```

Decoded samples contain `prediction` and `target` entries with `allies` and
`enemies`. Each unit record includes normalized `hp`, `dx`, `dy`,
`cooldown_or_energy` for allies, optional shield fields, and unit-type values.
`dx` and `dy` are SMACLite normalized offsets from the map center.

A useful first signal is finite training loss that trends downward over a few
epochs on collected trajectories.

## Predict Next State

Use `smac_jepa.predict_next` to feed one raw SMACLite global state and one joint
action into a trained checkpoint. The command decodes the predicted next state
into human-readable ally and enemy unit records.

Prepare inputs as:

- State: a `.npy` vector from `env.unwrapped.get_state()` or a JSON list of floats.
- Actions: a JSON list of discrete action ids, one per controlled ally.
- Metadata: pass the source `.npz` with `--metadata-npz` for generated configs
  so prediction receives the same static terrain/unit conditioning used during
  training.

Example using a known collected transition:

```bash
uv run python -m smac_jepa.predict_next \
  --checkpoint runs/original_entity/checkpoint.pt \
  --state-npy /path/to/state_t.npy \
  --actions-json /path/to/actions_t.json \
  --metadata-npz data/original/2s3z.npz \
  --out runs/original_entity/pred_next.json \
  --device cpu
```

The output JSON contains:

- `prediction.allies` and `prediction.enemies`: decoded predicted next-state units.
- `current_state`: the decoded input state for comparison.
- `input_action_ids`: the joint action used for conditioning.
- `metadata`: the dimensions and padding caps used by the model.

For a collected `.npz`, the actual next state for comparison is
`states[episode, step + 1]` after applying `actions[episode, step]` to
`states[episode, step]`.

## Generated Config Smoke Run

For a quick local check using a few generated configs, copy a small subset into
an ignored run folder, collect data, train with random windows, evaluate, then run
one prediction:

```bash
mkdir -p runs/config_smoke/configs
cp configs/generated/balanced_mirrors_var_001.json runs/config_smoke/configs/
cp configs/generated/balanced_mirrors_var_002.json runs/config_smoke/configs/
cp configs/generated/balanced_mirrors_var_003.json runs/config_smoke/configs/
cp configs/generated/balanced_mirrors_var_004.json runs/config_smoke/configs/

uv run python simulator/collect_generated_configs.py \
  --config-dir runs/config_smoke/configs \
  --out-dir runs/config_smoke/data \
  --manifest-out runs/config_smoke/split.json \
  --episodes 4 \
  --max-steps 30 \
  --seed 7 \
  --eval-fraction 0.5

uv run python -m smac_jepa.train \
  --manifest runs/config_smoke/split.json \
  --out-dir runs/config_smoke/run \
  --model-size smoke \
  --epochs 1 \
  --batch-size 4 \
  --device cpu \
  --no-amp \
  --window-mode random \
  --window-len 4 \
  --samples-per-epoch 16

uv run python -m smac_jepa.evaluate \
  --manifest runs/config_smoke/split.json \
  --split eval \
  --checkpoint runs/config_smoke/run/checkpoint.pt \
  --out runs/config_smoke/run/eval_metrics.json \
  --device cpu \
  --num-decode-samples 4 \
  --decode-sample-out runs/config_smoke/run/decoded_samples.json
```

This smoke run is meant to verify the pipeline, not model quality. Use more
configs, more episodes, and more epochs for meaningful predictions.

## Synthetic Smoke Test

If SMACLite is not installed yet, create tiny entity-layout synthetic datasets,
write a manifest, and run the smoke preset:

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
  --epochs 2
```
