# SMAC-JEPA Architecture Guide

This document explains how the repository works end to end. It is written for the current version of the project: an entity-slot JEPA world model for SMACLite with decoded human-readable predictions, presence prediction, static map/config conditioning, random-window training, rollout evaluation, and generated-config generalization tooling.

The short version:

```text
SMACLite environment/config
  -> trajectory collection as .npz files
  -> train/eval manifest
  -> dataset converts global state vectors into padded entity tokens
  -> entity-slot JEPA predicts next entity latents from current entity latents + joint action
  -> decoder + presence head produce human-readable next-state predictions
  -> evaluation measures one-step and rollout error
```

The project is not yet a full RL agent. It is the world-model layer intended to support later model-based RL or planning.

## Project Goal

SMACLite is a multi-agent combat simulator, but this project treats control as a single-agent problem: one controller chooses the full joint action for all allied agents.

That means the action is:

```text
a_t = [action_for_ally_0, action_for_ally_1, ..., action_for_ally_N]
```

However, the world state is still naturally multi-entity. Allies and enemies each have position, health, cooldown/shield/type features, and static unit stats. So the model uses:

```text
single-agent joint-action control
multi-entity internal world representation
```

The current model is inspired by JEPA and LeWM:

- encode current observation/state into latent representations;
- condition a predictor on actions and static context;
- predict future latent representations;
- regularize representations with SIGReg to reduce collapse risk;
- decode latents only as an auxiliary and interpretability head.

The biggest architectural choice is that the model keeps one latent per entity slot instead of pooling the full battlefield into one vector. This is important for route planning and rollout use because identity, position, death, and local interactions need to stay attached to individual units.

## Repository Layout

Top-level directories:

```text
configs/generated/      generated SMACLite JSON map/config files
data/                   local collected .npz trajectory datasets
docs/                   architecture and project documentation
reports/                ignored audit and analysis outputs
runs/                   ignored model checkpoints and training artifacts
scripts/                convenience shell scripts and synthetic-data generator
simulator/              SMACLite data collection scripts
smac_jepa/              main Python package
splits/                 train/eval manifest JSON files
tests/                  unit tests for dataset/model/prediction behavior
```

Important root files:

- `README.md`: quick workflow instructions.
- `implementation.md`: original project brief and design motivation.
- `requirements.txt`: minimal Python dependencies.

## Data Collection

The main collector is:

```text
simulator/collect_smaclite_data.py
```

It creates a SMACLite environment, runs a random valid-action policy, and writes one compressed `.npz` file.

Example:

```bash
python simulator/collect_smaclite_data.py \
  --env-key smaclite:smaclite/2s3z-v0 \
  --episodes 100 \
  --max-steps 120 \
  --out data/2s3z_random.npz \
  --seed 1
```

For generated configs:

```bash
python simulator/collect_smaclite_data.py \
  --env-key smaclite:smaclite/custom-v0 \
  --map-file configs/generated/balanced_mirrors_var_001.json \
  --scenario-name balanced_mirrors_var_001 \
  --episodes 64 \
  --max-steps 120 \
  --out data/generated/balanced_mirrors_var_001.npz \
  --seed 1
```

### What The Collector Stores

Each `.npz` contains dynamic trajectory arrays:

```text
states          [episodes, max_steps + 1, state_dim]
actions         [episodes, max_steps, n_agents]
action_onehot   [episodes, max_steps, n_agents, n_actions]
rewards         [episodes, max_steps]
dones           [episodes, max_steps]
valid           [episodes, max_steps]
avail_actions   [episodes, max_steps, n_agents, n_actions]
```

The key transition convention is:

```text
states[episode, step]
actions[episode, step]
  -> produces
states[episode, step + 1]
```

The collector also stores state-layout metadata:

```text
state_dim
n_agents
n_enemies
n_actions
ally_state_feat_size
enemy_state_feat_size
ally_has_shields
enemy_has_shields
num_unit_types
```

For generated configs, it also stores static context:

```text
static_condition
static_dim
entity_static
entity_static_feat_size
terrain_preset
map_width
map_height
attack_point
```

`static_condition` is a vector containing map/config-level information:

- normalized map width and height;
- normalized attack point;
- ally/enemy shield flags;
- number of unit types;
- ally/enemy counts;
- terrain-type counts;
- flattened 32x32 terrain grid.

`entity_static` stores per-slot unit stats:

- max HP;
- shield;
- damage;
- cooldown;
- speed;
- attack range;
- size;
- armor;
- energy;
- attack count;
- healing flag;
- air-unit flag.

These static features are critical for generalization because two configs can have similar dynamic state vectors but different terrain or unit stats.

### Generated Config Collection

For a directory of generated JSON configs:

```bash
python simulator/collect_generated_configs.py \
  --config-dir configs/generated \
  --out-dir data/generated \
  --manifest-out splits/generated_seed1.json \
  --episodes 64 \
  --max-steps 120 \
  --seed 1
```

This script calls the single-config collector for every JSON file, skips failed configs, and writes a manifest over successful `.npz` files.

Some generated configs may fail under the installed SMACLite package. In previous runs, common failure modes were:

- zero-shield division inside SMACLite;
- missing custom unit JSONs such as `ROACH.json` or `HYDRALISK.json`.

The newer generalization probe script filters some of these cases before collection.

## Manifests And Audits

Training and evaluation use manifest JSON files. A manifest says which `.npz` files belong to each split.

Shape:

```json
{
  "preset": "generated",
  "seed": 1,
  "split_unit": "configuration",
  "datasets": {
    "train": ["data/generated/a.npz"],
    "eval": ["data/generated/b.npz"]
  }
}
```

The split writer is:

```text
smac_jepa/splits.py
```

For original SMACLite scenarios:

```bash
python -m smac_jepa.splits \
  --preset original \
  --data-dir data/original \
  --out splits/original_seed1.json
```

For generated configs:

```bash
python -m smac_jepa.splits \
  --preset generated \
  --data-dir data/generated \
  --out splits/generated_seed1.json \
  --eval-fraction 0.2
```

Before training across generated configs, audit the manifest:

```bash
python -m smac_jepa.audit_dataset \
  --manifest splits/generated_seed1.json \
  --out reports/generated_audit.json
```

The audit reports:

- number of datasets;
- max allies;
- max enemies;
- max action count;
- max ally/enemy feature width;
- static metadata coverage;
- entity static feature width;
- terrain distribution;
- valid step counts;
- simulator step errors.

The model depends on this because it pads every config to manifest-wide caps.

## Dataset Conversion

The dataset implementation is:

```text
smac_jepa/data/dataset.py
```

The central class is:

```python
SMACJEPADataset
```

It reads one or more `.npz` files and returns training windows.

The raw SMACLite global state is a flat vector. The dataset converts it into entity tokens:

```text
flat SMACLite state vector
  -> ally_0 token
  -> ally_1 token
  -> ...
  -> enemy_0 token
  -> enemy_1 token
  -> ...
  -> padded slots
```

For a batch item:

```text
entity_t              current state entity tokens
entity_mask           current alive/present mask
target_entity         next state entity tokens
target_entity_mask    next alive/present mask
entity_slot_mask      configured slot mask
action_t              joint actions
action_mask           valid ally action slots
mask                  valid timestep mask
static_condition      terrain/config vector
```

### Entity Masks

There are two important masks:

```text
entity_mask
target_entity_mask
```

These represent whether an entity is currently nonzero/alive/present in the dynamic state.

There is also:

```text
entity_slot_mask
```

This represents whether the slot exists in the current config at all, even if the unit is dead later. This matters for presence prediction. Without `entity_slot_mask`, dead units disappear from the loss and the model is not explicitly trained to predict death/presence transitions.

### Window Modes

The dataset supports sequential and random windows.

Sequential:

```text
start at timestep 0
then timestep 1
then timestep 2
...
```

Random:

```text
sample a random valid starting point in an episode
return the next window_len steps
pad if the episode ends early
mask padded steps out of the loss
```

Training with random windows:

```bash
python -m smac_jepa.train \
  --manifest splits/generated_seed1.json \
  --out-dir runs/generated_random \
  --window-mode random \
  --window-len 8 \
  --samples-per-epoch 2000
```

Random windows reduce overfitting to early-episode states and give better coverage across long episodes.

## Model Architecture

The main model is:

```text
smac_jepa/jepa.py
```

Class:

```python
SMACJEPA
```

Current architecture:

```text
entity_t
  -> EntityStateEncoder
  -> current entity-slot latents

target_entity
  -> EntityStateEncoder
  -> target next entity-slot latents

current entity latents + action_t + static_condition
  -> EntityJEPAActionPredictor
  -> predicted next entity-slot latents

predicted next entity-slot latents
  -> decoder
  -> decoded next entity tokens

predicted next entity-slot latents
  -> presence_head
  -> alive/present logits
```

### Why Entity Slots?

The earlier architecture pooled all entities into one global scene latent. That was simple but lossy:

```text
all allies/enemies -> one vector -> predict one next vector
```

The current model instead preserves one latent per entity:

```text
allies/enemies -> one latent per slot -> predict one next latent per slot
```

This is more appropriate for SMACLite because route planning needs object permanence:

- ally 0 remains ally 0;
- enemy 3 remains enemy 3;
- positions move coherently;
- HP and cooldowns evolve per unit;
- dead/present status remains tied to the slot.

This is still JEPA-style because the objective is to predict future latent representations. JEPA does not require one global latent vector.

## Encoder

The encoder is:

```text
smac_jepa/modules/encoders.py
```

Class:

```python
EntityStateEncoder
```

It performs:

```text
entity token features
  -> linear projection
  -> ally/enemy type embedding
  -> slot identity embedding
  -> masked self-attention over entities
  -> layer norm
  -> one latent per entity slot
```

Output shape:

```text
[batch, time, entities, latent_dim]
```

Where:

```text
entities = max_agents + max_enemies
```

The encoder uses masks so padded slots do not contribute.

## Predictor

The predictor is:

```text
smac_jepa/modules/predictor.py
```

Classes:

```python
ActionHistoryEncoder
EntityJEPAActionPredictor
```

The action encoder maps each ally action to an action token:

```text
action_t
  -> one-hot action vectors per ally
  -> action token embeddings
  -> masked action self-attention
```

The predictor combines:

- entity-slot latents;
- per-agent action tokens;
- static config embedding;
- timestep embeddings.

It then runs causal temporal attention over flattened time/entity/action tokens.

Causality means predictions at timestep `t` can see:

```text
timesteps <= t
```

but cannot attend to future timesteps.

This matters because training windows can contain multiple steps, but the model should not cheat by reading future states/actions.

## Static Conditioning

Static conditioning enters through the predictor. The static vector is encoded and added to action tokens.

This gives the predictor access to:

- terrain;
- map size;
- attack point;
- unit count;
- shield flags;
- unit type count.

Entity static stats are appended directly into entity token features by the dataset. That lets each entity token carry information about the underlying unit type/stats.

## Decoder And Presence Head

The decoder is inside:

```text
smac_jepa/jepa.py
```

It maps each predicted entity latent back to token features:

```text
predicted entity latent
  -> decoded entity token
```

Decoded output shape:

```text
[batch, time, entities, token_dim]
```

The presence head predicts whether an entity slot should be present/alive:

```text
predicted entity latent
  -> presence logit
  -> sigmoid presence_score
```

Presence is supervised using `entity_slot_mask` and `target_entity_mask`.

This distinction is important:

```text
entity_slot_mask: this slot exists in the config
target_entity_mask: this slot is alive/present in the next state
```

The model can now learn:

```text
existing unit slot -> present or dead next step
```

instead of simply ignoring dead units.

## Losses

The model loss is computed in:

```text
SMACJEPA.loss()
```

The losses are:

```text
pred_loss
sigreg_loss
decoded_loss
presence_loss
```

### `pred_loss`

Latent prediction MSE:

```text
predicted next entity latent
vs
target encoder next entity latent
```

This is the main JEPA objective.

### `sigreg_loss`

SIGReg regularizes the latent distribution to reduce representation collapse.

Implementation:

```text
smac_jepa/modules/sigreg.py
```

It is inspired by LeWM's SIGReg: random projections of the latent distribution are matched to a Gaussian characteristic-function statistic.

### `decoded_loss`

Decoded token MSE:

```text
decoder(predicted latent)
vs
target entity token
```

This is not the pure JEPA objective, but it is useful here because:

- SMACLite states are structured and low-dimensional;
- decoded output gives human-readable predictions;
- it makes evaluation easier;
- it gives a direct state-space training signal.

### `presence_loss`

Binary cross entropy over entity slots:

```text
presence logits
vs
target_entity_mask
```

Masked by `entity_slot_mask`.

This trains death/presence prediction explicitly.

## Training

Training entrypoint:

```text
smac_jepa/train.py
```

Typical command:

```bash
python -m smac_jepa.train \
  --manifest splits/generated_seed1.json \
  --out-dir runs/generated_entity \
  --model-size default \
  --epochs 5 \
  --context-len 4 \
  --window-mode random \
  --window-len 8 \
  --samples-per-epoch 2000
```

The trainer:

1. Parses CLI args into `TrainConfig`.
2. Resolves model-size preset.
3. Loads train split paths from the manifest.
4. Loads all manifest paths to compute padding caps.
5. Builds `SMACJEPADataset`.
6. Builds `SMACJEPA`.
7. Trains with AdamW.
8. Logs per-step and per-epoch losses.
9. Writes SVG loss plots.
10. Saves `checkpoint.pt`.

The checkpoint stores:

```text
model_state
metadata
config
resolved_config
optimizer_state
scaler_state
epoch
global_step
```

The metadata is important because evaluation/prediction must reconstruct the model with the same padding caps and feature dimensions.

## Model Presets

Defined in:

```text
smac_jepa/presets.py
```

Available presets:

```text
smoke    small CPU-friendly test model
default  main training preset
large    heavier GPU-oriented preset
```

You can override individual dimensions with CLI flags such as:

```text
--latent-dim
--hidden-dim
--action-dim
--num-heads
--encoder-layers
--action-layers
--predictor-layers
--batch-size
--lr
```

## Evaluation

Evaluation entrypoint:

```text
smac_jepa/evaluate.py
```

Basic eval:

```bash
python -m smac_jepa.evaluate \
  --manifest splits/generated_seed1.json \
  --split eval \
  --checkpoint runs/generated_entity/checkpoint.pt \
  --out runs/generated_entity/eval_metrics.json
```

Main one-step metrics:

```text
next_state_embedding_mse
decoded_mae
decoded_mse
decoded_r2
presence_acc
tol_acc_0.01
tol_acc_0.05
tol_acc_0.10
```

`next_state_embedding_mse` measures latent-space prediction error.

Decoded metrics measure state-space error after decoding.

Presence accuracy measures whether the model predicts entity slots as present/dead correctly.

### Decoded Samples

To export human-readable examples:

```bash
python -m smac_jepa.evaluate \
  --manifest splits/generated_seed1.json \
  --split eval \
  --checkpoint runs/generated_entity/checkpoint.pt \
  --out runs/generated_entity/eval_metrics.json \
  --decode-sample-out runs/generated_entity/decoded_samples.json \
  --num-decode-samples 16
```

Each decoded sample contains:

```text
prediction
target
absolute_error_mean
```

Both prediction and target are split into:

```text
allies
enemies
```

### Rollout Evaluation

For planning, one-step accuracy is not enough. The evaluator can perform rollout-style evaluation:

```bash
python -m smac_jepa.evaluate \
  --manifest splits/generated_seed1.json \
  --split eval \
  --checkpoint runs/generated_entity/checkpoint.pt \
  --out runs/generated_entity/eval_rollout.json \
  --rollout-horizons 1,2,4,8 \
  --context-len 8
```

Rollout evaluation repeatedly feeds decoded predictions forward:

```text
s_t
  -> predicted s_t+1
  -> predicted s_t+2
  -> predicted s_t+3
  ...
```

Reported rollout metrics include:

```text
rollout_h1_decoded_mae
rollout_h2_decoded_mae
rollout_h4_decoded_mae
rollout_h8_decoded_mae

rollout_h*_presence_acc
rollout_h*_hp_mae
rollout_h*_xy_mae
```

These metrics are more relevant to model-based RL than one-step decoded MAE.

### Per-Config Evaluation

To see which configs generalize poorly:

```bash
python -m smac_jepa.evaluate \
  --manifest splits/generated_seed1.json \
  --split eval \
  --checkpoint runs/generated_entity/checkpoint.pt \
  --out runs/generated_entity/eval_metrics.json \
  --per-config-out runs/generated_entity/per_config.json
```

This writes one metric row per `.npz` dataset/config.

## Human-Readable Prediction

Prediction entrypoint:

```text
smac_jepa/predict_next.py
```

It predicts one next state from:

- a checkpoint;
- one raw SMACLite global state vector;
- one joint action;
- optional source metadata `.npz`.

Example:

```bash
python -m smac_jepa.predict_next \
  --checkpoint runs/generated_entity/checkpoint.pt \
  --state-npy /path/to/state_t.npy \
  --actions-json /path/to/actions_t.json \
  --metadata-npz data/generated/balanced_mirrors_var_001.npz \
  --out pred_next.json \
  --device cpu
```

The state should be:

```text
env.unwrapped.get_state()
```

The action JSON should be a list:

```json
[0, 3, 5, 1]
```

one discrete action id per allied unit.

The output contains:

```text
prediction
current_state
raw_decoded_shape
metadata
input_action_ids
```

Each predicted unit has fields like:

```text
unit_id
faction
present
target_present
presence_score
alive_score
hp
cooldown_or_energy
dx
dy
shield
unit_type_values
unit_type_index
```

`presence_score` is the model's sigmoid probability for whether the slot is present.

`present` is derived from `presence_score >= 0.5` when prediction scores are available.

`target_present` is the mask passed into formatting. In prediction mode, this usually reflects whether the unit was present in the input/current state.

`dx` and `dy` are normalized SMACLite offsets from the map center.

## Decoder

The decoder formatter is:

```text
smac_jepa/decoder.py
```

It converts decoded tensors into readable dictionaries.

It infers:

- whether ally shield features exist;
- whether enemy shield features exist;
- how many unit-type features exist;
- ally feature order;
- enemy feature order.

It does not currently clamp values. So early or undertrained models can output:

```text
hp < 0
hp > 1
presence_score around 0.5
```

That is useful diagnostically because it exposes raw model behavior. For an RL planner, a later wrapper should probably clamp or transform decoded state into valid simulator-compatible ranges.

## Generalization Probe

The reusable generated-config experiment runner is:

```text
smac_jepa/run_generalization_probe.py
```

It automates the workflow we previously ran manually:

```text
select generated configs
filter configs by max allies/enemies
filter unavailable custom units
copy selected configs
collect missing .npz files
write split manifest
audit manifest
train
evaluate train split
evaluate held-out split
write per-config metrics
write decoded samples
```

Example:

```bash
python -m smac_jepa.run_generalization_probe \
  --config-dir configs/generated \
  --out-dir runs/smac_jepa_generalization_probe \
  --train-configs 50 \
  --eval-configs 10 \
  --max-agents 50 \
  --max-enemies 50 \
  --episodes 8 \
  --max-steps 80 \
  --epochs 3 \
  --samples-per-epoch 2000 \
  --batch-size 16 \
  --context-len 4 \
  --rollout-horizons 1,2,4 \
  --device cpu
```

This produces:

```text
selected_configs.json
split.json
audit.json
run/checkpoint.pt
train_eval.json
eval_eval.json
train_per_config.json
eval_per_config.json
train_decoded.json
eval_decoded.json
```

This is the easiest way to gauge whether a model generalizes across generated SMACLite configs.

## Tests

Tests live in:

```text
tests/
```

Current test files:

```text
test_dataset_windows.py
test_decoder.py
test_jepa_architecture.py
test_predict_next.py
```

They cover:

- sequential and random dataset windows;
- tail padding and masks;
- static condition and entity static feature insertion;
- entity slot masks;
- decoder formatting;
- SIGReg finite loss and gradient flow;
- encoder slot identity;
- predictor temporal causality;
- static conditioning;
- presence loss/metrics;
- prediction input preprocessing.

Focused test command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
uv run --with pytest --with numpy --with torch \
pytest tests/test_decoder.py tests/test_dataset_windows.py tests/test_predict_next.py tests/test_jepa_architecture.py -q
```

## How The Main Files Fit Together

Data path:

```text
simulator/collect_smaclite_data.py
  -> .npz file
  -> smac_jepa/splits.py
  -> manifest JSON
  -> smac_jepa/audit_dataset.py
  -> SMACJEPADataset
```

Training path:

```text
SMACJEPADataset
  -> DataLoader
  -> SMACJEPA.loss()
  -> AdamW update
  -> logs/plots/checkpoint
```

Model path:

```text
entity_t
  -> EntityStateEncoder
  -> entity latents
  -> EntityJEPAActionPredictor with action_t + static_condition
  -> predicted next entity latents
  -> decoder + presence_head
```

Evaluation path:

```text
checkpoint + manifest
  -> SMACJEPADataset
  -> SMACJEPA.forward()
  -> one-step metrics
  -> optional rollout metrics
  -> optional decoded samples
  -> optional per-config JSON
```

Prediction path:

```text
checkpoint + raw state + joint action + metadata npz
  -> encode_state_vector()
  -> encode_action_ids()
  -> encoder/predictor
  -> decoder/presence head
  -> human-readable JSON
```

## Current Strengths

The repo now has:

- entity-slot latents instead of global pooled latents;
- variable ally/enemy support through padding caps;
- generated-config static conditioning;
- per-unit static stats;
- SIGReg anti-collapse regularization;
- decoded human-readable predictions;
- presence prediction;
- random-start window training;
- rollout evaluation;
- per-config evaluation;
- reusable generalization probe tooling.

This is enough to run meaningful world-model experiments on SMACLite.

## Current Limitations

The repo is not yet a complete model-based RL system.

Important limitations:

- data collection is mostly random valid-action policy;
- no reward prediction head;
- no termination prediction head;
- no planner interface yet;
- rollout evaluation exists, but rollout-aware training is still limited;
- decoded predictions are not clamped to simulator-valid ranges;
- static terrain conditioning is still a flat vector, not a spatial terrain-attention module;
- generated config validity is imperfect because some configs are incompatible with the installed SMACLite package;
- checkpoint compatibility is broken across major architecture changes.

## Recommended Next Steps

For world-model quality:

```text
add reward prediction
add done/termination prediction
add rollout-consistency training
add stronger terrain encoder
add scripted route/attack/kiting data collection policies
add per-feature metric dashboards
```

For RL integration:

```text
define planner-facing state format
define valid action masking for imagined states
build short-horizon planner using the world model
compare real env transition vs imagined transition
train/evaluate policy with model rollouts
```

For code hygiene:

```text
remove tracked __pycache__ files
remove legacy report artifacts
keep generated experiment outputs outside git
add README links to this architecture guide
```

## Mental Model

Think of the project as three layers.

Layer 1: data.

```text
SMACLite -> .npz trajectories -> manifests
```

Layer 2: world model.

```text
entity state + joint action + static config
  -> predicted next entity state
```

Layer 3: future RL/planning.

```text
policy/planner proposes joint actions
world model imagines outcomes
planner selects useful action sequence
real SMACLite env executes first action
```

The current repo has Layer 1 and Layer 2 in usable shape. Layer 3 is the next major stage.
