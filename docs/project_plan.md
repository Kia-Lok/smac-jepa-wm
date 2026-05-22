# SMAC-JEPA Project Plan: From World Model To Dreamer-Style RL

This document is the implementation roadmap for turning the current SMAC-JEPA repo into a model-based RL system for SMACLite.

The current repo already has a usable entity-slot JEPA world model:

- it collects SMACLite data into `.npz` files;
- it trains on variable generated configs;
- it predicts next entity-slot latents;
- it decodes predictions into human-readable ally/enemy state;
- it predicts presence/alive scores;
- it evaluates one-step and rollout error.

The next goal is larger: make this world model usable inside a DreamerV3-style RL workflow, where an actor-critic policy is trained from imagined rollouts inside the learned world model.

References:

- DreamerV3 paper: https://arxiv.org/abs/2301.04104
- DreamerV3 official repository: https://github.com/danijar/dreamerv3
- Current architecture guide: `docs/architecture.md`

## 1. Target End State

The target system should eventually look like this:

```text
SMACLite env/config
  -> online/offline replay buffer
  -> entity-slot world model training
  -> imagined latent rollouts
  -> actor-critic training from imagined rollouts
  -> policy acts in real SMACLite env
  -> new experience goes back into replay
```

The repo should support two modes:

1. Offline world-model training.
2. Online Dreamer-style model-based RL training.

Offline world-model training means:

```text
collected .npz trajectories
  -> train world model
  -> evaluate one-step and rollout prediction quality
```

Online Dreamer-style training means:

```text
policy interacts with SMACLite
  -> replay buffer stores transitions
  -> world model trains from replay
  -> actor/critic train from imagined rollouts
  -> policy improves
```

The current code is mostly at stage 1. The rest of this plan explains how to reach stage 2.

## 2. DreamerV3 Workflow Summary

DreamerV3 learns a world model from environment experience and trains an actor-critic policy inside imagined latent rollouts.

The important DreamerV3 components are:

```text
encoder
RSSM latent dynamics
decoder/reconstruction head
reward head
continuation head
actor
critic/value model
replay buffer
imagined rollouts
lambda-return actor-critic loss
```

The DreamerV3 official README describes the core idea as:

```text
learn a world model from experiences
use it to train an actor critic policy from imagined trajectories
predict future representations and rewards given actions
```

The paper frames Dreamer as learning a model of the environment and improving behavior by imagining future scenarios.

For this repo, we do not need to copy DreamerV3's pixel encoder or RSSM exactly. SMACLite is not a pixel task; it exposes vector/global state and structured entities. The goal is to implement the same functional workflow using a SMACLite-specific world model:

```text
DreamerV3 pixel/RSSM world model
  -> replaced by
SMAC-JEPA entity-slot latent world model
```

## 3. DreamerV3 To SMAC-JEPA Mapping

| DreamerV3 Concept | Current / Future SMAC-JEPA Equivalent |
|---|---|
| Observation encoder | `EntityStateEncoder` |
| RSSM latent dynamics | `EntityJEPAActionPredictor` over entity slots |
| Stochastic latent state | Optional future latent distribution head |
| Deterministic recurrent state | Current temporal attention context / future recurrent state wrapper |
| Decoder | Entity decoder in `SMACJEPA` |
| Reward head | Must be added |
| Continuation head | Must be added |
| Actor | Must be added: joint-action policy over allies |
| Critic/value model | Must be added |
| Replay buffer | Must be added for online training |
| Imagination | Partially exists through rollout eval; must become trainable imagination API |
| Actor-critic loss | Must be added |

The replacement should not be a literal file-level drop-in for the JAX DreamerV3 repo. Instead, this PyTorch repo should expose equivalent world-model capabilities:

```text
encode
observe
imagine
decode
predict_reward
predict_continue
predict_presence
```

That gives the rest of the RL algorithm the same conceptual primitives that DreamerV3 expects from its world model.

## 4. Current Repo Status

Already implemented:

- entity-token dataset;
- generated-config data collection;
- static config conditioning;
- entity static unit stats;
- entity-slot encoder;
- action-token predictor;
- SIGReg regularization;
- decoder;
- presence head;
- one-step evaluation;
- rollout evaluation;
- per-config evaluation;
- generalization probe runner.

Still missing for Dreamer-style RL:

- reward prediction head;
- continuation/done prediction head;
- actor network;
- critic/value network;
- replay buffer;
- online env loop;
- imagined rollout training;
- lambda-return actor-critic losses;
- valid-action-aware policy sampling;
- GPU-optimized large-scale training path;
- proper dataset sharding for millions of transitions.

## 5. Sequential Implementation Roadmap

The implementation should proceed in order. Do not jump directly to actor-critic training until world-model rollout quality is measurable.

## Phase 1: Harden The World Model

Goal: make the world model predict all quantities needed by a Dreamer-style planner.

### 1.1 Add Reward Prediction

Current collected data already stores:

```text
rewards [episodes, max_steps]
```

Add the reward target to dataset items:

```text
reward_t: reward after applying action_t
```

Add a reward head to `SMACJEPA`:

```text
predicted next latent/entity rollout state
  -> scalar reward prediction
```

Recommended first implementation:

```text
pool valid predicted entity latents
concatenate/static-condition embedding if useful
MLP -> scalar reward
MSE loss
```

Later improvement:

```text
two-hot symlog reward distribution
```

DreamerV3 uses transformed distributional scalar prediction for reward/value stability. For the first PyTorch implementation, scalar MSE is acceptable. If reward scale becomes unstable, switch to symlog/two-hot.

Acceptance criteria:

- `reward_loss` appears in logs.
- Evaluation reports `reward_mae` and `reward_mse`.
- Reward prediction is better than predicting the train-set mean reward.

### 1.2 Add Continuation / Done Prediction

Current collected data stores:

```text
dones [episodes, max_steps]
```

Dreamer-style algorithms usually predict continuation:

```text
continue = not terminal
```

Add dataset target:

```text
continue_t = 1.0 - done_t
```

Add continuation head:

```text
predicted latent state -> continuation logit
```

Loss:

```text
binary cross entropy with logits
```

Acceptance criteria:

- `continue_loss` appears in logs.
- Evaluation reports `continue_acc`, `continue_bce`.
- Terminal-heavy configs are tested separately.

### 1.3 Add Rollout-Consistency Training

Current rollout evaluation feeds decoded predictions forward, but training is still mainly one-step.

Add optional training flag:

```text
--rollout-loss-horizons 2,4,8
```

Training behavior:

```text
encode s_t
predict s_t+1
decode predicted s_t+1
feed predicted s_t+1 back into encoder/predictor
predict s_t+2
...
compare predicted decoded state/reward/continue against real sequence
```

Loss terms:

```text
rollout_h*_decoded_loss
rollout_h*_presence_loss
rollout_h*_reward_loss
rollout_h*_continue_loss
```

Start with short horizons:

```text
2 and 4
```

Only use horizon 8 or 16 after one-step quality is stable.

Acceptance criteria:

- rollout error at horizon 4 improves versus one-step-only baseline;
- rollout training does not destabilize one-step metrics;
- decoded values remain in plausible ranges.

### 1.4 Add Valid-Action Support

Collected data stores:

```text
avail_actions [episodes, steps, n_agents, n_actions]
```

For RL, the actor must not sample invalid actions.

Add dataset return:

```text
avail_action_t
target_avail_action
```

Initial use:

- pass `avail_action_t` to actor;
- mask logits before sampling.

Optional world-model use:

- predict next valid action mask from imagined state;
- or compute valid action mask from decoded state with environment rules later.

Recommended first implementation:

```text
actor uses real env valid action masks during online interaction
imagined actor uses conservative mask from current/last known valid mask
```

Better later implementation:

```text
valid-action head predicts next availability
```

Acceptance criteria:

- actor never samples invalid real-env actions;
- imagined rollout code has a clear masking policy.

## Phase 2: Build Replay And Online Data Loop

Goal: move from offline `.npz` training to Dreamer-style replay training.

### 2.1 Replay Buffer Schema

Replay items must contain:

```text
state
action
reward
done
valid
avail_actions
static_condition
entity_static
scenario/config id
metadata
```

The replay buffer should support:

- append new online transitions;
- sample fixed-length sequences;
- preserve config-level static data;
- save/load shards to disk;
- optionally mix offline and online data.

Recommended implementation:

```text
smac_jepa/replay.py
```

Core methods:

```python
add_episode(...)
sample(batch_size, sequence_length)
save_shard(path)
load_shards(paths)
```

Acceptance criteria:

- replay can load existing `.npz` files;
- replay can append online SMACLite episodes;
- replay samples match `SMACJEPADataset` tensor shapes.

### 2.2 Online SMACLite Driver

Add an environment interaction loop:

```text
reset env
get state
actor/policy chooses joint action
step env
store transition in replay
repeat
```

Initial policy choices:

1. random valid policy;
2. scripted route/attack policy;
3. learned actor once actor exists.

Recommended file:

```text
smac_jepa/online_train.py
```

Acceptance criteria:

- can collect online episodes into replay;
- can periodically train world model from replay;
- can periodically evaluate policy in real SMACLite.

### 2.3 Train Ratio

DreamerV3 uses a train ratio: number of gradient updates per environment step.

For this repo, expose:

```text
--train-ratio
```

Definition:

```text
gradient updates = train_ratio * environment_steps
```

Suggested starting values:

```text
offline pretrain: not applicable; use epochs/samples_per_epoch
online debug: train_ratio 4-8
small SMACLite: train_ratio 16-32
large generated configs: train_ratio 32-64
```

DreamerV3 often uses high train ratios for data efficiency. The right value depends on GPU throughput and replay diversity.

Acceptance criteria:

- logs report env steps, updates, updates/env-step, transitions/sec, updates/sec.

## Phase 3: Add Actor-Critic On Imagined Rollouts

Goal: train a policy from imagined trajectories generated by the world model.

### 3.1 Actor Network

The actor receives world-model latent state:

```text
entity latents
entity masks
static condition
available action mask
```

It outputs one discrete action distribution per ally:

```text
[batch, max_agents, max_actions]
```

Invalid actions must be masked before sampling.

Recommended actor structure:

```text
entity latents
  -> attention over ally/enemy slots
  -> ally-slot policy logits
  -> mask invalid action logits
  -> categorical distributions
```

Actor output:

```text
joint action sample
log probability per ally
entropy per ally
```

Because we treat control as single-agent, the joint policy is:

```text
pi(a_joint | state) = product over ally action distributions
```

or, later, a more coupled autoregressive policy.

Start with independent per-ally categorical heads. It is simpler and likely sufficient for the first RL integration.

Acceptance criteria:

- actor samples valid actions only;
- actor log-prob and entropy are available for loss computation;
- actor supports deterministic eval mode.

### 3.2 Critic / Value Network

The critic receives latent state and predicts scalar value:

```text
entity latents + masks + static condition -> V(s)
```

Initial implementation:

```text
masked mean pool entity latents
concat static embedding
MLP -> scalar value
```

Later:

```text
attention pooling or learned query token
```

Acceptance criteria:

- critic returns `[batch, time]` values;
- value loss trains on imagined lambda returns.

### 3.3 Imagination API

Add a world-model method:

```python
imagine(start_latent, policy, horizon, static_condition, masks)
```

It should return:

```text
latent_rollout
decoded_rollout
presence_rollout
reward_rollout
continue_rollout
action_rollout
logprob_rollout
entropy_rollout
```

Pseudo-flow:

```text
latent = start_latent
for h in horizon:
    action = actor(latent)
    next_latent = predictor(latent, action)
    reward = reward_head(next_latent)
    continue = continue_head(next_latent)
    store transition
    latent = next_latent
```

Acceptance criteria:

- imagined rollout tensors have stable shapes;
- gradients flow from actor loss through actor and optionally through world model depending on chosen setting;
- no future real observations are used during imagination.

### 3.4 Lambda Returns

Implement Dreamer-style lambda returns:

```text
reward predictions
continuation probabilities
critic value predictions
discount
lambda
  -> target returns
```

Suggested defaults:

```text
discount: 0.99
lambda: 0.95
imagination horizon: 15
```

DreamerV3 uses a long effective horizon and continuation discounting. For SMACLite, start smaller:

```text
horizon 5 for debug
horizon 15 for real training
horizon 30 for route-planning experiments
```

Acceptance criteria:

- actor loss decreases on synthetic reward tasks;
- critic loss decreases;
- value predictions are finite and stable.

## Phase 4: Full Online RL Loop

Goal: train an agent that uses the world model to improve policy in SMACLite.

Loop:

```text
initialize replay with random/scripted episodes
initialize world model, actor, critic

while env_steps < target:
    collect real env steps using current policy
    add episodes/transitions to replay

    for update in train_ratio * collected_steps:
        sample sequence batch from replay
        train world model
        sample latent starts from posterior states
        imagine rollouts with actor
        train actor and critic

    periodically:
        evaluate policy in real SMACLite
        evaluate world model on held-out configs
        save checkpoint
```

Logging:

```text
env_steps
episodes
mean_return
win_rate if available
world_model losses
actor loss
critic loss
entropy
valid-action violation count
rollout metrics
GPU utilization
updates/sec
transitions/sec
```

Acceptance criteria:

- random baseline is logged;
- scripted baseline is logged;
- learned policy exceeds random baseline on at least one small scenario;
- model rollout metrics do not collapse during actor learning.

## Phase 5: Scaling Across Thousands Of Configs

Goal: train a world model and policy that generalize across many SMACLite generated configs.

### 5.1 Config Selection

Use config-level splits, not random episode-level splits.

Keep held-out configs separated by:

- family;
- terrain;
- ally/enemy count;
- unit type mix;
- shield settings.

Recommended split:

```text
80% train configs
10% validation configs
10% test configs
```

Validation is for model selection. Test should stay untouched until final reporting.

### 5.2 Dataset Size Tiers

Assume roughly 100 valid steps per episode for estimates.

Smoke:

```text
5-10 configs
4-8 episodes/config
~2k-8k transitions
```

Prototype generalization:

```text
50 configs
8-16 episodes/config
~40k-80k transitions
```

Serious offline world model:

```text
500 configs
16-32 episodes/config
~0.8M-1.6M transitions
```

Broad generated-config training:

```text
1,000-2,000 configs
32-64 episodes/config
~3.2M-12.8M transitions
```

High-quality model for RL planning:

```text
2,000+ configs
64-128 episodes/config
~12.8M-25.6M transitions
```

If using max steps 120 and episodes often last close to max length, multiply these estimates by about 1.2.

### 5.3 How Many Episodes Per Config?

For early world-model generalization:

```text
more configs beats more episodes per config
```

Recommended staged approach:

1. Start with 32 episodes/config over as many valid configs as possible.
2. Evaluate per-config rollout metrics.
3. Add episodes only to weak families/terrains.
4. Add scripted/policy data before blindly increasing random data.

Suggested default for thousands of configs:

```text
1,000 configs x 32 episodes/config
```

Then scale to:

```text
2,000 configs x 64 episodes/config
```

For final model-based RL:

```text
2,000 configs x 64-128 episodes/config
+ online policy-generated rollouts
```

Random-only data will eventually saturate. Planning needs purposeful route and combat behavior.

### 5.4 Data Policy Mix

Do not rely only on random valid actions.

Use a mixture:

```text
40% random valid actions
20% move-to-attack-point scripted policy
20% attack-nearest / focus-fire scripted policy
10% retreat/kiting scripted policy
10% current learned policy
```

As the policy improves, shift toward:

```text
20% random/scripted
80% learned policy rollouts
```

This gives the world model data near the policy distribution that the actor will actually use.

## 6. GPU Training Plan

The current code supports:

```text
--device cuda
--amp
```

For serious training, use GPU.

### 6.1 Starting GPU Commands

Small GPU or first CUDA run:

```bash
python -m smac_jepa.train \
  --manifest splits/generated_seed1.json \
  --out-dir runs/generated_cuda_default \
  --model-size default \
  --epochs 10 \
  --batch-size 64 \
  --context-len 8 \
  --window-mode random \
  --window-len 8 \
  --samples-per-epoch 20000 \
  --device cuda \
  --amp
```

24GB GPU:

```bash
python -m smac_jepa.train \
  --manifest splits/generated_seed1.json \
  --out-dir runs/generated_cuda_large \
  --model-size large \
  --epochs 20 \
  --batch-size 128 \
  --context-len 8 \
  --window-mode random \
  --window-len 8 \
  --samples-per-epoch 50000 \
  --device cuda \
  --amp
```

If VRAM allows, try:

```text
batch_size 256
context_len 16
```

Increase one at a time.

### 6.2 GPU Utilization Rules

Track:

```text
GPU utilization
VRAM usage
updates/sec
tokens/sec or transitions/sec
data loading time
```

If GPU utilization is low:

- increase `num_workers`;
- increase `batch_size`;
- increase `samples_per_epoch`;
- prepack `.npz` data into larger replay shards;
- avoid repeatedly opening many small `.npz` files;
- pin memory in DataLoader;
- use persistent workers;
- keep data on local SSD.

If out of memory:

- reduce `batch_size`;
- reduce `context_len`;
- reduce `latent_dim`;
- use `model-size default` instead of `large`;
- bucket configs by entity count;
- cap max agents/enemies.

### 6.3 Entity Count And Attention Cost

Attention cost grows with token count.

Current predictor tokens per timestep are approximately:

```text
entity tokens + action tokens
= max_agents + max_enemies + max_agents
```

So if:

```text
max_agents = 50
max_enemies = 50
```

then:

```text
tokens per timestep ~= 150
```

With context length 16:

```text
sequence length ~= 2400 tokens
```

That can become expensive.

Recommended scaling strategy:

```text
bucket configs by entity count
train small/medium/large buckets separately first
only combine buckets once memory and speed are understood
```

Suggested buckets:

```text
small:   <= 20 total units
medium:  <= 60 total units
large:   <= 100 total units
huge:    > 100 total units
```

For your proposed 50/50 cap, the practical maximum is 100 total units. That is manageable, but batching many max-size configs with long context will be expensive.

### 6.4 Future GPU Optimizations

After correctness is stable:

- add pinned memory and persistent workers to training DataLoader;
- add a packed replay dataset to avoid repeated `.npz` reads;
- add `torch.compile` option;
- add gradient accumulation;
- add DDP only after single-GPU training is stable;
- add mixed precision by default on CUDA;
- log throughput metrics.

Do not start with multi-GPU. First make one GPU saturated and stable.

## 7. Required World-Model API

To replace DreamerV3's world model functionally, this repo should expose a clean API.

Recommended interface:

```python
class SMACWorldModel:
    def encode(self, obs, metadata):
        ...

    def observe(self, prev_latent, action, obs, static_condition, masks):
        ...

    def imagine(self, start_latent, policy, horizon, static_condition, masks):
        ...

    def decode(self, latent):
        ...

    def predict_reward(self, latent):
        ...

    def predict_continue(self, latent):
        ...

    def predict_presence(self, latent):
        ...
```

### `encode`

Input:

```text
raw global state or entity tokens
metadata
static/entity static features
```

Output:

```text
entity-slot latent state
entity masks
```

### `observe`

This is the posterior update equivalent.

Input:

```text
previous latent
previous action
current real observation
```

Output:

```text
posterior latent informed by real observation
```

In the current repo, this is basically:

```text
encoder(current observation)
```

Later, it can be made recurrent/stateful.

### `imagine`

Input:

```text
start latent
policy function
horizon
static condition
entity/action masks
```

Output:

```text
latent rollout
action rollout
reward rollout
continue rollout
presence rollout
decoded rollout
```

This is the core API actor-critic will use.

### `decode`

Input:

```text
latent entity slots
```

Output:

```text
decoded entity tokens
human-readable optional formatting
```

### `predict_reward`

Input:

```text
latent state
```

Output:

```text
scalar reward prediction
```

### `predict_continue`

Input:

```text
latent state
```

Output:

```text
probability episode continues
```

### `predict_presence`

Already partially exists.

Input:

```text
entity latents
```

Output:

```text
presence probability per entity slot
```

## 8. Actor-Critic API

Recommended actor interface:

```python
class SMACActor(nn.Module):
    def forward(self, latents, entity_mask, action_mask, static_condition):
        return action_distribution
```

Output:

```text
logits: [batch, max_agents, max_actions]
distribution: categorical per ally
sampled_action: [batch, max_agents]
logprob: [batch]
entropy: [batch]
```

Recommended critic interface:

```python
class SMACCritic(nn.Module):
    def forward(self, latents, entity_mask, static_condition):
        return value
```

Output:

```text
value: [batch]
```

For imagined rollouts:

```text
latents: [batch, horizon, entities, latent_dim]
values:  [batch, horizon]
```

## 9. Training Acceptance Criteria

### World Model Acceptance

Before actor-critic integration:

- held-out one-step decoded R2 is positive and improving;
- rollout horizon 4 and 8 errors are measured;
- rollout error does not explode immediately;
- reward prediction beats mean-reward baseline;
- continuation prediction is calibrated;
- presence accuracy is high on configs with deaths;
- per-config metrics identify weak families/terrains.

Suggested minimum bar:

```text
held-out decoded_r2 > 0.70 on prototype split
rollout_h4_decoded_mae < 2x one-step decoded_mae
presence_acc > 0.95 on slot-balanced eval
reward_mae better than constant baseline
```

These numbers are starting points, not final research claims.

### Actor-Critic Acceptance

Before large-scale training:

- actor samples only valid actions;
- imagined rollouts have finite rewards/values;
- actor loss, critic loss, entropy are logged;
- policy improves over random on a tiny scenario;
- policy does not exploit invalid decoded states.

### Scaling Acceptance

For large generated-config training:

- audit passes before training;
- invalid configs are filtered before collection;
- GPU utilization is measured;
- transitions/sec and updates/sec are logged;
- train/eval/test config splits are frozen;
- per-config metrics are tracked over time.

## 10. Evaluation Protocol

Use three evaluation layers.

### 10.1 World Model Evaluation

Run:

```bash
python -m smac_jepa.evaluate \
  --manifest splits/generated_seed1.json \
  --split eval \
  --checkpoint runs/generated/checkpoint.pt \
  --out runs/generated/eval.json \
  --decode-sample-out runs/generated/decoded.json \
  --per-config-out runs/generated/per_config.json \
  --rollout-horizons 1,2,4,8,16 \
  --context-len 16
```

Track:

```text
one-step decoded metrics
rollout metrics
presence accuracy
reward metrics once added
continue metrics once added
per-config error
```

### 10.2 Policy Evaluation

Evaluate in the real SMACLite environment:

```text
mean episode return
win rate if available
episode length
invalid action count
death/survival stats
terrain/family breakdown
```

Compare against:

- random valid policy;
- scripted attack policy;
- scripted route policy;
- previous checkpoint.

### 10.3 Generalization Evaluation

Always evaluate by held-out config, not just held-out episodes.

Report:

```text
train configs
validation configs
test configs
family breakdown
terrain breakdown
unit-count range
episode count
transition count
```

## 11. Concrete Milestones

### Milestone A: World Model Plus Reward/Continue

Implement:

- reward target in dataset;
- continuation target in dataset;
- reward head;
- continuation head;
- reward/continue losses;
- reward/continue eval metrics.

Run:

```text
50 train configs / 10 eval configs
8-16 episodes/config
rollout horizons 1,2,4
```

Exit criteria:

- all tests pass;
- reward and continuation metrics are finite;
- held-out eval does not regress badly.

### Milestone B: Rollout-Aware Training

Implement:

- optional rollout loss;
- horizon schedule;
- rollout decoded/reward/continue losses;
- config flag to enable/disable.

Run:

```text
compare one-step-only vs rollout-loss model
same data split
same seed
```

Exit criteria:

- rollout horizon 4/8 improves;
- one-step metrics remain acceptable.

### Milestone C: Replay Buffer

Implement:

- replay buffer class;
- load existing `.npz`;
- append online episodes;
- sample sequence batches;
- save/load replay shards.

Exit criteria:

- replay samples match existing dataset batch format;
- train can use replay instead of static dataset.

### Milestone D: Actor/Critic And Imagination

Implement:

- actor;
- critic;
- imagination API;
- lambda returns;
- actor loss;
- critic loss.

Exit criteria:

- actor learns on synthetic/simple SMACLite scenario;
- no invalid action samples;
- losses finite.

### Milestone E: Online Dreamer-Style Training

Implement:

- online env loop;
- replay collection;
- model updates;
- actor/critic updates;
- periodic real-env evaluation.

Exit criteria:

- learned policy beats random on at least one small original SMACLite scenario;
- checkpoints can resume;
- logs are sufficient to debug failures.

### Milestone F: Large-Scale Generated Config Training

Run:

```text
1,000+ configs
32+ episodes/config
GPU training
held-out config eval
policy eval
```

Exit criteria:

- world model generalizes to held-out configs;
- policy improves across multiple config families;
- weak config families are identified.

## 12. Recommended Immediate Next Task

The next code task should be:

```text
Add reward and continuation heads to the world model.
```

Why:

- Dreamer-style actor-critic cannot train without reward and continuation predictions.
- The data already contains `rewards` and `dones`.
- This is the smallest change that moves the repo materially closer to DreamerV3 integration.

Implementation scope:

- dataset returns `reward_t` and `continue_t`;
- model predicts reward and continue from predicted latents;
- training logs `reward_loss` and `continue_loss`;
- evaluation reports reward and continuation metrics;
- tests cover shapes and finite losses.

After that:

```text
Add imagination API and actor/critic.
```

## 13. Practical Notes For Thousands Of Configs

If you have thousands of generated configs:

1. First filter invalid configs.
2. Cap entity count for the first large run.
3. Collect broad but shallow data first.
4. Train a world model.
5. Evaluate per-config.
6. Add more data to weak config groups.
7. Introduce scripted/policy rollouts.

Recommended first serious run:

```text
1,000 configs
32 episodes/config
100 average valid steps
~3.2M transitions
```

Recommended stronger run:

```text
2,000 configs
64 episodes/config
100 average valid steps
~12.8M transitions
```

Recommended high-quality RL-pretraining run:

```text
2,000 configs
128 episodes/config
100 average valid steps
~25.6M transitions
```

Do not collect all data with random actions. Use random + scripted + policy data.

## 14. Summary

The current repo has a strong foundation:

```text
entity-slot JEPA world model
SMACLite dataset support
generated-config generalization tools
decoded prediction outputs
rollout evaluation
```

To become DreamerV3-like, it needs:

```text
reward head
continuation head
replay buffer
imagination API
actor
critic
lambda returns
online env loop
GPU-scale training pipeline
```

The guiding principle is:

```text
keep the SMACLite-specific entity-slot world model
but build the surrounding training loop like DreamerV3
```

That lets the project use DreamerV3's proven model-based RL workflow without forcing a pixel/RSSM architecture onto a vector/entity simulator.
