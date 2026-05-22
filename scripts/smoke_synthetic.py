from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic SMAC-JEPA NPZ")
    parser.add_argument("--out", default="data/synthetic.npz")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=10)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--n-enemies", type=int, default=2)
    parser.add_argument("--n-actions", type=int, default=5)
    parser.add_argument("--entity-layout", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ally_state_feat_size = 4
    enemy_state_feat_size = 3
    state_dim = args.state_dim
    if args.entity_layout:
        state_dim = (
            args.n_agents * ally_state_feat_size
            + args.n_enemies * enemy_state_feat_size
            + args.n_agents * args.n_actions
        )
    states = np.zeros((args.episodes, args.steps + 1, state_dim), dtype=np.float32)
    actions = rng.integers(
        0,
        args.n_actions,
        size=(args.episodes, args.steps, args.n_agents),
        dtype=np.int64,
    )
    action_onehot = np.eye(args.n_actions, dtype=np.float32)[actions]
    rewards = rng.normal(size=(args.episodes, args.steps)).astype(np.float32)
    dones = np.zeros((args.episodes, args.steps), dtype=bool)
    valid = np.ones((args.episodes, args.steps), dtype=bool)
    avail_actions = np.ones(
        (args.episodes, args.steps, args.n_agents, args.n_actions), dtype=np.float32
    )

    dynamics = rng.normal(scale=0.05, size=(args.n_agents * args.n_actions, state_dim)).astype(
        np.float32
    )
    states[:, 0] = rng.normal(size=(args.episodes, state_dim))
    for step in range(args.steps):
        action_features = action_onehot[:, step].reshape(args.episodes, -1)
        states[:, step + 1] = (
            states[:, step]
            + action_features @ dynamics
            + rng.normal(scale=0.01, size=(args.episodes, state_dim))
        )
        if args.entity_layout:
            tail = args.n_agents * ally_state_feat_size + args.n_enemies * enemy_state_feat_size
            states[:, step + 1, tail:] = action_features

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        states=states,
        actions=actions,
        action_onehot=action_onehot,
        rewards=rewards,
        dones=dones,
        valid=valid,
        avail_actions=avail_actions,
        scenario=np.asarray("synthetic"),
        state_dim=np.asarray(state_dim, dtype=np.int64),
        n_agents=np.asarray(args.n_agents, dtype=np.int64),
        n_enemies=np.asarray(args.n_enemies if args.entity_layout else 0, dtype=np.int64),
        n_actions=np.asarray(args.n_actions, dtype=np.int64),
        ally_state_feat_size=np.asarray(
            ally_state_feat_size if args.entity_layout else 0, dtype=np.int64
        ),
        enemy_state_feat_size=np.asarray(
            enemy_state_feat_size if args.entity_layout else 0, dtype=np.int64
        ),
        ally_has_shields=np.asarray(False, dtype=bool),
        enemy_has_shields=np.asarray(False, dtype=bool),
        num_unit_types=np.asarray(0, dtype=np.int64),
        static_condition=np.zeros((0,), dtype=np.float32),
        static_dim=np.asarray(0, dtype=np.int64),
        entity_static=np.zeros(
            (args.n_agents + (args.n_enemies if args.entity_layout else 0), 0),
            dtype=np.float32,
        ),
        entity_static_feat_size=np.asarray(0, dtype=np.int64),
        max_steps=np.asarray(args.steps, dtype=np.int64),
    )
    print(f"Saved synthetic dataset to {out}")


if __name__ == "__main__":
    main()
