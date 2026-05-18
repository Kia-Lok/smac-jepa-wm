from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np


TERRAIN_TYPES = {"_": 0.0, "C": 1.0, "X": 2.0}
ENTITY_STATIC_FEAT_SIZE = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect random-policy SMACLite trajectories")
    parser.add_argument("--env-key", required=True, help="Example: smaclite:smaclite/2s3z-v0")
    parser.add_argument("--scenario-name", help="Readable scenario name to store in metadata")
    parser.add_argument("--map-file", help="Optional custom SMACLite JSON map file")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stop-on-step-error", action="store_true")
    return parser.parse_args()


def sample_valid_actions(avail_actions: list[np.ndarray], rng: np.random.Generator) -> list[int]:
    actions: list[int] = []
    for avail in avail_actions:
        valid = np.flatnonzero(np.asarray(avail) > 0)
        if len(valid) == 0:
            actions.append(0)
        else:
            actions.append(int(rng.choice(valid)))
    return actions


def static_condition_from_env(smac_env) -> np.ndarray:
    map_info = smac_env.map_info
    terrain = np.asarray(
        [
            [TERRAIN_TYPES.get(getattr(cell, "value", str(cell)), 0.0) / 2.0 for cell in row]
            for row in map_info.terrain
        ],
        dtype=np.float32,
    )
    terrain_counts = np.asarray(
        [(terrain == value).mean() for value in (0.0, 0.5, 1.0)],
        dtype=np.float32,
    )
    flat_terrain = np.zeros((32 * 32,), dtype=np.float32)
    height = min(terrain.shape[0], 32)
    width = min(terrain.shape[1], 32)
    flat_terrain.reshape(32, 32)[:height, :width] = terrain[:height, :width]
    base = np.asarray(
        [
            float(map_info.width) / 64.0,
            float(map_info.height) / 64.0,
            float(map_info.attack_point[0]) / max(float(map_info.width), 1.0),
            float(map_info.attack_point[1]) / max(float(map_info.height), 1.0),
            float(bool(map_info.ally_has_shields)),
            float(bool(map_info.enemy_has_shields)),
            float(map_info.num_unit_types) / 16.0,
            float(smac_env.n_agents) / 64.0,
            float(smac_env.n_enemies) / 256.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, terrain_counts, flat_terrain]).astype(np.float32)


def entity_static_from_env(smac_env) -> np.ndarray:
    rows: list[np.ndarray] = []
    for collection in (smac_env.agents, smac_env.enemies):
        for _, unit in sorted(collection.items()):
            stats = unit.type.stats
            rows.append(
                np.asarray(
                    [
                        float(stats.hp) / 1000.0,
                        float(stats.shield) / 1000.0,
                        float(stats.damage) / 100.0,
                        float(stats.cooldown) / 100.0,
                        float(stats.speed) / 10.0,
                        float(stats.attack_range) / 20.0,
                        float(stats.size) / 10.0,
                        float(stats.armor) / 10.0,
                        float(stats.energy) / 200.0,
                        float(stats.attacks) / 10.0,
                        1.0 if str(stats.combat_type).endswith("HEALING") else 0.0,
                        1.0 if str(stats.plane).endswith("AIR") else 0.0,
                    ],
                    dtype=np.float32,
                )
            )
    if not rows:
        return np.zeros((0, ENTITY_STATIC_FEAT_SIZE), dtype=np.float32)
    return np.stack(rows, axis=0)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    make_kwargs = {"map_file": args.map_file} if args.map_file else {}
    env = gym.make(args.env_key, **make_kwargs)
    smac_env = env.unwrapped
    try:
        env.reset(seed=args.seed)
        state = np.asarray(smac_env.get_state(), dtype=np.float32)
        state_dim = state.shape[0]
        n_agents = int(smac_env.n_agents)
        n_enemies = int(getattr(smac_env, "n_enemies", 0))
        n_actions = int(smac_env.n_actions)
        ally_state_feat_size = int(getattr(smac_env, "ally_state_feat_size", 0))
        enemy_state_feat_size = int(getattr(smac_env, "enemy_state_feat_size", 0))
        map_info = getattr(smac_env, "map_info", None)
        terrain_preset = ""
        if args.map_file:
            try:
                terrain_preset = str(json.loads(Path(args.map_file).read_text()).get("terrain_preset", ""))
            except (OSError, json.JSONDecodeError):
                terrain_preset = ""
        ally_has_shields = bool(getattr(map_info, "ally_has_shields", False))
        enemy_has_shields = bool(getattr(map_info, "enemy_has_shields", False))
        num_unit_types = int(getattr(map_info, "num_unit_types", 0))
        static_condition = static_condition_from_env(smac_env)
        entity_static = entity_static_from_env(smac_env)

        states = np.zeros((args.episodes, args.max_steps + 1, state_dim), dtype=np.float32)
        actions = np.zeros((args.episodes, args.max_steps, n_agents), dtype=np.int64)
        action_onehot = np.zeros(
            (args.episodes, args.max_steps, n_agents, n_actions), dtype=np.float32
        )
        rewards = np.zeros((args.episodes, args.max_steps), dtype=np.float32)
        dones = np.zeros((args.episodes, args.max_steps), dtype=bool)
        valid = np.zeros((args.episodes, args.max_steps), dtype=bool)
        avail_store = np.zeros(
            (args.episodes, args.max_steps, n_agents, n_actions), dtype=np.float32
        )
        step_errors = 0

        for episode in range(args.episodes):
            env.reset(seed=args.seed + episode)
            states[episode, 0] = np.asarray(smac_env.get_state(), dtype=np.float32)
            done = False
            for step in range(args.max_steps):
                avail_actions = [
                    np.asarray(a, dtype=np.float32) for a in smac_env.get_avail_actions()
                ]
                joint_action = sample_valid_actions(avail_actions, rng)
                try:
                    _, reward, terminated, truncated, _ = env.step(joint_action)
                except Exception:
                    step_errors += 1
                    if args.stop_on_step_error:
                        raise
                    break
                done = bool(terminated or truncated)

                actions[episode, step] = np.asarray(joint_action, dtype=np.int64)
                action_onehot[episode, step] = np.eye(n_actions, dtype=np.float32)[joint_action]
                rewards[episode, step] = float(reward)
                dones[episode, step] = done
                valid[episode, step] = True
                avail_store[episode, step] = np.asarray(avail_actions, dtype=np.float32)
                states[episode, step + 1] = np.asarray(smac_env.get_state(), dtype=np.float32)

                if done:
                    break

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
            avail_actions=avail_store,
            scenario=np.asarray(args.scenario_name or args.env_key),
            env_key=np.asarray(args.env_key),
            map_file=np.asarray(args.map_file or ""),
            terrain_preset=np.asarray(terrain_preset),
            map_width=np.asarray(int(getattr(map_info, "width", 0)), dtype=np.int64),
            map_height=np.asarray(int(getattr(map_info, "height", 0)), dtype=np.int64),
            attack_point=np.asarray(getattr(map_info, "attack_point", (0, 0)), dtype=np.float32),
            state_dim=np.asarray(state_dim, dtype=np.int64),
            n_agents=np.asarray(n_agents, dtype=np.int64),
            n_enemies=np.asarray(n_enemies, dtype=np.int64),
            n_actions=np.asarray(n_actions, dtype=np.int64),
            ally_state_feat_size=np.asarray(ally_state_feat_size, dtype=np.int64),
            enemy_state_feat_size=np.asarray(enemy_state_feat_size, dtype=np.int64),
            ally_has_shields=np.asarray(ally_has_shields, dtype=bool),
            enemy_has_shields=np.asarray(enemy_has_shields, dtype=bool),
            num_unit_types=np.asarray(num_unit_types, dtype=np.int64),
            static_condition=static_condition,
            static_dim=np.asarray(static_condition.shape[0], dtype=np.int64),
            entity_static=entity_static,
            entity_static_feat_size=np.asarray(entity_static.shape[1], dtype=np.int64),
            max_steps=np.asarray(args.max_steps, dtype=np.int64),
            step_errors=np.asarray(step_errors, dtype=np.int64),
        )
        print(f"Saved {args.episodes} episodes to {out} step_errors={step_errors}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
