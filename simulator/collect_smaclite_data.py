from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np


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
            state_dim=np.asarray(state_dim, dtype=np.int64),
            n_agents=np.asarray(n_agents, dtype=np.int64),
            n_enemies=np.asarray(n_enemies, dtype=np.int64),
            n_actions=np.asarray(n_actions, dtype=np.int64),
            ally_state_feat_size=np.asarray(ally_state_feat_size, dtype=np.int64),
            enemy_state_feat_size=np.asarray(enemy_state_feat_size, dtype=np.int64),
            max_steps=np.asarray(args.max_steps, dtype=np.int64),
            step_errors=np.asarray(step_errors, dtype=np.int64),
        )
        print(f"Saved {args.episodes} episodes to {out} step_errors={step_errors}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
