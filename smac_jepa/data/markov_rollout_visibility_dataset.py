from __future__ import annotations

"""
Visibility-masked Markov-rollout dataset for SMAC-JEPA.

Parallel dataset file. It does not modify the original dataset or the plain
markov_rollout_dataset.py.

Purpose:
    For partially observable training, the model input state sequence can hide
    enemy entity tokens that are outside ally sight range, while the target
    sequence remains full-state.

Returned tensors:
    entity_seq:
        visibility-masked input observations

    entity_mask_seq:
        masks for visibility-masked inputs

    target_entity_seq:
        full-state targets, not visibility masked

    target_entity_mask_seq:
        full-state target masks

The RNN-memory rollout script should:
    - encode entity_seq for initial rollout states
    - encode target_entity_seq for target latents/losses
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from smac_jepa.data.markov_rollout_dataset import MarkovRolloutSMACJEPADataset


class VisibilityMarkovRolloutSMACJEPADataset(MarkovRolloutSMACJEPADataset):
    explicit_visibility_mask_version = 1

    """
    Markov rollout dataset with optional enemy visibility masking on inputs only.

    This assumes the entity dynamic layout used by your SMACLite data:
        ally features occupy the first n_agents * ally_state_feat_size entries
        enemy features follow immediately after

    By default, xy coordinates are assumed to be feature indices 2 and 3 inside
    ally/enemy dynamic features, matching the usual SMACLite-like layout used in
    the project.
    """

    def __init__(
        self,
        paths: str | Path | Iterable[str | Path],
        rollout_window: int = 20,
        rollout_horizon: int = 5,
        mode: str = "entity",
        window_mode: str = "random",
        samples_per_epoch: int | None = None,
        seed: int = 1,
        max_agents: int | None = None,
        max_enemies: int | None = None,
        max_actions: int | None = None,
        token_dim: int | None = None,
        dynamic_token_dim: int | None = None,
        static_dim: int | None = None,
        entity_static_feat_size: int | None = None,
        enemy_visibility_mask: bool = False,
        enemy_sight_range: float = 9.0,
        xy_indices: tuple[int, int] = (2, 3),
    ):
        self.enemy_visibility_mask = bool(enemy_visibility_mask)
        self.enemy_sight_range = float(enemy_sight_range)
        self.xy_indices = tuple(xy_indices)

        super().__init__(
            paths=paths,
            rollout_window=rollout_window,
            rollout_horizon=rollout_horizon,
            mode=mode,
            window_mode=window_mode,
            samples_per_epoch=samples_per_epoch,
            seed=seed,
            max_agents=max_agents,
            max_enemies=max_enemies,
            max_actions=max_actions,
            token_dim=token_dim,
            dynamic_token_dim=dynamic_token_dim,
            static_dim=static_dim,
            entity_static_feat_size=entity_static_feat_size,
        )

    def _entity_markov_rollout_item(
        self,
        episode_idx: int,
        start: int,
    ) -> dict[str, torch.Tensor]:
        episode = self.episodes[episode_idx]
        states = episode["states"]
        actions = episode["actions"]
        valid = episode["valid"]
        meta = episode["metadata"]

        action_end = start + self.segment_action_len
        state_end = action_end + 1

        full_state_seq = states[start:state_end]
        enemy_visible_seq = self._compute_enemy_visibility(
            full_state_seq,
            meta,
            episode.get("static_condition"),
        )
        input_state_seq = full_state_seq.copy()

        if self.enemy_visibility_mask:
            input_state_seq = self._apply_enemy_visibility_mask_to_states(
                input_state_seq,
                meta,
                episode.get("static_condition"),
            )

        action_seq = actions[start:action_end]
        valid_actions = valid[start:action_end].astype(np.float32)

        if full_state_seq.shape[0] != self.segment_state_len:
            raise IndexError(
                f"Expected {self.segment_state_len} states, got {full_state_seq.shape[0]}"
            )
        if action_seq.shape[0] != self.segment_action_len:
            raise IndexError(
                f"Expected {self.segment_action_len} actions, got {action_seq.shape[0]}"
            )

        # Input observations may be visibility-masked.
        entity_seq, entity_mask_seq = self._encode_state_window(
            input_state_seq,
            meta,
            episode["entity_static"],
        )

        # Targets stay full-state.
        target_entity_seq, target_entity_mask_seq = self._encode_state_window(
            full_state_seq,
            meta,
            episode["entity_static"],
        )

        # Explicit observation mask. The legacy entity mask may remain 1 for a
        # zeroed enemy token because static entity features are still present.
        # Hidden enemies must instead be 0 here so the encoder and recurrent
        # memory do not treat an unobserved zero token as a real observation.
        observation_mask_seq = target_entity_mask_seq.copy()
        if self.enemy_visibility_mask and meta.n_enemies > 0:
            enemy_start = int(self.metadata.max_agents)
            enemy_stop = enemy_start + int(meta.n_enemies)
            observation_mask_seq[:, enemy_start:enemy_stop] *= (
                enemy_visible_seq.astype(np.float32)
            )
        entity_mask_seq = observation_mask_seq.copy()

        action_t, action_mask = self._pad_actions(action_seq, meta)
        action_mask *= valid_actions[:, None]

        state_valid = np.ones((self.segment_state_len,), dtype=np.float32)
        state_valid[0] = valid_actions[0]
        state_valid[1:] = valid_actions

        observation_mask_seq *= state_valid[:, None]
        entity_mask_seq = observation_mask_seq.copy()
        target_entity_mask_seq *= state_valid[:, None]

        slot_mask = self._slot_mask(meta)
        entity_slot_mask = np.repeat(slot_mask[None, :], self.segment_state_len, axis=0)
        entity_slot_mask *= state_valid[:, None]

        return {
            "entity_seq": torch.from_numpy(entity_seq),
            # Backward compatibility: entity_mask_seq is now the true
            # observation mask rather than a structural/presence mask.
            "entity_mask_seq": torch.from_numpy(entity_mask_seq),
            "observation_mask_seq": torch.from_numpy(
                observation_mask_seq
            ),
            "target_entity_seq": torch.from_numpy(target_entity_seq),
            "target_entity_mask_seq": torch.from_numpy(target_entity_mask_seq),
            "entity_slot_mask_seq": torch.from_numpy(entity_slot_mask),
            "action_seq": torch.from_numpy(action_t),
            "action_mask_seq": torch.from_numpy(action_mask),
            "state_mask": torch.from_numpy(state_valid),
            "action_valid_mask": torch.from_numpy(valid_actions),
            "static_condition": torch.from_numpy(
                self._pad_static_condition(episode["static_condition"], meta)
            ),
            "segment_start": torch.tensor(start, dtype=torch.long),
            "episode_index": torch.tensor(episode_idx, dtype=torch.long),
        }

    def _compute_enemy_visibility(
        self,
        state_seq: np.ndarray,
        meta,
        static_condition: np.ndarray | None,
    ) -> np.ndarray:
        """Return a [T, n_enemies] boolean visibility matrix."""
        steps = int(state_seq.shape[0])
        if meta.n_enemies <= 0:
            return np.zeros((steps, 0), dtype=bool)
        if meta.n_agents <= 0:
            return np.zeros((steps, meta.n_enemies), dtype=bool)

        x_idx, y_idx = self.xy_indices
        if (
            meta.ally_state_feat_size <= max(x_idx, y_idx)
            or meta.enemy_state_feat_size <= max(x_idx, y_idx)
        ):
            # Fall back conservatively when positions cannot be inferred.
            ally_end = meta.n_agents * meta.ally_state_feat_size
            enemy_end = (
                ally_end + meta.n_enemies * meta.enemy_state_feat_size
            )
            enemies = state_seq[:, ally_end:enemy_end].reshape(
                steps,
                meta.n_enemies,
                meta.enemy_state_feat_size,
            )
            return np.abs(enemies).sum(axis=-1) > 0

        ally_end = meta.n_agents * meta.ally_state_feat_size
        enemy_end = ally_end + meta.n_enemies * meta.enemy_state_feat_size
        allies = state_seq[:, :ally_end].reshape(
            steps, meta.n_agents, meta.ally_state_feat_size
        )
        enemies = state_seq[:, ally_end:enemy_end].reshape(
            steps, meta.n_enemies, meta.enemy_state_feat_size
        )

        ally_present = np.abs(allies).sum(axis=-1) > 0
        enemy_present = np.abs(enemies).sum(axis=-1) > 0
        ally_alive = ally_present
        if meta.ally_state_feat_size >= 1:
            ally_alive = ally_alive & (allies[..., 0] > 0)

        ally_xy = allies[..., [x_idx, y_idx]]
        enemy_xy = enemies[..., [x_idx, y_idx]]
        scale_x = 1.0
        scale_y = 1.0

        coords = np.concatenate(
            [ally_xy.reshape(-1, 2), enemy_xy.reshape(-1, 2)], axis=0
        )
        finite_coords = coords[np.isfinite(coords).all(axis=1)]
        looks_normalized = (
            finite_coords.size > 0
            and np.nanmax(np.abs(finite_coords)) <= 2.0
        )
        if (
            looks_normalized
            and static_condition is not None
            and len(static_condition) >= 2
        ):
            sx = float(static_condition[0])
            sy = float(static_condition[1])
            if np.isfinite(sx) and sx > 0:
                scale_x = sx
            if np.isfinite(sy) and sy > 0:
                scale_y = sy

        visible = np.zeros((steps, meta.n_enemies), dtype=bool)
        for timestep in range(steps):
            alive_indices = np.flatnonzero(ally_alive[timestep])
            if alive_indices.size == 0:
                continue
            ally_pos = ally_xy[timestep, alive_indices]
            enemy_pos = enemy_xy[timestep]
            dx = (
                enemy_pos[:, None, 0] - ally_pos[None, :, 0]
            ) * scale_x
            dy = (
                enemy_pos[:, None, 1] - ally_pos[None, :, 1]
            ) * scale_y
            distance = np.sqrt(dx * dx + dy * dy)
            visible[timestep] = enemy_present[timestep] & (
                distance.min(axis=1) <= self.enemy_sight_range
            )
        return visible

    def _apply_enemy_visibility_mask_to_states(
        self,
        state_seq: np.ndarray,
        meta,
        static_condition: np.ndarray | None,
    ) -> np.ndarray:
        """
        Zero enemy dynamic features when no alive ally is within sight range.

        Targets are not passed through this function.
        """
        if meta.n_enemies <= 0 or meta.n_agents <= 0:
            return state_seq

        x_idx, y_idx = self.xy_indices
        if meta.ally_state_feat_size <= max(x_idx, y_idx) or meta.enemy_state_feat_size <= max(x_idx, y_idx):
            # Cannot infer positions safely; leave unchanged rather than silently breaking.
            return state_seq

        masked = state_seq.copy()
        steps = masked.shape[0]

        ally_end = meta.n_agents * meta.ally_state_feat_size
        enemy_end = ally_end + meta.n_enemies * meta.enemy_state_feat_size

        allies = masked[:, :ally_end].reshape(steps, meta.n_agents, meta.ally_state_feat_size)
        enemies = masked[:, ally_end:enemy_end].reshape(steps, meta.n_enemies, meta.enemy_state_feat_size)

        ally_present = (np.abs(allies).sum(axis=-1) > 0)
        enemy_present = (np.abs(enemies).sum(axis=-1) > 0)

        # If feature 0 behaves like health/alive, also require it to be positive.
        ally_alive = ally_present
        if meta.ally_state_feat_size >= 1:
            ally_alive = ally_alive & (allies[..., 0] > 0)

        ally_xy = allies[..., [x_idx, y_idx]]
        enemy_xy = enemies[..., [x_idx, y_idx]]

        scale_x = 1.0
        scale_y = 1.0

        # Heuristic for normalized coordinates:
        # if coordinates look normalized and static_condition has map dimensions,
        # convert coordinate differences back to approximate map units.
        coords = np.concatenate(
            [ally_xy.reshape(-1, 2), enemy_xy.reshape(-1, 2)],
            axis=0,
        )
        finite_coords = coords[np.isfinite(coords).all(axis=1)]
        looks_normalized = finite_coords.size > 0 and np.nanmax(np.abs(finite_coords)) <= 2.0

        if looks_normalized and static_condition is not None and len(static_condition) >= 2:
            sx = float(static_condition[0])
            sy = float(static_condition[1])
            if np.isfinite(sx) and sx > 0:
                scale_x = sx
            if np.isfinite(sy) and sy > 0:
                scale_y = sy

        visible = np.zeros((steps, meta.n_enemies), dtype=bool)

        for t in range(steps):
            alive_ally_idx = np.flatnonzero(ally_alive[t])
            if alive_ally_idx.size == 0:
                continue

            ally_pos = ally_xy[t, alive_ally_idx]  # [A_alive, 2]
            enemy_pos = enemy_xy[t]                # [E, 2]

            dx = (enemy_pos[:, None, 0] - ally_pos[None, :, 0]) * scale_x
            dy = (enemy_pos[:, None, 1] - ally_pos[None, :, 1]) * scale_y
            dist = np.sqrt(dx * dx + dy * dy)

            visible[t] = enemy_present[t] & (dist.min(axis=1) <= self.enemy_sight_range)

        invisible = enemy_present & ~visible
        enemies[invisible] = 0.0

        return masked
