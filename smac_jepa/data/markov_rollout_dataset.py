from __future__ import annotations

"""
Markov-rollout dataset for SMAC-JEPA.

Parallel dataset. It does not modify the original SMACJEPADataset.

For rollout_window=p and rollout_horizon=n, one item contains:
    states:  s_t, ..., s_{t+p+n}
    actions: a_t, ..., a_{t+p+n-1}

The training script uses every start point inside the first p states and rolls
out n recursive prediction steps from each start.
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from smac_jepa.data import SMACJEPADataset


class MarkovRolloutSMACJEPADataset(SMACJEPADataset):
    """Dataset returning random/sequential full rollout segments."""

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
    ):
        if rollout_window < 1:
            raise ValueError("rollout_window must be >= 1")
        if rollout_horizon < 1:
            raise ValueError("rollout_horizon must be >= 1")

        self.rollout_window = int(rollout_window)
        self.rollout_horizon = int(rollout_horizon)
        self.segment_action_len = self.rollout_window + self.rollout_horizon
        self.segment_state_len = self.segment_action_len + 1

        # Parent handles loading/caps/metadata/utilities. We override indexing/items.
        super().__init__(
            paths=paths,
            context_len=self.segment_state_len,
            mode=mode,
            window_mode=window_mode,
            window_len=self.segment_state_len,
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

        self.index: list[tuple[int, int]] = []
        for episode_idx, episode in enumerate(self.episodes):
            valid = episode["valid"]
            horizon = episode["actions"].shape[0]
            max_start = horizon - self.segment_action_len
            if max_start < 0:
                continue
            for start in range(0, max_start + 1):
                window_valid = valid[start : start + self.segment_action_len]
                if window_valid.size == self.segment_action_len and np.all(window_valid):
                    self.index.append((episode_idx, start))

        if not self.index:
            raise ValueError(
                "No valid Markov-rollout segments found. "
                f"Need at least rollout_window + rollout_horizon = {self.segment_action_len} "
                "valid transitions per sampled segment."
            )

        self.random_indices = list(range(len(self.index)))
        if self.window_mode == "random" and self.samples_per_epoch is None:
            self.samples_per_epoch = len(self.index)

    def __len__(self) -> int:
        if self.window_mode == "random":
            assert self.samples_per_epoch is not None
            return self.samples_per_epoch
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.window_mode == "random":
            idx = int(self.rng.choice(self.random_indices))
        episode_idx, start = self.index[idx]
        return self._entity_markov_rollout_item(episode_idx, start)

    def _entity_markov_rollout_item(self, episode_idx: int, start: int) -> dict[str, torch.Tensor]:
        episode = self.episodes[episode_idx]
        states = episode["states"]
        actions = episode["actions"]
        valid = episode["valid"]
        meta = episode["metadata"]

        action_end = start + self.segment_action_len
        state_end = action_end + 1

        state_seq = states[start:state_end]
        action_seq = actions[start:action_end]
        valid_actions = valid[start:action_end].astype(np.float32)

        if state_seq.shape[0] != self.segment_state_len:
            raise IndexError(f"Expected {self.segment_state_len} states, got {state_seq.shape[0]}")
        if action_seq.shape[0] != self.segment_action_len:
            raise IndexError(f"Expected {self.segment_action_len} actions, got {action_seq.shape[0]}")

        entity_seq, entity_mask_seq = self._encode_state_window(
            state_seq,
            meta,
            episode["entity_static"],
        )
        action_t, action_mask = self._pad_actions(action_seq, meta)
        action_mask *= valid_actions[:, None]

        state_valid = np.ones((self.segment_state_len,), dtype=np.float32)
        state_valid[0] = valid_actions[0]
        state_valid[1:] = valid_actions
        entity_mask_seq *= state_valid[:, None]

        slot_mask = self._slot_mask(meta)
        entity_slot_mask = np.repeat(slot_mask[None, :], self.segment_state_len, axis=0)
        entity_slot_mask *= state_valid[:, None]

        return {
            "entity_seq": torch.from_numpy(entity_seq),
            "entity_mask_seq": torch.from_numpy(entity_mask_seq),
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
