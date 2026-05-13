from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetMetadata:
    state_dim: int
    n_agents: int
    n_actions: int
    n_enemies: int = 0
    ally_state_feat_size: int = 0
    enemy_state_feat_size: int = 0
    max_agents: int = 0
    max_enemies: int = 0
    max_actions: int = 0
    token_dim: int = 0
    mode: str = "flat"


def _as_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def load_npz_metadata(path: str | Path) -> DatasetMetadata:
    with np.load(path, allow_pickle=False) as data:
        return _read_metadata(data)


def load_manifest(path: str | Path, split: str = "train") -> list[Path]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text())
    if "datasets" in manifest and split in manifest["datasets"]:
        values = manifest["datasets"][split]
    else:
        values = manifest[f"{split}_data"]
    return [
        (manifest_path.parent / value).resolve() if not Path(value).is_absolute() else Path(value)
        for value in values
    ]


def _scalar(data: np.lib.npyio.NpzFile, key: str, default: int = 0) -> int:
    if key not in data:
        return default
    return int(np.asarray(data[key]).item())


def _read_metadata(data: np.lib.npyio.NpzFile) -> DatasetMetadata:
    n_agents = _scalar(data, "n_agents")
    n_actions = _scalar(data, "n_actions")
    n_enemies = _scalar(data, "n_enemies")
    ally_size = _scalar(data, "ally_state_feat_size")
    enemy_size = _scalar(data, "enemy_state_feat_size")
    has_entity_layout = n_enemies > 0 and ally_size > 0 and enemy_size > 0
    return DatasetMetadata(
        state_dim=_scalar(data, "state_dim"),
        n_agents=n_agents,
        n_actions=n_actions,
        n_enemies=n_enemies,
        ally_state_feat_size=ally_size,
        enemy_state_feat_size=enemy_size,
        max_agents=n_agents,
        max_enemies=n_enemies,
        max_actions=n_actions,
        token_dim=max(ally_size, enemy_size, 1),
        mode="entity" if has_entity_layout else "flat",
    )


def _merge_metadata(items: list[DatasetMetadata], mode: str) -> DatasetMetadata:
    if not items:
        raise ValueError("At least one metadata item is required")
    if mode == "flat":
        first = items[0]
        for item in items[1:]:
            if (
                item.state_dim != first.state_dim
                or item.n_agents != first.n_agents
                or item.n_actions != first.n_actions
            ):
                raise ValueError(f"Flat datasets need matching metadata: {item} != {first}")
        return first

    return DatasetMetadata(
        state_dim=max(item.state_dim for item in items),
        n_agents=max(item.n_agents for item in items),
        n_actions=max(item.n_actions for item in items),
        n_enemies=max(item.n_enemies for item in items),
        ally_state_feat_size=max(item.ally_state_feat_size for item in items),
        enemy_state_feat_size=max(item.enemy_state_feat_size for item in items),
        max_agents=max(item.n_agents for item in items),
        max_enemies=max(item.n_enemies for item in items),
        max_actions=max(item.n_actions for item in items),
        token_dim=max(
            max(item.ally_state_feat_size, item.enemy_state_feat_size) for item in items
        ),
        mode="entity",
    )


def _one_hot(actions: np.ndarray, n_actions: int) -> np.ndarray:
    clipped = np.clip(actions.astype(np.int64), 0, n_actions - 1)
    return np.eye(n_actions, dtype=np.float32)[clipped]


class SMACJEPADataset(Dataset):
    """Windowed JEPA dataset.

    Each item contains an observation sequence, an action-history conditioning
    sequence, and the next observation sequence shifted by one step.
    """

    def __init__(
        self,
        paths: str | Path | Iterable[str | Path],
        context_len: int = 1,
        mode: str = "auto",
        max_agents: int | None = None,
        max_enemies: int | None = None,
        max_actions: int | None = None,
        token_dim: int | None = None,
    ):
        if context_len < 1:
            raise ValueError("context_len must be at least 1")
        self.paths = _as_paths(paths)
        self.context_len = context_len
        self.mode = mode
        if not self.paths:
            raise ValueError("At least one dataset path is required")

        states_list: list[np.ndarray] = []
        actions_list: list[np.ndarray] = []
        valid_list: list[np.ndarray] = []
        metadata_items: list[DatasetMetadata] = []

        for path in self.paths:
            with np.load(path, allow_pickle=False) as data:
                current = _read_metadata(data)
                metadata_items.append(current)

                states = data["states"].astype(np.float32)
                if "action_onehot" in data:
                    actions = data["action_onehot"].astype(np.float32)
                else:
                    actions = _one_hot(data["actions"], current.n_actions)
                if "valid" in data:
                    valid = data["valid"].astype(bool)
                else:
                    valid = np.ones(actions.shape[:2], dtype=bool)

                states_list.append(states)
                actions_list.append(actions)
                valid_list.append(valid)

        available_modes = {item.mode for item in metadata_items}
        if mode == "auto":
            resolved_mode = "entity" if available_modes == {"entity"} else "flat"
        else:
            resolved_mode = mode
        if resolved_mode == "entity" and "flat" in available_modes:
            raise ValueError("Entity mode requires datasets with SMACLite state layout metadata")
        self.mode = resolved_mode
        self.metadata = _merge_metadata(metadata_items, resolved_mode)
        if resolved_mode == "entity":
            override = {
                "max_agents": max_agents or self.metadata.max_agents,
                "max_enemies": max_enemies or self.metadata.max_enemies,
                "max_actions": max_actions or self.metadata.max_actions,
                "token_dim": token_dim or self.metadata.token_dim,
            }
            if override["max_agents"] < self.metadata.max_agents:
                raise ValueError("max_agents override is smaller than dataset requirement")
            if override["max_enemies"] < self.metadata.max_enemies:
                raise ValueError("max_enemies override is smaller than dataset requirement")
            if override["max_actions"] < self.metadata.max_actions:
                raise ValueError("max_actions override is smaller than dataset requirement")
            if override["token_dim"] < self.metadata.token_dim:
                raise ValueError("token_dim override is smaller than dataset requirement")
            self.metadata = replace(self.metadata, **override)
        self.scenarios: list[str] = []
        self.episodes: list[dict[str, Any]] = []
        self.index: list[tuple[int, int]] = []

        for path in self.paths:
            with np.load(path, allow_pickle=False) as data:
                item_meta = _read_metadata(data)
                scenario = str(np.asarray(data["scenario"]).item()) if "scenario" in data else path.stem
                states = data["states"].astype(np.float32)
                if "action_onehot" in data:
                    actions = data["action_onehot"].astype(np.float32)
                else:
                    actions = _one_hot(data["actions"], item_meta.n_actions)
                if "valid" in data:
                    valid = data["valid"].astype(bool)
                else:
                    valid = np.ones(actions.shape[:2], dtype=bool)
                for episode_idx in range(actions.shape[0]):
                    record: dict[str, Any] = {
                        "states": states[episode_idx],
                        "actions": actions[episode_idx],
                        "valid": valid[episode_idx],
                        "metadata": item_meta,
                        "scenario": scenario,
                    }
                    self.episodes.append(record)
                    self.scenarios.append(scenario)

        if resolved_mode == "flat":
            self.states = np.concatenate(states_list, axis=0)
            self.actions = np.concatenate(actions_list, axis=0)
            self.valid = np.concatenate(valid_list, axis=0)
        else:
            self.states = None
            self.actions = None
            self.valid = None

        self.index: list[tuple[int, int]] = []

        for episode_idx, episode in enumerate(self.episodes):
            horizon = episode["actions"].shape[0]
            for start in range(0, horizon - context_len + 1):
                window_valid = episode["valid"][start : start + context_len]
                if np.all(window_valid):
                    self.index.append((episode_idx, start))

        if not self.index:
            raise ValueError("No valid training windows found")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode_idx, start = self.index[idx]
        end = start + self.context_len
        if self.mode == "entity":
            return self._entity_item(episode_idx, start, end)

        assert self.states is not None and self.actions is not None and self.valid is not None
        return {
            # Observation sequence: state_t ... state_{t+K-1}
            "state_t": torch.from_numpy(self.states[episode_idx, start:end]),
            # Conditioning sequence: actions taken from those previous states.
            "action_t": torch.from_numpy(self.actions[episode_idx, start:end]),
            # Prediction target sequence: state_{t+1} ... state_{t+K}
            "target_state": torch.from_numpy(self.states[episode_idx, start + 1 : end + 1]),
            "mask": torch.from_numpy(self.valid[episode_idx, start:end].astype(np.float32)),
        }

    def _entity_item(self, episode_idx: int, start: int, end: int) -> dict[str, torch.Tensor]:
        episode = self.episodes[episode_idx]
        states = episode["states"]
        actions = episode["actions"]
        valid = episode["valid"]
        meta = episode["metadata"]
        entity_t, entity_mask = self._encode_state_window(states[start:end], meta)
        target_entity, target_entity_mask = self._encode_state_window(
            states[start + 1 : end + 1], meta
        )
        action_t, action_mask = self._pad_actions(actions[start:end], meta)
        return {
            "entity_t": torch.from_numpy(entity_t),
            "entity_mask": torch.from_numpy(entity_mask),
            "action_t": torch.from_numpy(action_t),
            "action_mask": torch.from_numpy(action_mask),
            "target_entity": torch.from_numpy(target_entity),
            "target_entity_mask": torch.from_numpy(target_entity_mask),
            "mask": torch.from_numpy(valid[start:end].astype(np.float32)),
            "scenario_id": torch.tensor(0, dtype=torch.long),
        }

    def _encode_state_window(
        self, states: np.ndarray, meta: DatasetMetadata
    ) -> tuple[np.ndarray, np.ndarray]:
        steps = states.shape[0]
        token_count = self.metadata.max_agents + self.metadata.max_enemies
        features = np.zeros((steps, token_count, self.metadata.token_dim), dtype=np.float32)
        mask = np.zeros((steps, token_count), dtype=np.float32)
        ally_end = meta.n_agents * meta.ally_state_feat_size
        enemy_end = ally_end + meta.n_enemies * meta.enemy_state_feat_size

        allies = states[:, :ally_end].reshape(steps, meta.n_agents, meta.ally_state_feat_size)
        enemies = states[:, ally_end:enemy_end].reshape(
            steps, meta.n_enemies, meta.enemy_state_feat_size
        )
        features[:, : meta.n_agents, : meta.ally_state_feat_size] = allies
        mask[:, : meta.n_agents] = (np.abs(allies).sum(axis=-1) > 0).astype(np.float32)
        enemy_start = self.metadata.max_agents
        features[
            :,
            enemy_start : enemy_start + meta.n_enemies,
            : meta.enemy_state_feat_size,
        ] = enemies
        mask[:, enemy_start : enemy_start + meta.n_enemies] = (
            np.abs(enemies).sum(axis=-1) > 0
        ).astype(np.float32)
        return features, mask

    def _pad_actions(
        self, actions: np.ndarray, meta: DatasetMetadata
    ) -> tuple[np.ndarray, np.ndarray]:
        padded = np.zeros(
            (actions.shape[0], self.metadata.max_agents, self.metadata.max_actions),
            dtype=np.float32,
        )
        mask = np.zeros((actions.shape[0], self.metadata.max_agents), dtype=np.float32)
        padded[:, : meta.n_agents, : meta.n_actions] = actions
        mask[:, : meta.n_agents] = 1.0
        return padded, mask
