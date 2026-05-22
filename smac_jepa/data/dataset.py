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
    ally_has_shields: bool = False
    enemy_has_shields: bool = False
    num_unit_types: int = 0
    max_agents: int = 0
    max_enemies: int = 0
    max_actions: int = 0
    token_dim: int = 0
    dynamic_token_dim: int = 0
    static_dim: int = 0
    entity_static_feat_size: int = 0
    mode: str = "entity"


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


def load_manifest_all(path: str | Path) -> list[Path]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text())
    values: list[str] = []
    if "datasets" in manifest:
        for split_values in manifest["datasets"].values():
            values.extend(split_values)
    else:
        for key, split_values in manifest.items():
            if key.endswith("_data") and isinstance(split_values, list):
                values.extend(split_values)
    seen: set[Path] = set()
    paths: list[Path] = []
    for value in values:
        path_value = (manifest_path.parent / value).resolve() if not Path(value).is_absolute() else Path(value)
        if path_value not in seen:
            seen.add(path_value)
            paths.append(path_value)
    return paths


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
    ally_has_shields = bool(_scalar(data, "ally_has_shields", 0))
    enemy_has_shields = bool(_scalar(data, "enemy_has_shields", 0))
    num_unit_types = _scalar(data, "num_unit_types", 0)
    static_dim = _scalar(data, "static_dim", 0)
    entity_static_feat_size = _scalar(data, "entity_static_feat_size", 0)
    has_entity_layout = n_enemies > 0 and ally_size > 0 and enemy_size > 0
    dynamic_token_dim = max(ally_size, enemy_size, 1)
    return DatasetMetadata(
        state_dim=_scalar(data, "state_dim"),
        n_agents=n_agents,
        n_actions=n_actions,
        n_enemies=n_enemies,
        ally_state_feat_size=ally_size,
        enemy_state_feat_size=enemy_size,
        ally_has_shields=ally_has_shields,
        enemy_has_shields=enemy_has_shields,
        num_unit_types=num_unit_types,
        max_agents=n_agents,
        max_enemies=n_enemies,
        max_actions=n_actions,
        token_dim=dynamic_token_dim + entity_static_feat_size,
        dynamic_token_dim=dynamic_token_dim,
        static_dim=static_dim,
        entity_static_feat_size=entity_static_feat_size,
        mode="entity" if has_entity_layout else "invalid",
    )


def _merge_metadata(items: list[DatasetMetadata]) -> DatasetMetadata:
    if not items:
        raise ValueError("At least one metadata item is required")
    return DatasetMetadata(
        state_dim=max(item.state_dim for item in items),
        n_agents=max(item.n_agents for item in items),
        n_actions=max(item.n_actions for item in items),
        n_enemies=max(item.n_enemies for item in items),
        ally_state_feat_size=max(item.ally_state_feat_size for item in items),
        enemy_state_feat_size=max(item.enemy_state_feat_size for item in items),
        ally_has_shields=any(item.ally_has_shields for item in items),
        enemy_has_shields=any(item.enemy_has_shields for item in items),
        num_unit_types=max(item.num_unit_types for item in items),
        max_agents=max(item.n_agents for item in items),
        max_enemies=max(item.n_enemies for item in items),
        max_actions=max(item.n_actions for item in items),
        token_dim=max(item.dynamic_token_dim for item in items)
        + max(item.entity_static_feat_size for item in items),
        dynamic_token_dim=max(item.dynamic_token_dim for item in items),
        static_dim=max(item.static_dim for item in items),
        entity_static_feat_size=max(item.entity_static_feat_size for item in items),
        mode="entity",
    )


def _one_hot(actions: np.ndarray, n_actions: int) -> np.ndarray:
    clipped = np.clip(actions.astype(np.int64), 0, n_actions - 1)
    return np.eye(n_actions, dtype=np.float32)[clipped]


def _load_static_condition(data: np.lib.npyio.NpzFile, meta: DatasetMetadata) -> np.ndarray:
    if "static_condition" not in data or meta.static_dim <= 0:
        return np.zeros((meta.static_dim,), dtype=np.float32)
    value = np.asarray(data["static_condition"], dtype=np.float32).reshape(-1)
    if value.shape[0] != meta.static_dim:
        raise ValueError(f"static_condition has size {value.shape[0]}, expected {meta.static_dim}")
    return value


def _load_entity_static(data: np.lib.npyio.NpzFile, meta: DatasetMetadata) -> np.ndarray:
    token_count = meta.n_agents + meta.n_enemies
    if "entity_static" not in data or meta.entity_static_feat_size <= 0:
        return np.zeros((token_count, meta.entity_static_feat_size), dtype=np.float32)
    value = np.asarray(data["entity_static"], dtype=np.float32)
    expected = (token_count, meta.entity_static_feat_size)
    if value.shape != expected:
        raise ValueError(f"entity_static has shape {value.shape}, expected {expected}")
    return value


class SMACJEPADataset(Dataset):
    """Windowed JEPA dataset.

    Each item contains an observation sequence, an action-history conditioning
    sequence, and the next observation sequence shifted by one step.
    """

    def __init__(
        self,
        paths: str | Path | Iterable[str | Path],
        context_len: int = 1,
        mode: str = "entity",
        window_mode: str = "sequential",
        window_len: int | None = None,
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
        if context_len < 1:
            raise ValueError("context_len must be at least 1")
        if window_mode not in {"sequential", "random"}:
            raise ValueError("window_mode must be 'sequential' or 'random'")
        resolved_window_len = window_len or context_len
        if resolved_window_len < 1:
            raise ValueError("window_len must be at least 1")
        if samples_per_epoch is not None and samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be at least 1")
        self.paths = _as_paths(paths)
        self.context_len = resolved_window_len
        self.window_mode = window_mode
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)
        if mode != "entity":
            raise ValueError("SMACJEPADataset only supports entity mode")
        self.mode = "entity"
        if not self.paths:
            raise ValueError("At least one dataset path is required")

        metadata_items: list[DatasetMetadata] = []

        for path in self.paths:
            with np.load(path, allow_pickle=False) as data:
                current = _read_metadata(data)
                metadata_items.append(current)

        available_modes = {item.mode for item in metadata_items}
        if available_modes != {"entity"}:
            raise ValueError("Entity mode requires datasets with SMACLite state layout metadata")
        self.metadata = _merge_metadata(metadata_items)
        override = {
            "max_agents": max_agents or self.metadata.max_agents,
            "max_enemies": max_enemies or self.metadata.max_enemies,
            "max_actions": max_actions or self.metadata.max_actions,
            "token_dim": token_dim or self.metadata.token_dim,
            "dynamic_token_dim": dynamic_token_dim or self.metadata.dynamic_token_dim,
            "static_dim": static_dim if static_dim is not None else self.metadata.static_dim,
            "entity_static_feat_size": (
                entity_static_feat_size
                if entity_static_feat_size is not None
                else self.metadata.entity_static_feat_size
            ),
        }
        if override["max_agents"] < self.metadata.max_agents:
            raise ValueError("max_agents override is smaller than dataset requirement")
        if override["max_enemies"] < self.metadata.max_enemies:
            raise ValueError("max_enemies override is smaller than dataset requirement")
        if override["max_actions"] < self.metadata.max_actions:
            raise ValueError("max_actions override is smaller than dataset requirement")
        if override["token_dim"] < self.metadata.token_dim:
            raise ValueError("token_dim override is smaller than dataset requirement")
        if override["dynamic_token_dim"] < self.metadata.dynamic_token_dim:
            raise ValueError("dynamic_token_dim override is smaller than dataset requirement")
        if override["static_dim"] < self.metadata.static_dim:
            raise ValueError("static_dim override is smaller than dataset requirement")
        if override["entity_static_feat_size"] < self.metadata.entity_static_feat_size:
            raise ValueError("entity_static_feat_size override is smaller than dataset requirement")
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
                static_condition = _load_static_condition(data, item_meta)
                entity_static = _load_entity_static(data, item_meta)
                for episode_idx in range(actions.shape[0]):
                    record: dict[str, Any] = {
                        "states": states[episode_idx],
                        "actions": actions[episode_idx],
                        "valid": valid[episode_idx],
                        "metadata": item_meta,
                        "scenario": scenario,
                        "static_condition": static_condition,
                        "entity_static": entity_static,
                    }
                    self.episodes.append(record)
                    self.scenarios.append(scenario)

        self.index: list[tuple[int, int]] = []
        self.random_episode_indices: list[int] = []

        for episode_idx, episode in enumerate(self.episodes):
            horizon = episode["actions"].shape[0]
            valid_starts = np.flatnonzero(episode["valid"])
            if len(valid_starts):
                self.random_episode_indices.append(episode_idx)
            for start in range(0, horizon - self.context_len + 1):
                window_valid = episode["valid"][start : start + self.context_len]
                if np.all(window_valid):
                    self.index.append((episode_idx, start))

        if self.window_mode == "sequential" and not self.index:
            raise ValueError("No valid training windows found")
        if self.window_mode == "random" and not self.random_episode_indices:
            raise ValueError("No valid random-start episodes found")
        if self.window_mode == "random" and self.samples_per_epoch is None:
            self.samples_per_epoch = max(len(self.index), len(self.random_episode_indices))

    def __len__(self) -> int:
        if self.window_mode == "random":
            assert self.samples_per_epoch is not None
            return self.samples_per_epoch
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.window_mode == "random":
            episode_idx, start = self._sample_random_window()
        else:
            episode_idx, start = self.index[idx]
        end = start + self.context_len
        return self._entity_item(episode_idx, start, end)

    def _sample_random_window(self) -> tuple[int, int]:
        episode_idx = int(self.rng.choice(self.random_episode_indices))
        valid = self.episodes[episode_idx]["valid"]
        starts = np.flatnonzero(valid)
        start = int(self.rng.choice(starts))
        return episode_idx, start

    def _entity_item(self, episode_idx: int, start: int, end: int) -> dict[str, torch.Tensor]:
        episode = self.episodes[episode_idx]
        states = episode["states"]
        actions = episode["actions"]
        valid = episode["valid"]
        meta = episode["metadata"]
        slot_mask = self._slot_mask(meta)
        entity_t, entity_mask = self._encode_state_window(
            self._pad_steps(states[start:end], self.context_len),
            meta,
            episode["entity_static"],
        )
        target_entity, target_entity_mask = self._encode_state_window(
            self._pad_steps(states[start + 1 : end + 1], self.context_len),
            meta,
            episode["entity_static"],
        )
        real_mask = self._pad_steps(valid[start:end].astype(np.float32), self.context_len)
        entity_mask *= real_mask[:, None]
        target_entity_mask *= real_mask[:, None]
        entity_slot_mask = np.repeat(slot_mask[None, :], self.context_len, axis=0) * real_mask[:, None]
        action_t, action_mask = self._pad_actions(
            self._pad_steps(actions[start:end], self.context_len), meta
        )
        action_mask *= real_mask[:, None]
        return {
            "entity_t": torch.from_numpy(entity_t),
            "entity_mask": torch.from_numpy(entity_mask),
            "action_t": torch.from_numpy(action_t),
            "action_mask": torch.from_numpy(action_mask),
            "target_entity": torch.from_numpy(target_entity),
            "target_entity_mask": torch.from_numpy(target_entity_mask),
            "entity_slot_mask": torch.from_numpy(entity_slot_mask),
            "mask": torch.from_numpy(real_mask),
            "static_condition": torch.from_numpy(
                self._pad_static_condition(episode["static_condition"], meta)
            ),
            "scenario_id": torch.tensor(0, dtype=torch.long),
        }

    def _pad_steps(self, values: np.ndarray, steps: int) -> np.ndarray:
        if values.shape[0] == steps:
            return values
        padded = np.zeros((steps, *values.shape[1:]), dtype=values.dtype)
        padded[: values.shape[0]] = values
        return padded

    def _encode_state_window(
        self, states: np.ndarray, meta: DatasetMetadata, entity_static: np.ndarray
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
        self._insert_entity_static(features, entity_static, meta)
        return features, mask

    def _slot_mask(self, meta: DatasetMetadata) -> np.ndarray:
        mask = np.zeros((self.metadata.max_agents + self.metadata.max_enemies,), dtype=np.float32)
        mask[: meta.n_agents] = 1.0
        enemy_start = self.metadata.max_agents
        mask[enemy_start : enemy_start + meta.n_enemies] = 1.0
        return mask

    def _insert_entity_static(
        self,
        features: np.ndarray,
        episode_static: np.ndarray | None,
        meta: DatasetMetadata,
    ) -> None:
        if episode_static is None or meta.entity_static_feat_size <= 0:
            return
        offset = self.metadata.dynamic_token_dim
        if offset + meta.entity_static_feat_size > self.metadata.token_dim:
            raise ValueError("entity static features exceed token_dim")
        features[:, : meta.n_agents, offset : offset + meta.entity_static_feat_size] = (
            episode_static[: meta.n_agents, : meta.entity_static_feat_size]
        )
        enemy_start = self.metadata.max_agents
        source_enemy_start = meta.n_agents
        features[
            :,
            enemy_start : enemy_start + meta.n_enemies,
            offset : offset + meta.entity_static_feat_size,
        ] = episode_static[
            source_enemy_start : source_enemy_start + meta.n_enemies,
            : meta.entity_static_feat_size,
        ]

    def _pad_static_condition(self, value: np.ndarray, meta: DatasetMetadata) -> np.ndarray:
        padded = np.zeros((self.metadata.static_dim,), dtype=np.float32)
        if meta.static_dim:
            padded[: meta.static_dim] = value[: meta.static_dim]
        return padded

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
