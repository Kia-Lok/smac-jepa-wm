from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from smac_jepa.data.dataset import DatasetMetadata


@dataclass(frozen=True)
class EntityFeatureLayout:
    ally_has_shield: bool
    enemy_has_shield: bool
    num_unit_types: int


def infer_entity_feature_layout(metadata: DatasetMetadata | dict[str, Any]) -> EntityFeatureLayout:
    explicit_num_types = _metadata_int(metadata, "num_unit_types", 0)
    explicit_ally_shield = _metadata_bool(metadata, "ally_has_shields")
    explicit_enemy_shield = _metadata_bool(metadata, "enemy_has_shields")
    if explicit_num_types or explicit_ally_shield or explicit_enemy_shield:
        return EntityFeatureLayout(
            ally_has_shield=explicit_ally_shield,
            enemy_has_shield=explicit_enemy_shield,
            num_unit_types=explicit_num_types,
        )
    ally_size = _metadata_int(metadata, "ally_state_feat_size")
    enemy_size = _metadata_int(metadata, "enemy_state_feat_size")
    candidates: list[EntityFeatureLayout] = []
    for ally_has_shield in (False, True):
        for enemy_has_shield in (False, True):
            ally_types = ally_size - 4 - int(ally_has_shield)
            enemy_types = enemy_size - 3 - int(enemy_has_shield)
            if ally_types == enemy_types and ally_types >= 0:
                candidates.append(
                    EntityFeatureLayout(
                        ally_has_shield=ally_has_shield,
                        enemy_has_shield=enemy_has_shield,
                        num_unit_types=ally_types,
                    )
                )
    if not candidates:
        return EntityFeatureLayout(False, False, 0)
    return max(candidates, key=lambda item: item.num_unit_types)


def format_entity_predictions(
    decoded: torch.Tensor | np.ndarray,
    metadata: DatasetMetadata | dict[str, Any],
    mask: torch.Tensor | np.ndarray | None = None,
    presence_scores: torch.Tensor | np.ndarray | None = None,
    presence_threshold: float = 0.5,
) -> dict[str, Any]:
    values = _as_numpy(decoded)
    if values.ndim != 2:
        raise ValueError(f"Expected one decoded step with shape (tokens, features), got {values.shape}")
    mask_values = None if mask is None else _as_numpy(mask)
    score_values = None if presence_scores is None else _as_numpy(presence_scores)
    max_agents = _metadata_int(metadata, "max_agents", _metadata_int(metadata, "n_agents"))
    max_enemies = _metadata_int(metadata, "max_enemies", _metadata_int(metadata, "n_enemies"))
    ally_size = _metadata_int(metadata, "ally_state_feat_size")
    enemy_size = _metadata_int(metadata, "enemy_state_feat_size")
    layout = infer_entity_feature_layout(metadata)
    enemy_start = max_agents
    return {
        "feature_layout": {
            **asdict(layout),
            "ally_order": _ally_feature_order(layout),
            "enemy_order": _enemy_feature_order(layout),
            "position_note": "dx and dy are normalized offsets from the map center.",
        },
        "allies": [
            _format_unit(
                values[idx, :ally_size],
                idx,
                "ally",
                layout.ally_has_shield,
                layout.num_unit_types,
                bool(mask_values[idx] > 0) if mask_values is not None else None,
                float(score_values[idx]) if score_values is not None else None,
                presence_threshold,
            )
            for idx in range(max_agents)
        ],
        "enemies": [
            _format_unit(
                values[enemy_start + idx, :enemy_size],
                idx,
                "enemy",
                layout.enemy_has_shield,
                layout.num_unit_types,
                bool(mask_values[enemy_start + idx] > 0) if mask_values is not None else None,
                float(score_values[enemy_start + idx]) if score_values is not None else None,
                presence_threshold,
            )
            for idx in range(max_enemies)
        ],
    }


def _format_unit(
    features: np.ndarray,
    unit_id: int,
    faction: str,
    has_shield: bool,
    num_unit_types: int,
    present: bool | None,
    presence_score: float | None = None,
    presence_threshold: float = 0.5,
) -> dict[str, Any]:
    offset = 0
    record: dict[str, Any] = {
        "unit_id": unit_id,
        "faction": faction,
        "present": bool(presence_score >= presence_threshold) if presence_score is not None else present,
        "target_present": present,
        "presence_score": presence_score,
        "alive_score": float(features[0]) if features.size > 0 else 0.0,
        "hp": float(features[0]) if features.size > 0 else 0.0,
    }
    offset = 1
    if faction == "ally":
        record["cooldown_or_energy"] = float(features[offset])
        offset += 1
    record["dx"] = float(features[offset])
    record["dy"] = float(features[offset + 1])
    offset += 2
    if has_shield:
        record["shield"] = float(features[offset])
        offset += 1
    type_values = features[offset : offset + num_unit_types]
    record["unit_type_values"] = [float(value) for value in type_values]
    record["unit_type_index"] = int(np.argmax(type_values)) if len(type_values) else None
    return record


def _ally_feature_order(layout: EntityFeatureLayout) -> list[str]:
    order = ["hp", "cooldown_or_energy", "dx", "dy"]
    if layout.ally_has_shield:
        order.append("shield")
    order.extend(f"unit_type_{idx}" for idx in range(layout.num_unit_types))
    return order


def _enemy_feature_order(layout: EntityFeatureLayout) -> list[str]:
    order = ["hp", "dx", "dy"]
    if layout.enemy_has_shield:
        order.append("shield")
    order.extend(f"unit_type_{idx}" for idx in range(layout.num_unit_types))
    return order


def _metadata_int(metadata: DatasetMetadata | dict[str, Any], key: str, default: int = 0) -> int:
    if isinstance(metadata, dict):
        return int(metadata.get(key, default))
    return int(getattr(metadata, key, default))


def _metadata_bool(metadata: DatasetMetadata | dict[str, Any], key: str) -> bool:
    if isinstance(metadata, dict):
        return bool(metadata.get(key, False))
    return bool(getattr(metadata, key, False))


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
