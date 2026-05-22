from __future__ import annotations

import numpy as np

from smac_jepa.data.dataset import DatasetMetadata
from smac_jepa.decoder import format_entity_predictions, infer_entity_feature_layout


def test_infer_entity_feature_layout_without_shields() -> None:
    metadata = DatasetMetadata(
        state_dim=0,
        n_agents=2,
        n_actions=5,
        n_enemies=1,
        ally_state_feat_size=7,
        enemy_state_feat_size=6,
        max_agents=2,
        max_enemies=1,
        max_actions=5,
        token_dim=7,
        mode="entity",
    )

    layout = infer_entity_feature_layout(metadata)

    assert layout.ally_has_shield is False
    assert layout.enemy_has_shield is False
    assert layout.num_unit_types == 3


def test_format_entity_predictions_splits_allies_and_enemies() -> None:
    metadata = DatasetMetadata(
        state_dim=0,
        n_agents=2,
        n_actions=5,
        n_enemies=1,
        ally_state_feat_size=7,
        enemy_state_feat_size=6,
        max_agents=2,
        max_enemies=1,
        max_actions=5,
        token_dim=7,
        mode="entity",
    )
    tokens = np.zeros((3, 7), dtype=np.float32)
    tokens[0, :7] = [0.8, 0.2, -0.1, 0.3, 0.1, 0.7, 0.2]
    tokens[1, :7] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    tokens[2, :6] = [0.5, 0.4, -0.2, 0.9, 0.05, 0.05]
    mask = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)

    formatted = format_entity_predictions(tokens, metadata, mask)

    assert formatted["allies"][0]["hp"] == np.float32(0.8).item()
    assert formatted["allies"][0]["cooldown_or_energy"] == np.float32(0.2).item()
    assert formatted["allies"][0]["unit_type_index"] == 1
    assert formatted["allies"][1]["present"] is False
    assert formatted["enemies"][0]["dx"] == np.float32(0.4).item()
    assert formatted["enemies"][0]["unit_type_index"] == 0
