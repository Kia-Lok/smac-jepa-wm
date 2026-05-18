from __future__ import annotations

import numpy as np
import pytest

from smac_jepa.data.dataset import DatasetMetadata
from smac_jepa.predict_next import (
    encode_action_ids,
    encode_state_vector,
    load_entity_static,
    metadata_for_model_caps,
)


def metadata() -> DatasetMetadata:
    return DatasetMetadata(
        state_dim=19,
        n_agents=2,
        n_actions=4,
        n_enemies=1,
        ally_state_feat_size=4,
        enemy_state_feat_size=3,
        max_agents=3,
        max_enemies=2,
        max_actions=5,
        token_dim=4,
        mode="entity",
    )


def test_encode_action_ids_pads_to_checkpoint_caps() -> None:
    actions, mask = encode_action_ids([1, 3], metadata())

    assert actions.shape == (3, 5)
    assert mask.tolist() == [1.0, 1.0, 0.0]
    assert actions[0].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert actions[1].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_encode_action_ids_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="Expected 2 action ids"):
        encode_action_ids([1], metadata())


def test_encode_state_vector_splits_allies_and_enemies() -> None:
    state = np.zeros(19, dtype=np.float32)
    state[0:4] = [1.0, 0.2, 0.3, 0.4]
    state[4:8] = [0.5, 0.1, -0.3, 0.2]
    state[8:11] = [0.8, -0.1, 0.6]

    tokens, mask = encode_state_vector(state, metadata())

    assert tokens.shape == (5, 4)
    assert mask.tolist() == [1.0, 1.0, 0.0, 1.0, 0.0]
    np.testing.assert_allclose(tokens[0], [1.0, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(tokens[3], [0.8, -0.1, 0.6, 0.0])


def test_metadata_for_model_caps_keeps_source_layout_with_checkpoint_padding() -> None:
    source = metadata()
    checkpoint = DatasetMetadata(
        state_dim=99,
        n_agents=9,
        n_actions=9,
        n_enemies=9,
        ally_state_feat_size=7,
        enemy_state_feat_size=6,
        max_agents=8,
        max_enemies=7,
        max_actions=6,
        token_dim=5,
        mode="entity",
    )

    merged = metadata_for_model_caps(source, checkpoint)

    assert merged.state_dim == source.state_dim
    assert merged.n_agents == source.n_agents
    assert merged.max_agents == checkpoint.max_agents
    assert merged.max_actions == checkpoint.max_actions


def test_load_entity_static_places_enemies_after_checkpoint_agent_cap(tmp_path) -> None:
    path = tmp_path / "sample.npz"
    np.savez_compressed(
        path,
        entity_static=np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
    )
    meta = DatasetMetadata(
        state_dim=0,
        n_agents=2,
        n_actions=4,
        n_enemies=1,
        ally_state_feat_size=4,
        enemy_state_feat_size=3,
        max_agents=4,
        max_enemies=2,
        max_actions=4,
        token_dim=5,
        dynamic_token_dim=4,
        entity_static_feat_size=1,
        mode="entity",
    )

    static = load_entity_static(str(path), meta)

    assert static[:, 0].tolist() == [1.0, 2.0, 0.0, 0.0, 3.0, 0.0]
