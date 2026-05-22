from __future__ import annotations

from pathlib import Path

import numpy as np

from smac_jepa.data import SMACJEPADataset


def write_npz(path: Path) -> None:
    episodes = 1
    steps = 5
    n_agents = 2
    n_enemies = 1
    n_actions = 4
    ally_size = 4
    enemy_size = 3
    state_dim = n_agents * ally_size + n_enemies * enemy_size + n_agents * n_actions
    states = np.zeros((episodes, steps + 1, state_dim), dtype=np.float32)
    for step in range(steps + 1):
        states[0, step, 0:4] = [1.0, 0.1, step / 10, step / 20]
        states[0, step, 4:8] = [1.0, 0.2, step / 30, step / 40]
        states[0, step, 8:11] = [1.0, -step / 10, step / 20]
    actions = np.zeros((episodes, steps, n_agents), dtype=np.int64)
    action_onehot = np.eye(n_actions, dtype=np.float32)[actions]
    valid = np.asarray([[True, True, True, True, False]], dtype=bool)
    np.savez_compressed(
        path,
        states=states,
        actions=actions,
        action_onehot=action_onehot,
        rewards=np.zeros((episodes, steps), dtype=np.float32),
        dones=np.zeros((episodes, steps), dtype=bool),
        valid=valid,
        avail_actions=np.ones((episodes, steps, n_agents, n_actions), dtype=np.float32),
        scenario=np.asarray("unit"),
        state_dim=np.asarray(state_dim, dtype=np.int64),
        n_agents=np.asarray(n_agents, dtype=np.int64),
        n_enemies=np.asarray(n_enemies, dtype=np.int64),
        n_actions=np.asarray(n_actions, dtype=np.int64),
        ally_state_feat_size=np.asarray(ally_size, dtype=np.int64),
        enemy_state_feat_size=np.asarray(enemy_size, dtype=np.int64),
        max_steps=np.asarray(steps, dtype=np.int64),
    )


def test_sequential_windows_keep_existing_full_window_behavior(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    write_npz(path)

    dataset = SMACJEPADataset(path, context_len=3, mode="entity")

    assert len(dataset) == 2
    item = dataset[0]
    assert item["entity_t"].shape == (3, 3, 4)
    assert item["action_t"].shape == (3, 2, 4)
    assert item["entity_slot_mask"].shape == (3, 3)
    assert item["entity_slot_mask"][0].tolist() == [1.0, 1.0, 1.0]
    assert item["mask"].tolist() == [1.0, 1.0, 1.0]


def test_random_windows_pad_short_episode_tails(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    write_npz(path)

    dataset = SMACJEPADataset(
        path,
        context_len=3,
        mode="entity",
        window_mode="random",
        window_len=4,
        samples_per_epoch=16,
        seed=3,
    )
    masks = [dataset[idx]["mask"].tolist() for idx in range(len(dataset))]

    assert len(dataset) == 16
    assert all(len(mask) == 4 for mask in masks)
    assert any(mask[-1] == 0.0 for mask in masks)


def test_entity_dataset_adds_static_condition_and_entity_static_features(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    write_npz(path)
    with np.load(path, allow_pickle=False) as data:
        values = dict(data.items())
    values["static_condition"] = np.asarray([0.5, 1.0, 0.25], dtype=np.float32)
    values["static_dim"] = np.asarray(3, dtype=np.int64)
    values["entity_static"] = np.asarray(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32
    )
    values["entity_static_feat_size"] = np.asarray(2, dtype=np.int64)
    np.savez_compressed(path, **values)

    dataset = SMACJEPADataset(path, context_len=3, mode="entity")
    item = dataset[0]

    assert dataset.metadata.dynamic_token_dim == 4
    assert dataset.metadata.token_dim == 6
    assert item["entity_t"].shape == (3, 3, 6)
    assert item["static_condition"].tolist() == [0.5, 1.0, 0.25]
    np.testing.assert_allclose(item["entity_t"][0, 0, 4:].numpy(), [0.1, 0.2])
    np.testing.assert_allclose(item["entity_t"][0, 2, 4:].numpy(), [0.5, 0.6])
