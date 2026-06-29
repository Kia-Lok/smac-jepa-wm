from __future__ import annotations

import os

import torch

from smac_jepa.anchored_belief_memory import (
    AnchoredActionConditionedEntityRolloutGRUMemory,
)


def main() -> None:
    torch.manual_seed(7)
    model = AnchoredActionConditionedEntityRolloutGRUMemory(
        latent_dim=12,
        memory_dim=30,
        n_actions=7,
        max_agents=3,
        hidden_dim=24,
    )
    memory = model.initial_memory(
        2, 5, device=torch.device("cpu"), dtype=torch.float32
    )
    actions = torch.zeros(2, 3, 7)
    actions[:, 0, 2] = 1
    actions[:, 1, 4] = 1
    actions[:, 2, 1] = 1
    action_mask = torch.ones(2, 3)
    context = model.entity_action_context(
        actions, action_mask, entities=5, dtype=torch.float32
    )
    assert context.shape == (2, 5, model.recurrent_dim)

    visible_z = torch.randn(2, 5, 12)
    observed = torch.ones(2, 5)
    memory = model.update(
        visible_z,
        memory,
        observed,
        action_context=context,
    )
    recurrent, anchor, seen, age = model._split(memory)
    assert torch.allclose(anchor, visible_z)
    assert bool((seen == 1).all())
    assert bool((age == 0).all())

    hidden_z = visible_z.clone()
    hidden_z[:, 4] = 0
    hidden_observed = observed.clone()
    hidden_observed[:, 4] = 0
    model.reset_auxiliary_statistics()
    conditioned = model.condition(hidden_z, memory, observed)
    next_memory = model.update(
        hidden_z,
        memory,
        hidden_observed,
        action_context=context,
    )
    assert conditioned.shape == visible_z.shape
    assert next_memory.shape == memory.shape
    assert torch.isfinite(conditioned).all()
    assert torch.isfinite(next_memory).all()

    loss = next_memory.square().mean() + model.gate_open_mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

    os.environ["SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO"] = "1"
    frozen = AnchoredActionConditionedEntityRolloutGRUMemory(
        latent_dim=12,
        memory_dim=30,
        n_actions=7,
        max_agents=3,
        hidden_dim=24,
    )
    frozen.load_state_dict(model.state_dict())
    frozen_memory = frozen.initial_memory(
        2, 5, device=torch.device("cpu"), dtype=torch.float32
    )
    frozen_memory = frozen.update(
        visible_z,
        frozen_memory,
        observed,
        action_context=context.detach(),
    )
    _, before_anchor, _, _ = frozen._split(frozen_memory)
    frozen_memory = frozen.update(
        hidden_z,
        frozen_memory,
        hidden_observed,
        action_context=context.detach(),
    )
    _, after_anchor, _, _ = frozen._split(frozen_memory)
    assert torch.allclose(before_anchor[:, 4], after_anchor[:, 4])

    print(
        "anchored_exp33_self_test_passed "
        f"recurrent_dim={model.recurrent_dim} "
        f"gate_mean={float(model.gate_open_mean().detach()):.6f}"
    )


if __name__ == "__main__":
    main()
