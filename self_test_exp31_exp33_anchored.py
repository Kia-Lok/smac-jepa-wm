from __future__ import annotations

"""Fast CPU-only contract tests for the Exp31--Exp33 anchored package.

This test deliberately covers both evaluator branches. The earlier recursion bug
was caused by testing only the anchored branch, so ordinary Exp31/32 dispatch is
now an explicit regression test.
"""

import os
from types import SimpleNamespace

import torch

from smac_jepa.anchored_belief_memory import (
    AnchoredActionConditionedEntityRolloutGRUMemory,
)


def _memory_dynamics_test() -> None:
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
    assert torch.isfinite(context).all()

    # Visible observations must overwrite the anchor exactly.
    visible_z = torch.randn(2, 5, 12)
    observed = torch.ones(2, 5)
    memory = model.update(
        visible_z,
        memory,
        observed,
        action_context=context,
    )
    _, anchor, seen, age = model._split(memory)
    assert torch.allclose(anchor, visible_z)
    assert bool((seen == 1).all())
    assert bool((age == 0).all())

    # A previously seen hidden entity must continue to propagate even though
    # its observation token and observation gate are both zero.
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
    next_recurrent, next_anchor, next_seen, next_age = model._split(next_memory)
    assert conditioned.shape == visible_z.shape
    assert next_memory.shape == memory.shape
    assert torch.isfinite(conditioned).all()
    assert torch.isfinite(next_memory).all()
    assert bool((next_seen[:, 4] == 1).all())
    assert bool((next_age[:, 4] == 1).all())
    assert not torch.equal(next_recurrent[:, 4], torch.zeros_like(next_recurrent[:, 4]))

    # An entity that has never been observed must not be invented by memory.
    empty = model.initial_memory(
        1, 1, device=torch.device("cpu"), dtype=torch.float32
    )
    empty_z = torch.zeros(1, 1, 12)
    empty_next = model.update(
        empty_z,
        empty,
        torch.zeros(1, 1),
        action=torch.zeros(1, 3, dtype=torch.long),
        action_mask=torch.zeros(1, 3),
    )
    assert torch.equal(empty, empty_next)

    loss = next_memory.square().mean() + model.gate_open_mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    # State dictionaries must round-trip strictly; evaluator reconstruction
    # relies on strict loading.
    clone = AnchoredActionConditionedEntityRolloutGRUMemory(
        latent_dim=12,
        memory_dim=30,
        n_actions=7,
        max_agents=3,
        hidden_dim=24,
    )
    clone.load_state_dict(model.state_dict(), strict=True)
    for key, value in model.state_dict().items():
        assert torch.equal(value, clone.state_dict()[key]), key

    # Gate-zero ablation must preserve the hidden anchor exactly.
    previous = os.environ.get("SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO")
    os.environ["SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO"] = "1"
    try:
        frozen = AnchoredActionConditionedEntityRolloutGRUMemory(
            latent_dim=12,
            memory_dim=30,
            n_actions=7,
            max_agents=3,
            hidden_dim=24,
        )
        frozen.load_state_dict(model.state_dict(), strict=True)
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
    finally:
        if previous is None:
            os.environ.pop("SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO", None)
        else:
            os.environ["SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO"] = previous


def _module_contract_test() -> None:
    import smac_jepa.train_jepa_exp31_exp33 as train_base
    import smac_jepa.train_jepa_exp31_exp33_anchored as train_wrapper
    import eval_jepa_exp31_exp33 as eval_base
    import eval_jepa_exp31_exp33_anchored as eval_wrapper
    import eval_jepa_hidden_belief_exp31_exp33 as hidden_eval

    required_train = {
        "ActionConditionedEntityRolloutGRUMemory",
        "get_model_preset",
        "main",
        "markov_rollout_rnn_losses",
        "parse_args",
    }
    required_eval = {
        "add_exact_regression_statistics",
        "build_dataset",
        "build_memory_module",
        "build_model",
        "build_rollout_feature_masks",
        "clone_fresh_native_decoder",
        "decode_with_probe",
        "finalize_exact_regression_statistics",
        "get_config",
        "resolve_device",
        "rollout_outputs",
        "to_device",
    }
    missing_train = sorted(name for name in required_train if not hasattr(train_base, name))
    missing_eval = sorted(name for name in required_eval if not hasattr(eval_base, name))
    assert not missing_train, f"base trainer API missing: {missing_train}"
    assert not missing_eval, f"base evaluator API missing: {missing_eval}"
    assert hidden_eval.base is eval_wrapper
    assert train_wrapper._base is train_base
    assert eval_wrapper._base is eval_base
    assert eval_wrapper._BASE_BUILD_MEMORY_MODULE is not eval_wrapper.build_memory_module


def _evaluator_dispatch_regression_test() -> None:
    import eval_jepa_exp31_exp33_anchored as wrapper

    dataset = SimpleNamespace(
        metadata=SimpleNamespace(n_actions=7, max_agents=3)
    )
    device = torch.device("cpu")

    # Ordinary checkpoints must dispatch to the captured original builder.
    # Re-entering wrapper.build_memory_module here was the exact recursion bug.
    original_captured = wrapper._BASE_BUILD_MEMORY_MODULE
    calls: list[str] = []
    sentinel = object()

    def fake_base_builder(checkpoint, incoming_dataset, incoming_device):
        assert incoming_dataset is dataset
        assert incoming_device == device
        calls.append(str(checkpoint.get("name")))
        return sentinel

    wrapper._BASE_BUILD_MEMORY_MODULE = fake_base_builder
    try:
        ordinary_checkpoint = {
            "name": "ordinary",
            "resolved_config": {"anchored_belief_memory": False},
            "memory_module_state": {},
        }
        result = wrapper.build_memory_module(
            ordinary_checkpoint, dataset, device
        )
        assert result is sentinel
        assert calls == ["ordinary"]
    finally:
        wrapper._BASE_BUILD_MEMORY_MODULE = original_captured

    # Anchored checkpoints must reconstruct the new architecture and load all
    # parameters and registered buffers strictly.
    source = AnchoredActionConditionedEntityRolloutGRUMemory(
        latent_dim=12,
        memory_dim=30,
        n_actions=7,
        max_agents=3,
        hidden_dim=24,
    )
    anchored_checkpoint = {
        "name": "anchored",
        "resolved_config": {
            "anchored_belief_memory": True,
            "latent_dim": 12,
            "rollout_memory_dim": 30,
            "rollout_memory_hidden_dim": 24,
            "rollout_memory_no_residual": False,
            "n_actions": 7,
            "max_agents": 3,
        },
        "metadata": {"n_actions": 7, "max_agents": 3},
        "memory_module_state": source.state_dict(),
    }
    loaded = wrapper.build_memory_module(
        anchored_checkpoint, dataset, device
    )
    assert isinstance(loaded, AnchoredActionConditionedEntityRolloutGRUMemory)
    for key, value in source.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key

    # main() must restore the base evaluator function even when the delegated
    # evaluator raises. This prevents process-local patch leakage.
    base = wrapper._base
    original_main = base.main
    original_builder = base.build_memory_module
    observed_patch: list[bool] = []

    def failing_main() -> None:
        observed_patch.append(base.build_memory_module is wrapper.build_memory_module)
        raise RuntimeError("intentional restoration test")

    base.main = failing_main
    try:
        try:
            wrapper.main()
        except RuntimeError as exc:
            assert "intentional restoration test" in str(exc)
        else:
            raise AssertionError("wrapper.main() did not propagate delegated error")
        assert observed_patch == [True]
        assert base.build_memory_module is original_builder
    finally:
        base.main = original_main
        base.build_memory_module = original_builder


def main() -> None:
    _memory_dynamics_test()
    _module_contract_test()
    _evaluator_dispatch_regression_test()
    print("anchored_exp31_exp33_full_self_test_passed")


if __name__ == "__main__":
    main()
