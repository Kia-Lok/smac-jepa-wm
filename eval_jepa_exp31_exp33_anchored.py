from __future__ import annotations

"""Compatibility evaluator with anchored Exp33 checkpoint support."""

from typing import Any

import torch

import eval_jepa_exp31_exp33 as _base

# Capture the unpatched function once. Calling _base.build_memory_module after
# monkey-patching would recurse back into this wrapper.
_BASE_BUILD_MEMORY_MODULE = _base.build_memory_module
from smac_jepa.anchored_belief_memory import (
    AnchoredActionConditionedEntityRolloutGRUMemory,
)

for _name in dir(_base):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_base, _name))


def build_memory_module(
    checkpoint: dict[str, Any],
    dataset,
    device: torch.device,
) -> torch.nn.Module:
    cfg = _base.get_config(checkpoint)
    memory_state = checkpoint.get("memory_module_state", {})
    anchored = bool(cfg.get("anchored_belief_memory", False)) or any(
        str(key).startswith("hidden_gate_net.") for key in memory_state
    )
    if not anchored:
        return _BASE_BUILD_MEMORY_MODULE(checkpoint, dataset, device)

    metadata = checkpoint.get("metadata", {})
    latent_dim = int(cfg["latent_dim"])
    memory_dim = int(cfg["rollout_memory_dim"])
    hidden_dim = cfg.get("rollout_memory_hidden_dim", None)
    residual = not bool(cfg.get("rollout_memory_no_residual", False))
    n_actions = int(
        cfg.get(
            "n_actions",
            metadata.get("n_actions", dataset.metadata.n_actions),
        )
    )
    max_agents = int(
        cfg.get(
            "max_agents",
            metadata.get("max_agents", dataset.metadata.max_agents),
        )
    )
    module = AnchoredActionConditionedEntityRolloutGRUMemory(
        latent_dim=latent_dim,
        memory_dim=memory_dim,
        n_actions=n_actions,
        max_agents=max_agents,
        hidden_dim=hidden_dim,
        residual=residual,
    ).to(device)
    state = checkpoint.get("memory_module_state")
    if state is None:
        raise RuntimeError("Anchored checkpoint has no memory_module_state")
    module.load_state_dict(state, strict=True)
    return module


def main() -> None:
    # Patch only for the duration of this evaluator invocation and restore the
    # original function even if evaluation raises. This prevents import-state
    # leakage when the module is reused by another evaluator.
    original = _base.build_memory_module
    _base.build_memory_module = build_memory_module
    try:
        _base.main()
    finally:
        _base.build_memory_module = original


if __name__ == "__main__":
    main()
