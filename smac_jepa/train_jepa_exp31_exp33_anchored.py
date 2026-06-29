from __future__ import annotations

"""Compatibility wrapper for Exp31/32 plus anchored-memory Exp33.

The installer stores the latest June-29 trainer as
``smac_jepa.train_jepa_exp31_exp33``. Without
SMAC_JEPA_ANCHORED_MEMORY=1 this module delegates to it unchanged.
"""

import os
from typing import Any

import torch

from . import train_jepa_exp31_exp33 as _base
from .anchored_belief_memory import (
    AnchoredActionConditionedEntityRolloutGRUMemory,
)

# Re-export the base module so existing imports keep working.
for _name in dir(_base):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_base, _name))

BaseActionConditionedEntityRolloutGRUMemory = (
    _base.ActionConditionedEntityRolloutGRUMemory
)
ActionConditionedEntityRolloutGRUMemory = (
    BaseActionConditionedEntityRolloutGRUMemory
)


def _enabled() -> bool:
    return os.environ.get("SMAC_JEPA_ANCHORED_MEMORY", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _safe_load(path: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _patch_for_anchored_memory() -> None:
    original_parse_args = _base.parse_args
    original_loss = _base.markov_rollout_rnn_losses
    original_torch_save = torch.save

    def parse_args():
        args = original_parse_args()
        args.anchored_belief_memory = True
        args.anchored_belief_version = 1
        args.anchor_gate_init = float(
            os.environ.get("SMAC_JEPA_ANCHOR_GATE_INIT", "-3.0")
        )
        args.anchor_delta_scale = float(
            os.environ.get("SMAC_JEPA_ANCHOR_DELTA_SCALE", "0.25")
        )
        args.anchor_hidden_correction_scale = float(
            os.environ.get(
                "SMAC_JEPA_ANCHOR_HIDDEN_CORRECTION_SCALE", "0.10"
            )
        )
        args.anchor_gate_sparsity_weight = float(
            os.environ.get(
                "SMAC_JEPA_ANCHOR_GATE_SPARSITY_WEIGHT", "0.002"
            )
        )
        latent_dim = int(
            args.latent_dim
            or _base.get_model_preset(args.model_size).latent_dim
        )
        recurrent_dim = int(args.rollout_memory_dim) - latent_dim - 2
        if not args.action_conditioned_memory:
            raise SystemExit(
                "Anchored Exp33 requires --action-conditioned-memory."
            )
        if recurrent_dim < 16:
            raise SystemExit(
                "Anchored Exp33 requires --rollout-memory-dim at least "
                f"latent_dim+18; latent_dim={latent_dim}, "
                f"memory_dim={args.rollout_memory_dim}."
            )
        args.anchored_recurrent_dim = recurrent_dim

        if args.init_from:
            raise SystemExit(
                "Anchored Exp33 changes memory architecture and must start "
                "from scratch; --init-from is intentionally rejected."
            )
        if args.resume:
            resume_checkpoint = _safe_load(args.resume)
            cfg = resume_checkpoint.get("resolved_config", {})
            state = resume_checkpoint.get("memory_module_state", {})
            anchored_resume = bool(
                cfg.get("anchored_belief_memory", False)
            ) or any(
                str(key).startswith("hidden_gate_net.") for key in state
            )
            if not anchored_resume:
                raise SystemExit(
                    "Anchored Exp33 cannot resume a non-anchored checkpoint."
                )
        return args

    def anchored_loss(*args, **kwargs):
        memory_module = args[1]
        if hasattr(memory_module, "reset_auxiliary_statistics"):
            memory_module.reset_auxiliary_statistics()
        losses = original_loss(*args, **kwargs)
        gate_mean = memory_module.gate_open_mean()
        weight = float(
            os.environ.get(
                "SMAC_JEPA_ANCHOR_GATE_SPARSITY_WEIGHT", "0.002"
            )
        )
        gate_penalty = gate_mean * weight
        losses["anchor_gate_open_mean"] = gate_mean.detach()
        losses["anchor_gate_sparsity_loss"] = gate_penalty
        losses["total_loss"] = losses["total_loss"] + gate_penalty
        return losses

    def anchored_torch_save(obj, *save_args, **save_kwargs):
        if isinstance(obj, dict):
            state = obj.get("memory_module_state", {})
            anchored_state = any(
                str(key).startswith("hidden_gate_net.") for key in state
            )
            if anchored_state:
                cfg = obj.setdefault("resolved_config", {})
                cfg.update(
                    {
                        "anchored_belief_memory": True,
                        "anchored_belief_version": 1,
                        "memory_architecture": (
                            "anchored_ordered_action_latent_filter_v1"
                        ),
                        "anchor_gate_init": float(
                            os.environ.get(
                                "SMAC_JEPA_ANCHOR_GATE_INIT", "-3.0"
                            )
                        ),
                        "anchor_delta_scale": float(
                            os.environ.get(
                                "SMAC_JEPA_ANCHOR_DELTA_SCALE", "0.25"
                            )
                        ),
                        "anchor_hidden_correction_scale": float(
                            os.environ.get(
                                "SMAC_JEPA_ANCHOR_HIDDEN_CORRECTION_SCALE",
                                "0.10",
                            )
                        ),
                        "anchor_gate_sparsity_weight": float(
                            os.environ.get(
                                "SMAC_JEPA_ANCHOR_GATE_SPARSITY_WEIGHT",
                                "0.002",
                            )
                        ),
                    }
                )
        return original_torch_save(obj, *save_args, **save_kwargs)

    _base.parse_args = parse_args
    _base.markov_rollout_rnn_losses = anchored_loss
    _base.ActionConditionedEntityRolloutGRUMemory = (
        AnchoredActionConditionedEntityRolloutGRUMemory
    )
    _base.torch.save = anchored_torch_save


def main() -> None:
    if _enabled():
        _patch_for_anchored_memory()
    _base.main()


if __name__ == "__main__":
    main()
