from __future__ import annotations

"""Explicit last-observed latent anchor for Exp33.

The public interface intentionally matches the existing blocker-fixed
ActionConditionedEntityRolloutGRUMemory so the current trainer and evaluators can
use it without changing rollout chronology.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class AnchoredActionConditionedEntityRolloutGRUMemory(nn.Module):
    """Action-conditioned recurrent memory with a conservative latent anchor.

    The externally visible memory tensor remains a single [B, E, M] tensor:

        [ recurrent state | anchored latent | has-seen | hidden age ]

    Visible observations overwrite the anchor. Hidden entities retain the
    anchor and receive a small, learned, gated, ordered-action-conditioned
    update. This prevents the generic GRU from having to both preserve exact
    last-seen information and model hidden changes in the same vector.
    """

    uses_action = True
    blocker_fixed_memory = True
    action_identity_preserved = True
    supports_precomputed_action_context = True
    anchored_belief_memory = True
    anchored_belief_version = 1

    def __init__(
        self,
        *,
        latent_dim: int,
        memory_dim: int,
        n_actions: int,
        max_agents: int,
        hidden_dim: int | None = None,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.memory_dim = int(memory_dim)
        self.n_actions = int(n_actions)
        self.max_agents = int(max_agents)
        self.residual = bool(residual)

        self.recurrent_dim = self.memory_dim - self.latent_dim - 2
        if self.recurrent_dim < 16:
            raise ValueError(
                "Anchored memory requires memory_dim >= latent_dim + 18; "
                f"got memory_dim={self.memory_dim}, latent_dim={self.latent_dim}."
            )

        width = int(hidden_dim or max(self.latent_dim, self.recurrent_dim))
        action_width = self.recurrent_dim

        # Ordered joint action context: no averaging across agents.
        self.own_action_embedding = nn.Embedding(self.n_actions, action_width)
        self.joint_action_net = nn.Sequential(
            nn.Linear(self.max_agents * self.n_actions, width),
            nn.SiLU(),
            nn.Linear(width, action_width),
        )
        self.action_fuse = nn.Sequential(
            nn.Linear(action_width * 2, width),
            nn.SiLU(),
            nn.Linear(width, action_width),
        )

        self.gru = nn.GRUCell(
            self.latent_dim + action_width,
            self.recurrent_dim,
        )

        self.visible_condition_net = nn.Sequential(
            nn.Linear(self.latent_dim + self.recurrent_dim, width),
            nn.SiLU(),
            nn.Linear(width, self.latent_dim),
        )
        self.hidden_condition_net = nn.Sequential(
            nn.Linear(
                self.latent_dim + self.recurrent_dim + action_width + 2,
                width,
            ),
            nn.SiLU(),
            nn.Linear(width, self.latent_dim),
        )

        hidden_update_in = (
            self.latent_dim + self.recurrent_dim + action_width + 2
        )
        self.hidden_delta_net = nn.Sequential(
            nn.Linear(hidden_update_in, width),
            nn.SiLU(),
            nn.Linear(width, self.latent_dim),
        )
        self.hidden_gate_net = nn.Sequential(
            nn.Linear(hidden_update_in, width),
            nn.SiLU(),
            nn.Linear(width, self.latent_dim),
        )

        # Begin close to identity/last-seen rather than random hidden drift.
        nn.init.zeros_(self.visible_condition_net[-1].weight)
        nn.init.zeros_(self.visible_condition_net[-1].bias)
        nn.init.zeros_(self.hidden_condition_net[-1].weight)
        nn.init.zeros_(self.hidden_condition_net[-1].bias)
        nn.init.zeros_(self.hidden_delta_net[-1].weight)
        nn.init.zeros_(self.hidden_delta_net[-1].bias)
        nn.init.zeros_(self.hidden_gate_net[-1].weight)
        nn.init.constant_(
            self.hidden_gate_net[-1].bias,
            _env_float("SMAC_JEPA_ANCHOR_GATE_INIT", -3.0),
        )

        self.register_buffer(
            "anchor_delta_scale",
            torch.tensor(
                _env_float("SMAC_JEPA_ANCHOR_DELTA_SCALE", 0.25),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "anchor_hidden_correction_scale",
            torch.tensor(
                _env_float(
                    "SMAC_JEPA_ANCHOR_HIDDEN_CORRECTION_SCALE", 0.10
                ),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "anchor_max_age",
            torch.tensor(
                max(
                    _env_float("SMAC_JEPA_ANCHOR_MAX_AGE", 16.0),
                    1.0,
                ),
                dtype=torch.float32,
            ),
        )
        self.force_gate_zero = _env_bool(
            "SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO", False
        )
        self._gate_terms: list[torch.Tensor] = []

    def _split(
        self, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self.recurrent_dim
        l = self.latent_dim
        recurrent = memory[..., :r]
        anchor = memory[..., r : r + l]
        seen = memory[..., r + l : r + l + 1]
        age = memory[..., r + l + 1 : r + l + 2]
        return recurrent, anchor, seen, age

    def _join(
        self,
        recurrent: torch.Tensor,
        anchor: torch.Tensor,
        seen: torch.Tensor,
        age: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([recurrent, anchor, seen, age], dim=-1)

    def initial_memory(
        self,
        batch_size: int,
        entities: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            entities,
            self.memory_dim,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _visibility_from_latent(z: torch.Tensor) -> torch.Tensor:
        # Corrected visibility masking zeroes hidden entity tokens before the
        # encoder, so their encoded vector is zero. A tiny tolerance avoids
        # dtype noise while preserving explicit masking semantics.
        return z.detach().float().abs().amax(dim=-1, keepdim=True) > 1.0e-8

    def _action_indices_and_mask(
        self,
        action: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action.dim() == 3:
            if action.shape[-1] != self.n_actions:
                raise RuntimeError(
                    "Action one-hot width does not match anchored memory: "
                    f"tensor={action.shape[-1]} expected={self.n_actions}"
                )
            indices = action.argmax(dim=-1).long()
            nonzero = action.float().sum(dim=-1) > 0
        elif action.dim() == 2:
            indices = action.long()
            nonzero = torch.ones_like(indices, dtype=torch.bool)
        else:
            raise ValueError(f"Unsupported action shape: {tuple(action.shape)}")

        if indices.shape[1] > self.max_agents:
            raise RuntimeError(
                f"actions contain {indices.shape[1]} agents but "
                f"max_agents={self.max_agents}"
            )
        indices = indices.clamp(min=0, max=self.n_actions - 1)
        valid = nonzero
        if action_mask is not None:
            mask = action_mask.bool()
            if mask.dim() == 3:
                mask = mask.squeeze(-1)
            valid = valid & mask

        if indices.shape[1] < self.max_agents:
            pad = self.max_agents - indices.shape[1]
            indices = F.pad(indices, (0, pad))
            valid = F.pad(valid, (0, pad), value=False)
        return indices, valid

    def entity_action_context(
        self,
        action: torch.Tensor,
        action_mask: torch.Tensor | None,
        *,
        entities: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        indices, valid = self._action_indices_and_mask(action, action_mask)
        batch = indices.shape[0]

        one_hot = F.one_hot(indices, num_classes=self.n_actions).to(dtype)
        one_hot = one_hot * valid.to(dtype).unsqueeze(-1)
        joint = self.joint_action_net(one_hot.reshape(batch, -1))

        own = self.own_action_embedding(indices)
        own = own * valid.to(own.dtype).unsqueeze(-1)
        if entities <= self.max_agents:
            own_per_entity = own[:, :entities]
        else:
            own_per_entity = F.pad(
                own,
                (0, 0, 0, entities - self.max_agents),
            )
        joint_per_entity = joint[:, None, :].expand(-1, entities, -1)
        return self.action_fuse(
            torch.cat([own_per_entity, joint_per_entity], dim=-1)
        ).to(dtype)

    def precompute_action_context_sequence(
        self,
        action_seq: torch.Tensor,
        action_mask_seq: torch.Tensor | None,
        *,
        entities: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if action_seq.dim() < 3:
            raise ValueError(
                f"Expected batched action sequence, got {tuple(action_seq.shape)}"
            )
        bsz, timesteps = action_seq.shape[:2]
        flat_action = action_seq.reshape(
            bsz * timesteps, *action_seq.shape[2:]
        )
        flat_mask = None
        if action_mask_seq is not None:
            flat_mask = action_mask_seq.reshape(
                bsz * timesteps, *action_mask_seq.shape[2:]
            )
        flat = self.entity_action_context(
            flat_action,
            flat_mask,
            entities=entities,
            dtype=dtype,
        )
        return flat.reshape(
            bsz, timesteps, entities, self.recurrent_dim
        )

    def reset_auxiliary_statistics(self) -> None:
        self._gate_terms = []

    def gate_open_mean(self) -> torch.Tensor:
        if self._gate_terms:
            return torch.stack(self._gate_terms).mean()
        return self.hidden_gate_net[-1].weight.sum() * 0.0

    def condition(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        belief_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        recurrent, anchor, seen, age = self._split(memory)
        visible = self._visibility_from_latent(z)
        seen_bool = seen > 0.5
        age_norm = (age / self.anchor_max_age.to(age.dtype)).clamp(0.0, 1.0)

        visible_correction = self.visible_condition_net(
            torch.cat([z, recurrent], dim=-1)
        )
        visible_belief = z + visible_correction if self.residual else visible_correction

        zero_action = recurrent.new_zeros(recurrent.shape)
        hidden_features = torch.cat(
            [anchor, recurrent, zero_action, seen, age_norm], dim=-1
        )
        hidden_correction = self.hidden_condition_net(hidden_features)
        hidden_belief = anchor + self.anchor_hidden_correction_scale.to(hidden_correction.dtype) * hidden_correction
        if self.force_gate_zero:
            hidden_belief = anchor

        out = torch.where(
            visible.expand_as(z),
            visible_belief,
            torch.where(seen_bool.expand_as(z), hidden_belief, z),
        )
        if belief_gate is not None:
            out = out * belief_gate.clamp(0.0, 1.0).unsqueeze(-1)
        return out

    def update(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        update_gate: torch.Tensor | None = None,
        *,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        action_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        recurrent, anchor, seen, age = self._split(memory)
        bsz, entities, _ = z.shape

        if action_context is None:
            if action is None:
                action = torch.zeros(
                    bsz,
                    self.max_agents,
                    device=z.device,
                    dtype=torch.long,
                )
                action_mask = torch.zeros(
                    bsz,
                    self.max_agents,
                    device=z.device,
                    dtype=z.dtype,
                )
            action_context = self.entity_action_context(
                action,
                action_mask,
                entities=entities,
                dtype=z.dtype,
            )

        latent_is_observed = self._visibility_from_latent(z)
        if update_gate is None:
            slot_active = torch.ones_like(latent_is_observed)
        else:
            slot_active = update_gate.unsqueeze(-1) > 0.0

        # Real-history hidden entities have a zero observation token and an
        # observation update gate of zero. Unlike the legacy GRU, an anchored
        # filter must still propagate a previously-seen hidden belief. Predicted
        # imagined latents are non-zero; if their predicted presence gate closes,
        # they are not updated.
        visible = latent_is_observed & slot_active
        seen_bool = seen > 0.5
        hidden = (~latent_is_observed) & seen_bool
        active = visible | hidden

        age_norm = (age / self.anchor_max_age.to(age.dtype)).clamp(0.0, 1.0)
        hidden_features = torch.cat(
            [anchor, recurrent, action_context, seen, age_norm], dim=-1
        )
        gate = torch.sigmoid(self.hidden_gate_net(hidden_features))
        delta = torch.tanh(self.hidden_delta_net(hidden_features))
        if self.force_gate_zero:
            gate = torch.zeros_like(gate)

        updated_anchor = anchor + self.anchor_delta_scale.to(delta.dtype) * gate * delta
        next_anchor = torch.where(
            visible.expand_as(anchor),
            z,
            torch.where(hidden.expand_as(anchor), updated_anchor, anchor),
        )
        next_seen = torch.where(
            visible,
            torch.ones_like(seen),
            seen,
        )
        next_age = torch.where(
            visible,
            torch.zeros_like(age),
            torch.where(
                hidden,
                torch.minimum(
                    age + 1.0,
                    self.anchor_max_age.to(age.dtype),
                ),
                age,
            ),
        )

        recurrent_input_latent = torch.where(
            visible.expand_as(z), z, next_anchor
        )
        candidate = self.gru(
            torch.cat([recurrent_input_latent, action_context], dim=-1).reshape(
                bsz * entities, -1
            ),
            recurrent.reshape(bsz * entities, -1),
        ).reshape(bsz, entities, self.recurrent_dim)
        next_recurrent = torch.where(
            active.expand_as(recurrent), candidate, recurrent
        )

        if self.training and bool(hidden.any().item()):
            selected = gate[hidden.expand_as(gate)]
            if selected.numel() > 0:
                self._gate_terms.append(selected.mean())

        return self._join(
            next_recurrent,
            next_anchor,
            next_seen.to(z.dtype),
            next_age.to(z.dtype),
        )
