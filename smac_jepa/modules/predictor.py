from __future__ import annotations

import torch
from torch import nn

from smac_jepa.modules.blocks import AttentionBlock, MLP


class ActionHistoryEncoder(nn.Module):
    """Encodes joint actions while preserving one token per ally slot."""

    def __init__(
        self,
        max_agents: int,
        max_actions: int,
        action_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.max_agents = max_agents
        self.max_actions = max_actions
        self.input = nn.Linear(max_actions, action_dim)
        self.agent_embedding = nn.Embedding(max_agents, action_dim)
        self.blocks = nn.ModuleList(
            AttentionBlock(action_dim, num_heads=num_heads, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(action_dim)

    def forward(self, actions: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        batch, steps, agents = actions.shape[:3]
        x = self.input(actions)
        agent_ids = torch.arange(agents, device=actions.device)
        x = x + self.agent_embedding(agent_ids).view(1, 1, agents, -1)
        flat_x = x.reshape(batch * steps, agents, -1)
        flat_mask = action_mask.reshape(batch * steps, agents) > 0
        key_padding_mask = ~flat_mask
        all_padded = key_padding_mask.all(dim=1)
        if all_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padded, 0] = False
        for block in self.blocks:
            flat_x = block(flat_x, key_padding_mask=key_padding_mask)
        flat_x = self.norm(flat_x) * flat_mask.unsqueeze(-1)
        return flat_x.reshape(batch, steps, agents, -1)


class EntityJEPAActionPredictor(nn.Module):
    """Predicts next entity-slot latents from slot latents and joint-action history."""

    def __init__(
        self,
        latent_dim: int,
        max_agents: int,
        max_actions: int,
        action_dim: int,
        hidden_dim: int,
        num_heads: int,
        action_layers: int = 1,
        predictor_layers: int = 1,
        max_context_len: int = 32,
        static_dim: int = 0,
    ):
        super().__init__()
        self.max_context_len = max_context_len
        self.static_dim = static_dim
        self.action_encoder = ActionHistoryEncoder(
            max_agents=max_agents,
            max_actions=max_actions,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=action_layers,
        )
        self.static_encoder = (
            MLP(static_dim, hidden_dim, action_dim) if static_dim > 0 else None
        )
        self.timestep_embedding = nn.Embedding(max_context_len, latent_dim)
        self.input_proj = nn.Linear(latent_dim + action_dim, latent_dim)
        self.action_token_proj = nn.Linear(action_dim, latent_dim)
        self.blocks = nn.ModuleList(
            AttentionBlock(latent_dim, num_heads=num_heads, hidden_dim=hidden_dim)
            for _ in range(predictor_layers)
        )
        self.output = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim))

    def forward(
        self,
        latents: torch.Tensor,
        conditioning_actions: torch.Tensor,
        action_mask: torch.Tensor,
        timestep_mask: torch.Tensor | None = None,
        entity_mask: torch.Tensor | None = None,
        static_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents.shape[1] > self.max_context_len:
            raise ValueError(
                f"context length {latents.shape[1]} exceeds max_context_len {self.max_context_len}"
            )
        batch, steps, entities, latent_dim = latents.shape
        action_tokens = self.action_encoder(conditioning_actions, action_mask)
        if self.static_encoder is not None:
            if static_condition is None:
                static_condition = torch.zeros(
                    batch,
                    self.static_dim,
                    dtype=latents.dtype,
                    device=latents.device,
                )
            static_emb = self.static_encoder(static_condition.to(latents.dtype))
            action_tokens = action_tokens + static_emb.view(batch, 1, 1, -1)
        action_weights = (action_mask > 0).to(action_tokens.dtype).unsqueeze(-1)
        action_context = (action_tokens * action_weights).sum(dim=2, keepdim=True)
        action_context = action_context / action_weights.sum(dim=2, keepdim=True).clamp_min(1.0)
        action_context = action_context.expand(batch, steps, entities, -1)
        entity_x = self.input_proj(torch.cat([latents, action_context], dim=-1))
        action_x = self.action_token_proj(action_tokens)
        step_ids = torch.arange(steps, device=latents.device)
        step_emb = self.timestep_embedding(step_ids).view(1, steps, 1, latent_dim)
        entity_x = entity_x + step_emb
        action_x = action_x + step_emb
        tokens_per_step = entities + action_x.shape[2]
        x = torch.cat([entity_x, action_x], dim=2).reshape(batch, steps * tokens_per_step, latent_dim)
        key_padding_mask = None
        if timestep_mask is not None or entity_mask is not None:
            if timestep_mask is None:
                timestep_mask = torch.ones(batch, steps, dtype=latents.dtype, device=latents.device)
            if entity_mask is None:
                entity_mask = torch.ones(
                    batch, steps, entities, dtype=latents.dtype, device=latents.device
                )
            action_valid = action_mask > 0
            valid_mask = torch.cat([entity_mask > 0, action_valid], dim=2)
            valid_mask = (timestep_mask.unsqueeze(-1) > 0) & valid_mask
            key_padding_mask = ~valid_mask.reshape(batch, steps * tokens_per_step)
            all_padded = key_padding_mask.all(dim=1)
            if all_padded.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_padded, 0] = False
        causal_mask = _entity_time_causal_mask(steps, tokens_per_step, latents.device)
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask, attn_mask=causal_mask)
        x = self.output(x).reshape(batch, steps, tokens_per_step, latent_dim)
        return x[:, :, :entities]


def _entity_time_causal_mask(steps: int, entities: int, device: torch.device) -> torch.Tensor:
    token_steps = torch.arange(steps, device=device).repeat_interleave(entities)
    return token_steps.view(1, -1) > token_steps.view(-1, 1)
