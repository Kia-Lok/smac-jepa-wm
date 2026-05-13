from __future__ import annotations

import torch
from torch import nn

from smac_jepa.modules.blocks import AttentionBlock, MLP


class JEPAActionPredictor(nn.Module):
    """Predicts next latents from observation latents and action-history conditioning."""

    def __init__(
        self,
        latent_dim: int,
        n_agents: int,
        n_actions: int,
        action_dim: int,
        hidden_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.action_encoder = MLP(n_agents * n_actions, hidden_dim, action_dim)
        self.input_proj = nn.Linear(latent_dim + action_dim, latent_dim)
        self.block = AttentionBlock(latent_dim, num_heads=num_heads, hidden_dim=hidden_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )
    
    def forward(self, latents: torch.Tensor, conditioning_actions: torch.Tensor) -> torch.Tensor:
        batch, steps = latents.shape[:2]
        #Basically action_emb is the conditioning variable. conditioning_actions store every action from previous state
        action_flat = conditioning_actions.reshape(batch, steps, self.n_agents * self.n_actions) #Flattens the vector to just 3 dimensions
        action_emb = self.action_encoder(action_flat) #Becomes action embedding
        x = torch.cat([latents, action_emb], dim=-1) #Combine to form (obs emb, action emb)
        x = self.input_proj(x)
        x = self.block(x) #Attention block to predict next action (Lowkey sus tho is this how they did it in LeWM?)
        return self.output(x)


class ActionHistoryEncoder(nn.Module):
    """Encodes all joint actions in the context window with masked attention."""

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
        denom = flat_mask.sum(dim=1, keepdim=True).clamp_min(1).to(flat_x.dtype)
        pooled = flat_x.sum(dim=1) / denom
        return pooled.reshape(batch, steps, -1)


class EntityJEPAActionPredictor(nn.Module):
    """Predicts next state latents from latents and full joint-action history."""

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
    ):
        super().__init__()
        self.max_context_len = max_context_len
        self.action_encoder = ActionHistoryEncoder(
            max_agents=max_agents,
            max_actions=max_actions,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=action_layers,
        )
        self.timestep_embedding = nn.Embedding(max_context_len, latent_dim)
        self.input_proj = nn.Linear(latent_dim + action_dim, latent_dim)
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
    ) -> torch.Tensor:
        if latents.shape[1] > self.max_context_len:
            raise ValueError(
                f"context length {latents.shape[1]} exceeds max_context_len {self.max_context_len}"
            )
        action_emb = self.action_encoder(conditioning_actions, action_mask)
        x = self.input_proj(torch.cat([latents, action_emb], dim=-1))
        step_ids = torch.arange(latents.shape[1], device=latents.device)
        x = x + self.timestep_embedding(step_ids).view(1, latents.shape[1], -1)
        key_padding_mask = None
        if timestep_mask is not None:
            key_padding_mask = ~(timestep_mask > 0)
            all_padded = key_padding_mask.all(dim=1)
            if all_padded.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_padded, 0] = False
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return self.output(x)
