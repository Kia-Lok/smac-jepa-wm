from __future__ import annotations

import torch
from torch import nn

from smac_jepa.modules.blocks import AttentionBlock, MLP


class StateEncoder(nn.Module):
    """Small vector-state encoder replacing LeWM's pixel encoder."""

    def __init__(self, state_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            MLP(state_dim, hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class EntityStateEncoder(nn.Module):
    """Masked self-attention encoder for padded SMACLite entity tokens."""

    def __init__(
        self,
        token_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_heads: int,
        max_agents: int,
        max_enemies: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.max_agents = max_agents
        self.max_enemies = max_enemies
        self.input = nn.Linear(token_dim, latent_dim)
        self.type_embedding = nn.Embedding(2, latent_dim)
        self.blocks = nn.ModuleList(
            AttentionBlock(latent_dim, num_heads=num_heads, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, steps, token_count = tokens.shape[:3]
        x = self.input(tokens)
        type_ids = torch.zeros(token_count, dtype=torch.long, device=tokens.device)
        type_ids[self.max_agents : self.max_agents + self.max_enemies] = 1
        x = x + self.type_embedding(type_ids).view(1, 1, token_count, -1)

        flat_x = x.reshape(batch * steps, token_count, -1)
        flat_mask = mask.reshape(batch * steps, token_count) > 0
        key_padding_mask = ~flat_mask
        all_padded = key_padding_mask.all(dim=1)
        if all_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padded, 0] = False

        for block in self.blocks:
            flat_x = block(flat_x, key_padding_mask=key_padding_mask)
        flat_x = self.norm(flat_x)
        flat_x = flat_x * flat_mask.unsqueeze(-1)
        denom = flat_mask.sum(dim=1, keepdim=True).clamp_min(1).to(flat_x.dtype)
        pooled = flat_x.sum(dim=1) / denom
        return pooled.reshape(batch, steps, -1)
