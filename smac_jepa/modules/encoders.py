from __future__ import annotations

import torch
from torch import nn

from smac_jepa.modules.blocks import AttentionBlock


class EntityStateEncoder(nn.Module):
    """Masked self-attention encoder that preserves one latent per entity slot."""

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
        self.ally_slot_embedding = nn.Embedding(max_agents, latent_dim)
        self.enemy_slot_embedding = nn.Embedding(max_enemies, latent_dim) if max_enemies else None
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
        slot_emb = torch.zeros(token_count, x.shape[-1], dtype=x.dtype, device=tokens.device)
        ally_ids = torch.arange(self.max_agents, device=tokens.device)
        slot_emb[: self.max_agents] = self.ally_slot_embedding(ally_ids)
        if self.max_enemies and self.enemy_slot_embedding is not None:
            enemy_ids = torch.arange(self.max_enemies, device=tokens.device)
            slot_emb[self.max_agents : self.max_agents + self.max_enemies] = (
                self.enemy_slot_embedding(enemy_ids)
            )
        x = x + slot_emb.view(1, 1, token_count, -1)

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
        return flat_x.reshape(batch, steps, token_count, -1)
