from __future__ import annotations

"""
Trainable rollout memory modules for recursive latent rollout.

First version:
    - one GRU memory vector per entity slot
    - initial memory is zeros
    - before each predictor call, memory is fused into the current rollout latent
    - after prediction, memory is updated using the latest predicted latent z_hat

No separate memory loss is required. The memory module is trained because its
output changes the predictor input, and the normal rollout loss backpropagates
through it.
"""

import torch
from torch import nn


class EntityRolloutGRUMemory(nn.Module):
    """
    Per-entity GRU memory for Markov rollout.

    We do not concatenate memory directly into the existing predictor because the
    predictor expects latent_dim inputs. Instead:

        concat([z, memory]) -> MLP -> latent_dim correction
        z_conditioned = LayerNorm(z + correction)

    So the predictor still receives [N, 1, E, latent_dim].
    """

    def __init__(
        self,
        latent_dim: int,
        memory_dim: int = 128,
        hidden_dim: int | None = None,
        residual: bool = True,
    ):
        super().__init__()
        if memory_dim < 1:
            raise ValueError("memory_dim must be >= 1")

        self.latent_dim = int(latent_dim)
        self.memory_dim = int(memory_dim)
        self.hidden_dim = int(hidden_dim or max(latent_dim, memory_dim))
        self.residual = bool(residual)

        self.gru = nn.GRUCell(self.latent_dim, self.memory_dim)

        self.fuse = nn.Sequential(
            nn.Linear(self.latent_dim + self.memory_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.norm = nn.LayerNorm(self.latent_dim)

    def initial_memory(
        self,
        batch_entities: int,
        entity_slots: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_entities,
            entity_slots,
            self.memory_dim,
            device=device,
            dtype=dtype,
        )

    def condition(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        entity_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Inject memory into latent before predictor call.

        z:           [N, E, D]
        memory:      [N, E, M]
        entity_mask: [N, E], optional
        returns:     [N, E, D]
        """
        if z.ndim != 3:
            raise ValueError(f"z must have shape [N, E, D], got {tuple(z.shape)}")
        if memory.ndim != 3:
            raise ValueError(f"memory must have shape [N, E, M], got {tuple(memory.shape)}")
        if z.shape[:2] != memory.shape[:2]:
            raise ValueError("z and memory must have matching [N, E] dimensions")

        correction = self.fuse(torch.cat([z, memory], dim=-1))
        if self.residual:
            out = self.norm(z + correction)
        else:
            out = self.norm(correction)

        if entity_mask is not None:
            out = out * entity_mask.unsqueeze(-1)

        return out

    def update(
        self,
        z_next: torch.Tensor,
        memory: torch.Tensor,
        entity_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Update memory using the latest predicted latent.

        z_next:      [N, E, D]
        memory:      [N, E, M]
        entity_mask: [N, E], optional
        returns:     [N, E, M]
        """
        if z_next.ndim != 3:
            raise ValueError(f"z_next must have shape [N, E, D], got {tuple(z_next.shape)}")
        if memory.ndim != 3:
            raise ValueError(f"memory must have shape [N, E, M], got {tuple(memory.shape)}")

        n, e, d = z_next.shape
        z_flat = z_next.reshape(n * e, d)
        mem_flat = memory.reshape(n * e, self.memory_dim)

        new_mem = self.gru(z_flat, mem_flat).reshape(n, e, self.memory_dim)

        if entity_mask is not None:
            new_mem = new_mem * entity_mask.unsqueeze(-1)

        return new_mem
