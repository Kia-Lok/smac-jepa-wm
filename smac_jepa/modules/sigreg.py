from __future__ import annotations

import torch
from torch import nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer used by LeWM.

    The statistic matches random one-dimensional projections of the latent
    distribution to a standard Gaussian characteristic function.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        if knots < 2:
            raise ValueError("knots must be at least 2")
        if num_proj < 1:
            raise ValueError("num_proj must be at least 1")
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, latents: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        z = _flatten_valid_latents(latents, mask)
        if z.shape[0] < 2:
            return latents.new_tensor(0.0)
        projections = torch.randn(z.shape[-1], self.num_proj, device=z.device, dtype=z.dtype)
        projections = projections / projections.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)
        projected = z @ projections
        x_t = projected.unsqueeze(-1) * self.t.to(z.dtype)
        err = (x_t.cos().mean(dim=0) - self.phi.to(z.dtype)).square()
        err = err + x_t.sin().mean(dim=0).square()
        statistic = (err @ self.weights.to(z.dtype)) * z.shape[0]
        return statistic.mean()


def sigreg_loss(
    latents: torch.Tensor,
    mask: torch.Tensor | None = None,
    knots: int = 17,
    num_proj: int = 1024,
) -> torch.Tensor:
    """Functional wrapper for LeWM-style SIGReg."""

    return SIGReg(knots=knots, num_proj=num_proj).to(latents.device)(latents, mask)


def _flatten_valid_latents(latents: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        flat_mask = mask.reshape(-1) > 0
        return latents.reshape(-1, latents.shape[-1])[flat_mask]
    return latents.reshape(-1, latents.shape[-1])
