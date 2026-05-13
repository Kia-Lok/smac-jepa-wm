from __future__ import annotations

import torch
import torch.nn.functional as F


def sigreg_loss(latents: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-4) -> torch.Tensor:
    """Variance/covariance regularizer for non-collapsed latent embeddings."""

    if mask is not None:
        flat_mask = mask.reshape(-1) > 0
        z = latents.reshape(-1, latents.shape[-1])[flat_mask]
    else:
        z = latents.reshape(-1, latents.shape[-1])

    if z.shape[0] < 2:
        return latents.new_tensor(0.0)

    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    variance_loss = F.relu(1.0 - std).mean()

    cov = (z.T @ z) / (z.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    covariance_loss = off_diag.pow(2).sum() / z.shape[-1]
    return variance_loss + covariance_loss

