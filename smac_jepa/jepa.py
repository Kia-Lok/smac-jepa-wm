from __future__ import annotations

import torch
from torch import nn

from smac_jepa.modules import (
    EntityJEPAActionPredictor,
    EntityStateEncoder,
    sigreg_loss,
)


class SMACJEPA(nn.Module):
    def __init__(
        self,
        state_dim: int, #Dimension of the input (Should vary so 
        n_agents: int, #set max number of agents
        n_actions: int, #Set all the actions that can be taken (Should be fixed)
        latent_dim: int = 64,  #Dimension of embedding space (Set manually)
        hidden_dim: int = 128,
        action_dim: int = 64, #Number of actions available
        num_heads: int = 2,
        mode: str = "entity",
        max_agents: int | None = None,
        max_enemies: int = 0,
        max_actions: int | None = None,
        token_dim: int | None = None,
        decoder_weight: float = 1.0,
        encoder_layers: int = 1,
        action_layers: int = 1,
        predictor_layers: int = 1,
        max_context_len: int = 32,
        static_dim: int = 0,
    ):
        super().__init__()
        if mode != "entity":
            raise ValueError("SMACJEPA only supports entity mode")
        if token_dim is None or max_agents is None or max_actions is None:
            raise ValueError("Entity mode requires token_dim, max_agents, and max_actions")
        self.mode = mode
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.latent_dim = latent_dim
        self.decoder_weight = decoder_weight
        self.static_dim = static_dim
        self.max_agents = max_agents
        self.max_enemies = max_enemies
        self.max_actions = max_actions
        self.token_dim = token_dim
        self.encoder = EntityStateEncoder(
            token_dim=token_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_agents=max_agents,
            max_enemies=max_enemies,
            num_layers=encoder_layers,
        )
        self.predictor = EntityJEPAActionPredictor(
            latent_dim=latent_dim,
            max_agents=max_agents,
            max_actions=max_actions,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            action_layers=action_layers,
            predictor_layers=predictor_layers,
            max_context_len=max_context_len,
            static_dim=static_dim,
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.presence_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        latents = self.encoder(batch["entity_t"], batch["entity_mask"])
        target_latent = self.encoder(batch["target_entity"], batch["target_entity_mask"])
        pred_latent = self.predictor(
            latents,
            batch["action_t"],
            batch["action_mask"],
            batch["mask"],
            batch["entity_mask"],
            batch.get("static_condition"),
        )
        decoded = self.decode_entities(pred_latent)
        presence_logits = self.predict_presence(pred_latent)
        current_latent_mask = batch["entity_mask"] * batch["mask"].unsqueeze(-1)
        target_latent_mask = batch["target_entity_mask"] * batch["mask"].unsqueeze(-1)
        slot_mask = batch.get("entity_slot_mask")
        if slot_mask is None:
            slot_mask = batch["target_entity_mask"] * batch["mask"].unsqueeze(-1)
        return {
            "pred_latent": pred_latent,
            "target_latent": target_latent,
            "reg_latent": torch.cat([latents, target_latent], dim=1),
            "reg_mask": torch.cat([current_latent_mask, target_latent_mask], dim=1),
            "decoded_target": decoded,
            "presence_logits": presence_logits,
            "target_entity": batch["target_entity"],
            "target_entity_mask": batch["target_entity_mask"],
            "entity_slot_mask": slot_mask,
            "mask": batch["mask"],
            "current_entity_mask": batch["entity_mask"],
        }

    def decode_entities(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def predict_presence(self, latents: torch.Tensor) -> torch.Tensor:
        return self.presence_head(latents).squeeze(-1)

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        sigreg_weight: float = 0.01,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(batch)
        mask = out["target_entity_mask"].unsqueeze(-1) * out["mask"].unsqueeze(-1).unsqueeze(-1)
        denom = mask.sum().clamp_min(1.0) * out["pred_latent"].shape[-1]
        pred_loss = ((out["pred_latent"] - out["target_latent"]).pow(2) * mask).sum() / denom
        reg_loss = sigreg_loss(out["reg_latent"], out["reg_mask"])
        entity_denom = mask.sum().clamp_min(1.0) * out["target_entity"].shape[-1]
        decoded_loss = (
            (out["decoded_target"] - out["target_entity"]).pow(2) * mask
        ).sum() / entity_denom
        slot_mask = out["entity_slot_mask"]
        presence_target = out["target_entity_mask"]
        presence_loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
            out["presence_logits"],
            presence_target,
            reduction="none",
        )
        presence_loss = (presence_loss_raw * slot_mask).sum() / slot_mask.sum().clamp_min(1.0)
        total = (
            pred_loss
            + sigreg_weight * reg_loss
            + self.decoder_weight * decoded_loss
            + presence_loss
        )
        losses = {
            "total_loss": total,
            "pred_loss": pred_loss,
            "sigreg_loss": reg_loss,
            "decoded_loss": decoded_loss,
            "presence_loss": presence_loss,
        }
        with torch.no_grad():
            losses.update(entity_prediction_metrics(out))
        return losses


def entity_prediction_metrics(out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    decoded = out["decoded_target"]
    target = out["target_entity"]
    mask = out["target_entity_mask"].unsqueeze(-1) * out["mask"].unsqueeze(-1).unsqueeze(-1)
    denom = (mask.sum() * target.shape[-1]).clamp_min(1.0)
    abs_err = (decoded - target).abs() * mask
    mae = abs_err.sum() / denom
    mse = ((decoded - target).pow(2) * mask).sum() / denom
    target_mean = (target * mask).sum() / denom
    ss_res = ((decoded - target).pow(2) * mask).sum()
    ss_tot = ((target - target_mean).pow(2) * mask).sum().clamp_min(1e-8)
    r2 = 1.0 - ss_res / ss_tot
    metrics = {
        "decoded_mae": mae,
        "decoded_mse": mse,
        "decoded_r2": r2,
    }
    if "presence_logits" in out and "entity_slot_mask" in out:
        slot_mask = out["entity_slot_mask"]
        target_presence = out["target_entity_mask"]
        pred_presence = (torch.sigmoid(out["presence_logits"]) >= 0.5).to(target.dtype)
        presence_correct = (pred_presence == target_presence).to(target.dtype) * slot_mask
        metrics["presence_acc"] = presence_correct.sum() / slot_mask.sum().clamp_min(1.0)
    for threshold in (0.01, 0.05, 0.10):
        correct = ((decoded - target).abs() <= threshold).to(target.dtype) * mask
        metrics[f"tol_acc_{threshold:.2f}"] = correct.sum() / denom
    return metrics
