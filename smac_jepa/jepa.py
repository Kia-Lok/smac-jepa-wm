from __future__ import annotations

import torch
from torch import nn

from smac_jepa.modules import (
    EntityJEPAActionPredictor,
    EntityStateEncoder,
    JEPAActionPredictor,
    StateEncoder,
    sigreg_loss,
)

#Loads both the encoder and predictor as part of the JEPA Model (Currently got issue where the encoder is somehow not an attention head
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
        mode: str = "flat",
        max_agents: int | None = None,
        max_enemies: int = 0,
        max_actions: int | None = None,
        token_dim: int | None = None,
        decoder_weight: float = 1.0,
        encoder_layers: int = 1,
        action_layers: int = 1,
        predictor_layers: int = 1,
        max_context_len: int = 32,
    ):
        super().__init__() #Need override nn.Module
        self.mode = mode
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.latent_dim = latent_dim #Flags to pass in as params when running the script
        self.decoder_weight = decoder_weight
        if mode == "entity":
            if token_dim is None or max_agents is None or max_actions is None:
                raise ValueError("Entity mode requires token_dim, max_agents, and max_actions")
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
            )
            
            #Fully connected layer
            self.decoder = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, (max_agents + max_enemies) * token_dim),
            )
        else:
            self.max_agents = n_agents
            self.max_enemies = 0
            self.max_actions = n_actions
            self.token_dim = state_dim
            self.encoder = StateEncoder(state_dim, hidden_dim, latent_dim)
            self.predictor = JEPAActionPredictor(
                latent_dim=latent_dim,
                n_agents=n_agents,
                n_actions=n_actions,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
            )
            self.decoder = None
    #Encode the obs into embeddings.
    def encode_state(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder(states)
    #Produce the prediction based on the current observation and conditioning variable (Past actions)
    def predict_next(
        self,
        latents: torch.Tensor,
        conditioning_actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.predictor(latents, conditioning_actions)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.mode == "entity":
            latents = self.encoder(batch["entity_t"], batch["entity_mask"])
            with torch.no_grad():
                target_latent = self.encoder(batch["target_entity"], batch["target_entity_mask"])
            pred_latent = self.predictor(
                latents,
                batch["action_t"],
                batch["action_mask"],
                batch["mask"],
            )
            decoded = self.decode_entities(pred_latent)
            return {
                "pred_latent": pred_latent,
                "target_latent": target_latent,
                "decoded_target": decoded,
                "target_entity": batch["target_entity"],
                "target_entity_mask": batch["target_entity_mask"],
                "mask": batch["mask"],
            }

        # observation sequence plus action-history conditioning predicts the
        # next observation sequence in latent space.
        latents = self.encode_state(batch["state_t"]) #Observed obs (state_t -> encoder -> latents)
        with torch.no_grad():
            target_latent = self.encode_state(batch["target_state"]) #Real next state (target_state -> encoder -> target_latents)
        pred_latent = self.predict_next(latents, batch["action_t"]) #Pred next state (latent + action_t -> predictor -> pred_latent)
        return {
            "pred_latent": pred_latent,
            "target_latent": target_latent,
            "mask": batch["mask"], #Masked MSE is used due to possibility of the latent space being invalid (actual < max)
        }

    def decode_entities(self, latents: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("Entity decoder is only available in entity mode")
        decoded = self.decoder(latents)
        return decoded.reshape(
            latents.shape[0],
            latents.shape[1],
            self.max_agents + self.max_enemies,
            self.token_dim,
        )

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        sigreg_weight: float = 0.01,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(batch)
        mask = out["mask"].unsqueeze(-1)
        denom = mask.sum().clamp_min(1.0) * out["pred_latent"].shape[-1]
        pred_loss = ((out["pred_latent"] - out["target_latent"]).pow(2) * mask).sum() / denom
        reg_loss = sigreg_loss(out["target_latent"], out["mask"])
        decoded_loss = pred_loss.new_tensor(0.0)
        if self.mode == "entity":
            entity_mask = out["target_entity_mask"].unsqueeze(-1) * out["mask"].unsqueeze(-1).unsqueeze(-1)
            entity_denom = entity_mask.sum().clamp_min(1.0) * out["target_entity"].shape[-1]
            decoded_loss = (
                (out["decoded_target"] - out["target_entity"]).pow(2) * entity_mask
            ).sum() / entity_denom
        total = pred_loss + sigreg_weight * reg_loss + self.decoder_weight * decoded_loss
        losses = {
            "total_loss": total,
            "pred_loss": pred_loss,
            "sigreg_loss": reg_loss,
            "decoded_loss": decoded_loss,
        }
        if self.mode == "entity":
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
    for threshold in (0.01, 0.05, 0.10):
        correct = ((decoded - target).abs() <= threshold).to(target.dtype) * mask
        metrics[f"tol_acc_{threshold:.2f}"] = correct.sum() / denom
    return metrics
