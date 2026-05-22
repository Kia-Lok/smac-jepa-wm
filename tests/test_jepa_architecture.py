from __future__ import annotations

import torch

from smac_jepa.jepa import SMACJEPA
from smac_jepa.modules import EntityJEPAActionPredictor, EntityStateEncoder, SIGReg


def test_lewm_sigreg_is_finite_and_backpropagates() -> None:
    latents = torch.randn(4, 3, 8, requires_grad=True)
    loss = SIGReg(knots=7, num_proj=16)(latents)

    assert torch.isfinite(loss)
    loss.backward()
    assert latents.grad is not None
    assert torch.isfinite(latents.grad).all()
    assert latents.grad.abs().sum() > 0


def test_model_loss_routes_sigreg_through_live_encoder_latents() -> None:
    model = SMACJEPA(
        state_dim=0,
        n_agents=2,
        n_actions=4,
        latent_dim=16,
        hidden_dim=32,
        action_dim=16,
        num_heads=4,
        mode="entity",
        max_agents=2,
        max_enemies=1,
        max_actions=4,
        token_dim=4,
        decoder_weight=0.0,
        encoder_layers=1,
        action_layers=1,
        predictor_layers=1,
        max_context_len=4,
    )
    batch = {
        "entity_t": torch.randn(2, 3, 3, 4),
        "entity_mask": torch.ones(2, 3, 3),
        "action_t": torch.zeros(2, 3, 2, 4),
        "action_mask": torch.ones(2, 3, 2),
        "target_entity": torch.randn(2, 3, 3, 4),
        "target_entity_mask": torch.ones(2, 3, 3),
        "entity_slot_mask": torch.ones(2, 3, 3),
        "mask": torch.ones(2, 3),
    }

    losses = model.loss(batch, sigreg_weight=1.0)
    assert "presence_loss" in losses
    assert "presence_acc" in losses
    losses["total_loss"].backward()

    encoder_grad = sum(
        param.grad.abs().sum().item()
        for param in model.encoder.parameters()
        if param.grad is not None
    )
    assert encoder_grad > 0


def test_entity_predictor_is_causal_for_early_steps() -> None:
    torch.manual_seed(1)
    predictor = EntityJEPAActionPredictor(
        latent_dim=16,
        max_agents=2,
        max_actions=4,
        action_dim=16,
        hidden_dim=32,
        num_heads=4,
        action_layers=1,
        predictor_layers=2,
        max_context_len=4,
    )
    predictor.eval()
    latents = torch.randn(2, 4, 3, 16)
    actions = torch.randn(2, 4, 2, 4)
    action_mask = torch.ones(2, 4, 2)
    timestep_mask = torch.ones(2, 4)
    entity_mask = torch.ones(2, 4, 3)

    base = predictor(latents, actions, action_mask, timestep_mask, entity_mask)
    changed_latents = latents.clone()
    changed_actions = actions.clone()
    changed_latents[:, 2:] = torch.randn_like(changed_latents[:, 2:]) * 10
    changed_actions[:, 2:] = torch.randn_like(changed_actions[:, 2:]) * 10
    changed = predictor(changed_latents, changed_actions, action_mask, timestep_mask, entity_mask)

    torch.testing.assert_close(base[:, :2], changed[:, :2])


def test_entity_encoder_uses_agent_slot_identity() -> None:
    torch.manual_seed(2)
    encoder = EntityStateEncoder(
        token_dim=4,
        latent_dim=16,
        hidden_dim=32,
        num_heads=4,
        max_agents=2,
        max_enemies=1,
        num_layers=1,
    )
    encoder.eval()
    tokens = torch.zeros(1, 1, 3, 4)
    tokens[0, 0, 0] = torch.tensor([1.0, 0.0, 0.2, 0.3])
    tokens[0, 0, 1] = torch.tensor([0.5, 0.1, -0.2, 0.4])
    tokens[0, 0, 2] = torch.tensor([0.8, -0.1, 0.0, 0.0])
    swapped = tokens.clone()
    swapped[0, 0, 0] = tokens[0, 0, 1]
    swapped[0, 0, 1] = tokens[0, 0, 0]
    mask = torch.ones(1, 1, 3)

    encoded = encoder(tokens, mask)
    encoded_swapped = encoder(swapped, mask)

    assert encoded.shape == (1, 1, 3, 16)
    assert not torch.allclose(encoded, encoded_swapped)
    assert not torch.allclose(encoded[:, :, 0], encoded_swapped[:, :, 0])


def test_entity_predictor_uses_static_conditioning() -> None:
    torch.manual_seed(3)
    predictor = EntityJEPAActionPredictor(
        latent_dim=16,
        max_agents=2,
        max_actions=4,
        action_dim=16,
        hidden_dim=32,
        num_heads=4,
        action_layers=1,
        predictor_layers=1,
        max_context_len=4,
        static_dim=5,
    )
    predictor.eval()
    latents = torch.randn(2, 3, 3, 16)
    actions = torch.zeros(2, 3, 2, 4)
    action_mask = torch.ones(2, 3, 2)
    timestep_mask = torch.ones(2, 3)
    entity_mask = torch.ones(2, 3, 3)
    static_a = torch.zeros(2, 5)
    static_b = torch.ones(2, 5)

    pred_a = predictor(latents, actions, action_mask, timestep_mask, entity_mask, static_a)
    pred_b = predictor(latents, actions, action_mask, timestep_mask, entity_mask, static_b)

    assert not torch.allclose(pred_a, pred_b)
