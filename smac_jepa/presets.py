from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPreset:
    latent_dim: int
    hidden_dim: int
    action_dim: int
    num_heads: int
    encoder_layers: int
    action_layers: int
    predictor_layers: int
    batch_size: int
    lr: float


MODEL_PRESETS: dict[str, ModelPreset] = {
    "smoke": ModelPreset(
        latent_dim=64,
        hidden_dim=128,
        action_dim=64,
        num_heads=4,
        encoder_layers=1,
        action_layers=1,
        predictor_layers=1,
        batch_size=32,
        lr=1e-3,
    ),
    "default": ModelPreset(
        latent_dim=192,
        hidden_dim=512,
        action_dim=192,
        num_heads=8,
        encoder_layers=4,
        action_layers=4,
        predictor_layers=4,
        batch_size=128,
        lr=3e-4,
    ),
    "large": ModelPreset(
        latent_dim=384,
        hidden_dim=1536,
        action_dim=384,
        num_heads=12,
        encoder_layers=8,
        action_layers=6,
        predictor_layers=8,
        batch_size=128,
        lr=2e-4,
    ),
}


def get_model_preset(name: str) -> ModelPreset:
    try:
        return MODEL_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unknown model preset {name!r}; expected one of: {choices}") from exc
