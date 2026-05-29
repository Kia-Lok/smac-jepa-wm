from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from smac_jepa.data.dataset import DatasetMetadata
from smac_jepa.data.dataset import load_npz_metadata
from smac_jepa.decoder import format_entity_predictions
from smac_jepa.jepa import SMACJEPA
from smac_jepa.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict one decoded next state from one SMACLite global state and joint action"
    )
    parser.add_argument("--checkpoint", required=True)
    state_group = parser.add_mutually_exclusive_group(required=True)
    state_group.add_argument("--state-npy", help="Path to a .npy state vector")
    state_group.add_argument("--state-json", help="Path to a JSON list of state floats")
    parser.add_argument("--actions-json", required=True, help="Path to a JSON list of action ids")
    parser.add_argument(
        "--metadata-npz",
        help="Optional source dataset .npz for the state/action layout when it differs from checkpoint caps",
    )
    parser.add_argument("--out", help="Optional output JSON path")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("resolved_config", checkpoint["config"])
    checkpoint_metadata = metadata_from_checkpoint(checkpoint["metadata"])
    model = build_model_from_checkpoint(checkpoint, config, checkpoint_metadata, device)
    input_metadata = checkpoint_metadata
    static_condition = np.zeros((checkpoint_metadata.static_dim,), dtype=np.float32)
    entity_static = np.zeros(
        (
            checkpoint_metadata.max_agents + checkpoint_metadata.max_enemies,
            checkpoint_metadata.entity_static_feat_size,
        ),
        dtype=np.float32,
    )
    if args.metadata_npz:
        source_metadata = load_npz_metadata(args.metadata_npz)
        input_metadata = metadata_for_model_caps(source_metadata, checkpoint_metadata)
        static_condition = load_static_condition(args.metadata_npz, input_metadata)
        entity_static = load_entity_static(args.metadata_npz, input_metadata)

    state = load_state_vector(args.state_npy, args.state_json)
    action_ids = load_action_ids(args.actions_json)
    result = predict_human_readable_next(
        model, input_metadata, state, action_ids, static_condition, entity_static, device
    )

    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
    else:
        print(output, end="")


def metadata_from_checkpoint(raw: dict[str, Any]) -> DatasetMetadata:
    return DatasetMetadata(
        state_dim=int(raw["state_dim"]),
        n_agents=int(raw["n_agents"]),
        n_actions=int(raw["n_actions"]),
        n_enemies=int(raw.get("n_enemies", 0)),
        ally_state_feat_size=int(raw.get("ally_state_feat_size", 0)),
        enemy_state_feat_size=int(raw.get("enemy_state_feat_size", 0)),
        ally_has_shields=bool(raw.get("ally_has_shields", False)),
        enemy_has_shields=bool(raw.get("enemy_has_shields", False)),
        num_unit_types=int(raw.get("num_unit_types", 0)),
        max_agents=int(raw.get("max_agents", raw["n_agents"])),
        max_enemies=int(raw.get("max_enemies", raw.get("n_enemies", 0))),
        max_actions=int(raw.get("max_actions", raw["n_actions"])),
        token_dim=int(raw.get("token_dim", raw["state_dim"])),
        dynamic_token_dim=int(raw.get("dynamic_token_dim", raw.get("token_dim", raw["state_dim"]))),
        static_dim=int(raw.get("static_dim", 0)),
        entity_static_feat_size=int(raw.get("entity_static_feat_size", 0)),
        mode=str(raw.get("mode", "entity")),
    )


def metadata_for_model_caps(source: DatasetMetadata, checkpoint: DatasetMetadata) -> DatasetMetadata:
    checkpoint_dynamic_dim = checkpoint.dynamic_token_dim or checkpoint.token_dim
    source_dynamic_dim = source.dynamic_token_dim or max(
        source.ally_state_feat_size, source.enemy_state_feat_size
    )
    if source.n_agents > checkpoint.max_agents:
        raise ValueError("Source metadata has more agents than the checkpoint supports")
    if source.n_enemies > checkpoint.max_enemies:
        raise ValueError("Source metadata has more enemies than the checkpoint supports")
    if source.n_actions > checkpoint.max_actions:
        raise ValueError("Source metadata has more actions than the checkpoint supports")
    if source_dynamic_dim > checkpoint_dynamic_dim:
        raise ValueError("Source metadata has wider entity tokens than the checkpoint supports")
    if source.static_dim > checkpoint.static_dim:
        raise ValueError("Source metadata has wider static conditioning than the checkpoint supports")
    if source.entity_static_feat_size > checkpoint.entity_static_feat_size:
        raise ValueError("Source entity static features exceed checkpoint support")
    return DatasetMetadata(
        state_dim=source.state_dim,
        n_agents=source.n_agents,
        n_actions=source.n_actions,
        n_enemies=source.n_enemies,
        ally_state_feat_size=source.ally_state_feat_size,
        enemy_state_feat_size=source.enemy_state_feat_size,
        ally_has_shields=source.ally_has_shields,
        enemy_has_shields=source.enemy_has_shields,
        num_unit_types=source.num_unit_types,
        max_agents=checkpoint.max_agents,
        max_enemies=checkpoint.max_enemies,
        max_actions=checkpoint.max_actions,
        token_dim=checkpoint.token_dim,
        dynamic_token_dim=checkpoint_dynamic_dim,
        static_dim=checkpoint.static_dim,
        entity_static_feat_size=checkpoint.entity_static_feat_size,
        mode=checkpoint.mode,
    )


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    metadata: DatasetMetadata,
    device: torch.device,
) -> SMACJEPA:
    model = SMACJEPA(
        state_dim=metadata.state_dim,
        n_agents=metadata.n_agents,
        n_actions=metadata.n_actions,
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        action_dim=int(config["action_dim"]),
        num_heads=int(config["num_heads"]),
        mode=metadata.mode,
        max_agents=metadata.max_agents,
        max_enemies=metadata.max_enemies,
        max_actions=metadata.max_actions,
        token_dim=metadata.token_dim,
        static_dim=metadata.static_dim,
        decoder_weight=float(config.get("decoder_weight", 1.0)),
        encoder_layers=int(config.get("encoder_layers", 1)),
        action_layers=int(config.get("action_layers", 1)),
        predictor_layers=int(config.get("predictor_layers", 1)),
        max_context_len=int(config.get("max_context_len", 32)),
        static_conditioning=str(
            checkpoint["metadata"].get("static_conditioning", config.get("static_conditioning", "action"))
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def predict_human_readable_next(
    model: SMACJEPA,
    metadata: DatasetMetadata,
    state: np.ndarray,
    action_ids: list[int],
    static_condition: np.ndarray,
    entity_static: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    if metadata.mode != "entity":
        raise ValueError("Human-readable prediction currently requires entity-mode metadata")
    entity_tokens, entity_mask = encode_state_vector(state, metadata, entity_static)
    action_tensor, action_mask = encode_action_ids(action_ids, metadata)
    with torch.no_grad():
        entity_t = torch.from_numpy(entity_tokens).unsqueeze(0).unsqueeze(0).to(device)
        entity_mask_t = torch.from_numpy(entity_mask).unsqueeze(0).unsqueeze(0).to(device)
        action_t = torch.from_numpy(action_tensor).unsqueeze(0).unsqueeze(0).to(device)
        action_mask_t = torch.from_numpy(action_mask).unsqueeze(0).unsqueeze(0).to(device)
        static_t = torch.from_numpy(static_condition).unsqueeze(0).to(device)
        timestep_mask = torch.ones((1, 1), dtype=torch.float32, device=device)
        latent = model.encoder(entity_t, entity_mask_t)
        pred_latent = model.predictor(
            latent, action_t, action_mask_t, timestep_mask, entity_mask_t, static_t
        )
        decoded = model.decode_entities(pred_latent)[0, 0]
        presence_scores = torch.sigmoid(model.predict_presence(pred_latent))[0, 0]

    return {
        "prediction": format_entity_predictions(decoded, metadata, entity_mask, presence_scores),
        "current_state": format_entity_predictions(entity_tokens, metadata, entity_mask),
        "raw_decoded_shape": list(decoded.shape),
        "metadata": asdict(metadata),
        "input_action_ids": action_ids,
    }


def encode_state_vector(
    state: np.ndarray,
    metadata: DatasetMetadata,
    entity_static: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] != metadata.state_dim:
        raise ValueError(f"Expected state_dim={metadata.state_dim}, got {state.shape[0]}")
    token_count = metadata.max_agents + metadata.max_enemies
    tokens = np.zeros((token_count, metadata.token_dim), dtype=np.float32)
    mask = np.zeros(token_count, dtype=np.float32)
    ally_end = metadata.n_agents * metadata.ally_state_feat_size
    enemy_end = ally_end + metadata.n_enemies * metadata.enemy_state_feat_size
    allies = state[:ally_end].reshape(metadata.n_agents, metadata.ally_state_feat_size)
    enemies = state[ally_end:enemy_end].reshape(metadata.n_enemies, metadata.enemy_state_feat_size)
    tokens[: metadata.n_agents, : metadata.ally_state_feat_size] = allies
    mask[: metadata.n_agents] = (np.abs(allies).sum(axis=-1) > 0).astype(np.float32)
    enemy_start = metadata.max_agents
    tokens[
        enemy_start : enemy_start + metadata.n_enemies,
        : metadata.enemy_state_feat_size,
    ] = enemies
    mask[enemy_start : enemy_start + metadata.n_enemies] = (
        np.abs(enemies).sum(axis=-1) > 0
    ).astype(np.float32)
    if entity_static is not None and metadata.entity_static_feat_size > 0:
        offset = metadata.dynamic_token_dim
        tokens[:, offset : offset + metadata.entity_static_feat_size] = entity_static[
            :, : metadata.entity_static_feat_size
        ]
    return tokens, mask


def load_static_condition(path: str, metadata: DatasetMetadata) -> np.ndarray:
    if metadata.static_dim <= 0:
        return np.zeros((0,), dtype=np.float32)
    with np.load(path, allow_pickle=False) as data:
        if "static_condition" not in data:
            return np.zeros((metadata.static_dim,), dtype=np.float32)
        value = np.asarray(data["static_condition"], dtype=np.float32).reshape(-1)
    padded = np.zeros((metadata.static_dim,), dtype=np.float32)
    padded[: min(metadata.static_dim, value.shape[0])] = value[: metadata.static_dim]
    return padded


def load_entity_static(path: str, metadata: DatasetMetadata) -> np.ndarray:
    token_count = metadata.max_agents + metadata.max_enemies
    padded = np.zeros((token_count, metadata.entity_static_feat_size), dtype=np.float32)
    if metadata.entity_static_feat_size <= 0:
        return padded
    with np.load(path, allow_pickle=False) as data:
        if "entity_static" not in data:
            return padded
        value = np.asarray(data["entity_static"], dtype=np.float32)
    width = min(metadata.entity_static_feat_size, value.shape[1])
    ally_count = min(metadata.n_agents, value.shape[0], metadata.max_agents)
    padded[:ally_count, :width] = value[:ally_count, :width]
    enemy_source_start = metadata.n_agents
    enemy_source_end = min(value.shape[0], enemy_source_start + metadata.n_enemies)
    enemy_count = max(0, min(enemy_source_end - enemy_source_start, metadata.max_enemies))
    if enemy_count:
        enemy_target_start = metadata.max_agents
        padded[enemy_target_start : enemy_target_start + enemy_count, :width] = value[
            enemy_source_start : enemy_source_start + enemy_count,
            :width,
        ]
    return padded


def encode_action_ids(
    action_ids: list[int],
    metadata: DatasetMetadata,
) -> tuple[np.ndarray, np.ndarray]:
    if len(action_ids) != metadata.n_agents:
        raise ValueError(f"Expected {metadata.n_agents} action ids, got {len(action_ids)}")
    actions = np.zeros((metadata.max_agents, metadata.max_actions), dtype=np.float32)
    mask = np.zeros(metadata.max_agents, dtype=np.float32)
    for agent_idx, action_id in enumerate(action_ids):
        if action_id < 0 or action_id >= metadata.n_actions:
            raise ValueError(
                f"Action id {action_id} for agent {agent_idx} is outside [0, {metadata.n_actions})"
            )
        actions[agent_idx, int(action_id)] = 1.0
        mask[agent_idx] = 1.0
    return actions, mask


def load_state_vector(state_npy: str | None, state_json: str | None) -> np.ndarray:
    if state_npy:
        return np.load(state_npy).astype(np.float32)
    if state_json:
        return np.asarray(json.loads(Path(state_json).read_text()), dtype=np.float32)
    raise ValueError("One of state_npy or state_json must be provided")


def load_action_ids(path: str) -> list[int]:
    values = json.loads(Path(path).read_text())
    if not isinstance(values, list):
        raise ValueError("actions-json must contain a JSON list")
    return [int(value) for value in values]


if __name__ == "__main__":
    main()
