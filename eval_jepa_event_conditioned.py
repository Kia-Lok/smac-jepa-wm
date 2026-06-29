from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from smac_jepa.data import SMACJEPADataset, load_manifest, load_manifest_all
from smac_jepa.data.markov_rollout_visibility_dataset import VisibilityMarkovRolloutSMACJEPADataset
from smac_jepa.jepa import SMACJEPA
from smac_jepa.train_jepa_exp17_exp19_fixed import (
    ActionConditionedEntityRolloutGRUMemory,
    BeliefEntityRolloutGRUMemory,
    merge_observed_presence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate blocker-fixed JEPA checkpoints with padding-aware decoded-state, action-sensitivity, "
            "memory-ablation, hidden-entity, and autonomous-mask diagnostics."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)

    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--window-mode", default="sequential", choices=["sequential", "random"])
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--target-mode", choices=["full", "observed"], default=None)

    visibility_group = parser.add_mutually_exclusive_group()
    visibility_group.add_argument(
        "--enemy-visibility-mask", dest="enemy_visibility_mask", action="store_true"
    )
    visibility_group.add_argument(
        "--no-enemy-visibility-mask", dest="enemy_visibility_mask", action="store_false"
    )
    parser.set_defaults(enemy_visibility_mask=None)
    parser.add_argument("--enemy-sight-range", type=float, default=None)

    parser.add_argument("--sigreg-weight", type=float, default=None)
    parser.add_argument("--decoder-weight", type=float, default=None)
    parser.add_argument("--presence-weight", type=float, default=None)
    parser.add_argument("--one-step-weight", type=float, default=None)
    parser.add_argument("--td-lambda", type=float, default=None)

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1],
        help=(
            "Absolute tolerances reported on meaningful features only. "
            "The evaluator also reports target-standard-deviation-normalized RMSE."
        ),
    )
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--eval-rollout-horizon",
        type=int,
        default=None,
        help="Force every checkpoint to be evaluated at the same horizon.",
    )
    parser.add_argument(
        "--require-hidden-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compatibility flag. Hidden coverage is reported explicitly, "
            "but zero hidden examples no longer discard the rest of the "
            "evaluation results."
        ),
    )

    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help=(
            "Run additional action-shuffle, zero-history-memory, and autonomous-mask rollouts. "
            "This is slower but much more informative for Dreamer integration."
        ),
    )
    parser.add_argument(
        "--diagnostic-max-batches",
        type=int,
        default=None,
        help="Optional smaller batch limit for extra diagnostics; defaults to --max-batches.",
    )

    probe_group = parser.add_mutually_exclusive_group()
    probe_group.add_argument(
        "--probe-decoder",
        dest="probe_decoder",
        action="store_true",
        help=(
            "Train/load an independent checkpoint-specific decoder probe. "
            "The world model is frozen; only a fresh copy of the native decoder MLP is trained."
        ),
    )
    probe_group.add_argument(
        "--no-probe-decoder",
        dest="probe_decoder",
        action="store_false",
    )
    parser.set_defaults(probe_decoder=True)

    parser.add_argument("--probe-dir", default=None)
    parser.add_argument("--probe-train-split", default="train")
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-max-batches-per-epoch", type=int, default=300)
    parser.add_argument("--probe-samples-per-epoch", type=int, default=20000)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-5)
    parser.add_argument("--probe-seed", type=int, default=123)
    parser.add_argument("--probe-grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--force-retrain-probe",
        action="store_true",
        help="Retrain probe decoders even when a saved probe file already exists.",
    )
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=0.01,
        help=(
            "A dynamic coordinate is considered changed when its absolute "
            "difference from the rollout-start state exceeds this value."
        ),
    )
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=0.01,
        help=(
            "A rollout target is considered eventful when any valid dynamic "
            "coordinate changes by more than this value."
        ),
    )
    parser.add_argument(
        "--attack-action-min",
        type=int,
        default=6,
        help=(
            "Minimum discrete action index treated as attack-like for the "
            "action-event subset."
        ),
    )
    return parser.parse_args()



def validate_dataset_visibility(
    dataset: VisibilityMarkovRolloutSMACJEPADataset,
    *,
    maximum_items: int = 1024,
) -> None:
    """Fail early if the explicit observation mask never marks hidden enemies."""
    if not dataset.enemy_visibility_mask:
        return

    if int(
        getattr(dataset, "explicit_visibility_mask_version", 0)
    ) < 1:
        raise RuntimeError(
            "Corrected visibility dataset was not installed."
        )

    segment_count = len(dataset.index)
    item_count = min(segment_count, int(maximum_items))
    if item_count <= 0:
        raise RuntimeError(
            "Visibility preflight received an empty dataset."
        )

    indices = torch.linspace(
        0,
        segment_count - 1,
        steps=item_count,
        dtype=torch.float64,
    ).round().to(torch.long)

    hidden = 0.0
    target = 0.0
    for raw_index in indices.tolist():
        episode_index, segment_start = dataset.index[int(raw_index)]
        sample = dataset._entity_markov_rollout_item(
            episode_index,
            segment_start,
        )
        observed_mask = sample["observation_mask_seq"].float()
        target_mask = sample["target_entity_mask_seq"].float()
        hidden += float(
            (target_mask * (1.0 - observed_mask)).sum()
        )
        target += float(target_mask.sum())

    if hidden <= 0.0:
        raise RuntimeError(
            "Visibility preflight found zero hidden target entities across "
            f"{item_count} deterministic segments. Refusing to evaluate."
        )

    print(
        "eval_visibility_preflight_passed "
        f"items={item_count} "
        f"hidden_fraction={hidden / max(target, 1.0):.6f}",
        flush=True,
    )



def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def get_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return checkpoint.get("resolved_config", checkpoint.get("config", {}))


def threshold_key(threshold: float) -> str:
    return f"{threshold:g}".replace(".", "p")


def is_r2_checkpoint_config(cfg: dict[str, Any]) -> bool:
    return bool(
        cfg.get("r2_objective_version", 0)
        or cfg.get("training_regime")
        == "markov_rollout_rnn_seqmem_r2offline"
    )


def normalize_entity_latent(
    latent: torch.Tensor,
    entity_mask: torch.Tensor,
    *,
    enabled: bool,
) -> torch.Tensor:
    """
    Match Exp14/15 training exactly.

    R2-offline checkpoints use non-affine per-entity LayerNorm before:
      - recurrent memory conditioning/update,
      - rollout prediction losses,
      - decoding and presence heads.

    Older checkpoints bypass this function unchanged.
    """
    if enabled:
        latent = torch.nn.functional.layer_norm(
            latent.float(),
            (latent.shape[-1],),
            weight=None,
            bias=None,
            eps=1e-5,
        ).to(dtype=latent.dtype)
    return latent * entity_mask.unsqueeze(-1)



def _episode_indices(batch: dict[str, torch.Tensor]) -> list[int]:
    if "episode_index" not in batch:
        raise KeyError(
            "Evaluation batch has no episode_index. The corrected decoded "
            "metrics require per-episode ally/enemy feature metadata."
        )
    return [
        int(value)
        for value in batch["episode_index"].detach().cpu().reshape(-1).tolist()
    ]


def _base_feature_masks(
    dataset: VisibilityMarkovRolloutSMACJEPADataset,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """
    Construct [B, E, D] masks from each sampled episode's true metadata.

    Entity token layout in SMACJEPADataset:
      - dynamic features start at dimension 0;
      - ally slots use ally_state_feat_size dimensions;
      - enemy slots use enemy_state_feat_size dimensions;
      - entity-static features start at dataset.metadata.dynamic_token_dim;
      - remaining token dimensions are padding and must not be scored.
    """
    episode_indices = _episode_indices(batch)
    batch_size = len(episode_indices)
    max_agents = int(dataset.metadata.max_agents)
    max_enemies = int(dataset.metadata.max_enemies)
    entities = max_agents + max_enemies
    token_dim = int(dataset.metadata.token_dim)
    static_offset = int(dataset.metadata.dynamic_token_dim)

    dynamic = torch.zeros(
        batch_size,
        entities,
        token_dim,
        device=device,
        dtype=dtype,
    )
    static = torch.zeros_like(dynamic)
    ally_dynamic = torch.zeros_like(dynamic)
    enemy_dynamic = torch.zeros_like(dynamic)

    for batch_idx, episode_idx in enumerate(episode_indices):
        meta = dataset.episodes[episode_idx]["metadata"]

        ally_count = min(int(meta.n_agents), max_agents)
        enemy_count = min(int(meta.n_enemies), max_enemies)
        ally_dim = min(int(meta.ally_state_feat_size), token_dim)
        enemy_dim = min(int(meta.enemy_state_feat_size), token_dim)
        static_dim = min(
            int(meta.entity_static_feat_size),
            max(token_dim - static_offset, 0),
        )

        if ally_count > 0 and ally_dim > 0:
            dynamic[batch_idx, :ally_count, :ally_dim] = 1.0
            ally_dynamic[batch_idx, :ally_count, :ally_dim] = 1.0

        if enemy_count > 0 and enemy_dim > 0:
            enemy_start = max_agents
            enemy_stop = enemy_start + enemy_count
            dynamic[
                batch_idx,
                enemy_start:enemy_stop,
                :enemy_dim,
            ] = 1.0
            enemy_dynamic[
                batch_idx,
                enemy_start:enemy_stop,
                :enemy_dim,
            ] = 1.0

        if static_dim > 0:
            static_start = static_offset
            static_stop = static_start + static_dim
            if ally_count > 0:
                static[
                    batch_idx,
                    :ally_count,
                    static_start:static_stop,
                ] = 1.0
            if enemy_count > 0:
                enemy_start = max_agents
                enemy_stop = enemy_start + enemy_count
                static[
                    batch_idx,
                    enemy_start:enemy_stop,
                    static_start:static_stop,
                ] = 1.0

    return {
        "dynamic": dynamic,
        "static": static,
        "meaningful": (dynamic + static).clamp_max(1.0),
        "ally_dynamic": ally_dynamic,
        "enemy_dynamic": enemy_dynamic,
    }


def build_rollout_feature_masks(
    dataset: VisibilityMarkovRolloutSMACJEPADataset,
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Return exact [B, P, H, E, D] masks for the current rollout targets.
    """
    target = outputs["target_entity"]
    base = _base_feature_masks(
        dataset,
        batch,
        device=target.device,
        dtype=target.dtype,
    )

    entity_valid = (
        outputs["target_entity_mask"]
        * outputs["valid_mask"].unsqueeze(-1)
    )
    visible_entity = entity_valid * outputs["observed_entity_mask"]
    hidden_entity = entity_valid * (
        1.0 - outputs["observed_entity_mask"]
    )

    def expand(mask: torch.Tensor) -> torch.Tensor:
        return (
            mask[:, None, None]
            .expand(
                target.shape[0],
                target.shape[1],
                target.shape[2],
                target.shape[3],
                target.shape[4],
            )
        )

    masks = {
        key: expand(value) * entity_valid.unsqueeze(-1)
        for key, value in base.items()
    }
    masks["visible_dynamic"] = (
        expand(base["dynamic"]) * visible_entity.unsqueeze(-1)
    )
    masks["hidden_dynamic"] = (
        expand(base["dynamic"]) * hidden_entity.unsqueeze(-1)
    )
    masks["entity_valid"] = entity_valid
    return masks


def build_sequence_feature_masks(
    dataset: VisibilityMarkovRolloutSMACJEPADataset,
    batch: dict[str, torch.Tensor],
    entity_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Return [B, T, E, D] meaningful masks for independent-probe training.
    """
    base = _base_feature_masks(
        dataset,
        batch,
        device=entity_mask.device,
        dtype=entity_mask.dtype,
    )

    def expand(mask: torch.Tensor) -> torch.Tensor:
        return mask[:, None].expand(
            entity_mask.shape[0],
            entity_mask.shape[1],
            entity_mask.shape[2],
            mask.shape[-1],
        )

    return {
        key: expand(value) * entity_mask.unsqueeze(-1)
        for key, value in base.items()
    }


def exact_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    feature_mask: torch.Tensor,
) -> torch.Tensor:
    valid = feature_mask.to(dtype=pred.dtype)
    return (
        (pred - target).square() * valid
    ).sum() / valid.sum().clamp_min(1.0)


def _add_scalar(
    accumulator: dict[str, float],
    key: str,
    value: torch.Tensor | float,
) -> None:
    if isinstance(value, torch.Tensor):
        number = float(value.detach().double().cpu())
    else:
        number = float(value)
    accumulator[key] = accumulator.get(key, 0.0) + number


def add_exact_regression_statistics(
    accumulator: dict[str, float],
    prefixes: set[str],
    prefix: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    feature_mask: torch.Tensor,
    thresholds: list[float],
) -> None:
    """
    Accumulate additive numerators and denominators.

    Unlike the legacy evaluator, this computes the final proportion as:
        total correct meaningful coordinates / total meaningful coordinates
    rather than averaging per-batch proportions by window count.
    """
    mask = feature_mask.bool()
    count = mask.sum()
    if int(count.detach().cpu()) == 0:
        return

    error = (pred - target).float()
    absolute_error = error.abs()
    target_float = target.float()

    prefixes.add(prefix)
    _add_scalar(accumulator, f"{prefix}__count", count)
    _add_scalar(
        accumulator,
        f"{prefix}__abs_sum",
        absolute_error[mask].double().sum(),
    )
    _add_scalar(
        accumulator,
        f"{prefix}__sq_sum",
        error[mask].double().square().sum(),
    )
    _add_scalar(
        accumulator,
        f"{prefix}__target_sum",
        target_float[mask].double().sum(),
    )
    _add_scalar(
        accumulator,
        f"{prefix}__target_sq_sum",
        target_float[mask].double().square().sum(),
    )

    feature_count_per_entity = mask.sum(dim=-1)
    valid_entity = feature_count_per_entity > 0
    _add_scalar(
        accumulator,
        f"{prefix}__entity_count",
        valid_entity.sum(),
    )

    for threshold in thresholds:
        key = threshold_key(threshold)
        within = absolute_error <= float(threshold)
        _add_scalar(
            accumulator,
            f"{prefix}__correct_{key}",
            (within & mask).sum(),
        )

        # An entity prediction is correct only when every meaningful feature
        # for that entity is within tolerance.
        entity_correct = (
            (within | ~mask).all(dim=-1) & valid_entity
        )
        _add_scalar(
            accumulator,
            f"{prefix}__entity_correct_{key}",
            entity_correct.sum(),
        )


def finalize_exact_regression_statistics(
    accumulator: dict[str, float],
    prefixes: set[str],
    thresholds: list[float],
) -> dict[str, float]:
    result: dict[str, float] = {}

    for prefix in sorted(prefixes):
        count = accumulator.get(f"{prefix}__count", 0.0)
        entity_count = accumulator.get(
            f"{prefix}__entity_count", 0.0
        )
        if count <= 0:
            continue

        mae_value = accumulator[f"{prefix}__abs_sum"] / count
        mse_value = accumulator[f"{prefix}__sq_sum"] / count
        rmse_value = mse_value ** 0.5

        target_mean = (
            accumulator[f"{prefix}__target_sum"] / count
        )
        target_variance = max(
            accumulator[f"{prefix}__target_sq_sum"] / count
            - target_mean * target_mean,
            0.0,
        )
        target_std = target_variance ** 0.5

        result[f"{prefix}_count"] = count
        result[f"{prefix}_mae"] = mae_value
        result[f"{prefix}_mse"] = mse_value
        result[f"{prefix}_rmse"] = rmse_value
        result[f"{prefix}_target_std"] = target_std
        result[f"{prefix}_target_std_normalized_rmse"] = (
            rmse_value / max(target_std, 1e-12)
        )

        for threshold in thresholds:
            key = threshold_key(threshold)
            result[f"{prefix}_coord_acc_{key}"] = (
                accumulator.get(
                    f"{prefix}__correct_{key}", 0.0
                )
                / count
            )
            result[f"{prefix}_entity_acc_{key}"] = (
                accumulator.get(
                    f"{prefix}__entity_correct_{key}", 0.0
                )
                / max(entity_count, 1.0)
            )

    return result


def add_prediction_family_statistics(
    accumulator: dict[str, float],
    prefixes: set[str],
    family: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    masks: dict[str, torch.Tensor],
    thresholds: list[float],
    *,
    include_detailed_groups: bool = True,
) -> None:
    groups = ["dynamic"]
    if include_detailed_groups:
        groups.extend(
            [
                "meaningful",
                "static",
                "ally_dynamic",
                "enemy_dynamic",
                "visible_dynamic",
                "hidden_dynamic",
            ]
        )

    for group in groups:
        add_exact_regression_statistics(
            accumulator,
            prefixes,
            f"corrected_{family}_{group}",
            pred,
            target,
            masks[group],
            thresholds,
        )

    horizon = pred.shape[2]
    for step in range(horizon):
        sl = slice(step, step + 1)
        add_exact_regression_statistics(
            accumulator,
            prefixes,
            f"corrected_{family}_dynamic_h{step + 1}",
            pred[:, :, sl],
            target[:, :, sl],
            masks["dynamic"][:, :, sl],
            thresholds,
        )



def build_event_conditioned_masks(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    base_masks: dict[str, torch.Tensor],
    *,
    change_threshold: float,
    event_threshold: float,
    attack_action_min: int,
) -> dict[str, torch.Tensor]:
    """
    Build evaluation-only masks that expose whether the model predicts actual
    changes rather than benefiting from unchanged coordinates.

    Shapes:
        target/persistence: [B, P, H, E, F]
        dynamic mask:       [B, P, H, E, F]
    """
    target = outputs["target_entity"]
    persistence = persistence_prediction(batch, outputs)
    dynamic_mask = base_masks["dynamic"].bool()

    absolute_delta = (target - persistence).abs()

    changed_dynamic = dynamic_mask & (
        absolute_delta > float(change_threshold)
    )
    unchanged_dynamic = dynamic_mask & ~changed_dynamic

    event_coordinate = dynamic_mask & (
        absolute_delta > float(event_threshold)
    )
    event_instance = event_coordinate.any(dim=(-1, -2), keepdim=True)
    event_window_dynamic = dynamic_mask & event_instance
    no_event_window_dynamic = dynamic_mask & ~event_instance

    # Action sequence contains the rollout-window actions followed by the
    # future rollout actions. For each prediction starting at p and horizon h,
    # the relevant action is p+h.
    action_seq = batch["action_seq"]
    action_mask_seq = batch["action_mask_seq"].bool()
    bsz, rollout_window, horizon = target.shape[:3]

    action_event = torch.zeros(
        (bsz, rollout_window, horizon),
        dtype=torch.bool,
        device=target.device,
    )
    for start in range(rollout_window):
        for step in range(horizon):
            action_index = start + step
            if action_index >= action_seq.shape[1]:
                continue
            actions = action_seq[:, action_index]
            valid_actions = action_mask_seq[:, action_index]
            attack_like = (
                (actions >= int(attack_action_min))
                & valid_actions
            ).any(dim=-1)
            action_event[:, start, step] = attack_like

    action_event_dynamic = dynamic_mask & action_event[
        :, :, :, None, None
    ]
    no_action_event_dynamic = dynamic_mask & ~action_event[
        :, :, :, None, None
    ]

    return {
        "changed_dynamic": changed_dynamic,
        "unchanged_dynamic": unchanged_dynamic,
        "event_window_dynamic": event_window_dynamic,
        "no_event_window_dynamic": no_event_window_dynamic,
        "action_event_dynamic": action_event_dynamic,
        "no_action_event_dynamic": no_action_event_dynamic,
    }


def add_event_conditioned_statistics(
    accumulator: dict[str, float],
    prefixes: set[str],
    family: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    event_masks: dict[str, torch.Tensor],
    thresholds: list[float],
) -> None:
    for group, mask in event_masks.items():
        add_exact_regression_statistics(
            accumulator,
            prefixes,
            f"event_{family}_{group}",
            pred,
            target,
            mask,
            thresholds,
        )

    horizon = pred.shape[2]
    for step in range(horizon):
        sl = slice(step, step + 1)
        for group, mask in event_masks.items():
            add_exact_regression_statistics(
                accumulator,
                prefixes,
                f"event_{family}_{group}_h{step + 1}",
                pred[:, :, sl],
                target[:, :, sl],
                mask[:, :, sl],
                thresholds,
            )



def decode_with_probe(
    outputs: dict[str, torch.Tensor],
    probe_decoder: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_latent = outputs["pred_latent"]
    target_latent = outputs["target_latent"]
    target_entity = outputs["target_entity"]

    shape = pred_latent.shape
    flat_count = shape[0] * shape[1] * shape[2]
    entities = shape[3]
    latent_dim = shape[4]

    rollout = probe_decoder(
        pred_latent.reshape(flat_count, entities, latent_dim)
    ).reshape_as(target_entity)
    reconstruction = probe_decoder(
        target_latent.reshape(flat_count, entities, latent_dim)
    ).reshape_as(target_entity)
    return rollout, reconstruction


def persistence_prediction(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Predict every future state as the full current state at rollout start.
    """
    full_sequence = batch.get(
        "target_entity_seq", batch["entity_seq"]
    )
    rollout_window = outputs["target_entity"].shape[1]
    horizon = outputs["target_entity"].shape[2]
    current = full_sequence[:, :rollout_window]
    return current.unsqueeze(2).expand(
        -1, -1, horizon, -1, -1
    )


def build_dataset(
    manifest: str,
    split: str,
    resolved_config: dict[str, Any],
    window_mode: str,
    samples_per_epoch: int | None,
    enemy_visibility_mask: bool | None,
    enemy_sight_range: float | None,
    eval_rollout_horizon: int | None,
) -> VisibilityMarkovRolloutSMACJEPADataset:
    data_paths = [str(path) for path in load_manifest(manifest, split)]
    cap_paths = load_manifest_all(manifest)
    cap_dataset = SMACJEPADataset(cap_paths, context_len=1, mode="entity")
    cap_metadata = cap_dataset.metadata

    resolved_enemy_visibility_mask = (
        bool(resolved_config.get("enemy_visibility_mask", True))
        if enemy_visibility_mask is None
        else bool(enemy_visibility_mask)
    )
    resolved_enemy_sight_range = (
        float(resolved_config.get("enemy_sight_range", 9.0))
        if enemy_sight_range is None
        else float(enemy_sight_range)
    )

    return VisibilityMarkovRolloutSMACJEPADataset(
        data_paths,
        rollout_window=int(resolved_config.get("rollout_window", 20)),
        rollout_horizon=int(
            eval_rollout_horizon
            if eval_rollout_horizon is not None
            else resolved_config.get("rollout_horizon", 5)
        ),
        mode="entity",
        window_mode=window_mode,
        samples_per_epoch=samples_per_epoch,
        seed=int(resolved_config.get("seed", 1)),
        max_agents=cap_metadata.max_agents,
        max_enemies=cap_metadata.max_enemies,
        max_actions=cap_metadata.max_actions,
        token_dim=cap_metadata.token_dim,
        dynamic_token_dim=cap_metadata.dynamic_token_dim,
        static_dim=cap_metadata.static_dim,
        entity_static_feat_size=cap_metadata.entity_static_feat_size,
        enemy_visibility_mask=resolved_enemy_visibility_mask,
        enemy_sight_range=resolved_enemy_sight_range,
    )


def build_model(checkpoint: dict[str, Any], dataset: VisibilityMarkovRolloutSMACJEPADataset, device: torch.device) -> SMACJEPA:
    cfg = get_config(checkpoint)
    metadata = checkpoint["metadata"]

    model = SMACJEPA(
        state_dim=metadata.get("state_dim", dataset.metadata.state_dim),
        n_agents=metadata.get("n_agents", dataset.metadata.n_agents),
        n_actions=metadata.get("n_actions", dataset.metadata.n_actions),
        latent_dim=int(cfg["latent_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        action_dim=int(cfg["action_dim"]),
        num_heads=int(cfg["num_heads"]),
        mode=metadata.get("mode", dataset.metadata.mode),
        max_agents=metadata.get("max_agents", dataset.metadata.max_agents),
        max_enemies=metadata.get("max_enemies", dataset.metadata.max_enemies),
        max_actions=metadata.get("max_actions", dataset.metadata.max_actions),
        token_dim=metadata.get("token_dim", dataset.metadata.token_dim),
        static_dim=metadata.get("static_dim", dataset.metadata.static_dim),
        decoder_weight=float(cfg.get("decoder_weight", 1.0)),
        encoder_layers=int(cfg["encoder_layers"]),
        action_layers=int(cfg["action_layers"]),
        predictor_layers=int(cfg["predictor_layers"]),
        max_context_len=int(cfg.get("max_context_len", 32)),
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model



def _module_linear_signature(module: nn.Module) -> list[tuple[int, int]]:
    return [
        (int(layer.in_features), int(layer.out_features))
        for layer in module.modules()
        if isinstance(layer, nn.Linear)
    ]


def _reset_module_parameters(module: nn.Module) -> None:
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def clone_fresh_native_decoder(
    model: SMACJEPA,
    *,
    latent_dim: int,
    target_dim: int,
    device: torch.device,
) -> tuple[nn.Module, str, list[tuple[int, int]]]:
    """
    Find the actual MLP used by model.decode_entities(), deep-copy it, and reset it.

    Selection is architecture based:
      first Linear input == latent_dim
      last Linear output == entity feature dimension

    The copied module therefore has the same layers, widths, activations, and output
    dimension as the decoder originally trained inside that checkpoint.
    """
    candidates: list[tuple[str, nn.Module, list[tuple[int, int]]]] = []
    decode_names = set()
    try:
        decode_names = set(model.decode_entities.__func__.__code__.co_names)
    except Exception:
        pass

    for name, module in model.named_modules():
        if not name:
            continue
        signature = _module_linear_signature(module)
        if not signature:
            continue
        if signature[0][0] != latent_dim or signature[-1][1] != target_dim:
            continue

        # Confirm that the candidate accepts entity-wise latent tensors directly.
        try:
            dummy = torch.zeros(2, 3, latent_dim, device=device)
            output = module(dummy)
            if not isinstance(output, torch.Tensor):
                continue
            if output.shape[-1] != target_dim:
                continue
        except Exception:
            continue
        candidates.append((name, module, signature))

    if not candidates:
        module_summary = [
            (name, _module_linear_signature(module))
            for name, module in model.named_modules()
            if name and _module_linear_signature(module)
        ]
        raise RuntimeError(
            "Could not identify the native entity decoder MLP. "
            f"Expected latent_dim={latent_dim}, target_dim={target_dim}. "
            f"Linear-module candidates were: {module_summary}"
        )

    # Prefer a top-level attribute explicitly referenced by decode_entities().
    preferred = [
        item for item in candidates
        if item[0].split(".")[0] in decode_names
    ]
    pool = preferred if preferred else candidates

    # Prefer the shallow wrapper when decode_entities references it; otherwise the
    # deepest matching module is usually the concrete Sequential MLP.
    if preferred:
        name, module, signature = min(pool, key=lambda item: item[0].count("."))
    else:
        name, module, signature = max(pool, key=lambda item: item[0].count("."))

    probe = copy.deepcopy(module).to(device)
    _reset_module_parameters(probe)
    probe.train()
    return probe, name, signature


def _masked_decoder_mse(
    decoded: torch.Tensor,
    target: torch.Tensor,
    feature_mask: torch.Tensor,
) -> torch.Tensor:
    return exact_masked_mse(decoded, target, feature_mask)


def probe_checkpoint_path(
    checkpoint_path: Path,
    args: argparse.Namespace,
    out_dir: Path,
) -> Path:
    root = Path(args.probe_dir) if args.probe_dir else out_dir / "probe_decoders"
    root.mkdir(parents=True, exist_ok=True)
    return root / (
        f"{checkpoint_path.parent.name}_{checkpoint_path.stem}"
        "_meaningful_features_v2_probe_decoder.pt"
    )


def train_or_load_probe_decoder(
    *,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    cfg: dict[str, Any],
    model: SMACJEPA,
    eval_dataset: VisibilityMarkovRolloutSMACJEPADataset,
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[nn.Module | None, dict[str, Any]]:
    if not args.probe_decoder:
        return None, {"probe_decoder_enabled": False}

    sample = eval_dataset[0]
    target_sample = sample.get("target_entity_seq", sample["entity_seq"])
    target_dim = int(target_sample.shape[-1])
    latent_dim = int(cfg["latent_dim"])
    r2_latent_normalize = bool(cfg.get("r2_latent_normalize", False))

    random.seed(args.probe_seed)
    torch.manual_seed(args.probe_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.probe_seed)

    probe, module_name, signature = clone_fresh_native_decoder(
        model,
        latent_dim=latent_dim,
        target_dim=target_dim,
        device=device,
    )

    save_path = probe_checkpoint_path(checkpoint_path, args, out_dir)
    info: dict[str, Any] = {
        "probe_decoder_enabled": True,
        "probe_decoder_path": str(save_path),
        "probe_decoder_module_name": module_name,
        "probe_decoder_linear_signature": signature,
        "probe_decoder_target_dim": target_dim,
        "probe_decoder_latent_dim": latent_dim,
        "probe_decoder_r2_latent_normalize": r2_latent_normalize,
        "probe_decoder_loaded_existing": False,
        "probe_decoder_train_updates": 0,
    }

    if save_path.exists() and not args.force_retrain_probe:
        saved = torch.load(save_path, map_location=device)
        probe.load_state_dict(saved["probe_decoder_state"])
        probe.eval()
        info.update(saved.get("probe_info", {}))
        info["probe_decoder_loaded_existing"] = True
        print(f"loaded probe decoder {save_path}", flush=True)
        return probe, info

    train_dataset = build_dataset(
        args.manifest,
        args.probe_train_split,
        cfg,
        "random",
        args.probe_samples_per_epoch,
        args.enemy_visibility_mask,
        args.enemy_sight_range,
        1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # The checkpoint remains fully frozen. Only the fresh decoder copy is optimized.
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in probe.parameters():
        parameter.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(args.probe_lr),
        weight_decay=float(args.probe_weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    epoch_losses: list[float] = []
    updates = 0

    print(
        f"training independent probe decoder module={module_name} "
        f"signature={signature} epochs={args.probe_epochs} "
        f"max_batches_per_epoch={args.probe_max_batches_per_epoch}",
        flush=True,
    )

    for epoch in range(1, args.probe_epochs + 1):
        probe.train()
        total_loss = 0.0
        total_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if (
                args.probe_max_batches_per_epoch is not None
                and batch_idx >= args.probe_max_batches_per_epoch
            ):
                break

            batch = to_device(batch, device)
            full_entity_seq = batch.get("target_entity_seq", batch["entity_seq"])
            full_entity_mask = batch.get(
                "target_entity_mask_seq", batch["entity_mask_seq"]
            )
            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.cuda.amp.autocast(enabled=amp_enabled)
                if device.type == "cuda"
                else nullcontext()
            )
            with torch.no_grad(), autocast_context:
                true_latent = model.encoder(
                    full_entity_seq, full_entity_mask
                )
                true_latent = normalize_entity_latent(
                    true_latent,
                    full_entity_mask,
                    enabled=r2_latent_normalize,
                )

            with autocast_context:
                reconstructed = probe(true_latent.detach())
                train_feature_masks = build_sequence_feature_masks(
                    train_dataset,
                    batch,
                    full_entity_mask,
                )
                loss = _masked_decoder_mse(
                    reconstructed,
                    full_entity_seq,
                    train_feature_masks["meaningful"],
                )

            scaler.scale(loss).backward()
            if args.probe_grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    probe.parameters(), float(args.probe_grad_clip)
                )
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().cpu())
            total_batches += 1
            updates += 1

        epoch_loss = total_loss / max(total_batches, 1)
        epoch_losses.append(epoch_loss)
        print(
            f"probe epoch={epoch}/{args.probe_epochs} "
            f"updates={updates} train_mse={epoch_loss:.6f}",
            flush=True,
        )

    probe.eval()
    info.update(
        {
            "probe_decoder_train_split": args.probe_train_split,
            "probe_decoder_epochs": int(args.probe_epochs),
            "probe_decoder_lr": float(args.probe_lr),
            "probe_decoder_weight_decay": float(args.probe_weight_decay),
            "probe_decoder_seed": int(args.probe_seed),
            "probe_decoder_train_updates": updates,
            "probe_decoder_final_train_mse": (
                float(epoch_losses[-1]) if epoch_losses else None
            ),
            "probe_decoder_epoch_train_mse": epoch_losses,
        }
    )
    torch.save(
        {
            "probe_decoder_state": probe.state_dict(),
            "probe_info": info,
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        },
        save_path,
    )
    print(f"saved probe decoder {save_path}", flush=True)
    return probe, info


def compute_probe_decoder_metrics(
    outputs: dict[str, torch.Tensor],
    probe_decoder: nn.Module,
    thresholds: list[float],
) -> dict[str, torch.Tensor]:
    pred_latent = outputs["pred_latent"]
    target_latent = outputs["target_latent"]
    target_entity = outputs["target_entity"]
    target_entity_mask = outputs["target_entity_mask"]
    observed_entity_mask = outputs["observed_entity_mask"]
    valid_mask = outputs["valid_mask"]

    shape = pred_latent.shape
    flat_count = shape[0] * shape[1] * shape[2]
    entities = shape[3]
    latent_dim = shape[4]

    probe_rollout = probe_decoder(
        pred_latent.reshape(flat_count, entities, latent_dim)
    ).reshape_as(target_entity)
    probe_reconstruction = probe_decoder(
        target_latent.reshape(flat_count, entities, latent_dim)
    ).reshape_as(target_entity)

    entity_valid = target_entity_mask * valid_mask.unsqueeze(-1)
    feature_mask = entity_valid.unsqueeze(-1)
    visible_valid = entity_valid * observed_entity_mask
    hidden_valid = entity_valid * (1.0 - observed_entity_mask)

    metrics: dict[str, torch.Tensor] = {}

    for prefix, decoded in (
        ("probe_rollout", probe_rollout),
        ("probe_reconstruction", probe_reconstruction),
    ):
        metrics[f"{prefix}_mse"] = mse(decoded, target_entity, feature_mask)
        metrics[f"{prefix}_mae"] = mae(decoded, target_entity, feature_mask)
        metrics[f"{prefix}_mae_visible"] = mae(
            decoded, target_entity, visible_valid.unsqueeze(-1)
        )
        metrics[f"{prefix}_mae_hidden"] = mae(
            decoded, target_entity, hidden_valid.unsqueeze(-1)
        )
        for threshold in thresholds:
            key = threshold_key(threshold)
            metrics[f"{prefix}_acc_{key}"] = threshold_acc(
                decoded, target_entity, feature_mask, threshold
            )
            metrics[f"{prefix}_acc_{key}_visible"] = threshold_acc(
                decoded, target_entity, visible_valid.unsqueeze(-1), threshold
            )
            metrics[f"{prefix}_acc_{key}_hidden"] = threshold_acc(
                decoded, target_entity, hidden_valid.unsqueeze(-1), threshold
            )

    metrics["probe_rollout_gap_mse"] = (
        metrics["probe_rollout_mse"] - metrics["probe_reconstruction_mse"]
    )
    metrics["probe_rollout_gap_mae"] = (
        metrics["probe_rollout_mae"] - metrics["probe_reconstruction_mae"]
    )
    metrics["probe_rollout_to_reconstruction_mse_ratio"] = (
        metrics["probe_rollout_mse"]
        / metrics["probe_reconstruction_mse"].clamp_min(1e-12)
    )

    for threshold in thresholds:
        key = threshold_key(threshold)
        metrics[f"probe_accuracy_drop_{key}"] = (
            metrics[f"probe_reconstruction_acc_{key}"]
            - metrics[f"probe_rollout_acc_{key}"]
        )

    horizon = pred_latent.shape[2]
    for step in range(horizon):
        name = f"h{step + 1}"
        sl = slice(step, step + 1)
        step_mask = feature_mask[:, :, sl]
        step_visible = visible_valid[:, :, sl].unsqueeze(-1)
        step_hidden = hidden_valid[:, :, sl].unsqueeze(-1)

        for prefix, decoded in (
            ("probe_rollout", probe_rollout),
            ("probe_reconstruction", probe_reconstruction),
        ):
            metrics[f"{prefix}_mse_{name}"] = mse(
                decoded[:, :, sl], target_entity[:, :, sl], step_mask
            )
            metrics[f"{prefix}_mae_{name}"] = mae(
                decoded[:, :, sl], target_entity[:, :, sl], step_mask
            )
            metrics[f"{prefix}_mae_visible_{name}"] = mae(
                decoded[:, :, sl], target_entity[:, :, sl], step_visible
            )
            metrics[f"{prefix}_mae_hidden_{name}"] = mae(
                decoded[:, :, sl], target_entity[:, :, sl], step_hidden
            )
            for threshold in thresholds:
                key = threshold_key(threshold)
                metrics[f"{prefix}_acc_{key}_{name}"] = threshold_acc(
                    decoded[:, :, sl],
                    target_entity[:, :, sl],
                    step_mask,
                    threshold,
                )

        metrics[f"probe_rollout_gap_mse_{name}"] = (
            metrics[f"probe_rollout_mse_{name}"]
            - metrics[f"probe_reconstruction_mse_{name}"]
        )

    if horizon >= 2:
        first = metrics["probe_rollout_mse_h1"].clamp_min(1e-12)
        last = metrics[f"probe_rollout_mse_h{horizon}"]
        metrics["probe_rollout_error_growth_ratio"] = last / first
        metrics["probe_rollout_error_growth_absolute"] = last - first

    return metrics


def build_memory_module(
    checkpoint: dict[str, Any],
    dataset: VisibilityMarkovRolloutSMACJEPADataset,
    device: torch.device,
) -> torch.nn.Module:
    cfg = get_config(checkpoint)
    if int(cfg.get("r2_blocker_fix_version", 0)) < 1:
        raise RuntimeError(
            "This evaluator supports blocker-fixed Exp16-Exp19 checkpoints only."
        )

    latent_dim = int(cfg["latent_dim"])
    memory_dim = int(cfg.get("rollout_memory_dim", 128))
    hidden_dim = cfg.get("rollout_memory_hidden_dim", None)
    residual = not bool(cfg.get("rollout_memory_no_residual", False))
    action_conditioned = bool(cfg.get("action_conditioned_memory", False))

    if action_conditioned:
        metadata = checkpoint.get("metadata", {})
        n_actions = int(
            cfg.get(
                "n_actions",
                metadata.get("n_actions", dataset.metadata.n_actions),
            )
        )
        max_agents = int(
            cfg.get(
                "max_agents",
                metadata.get("max_agents", dataset.metadata.max_agents),
            )
        )
        memory_module = ActionConditionedEntityRolloutGRUMemory(
            latent_dim=latent_dim,
            memory_dim=memory_dim,
            n_actions=n_actions,
            max_agents=max_agents,
            hidden_dim=hidden_dim,
            residual=residual,
        ).to(device)
    else:
        memory_module = BeliefEntityRolloutGRUMemory(
            latent_dim=latent_dim,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            residual=residual,
        ).to(device)

    if "memory_module_state" not in checkpoint:
        raise RuntimeError("Checkpoint does not contain memory_module_state")
    memory_module.load_state_dict(checkpoint["memory_module_state"])
    memory_module.eval()
    return memory_module


@torch.no_grad()
def rollout_outputs(
    model: SMACJEPA,
    memory_module: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    rollout_window: int,
    rollout_horizon: int,
    target_mode: str,
    *,
    action_mode: str = "correct",
    zero_history_memory: bool = False,
    mask_mode: str = "predicted",
    presence_threshold: float = 0.5,
    r2_latent_normalize: bool = False,
) -> dict[str, torch.Tensor]:
    """Open-loop rollout matching blocker-fixed training chronology exactly."""
    del presence_threshold

    entity_seq = batch["entity_seq"]
    if "observation_mask_seq" not in batch:
        raise RuntimeError(
            "Dataset is missing observation_mask_seq. Install the corrected "
            "visibility dataset before evaluation."
        )
    observation_mask_seq = batch["observation_mask_seq"]
    action_seq = batch["action_seq"]
    action_mask_seq = batch["action_mask_seq"]
    state_mask = batch["state_mask"]
    static_condition = batch.get("static_condition")
    slot_mask_seq = batch["entity_slot_mask_seq"]

    if target_mode == "observed":
        target_entity_seq = entity_seq
        target_entity_mask_seq = observation_mask_seq
    elif target_mode == "full":
        target_entity_seq = batch.get("target_entity_seq", entity_seq)
        target_entity_mask_seq = batch.get(
            "target_entity_mask_seq", observation_mask_seq
        )
    else:
        raise ValueError(f"Unknown target_mode={target_mode}")

    bsz = entity_seq.shape[0]
    p = int(rollout_window)
    h = int(rollout_horizon)

    input_latents = normalize_entity_latent(
        model.encoder(entity_seq, observation_mask_seq),
        observation_mask_seq,
        enabled=r2_latent_normalize,
    )
    target_latents = normalize_entity_latent(
        model.encoder(target_entity_seq, target_entity_mask_seq),
        target_entity_mask_seq,
        enabled=r2_latent_normalize,
    )
    _, _, entities, latent_dim = input_latents.shape

    main_memory = memory_module.initial_memory(
        bsz, entities, device=entity_seq.device, dtype=input_latents.dtype
    )
    main_presence = slot_mask_seq[:, 0].to(input_latents.dtype)
    static_flat = static_condition if static_condition is not None else None

    pred_by_start = []
    target_latent_by_start = []
    target_entity_by_start = []
    target_mask_by_start = []
    observed_mask_by_start = []
    slot_mask_by_start = []
    valid_by_start = []

    for start_idx in range(p):
        z_start = input_latents[:, start_idx]
        observed_start = observation_mask_seq[:, start_idx]
        slot_start = slot_mask_seq[:, start_idx]
        main_presence = merge_observed_presence(
            main_presence, observed_start, slot_start
        )

        rollout_memory = (
            memory_module.initial_memory(
                bsz, entities, device=entity_seq.device, dtype=input_latents.dtype
            )
            if zero_history_memory
            else main_memory
        )
        z = z_start
        current_presence = main_presence
        current_slot_mask = slot_start
        current_update_gate = observed_start

        pred_steps = []
        target_latent_steps = []
        target_entity_steps = []
        target_mask_steps = []
        observed_mask_steps = []
        slot_mask_steps = []
        valid_steps = []
        first_future_presence = None

        for step in range(h):
            action_idx = start_idx + step
            target_idx = start_idx + step + 1

            action_h = action_seq[:, action_idx : action_idx + 1]
            action_mask_h = action_mask_seq[:, action_idx : action_idx + 1]
            if action_mode == "shuffled":
                if bsz > 1:
                    action_h = torch.roll(action_h, shifts=1, dims=0)
                    action_mask_h = torch.roll(action_mask_h, shifts=1, dims=0)
                else:
                    alt_idx = min(action_idx + 1, action_seq.shape[1] - 1)
                    action_h = action_seq[:, alt_idx : alt_idx + 1]
                    action_mask_h = action_mask_seq[:, alt_idx : alt_idx + 1]
            elif action_mode != "correct":
                raise ValueError(f"Unknown action_mode={action_mode}")

            future_slot_mask = slot_mask_seq[:, target_idx]
            valid_h = state_mask[:, target_idx]
            timestep_mask = torch.ones(
                (bsz, 1), device=entity_seq.device, dtype=entity_seq.dtype
            )

            z_conditioned = memory_module.condition(
                z, rollout_memory, current_presence
            )
            pred_raw = model.predictor(
                z_conditioned.unsqueeze(1),
                action_h,
                action_mask_h,
                timestep_mask,
                current_slot_mask.unsqueeze(1),
                static_flat,
            )[:, 0]
            pred_raw = pred_raw * future_slot_mask.unsqueeze(-1)
            pred_h = normalize_entity_latent(
                pred_raw,
                future_slot_mask,
                enabled=r2_latent_normalize,
            )

            predicted_presence = (
                torch.sigmoid(model.predict_presence(pred_h).float())
                .to(dtype=pred_h.dtype)
                * future_slot_mask
            )
            target_mask_h = target_entity_mask_seq[:, target_idx]
            observed_mask_h = observation_mask_seq[:, target_idx]

            pred_steps.append(pred_h)
            target_latent_steps.append(target_latents[:, target_idx])
            target_entity_steps.append(target_entity_seq[:, target_idx])
            target_mask_steps.append(target_mask_h)
            observed_mask_steps.append(observed_mask_h)
            slot_mask_steps.append(future_slot_mask)
            valid_steps.append(valid_h)

            if getattr(memory_module, "uses_action", False):
                next_memory = memory_module.update(
                    z,
                    rollout_memory,
                    current_update_gate,
                    action=action_h[:, 0],
                    action_mask=action_mask_h[:, 0],
                )
            else:
                next_memory = memory_module.update(
                    z, rollout_memory, current_update_gate
                )

            if mask_mode == "predicted":
                next_presence = predicted_presence
            elif mask_mode == "oracle":
                next_presence = target_mask_h.to(pred_h.dtype)
            elif mask_mode == "carry":
                next_presence = current_presence
            else:
                raise ValueError(f"Unknown mask_mode={mask_mode}")

            if step == 0:
                first_future_presence = next_presence

            rollout_memory = next_memory
            z = pred_h
            current_presence = next_presence
            current_slot_mask = future_slot_mask
            current_update_gate = next_presence

        pred_by_start.append(torch.stack(pred_steps, dim=1))
        target_latent_by_start.append(torch.stack(target_latent_steps, dim=1))
        target_entity_by_start.append(torch.stack(target_entity_steps, dim=1))
        target_mask_by_start.append(torch.stack(target_mask_steps, dim=1))
        observed_mask_by_start.append(torch.stack(observed_mask_steps, dim=1))
        slot_mask_by_start.append(torch.stack(slot_mask_steps, dim=1))
        valid_by_start.append(torch.stack(valid_steps, dim=1))

        real_action = action_seq[:, start_idx]
        real_action_mask = action_mask_seq[:, start_idx]
        if getattr(memory_module, "uses_action", False):
            main_memory = memory_module.update(
                z_start,
                main_memory,
                observed_start,
                action=real_action,
                action_mask=real_action_mask,
            )
        else:
            main_memory = memory_module.update(
                z_start, main_memory, observed_start
            )
        if first_future_presence is not None:
            main_presence = first_future_presence

    pred_latent = torch.stack(pred_by_start, dim=1)
    target_latent = torch.stack(target_latent_by_start, dim=1)
    target_entity = torch.stack(target_entity_by_start, dim=1)
    target_entity_mask = torch.stack(target_mask_by_start, dim=1)
    observed_entity_mask = torch.stack(observed_mask_by_start, dim=1)
    entity_slot_mask = torch.stack(slot_mask_by_start, dim=1)
    valid_mask = torch.stack(valid_by_start, dim=1)

    decoded = model.decode_entities(
        pred_latent.reshape(bsz * p * h, entities, latent_dim)
    ).reshape(bsz, p, h, entities, -1)
    presence_logits = model.predict_presence(
        pred_latent.reshape(bsz * p * h, entities, latent_dim)
    ).reshape(bsz, p, h, entities)

    return {
        "pred_latent": pred_latent,
        "target_latent": target_latent,
        "decoded": decoded,
        "target_entity": target_entity,
        "presence_logits": presence_logits,
        "target_entity_mask": target_entity_mask,
        "observed_entity_mask": observed_entity_mask,
        "entity_slot_mask": entity_slot_mask,
        "valid_mask": valid_mask,
    }


def mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - target).pow(2) * mask).sum() / (mask.sum().clamp_min(1.0) * pred.shape[-1])


def mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - target).abs() * mask).sum() / (mask.sum().clamp_min(1.0) * pred.shape[-1])


def threshold_acc(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
    valid = mask.expand_as(pred).bool()
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return ((pred - target).abs() <= threshold)[valid].float().mean()


def presence_metrics(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, threshold: float) -> dict[str, torch.Tensor]:
    valid = mask.bool()
    if valid.sum() == 0:
        z = torch.tensor(0.0, device=logits.device)
        return {"presence_acc": z, "presence_precision": z, "presence_recall": z, "presence_f1": z}

    pred = (torch.sigmoid(logits) >= threshold) & valid
    tgt = (target >= 0.5) & valid
    tp = (pred & tgt).sum().float()
    tn = ((~pred) & (~tgt) & valid).sum().float()
    fp = (pred & (~tgt) & valid).sum().float()
    fn = ((~pred) & tgt).sum().float()

    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    acc = (tp + tn) / valid.sum().float().clamp_min(1.0)
    return {"presence_acc": acc, "presence_precision": precision, "presence_recall": recall, "presence_f1": f1}


def _subset_metrics(
    prefix: str,
    pred_latent: torch.Tensor,
    target_latent: torch.Tensor,
    decoded: torch.Tensor,
    target_entity: torch.Tensor,
    entity_mask: torch.Tensor,
    thresholds: list[float],
) -> dict[str, torch.Tensor]:
    feat_mask = entity_mask.unsqueeze(-1)
    result: dict[str, torch.Tensor] = {
        f"latent_mse_{prefix}": mse(pred_latent, target_latent, feat_mask),
        f"decoded_mse_{prefix}": mse(decoded, target_entity, feat_mask),
        f"decoded_mae_{prefix}": mae(decoded, target_entity, feat_mask),
        f"entity_fraction_{prefix}": entity_mask.float().mean(),
    }
    for threshold in thresholds:
        key = threshold_key(threshold)
        result[f"decoded_acc_{key}_{prefix}"] = threshold_acc(
            decoded, target_entity, feat_mask, threshold
        )
    return result


def compute_decoded_metrics(
    outputs: dict[str, torch.Tensor],
    thresholds: list[float],
    presence_threshold: float,
) -> dict[str, torch.Tensor]:
    pred_latent = outputs["pred_latent"]
    target_latent = outputs["target_latent"]
    decoded = outputs["decoded"]
    target_entity = outputs["target_entity"]
    presence_logits = outputs["presence_logits"]
    target_entity_mask = outputs["target_entity_mask"]
    observed_entity_mask = outputs["observed_entity_mask"]
    entity_slot_mask = outputs["entity_slot_mask"]
    valid_mask = outputs["valid_mask"]

    entity_valid = target_entity_mask * valid_mask.unsqueeze(-1)
    entity_valid_feat = entity_valid.unsqueeze(-1)
    slot_valid = entity_slot_mask * valid_mask.unsqueeze(-1)

    visible_valid = entity_valid * observed_entity_mask
    hidden_valid = entity_valid * (1.0 - observed_entity_mask)

    metrics: dict[str, torch.Tensor] = {}
    metrics["latent_mse"] = mse(
        pred_latent, target_latent, entity_valid_feat
    )

    latent_valid = entity_valid.bool()
    if latent_valid.sum() >= 2:
        pred_flat = pred_latent[latent_valid].float()
        target_flat = target_latent[latent_valid].float()
        pred_std = pred_flat.std(dim=0, unbiased=False)
        target_std = target_flat.std(dim=0, unbiased=False)
        metrics["pred_latent_std_mean"] = pred_std.mean()
        metrics["pred_latent_std_min"] = pred_std.min()
        metrics["target_latent_std_mean"] = target_std.mean()
        metrics["target_latent_std_min"] = target_std.min()
        metrics["pred_to_target_std_ratio"] = (
            pred_std.mean() / target_std.mean().clamp_min(1e-12)
        )
        metrics["variance_normalized_latent_mse"] = (
            metrics["latent_mse"]
            / target_std.square().mean().clamp_min(1e-12)
        )
    else:
        zero = pred_latent.sum() * 0.0
        metrics["pred_latent_std_mean"] = zero
        metrics["pred_latent_std_min"] = zero
        metrics["target_latent_std_mean"] = zero
        metrics["target_latent_std_min"] = zero
        metrics["pred_to_target_std_ratio"] = zero
        metrics["variance_normalized_latent_mse"] = zero

    metrics["decoded_mse"] = mse(decoded, target_entity, entity_valid_feat)
    metrics["decoded_mae"] = mae(decoded, target_entity, entity_valid_feat)

    for threshold in thresholds:
        key = threshold_key(threshold)
        metrics[f"decoded_acc_{key}"] = threshold_acc(
            decoded, target_entity, entity_valid_feat, threshold
        )

    metrics.update(
        presence_metrics(
            presence_logits, target_entity_mask, slot_valid, presence_threshold
        )
    )
    metrics.update(
        _subset_metrics(
            "visible",
            pred_latent,
            target_latent,
            decoded,
            target_entity,
            visible_valid,
            thresholds,
        )
    )
    metrics.update(
        _subset_metrics(
            "hidden",
            pred_latent,
            target_latent,
            decoded,
            target_entity,
            hidden_valid,
            thresholds,
        )
    )

    horizon = decoded.shape[2]
    for step in range(horizon):
        name = f"h{step + 1}"
        sl = slice(step, step + 1)
        step_entity_valid_feat = entity_valid[:, :, sl].unsqueeze(-1)
        step_slot_valid = slot_valid[:, :, sl]

        metrics[f"latent_mse_{name}"] = mse(
            pred_latent[:, :, sl], target_latent[:, :, sl], step_entity_valid_feat
        )
        metrics[f"decoded_mse_{name}"] = mse(
            decoded[:, :, sl], target_entity[:, :, sl], step_entity_valid_feat
        )
        metrics[f"decoded_mae_{name}"] = mae(
            decoded[:, :, sl], target_entity[:, :, sl], step_entity_valid_feat
        )

        for threshold in thresholds:
            key = threshold_key(threshold)
            metrics[f"decoded_acc_{key}_{name}"] = threshold_acc(
                decoded[:, :, sl],
                target_entity[:, :, sl],
                step_entity_valid_feat,
                threshold,
            )

        step_presence = presence_metrics(
            presence_logits[:, :, sl],
            target_entity_mask[:, :, sl],
            step_slot_valid,
            presence_threshold,
        )
        for key, value in step_presence.items():
            metrics[f"{key}_{name}"] = value

        # Hidden and visible decoded MAE by horizon are compact, control-relevant diagnostics.
        step_visible = visible_valid[:, :, sl]
        step_hidden = hidden_valid[:, :, sl]
        metrics[f"decoded_mae_visible_{name}"] = mae(
            decoded[:, :, sl], target_entity[:, :, sl], step_visible.unsqueeze(-1)
        )
        metrics[f"decoded_mae_hidden_{name}"] = mae(
            decoded[:, :, sl], target_entity[:, :, sl], step_hidden.unsqueeze(-1)
        )

    if horizon >= 2:
        h1 = metrics["latent_mse_h1"].clamp_min(1e-12)
        hlast = metrics[f"latent_mse_h{horizon}"]
        metrics["latent_error_growth_ratio"] = hlast / h1
        metrics["latent_error_growth_absolute"] = hlast - h1

        d1 = metrics["decoded_mae_h1"].clamp_min(1e-12)
        dlast = metrics[f"decoded_mae_h{horizon}"]
        metrics["decoded_mae_growth_ratio"] = dlast / d1
        metrics["decoded_mae_growth_absolute"] = dlast - d1

    return metrics


def evaluate_one_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = get_config(checkpoint)
    r2_checkpoint = is_r2_checkpoint_config(cfg)
    r2_latent_normalize = bool(
        cfg.get("r2_latent_normalize", False)
    )

    rollout_horizon = int(
        args.eval_rollout_horizon
        if args.eval_rollout_horizon is not None
        else cfg.get("rollout_horizon", 5)
    )
    rollout_window = int(cfg.get("rollout_window", 20))

    dataset = build_dataset(
        args.manifest,
        args.split,
        cfg,
        args.window_mode,
        args.samples_per_epoch,
        args.enemy_visibility_mask,
        args.enemy_sight_range,
        rollout_horizon,
    )
    if args.require_hidden_eval:
        validate_dataset_visibility(dataset)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(checkpoint, dataset, device)
    memory_module = build_memory_module(checkpoint, dataset, device)

    probe_decoder, probe_info = train_or_load_probe_decoder(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        cfg=cfg,
        model=model,
        eval_dataset=dataset,
        args=args,
        out_dir=Path(args.out_dir),
        device=device,
        amp_enabled=amp_enabled,
    )

    target_mode = str(args.target_mode or cfg.get("target_mode", "full"))
    sigreg_weight = float(
        args.sigreg_weight
        if args.sigreg_weight is not None
        else cfg.get("sigreg_weight", 0.01)
    )
    decoder_weight = float(
        args.decoder_weight
        if args.decoder_weight is not None
        else cfg.get("decoder_weight", 0.01)
    )
    presence_weight = float(
        args.presence_weight
        if args.presence_weight is not None
        else cfg.get("presence_weight", 0.01)
    )
    one_step_weight = float(
        args.one_step_weight
        if args.one_step_weight is not None
        else cfg.get("one_step_weight", 0.0)
    )
    td_lambda = float(
        args.td_lambda if args.td_lambda is not None else cfg.get("td_lambda", 0.9)
    )

    sums: dict[str, float] = {}

    # Corrected decoded-state metrics use additive coordinate/entity counts.
    corrected_sums: dict[str, float] = {}
    corrected_prefixes: set[str] = set()

    batches = 0
    windows = 0
    prediction_count = 0
    elapsed_seconds = 0.0
    diagnostic_windows = 0

    autocast_context = (
        torch.cuda.amp.autocast(enabled=amp_enabled)
        if device.type == "cuda"
        else nullcontext()
    )

    diagnostic_limit = (
        args.diagnostic_max_batches
        if args.diagnostic_max_batches is not None
        else args.max_batches
    )

    for batch in loader:
        if args.max_batches is not None and batches >= args.max_batches:
            break

        batch = to_device(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()

        with torch.no_grad(), autocast_context:
            # Keep the checkpoint's original training-objective losses only when the
            # requested eval horizon matches the training horizon. The neutral rollout
            # metrics below are always computed at the requested common horizon.
            if False:
                losses = markov_rollout_rnn_losses(
                    model,
                    memory_module,
                    batch,
                    rollout_window=rollout_window,
                    rollout_horizon=rollout_horizon,
                    temporal_loss_mode=str(cfg.get("temporal_loss", "lambda")),
                    td_lambda=td_lambda,
                    flat_decay_start=cfg.get("flat_decay_start", None),
                    flat_decay_final_weight=float(
                        cfg.get("flat_decay_final_weight", 0.5)
                    ),
                    sigreg_weight=sigreg_weight,
                    decoder_weight=decoder_weight,
                    presence_weight=presence_weight,
                    one_step_weight=one_step_weight,
                    target_mode=target_mode,
                    detach_rollout_targets=bool(
                        cfg.get("detach_rollout_targets", False)
                    ),
                    unweighted_aux_losses=bool(
                        cfg.get("unweighted_aux_losses", False)
                    ),
                )
            else:
                losses = {}

            baseline_outputs = rollout_outputs(
                model,
                memory_module,
                batch,
                rollout_window,
                rollout_horizon,
                target_mode,
                action_mode="correct",
                zero_history_memory=False,
                mask_mode="predicted",
                presence_threshold=float(args.presence_threshold),
                r2_latent_normalize=r2_latent_normalize,
            )
            baseline_metrics = compute_decoded_metrics(
                baseline_outputs,
                list(args.thresholds),
                float(args.presence_threshold),
            )
            probe_metrics: dict[str, torch.Tensor] = {}
            if probe_decoder is not None:
                probe_metrics = compute_probe_decoder_metrics(
                    baseline_outputs,
                    probe_decoder,
                    list(args.thresholds),
                )

            diagnostic_metrics: dict[str, torch.Tensor] = {}
            run_diagnostics_this_batch = (
                args.diagnostics
                and (diagnostic_limit is None or batches < diagnostic_limit)
            )
            if run_diagnostics_this_batch:
                shuffled_outputs = rollout_outputs(
                    model,
                    memory_module,
                    batch,
                    rollout_window,
                    rollout_horizon,
                    target_mode,
                    action_mode="shuffled",
                    zero_history_memory=False,
                    mask_mode="predicted",
                    presence_threshold=float(args.presence_threshold),
                    r2_latent_normalize=r2_latent_normalize,
                )
                shuffled_metrics = compute_decoded_metrics(
                    shuffled_outputs,
                    list(args.thresholds),
                    float(args.presence_threshold),
                )

                zero_memory_outputs = rollout_outputs(
                    model,
                    memory_module,
                    batch,
                    rollout_window,
                    rollout_horizon,
                    target_mode,
                    action_mode="correct",
                    zero_history_memory=True,
                    mask_mode="predicted",
                    presence_threshold=float(args.presence_threshold),
                    r2_latent_normalize=r2_latent_normalize,
                )
                zero_memory_metrics = compute_decoded_metrics(
                    zero_memory_outputs,
                    list(args.thresholds),
                    float(args.presence_threshold),
                )

                autonomous_outputs = rollout_outputs(
                    model,
                    memory_module,
                    batch,
                    rollout_window,
                    rollout_horizon,
                    target_mode,
                    action_mode="correct",
                    zero_history_memory=False,
                    mask_mode="oracle",
                    presence_threshold=float(args.presence_threshold),
                    r2_latent_normalize=r2_latent_normalize,
                )
                autonomous_metrics = compute_decoded_metrics(
                    autonomous_outputs,
                    list(args.thresholds),
                    float(args.presence_threshold),
                )

                for key, value in shuffled_metrics.items():
                    diagnostic_metrics[f"shuffled_action_{key}"] = value
                for key, value in zero_memory_metrics.items():
                    diagnostic_metrics[f"zero_memory_{key}"] = value
                for key, value in autonomous_metrics.items():
                    diagnostic_metrics[f"oracle_mask_{key}"] = value

                diagnostic_metrics["wrong_action_latent_penalty"] = (
                    shuffled_metrics["latent_mse"] - baseline_metrics["latent_mse"]
                )
                diagnostic_metrics["wrong_action_decoded_mae_penalty"] = (
                    shuffled_metrics["decoded_mae"] - baseline_metrics["decoded_mae"]
                )
                diagnostic_metrics["memory_latent_gain"] = (
                    zero_memory_metrics["latent_mse"] - baseline_metrics["latent_mse"]
                )
                diagnostic_metrics["memory_decoded_mae_gain"] = (
                    zero_memory_metrics["decoded_mae"]
                    - baseline_metrics["decoded_mae"]
                )
                diagnostic_metrics["oracle_mask_latent_advantage"] = (
                    baseline_metrics["latent_mse"] - autonomous_metrics["latent_mse"]
                )
                diagnostic_metrics["oracle_mask_decoded_mae_advantage"] = (
                    baseline_metrics["decoded_mae"]
                    - autonomous_metrics["decoded_mae"]
                )

            # --------------------------------------------------------------
            # Corrected decoded-state metrics.
            # --------------------------------------------------------------
            corrected_masks = build_rollout_feature_masks(
                dataset,
                batch,
                baseline_outputs,
            )
            corrected_target = baseline_outputs["target_entity"]

            add_prediction_family_statistics(
                corrected_sums,
                corrected_prefixes,
                "native",
                baseline_outputs["decoded"],
                corrected_target,
                corrected_masks,
                list(args.thresholds),
                include_detailed_groups=True,
            )

            persistence = persistence_prediction(
                batch, baseline_outputs
            )

            event_masks = build_event_conditioned_masks(
                batch,
                baseline_outputs,
                corrected_masks,
                change_threshold=float(args.change_threshold),
                event_threshold=float(args.event_threshold),
                attack_action_min=int(args.attack_action_min),
            )
            add_event_conditioned_statistics(
                corrected_sums,
                corrected_prefixes,
                "native",
                baseline_outputs["decoded"],
                corrected_target,
                event_masks,
                list(args.thresholds),
            )
            add_event_conditioned_statistics(
                corrected_sums,
                corrected_prefixes,
                "persistence",
                persistence,
                corrected_target,
                event_masks,
                list(args.thresholds),
            )

            add_prediction_family_statistics(
                corrected_sums,
                corrected_prefixes,
                "persistence",
                persistence,
                corrected_target,
                corrected_masks,
                list(args.thresholds),
                include_detailed_groups=False,
            )
            add_prediction_family_statistics(
                corrected_sums,
                corrected_prefixes,
                "zero",
                torch.zeros_like(corrected_target),
                corrected_target,
                corrected_masks,
                list(args.thresholds),
                include_detailed_groups=False,
            )

            if probe_decoder is not None:
                (
                    corrected_probe_rollout,
                    corrected_probe_reconstruction,
                ) = decode_with_probe(
                    baseline_outputs,
                    probe_decoder,
                )
                add_event_conditioned_statistics(
                    corrected_sums,
                    corrected_prefixes,
                    "probe_rollout",
                    corrected_probe_rollout,
                    corrected_target,
                    event_masks,
                    list(args.thresholds),
                )
                add_event_conditioned_statistics(
                    corrected_sums,
                    corrected_prefixes,
                    "probe_reconstruction",
                    corrected_probe_reconstruction,
                    corrected_target,
                    event_masks,
                    list(args.thresholds),
                )
                add_prediction_family_statistics(
                    corrected_sums,
                    corrected_prefixes,
                    "probe_rollout",
                    corrected_probe_rollout,
                    corrected_target,
                    corrected_masks,
                    list(args.thresholds),
                    include_detailed_groups=True,
                )
                add_prediction_family_statistics(
                    corrected_sums,
                    corrected_prefixes,
                    "probe_reconstruction",
                    corrected_probe_reconstruction,
                    corrected_target,
                    corrected_masks,
                    list(args.thresholds),
                    include_detailed_groups=True,
                )

            if run_diagnostics_this_batch:
                # Store the normal rollout on the exact same diagnostic subset.
                # Derived action/memory/oracle differences must never subtract a
                # 100-batch diagnostic metric from a 500-batch main metric.
                add_prediction_family_statistics(
                    corrected_sums,
                    corrected_prefixes,
                    "diagnostic_native",
                    baseline_outputs["decoded"],
                    corrected_target,
                    corrected_masks,
                    list(args.thresholds),
                    include_detailed_groups=False,
                )
                add_prediction_family_statistics(
                    corrected_sums,
                    corrected_prefixes,
                    "shuffled_action",
                    shuffled_outputs["decoded"],
                    corrected_target,
                    corrected_masks,
                    list(args.thresholds),
                    include_detailed_groups=False,
                )
                add_prediction_family_statistics(
                    corrected_sums,
                    corrected_prefixes,
                    "zero_memory",
                    zero_memory_outputs["decoded"],
                    corrected_target,
                    corrected_masks,
                    list(args.thresholds),
                    include_detailed_groups=False,
                )
                add_prediction_family_statistics(
                    corrected_sums,
                    corrected_prefixes,
                    "oracle_mask",
                    autonomous_outputs["decoded"],
                    corrected_target,
                    corrected_masks,
                    list(args.thresholds),
                    include_detailed_groups=False,
                )

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_seconds += time.perf_counter() - started

        bsz = next(iter(batch.values())).shape[0]
        batches += 1
        windows += bsz
        prediction_count += bsz * rollout_window * rollout_horizon

        for key, value in losses.items():
            sums[f"loss_{key}"] = sums.get(f"loss_{key}", 0.0) + float(
                value.detach().cpu()
            ) * bsz
        for key, value in baseline_metrics.items():
            sums[f"metric_{key}"] = sums.get(f"metric_{key}", 0.0) + float(
                value.detach().cpu()
            ) * bsz
        for key, value in probe_metrics.items():
            sums[key] = sums.get(key, 0.0) + float(
                value.detach().cpu()
            ) * bsz
        if diagnostic_metrics:
            diagnostic_windows += bsz
        for key, value in diagnostic_metrics.items():
            sums[f"diagnostic_{key}"] = sums.get(
                f"diagnostic_{key}", 0.0
            ) + float(value.detach().cpu()) * bsz

    if windows == 0:
        raise RuntimeError(f"No eval windows produced for {checkpoint_path}")

    row: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_dir": str(checkpoint_path.parent),
        "checkpoint_saved_epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "eval_batches": batches,
        "eval_windows": windows,
        "target_mode": target_mode,
        "action_conditioned_memory": bool(
            cfg.get("action_conditioned_memory", False)
        ),
        "one_step_weight": one_step_weight,
        "sigreg_weight": sigreg_weight,
        "decoder_weight": decoder_weight,
        "presence_weight": presence_weight,
        "td_lambda": td_lambda,
        "enemy_visibility_mask": bool(dataset.enemy_visibility_mask),
        "enemy_sight_range": float(dataset.enemy_sight_range),
        "rollout_window": rollout_window,
        "training_rollout_horizon": int(cfg.get("rollout_horizon", 5)),
        "eval_rollout_horizon": rollout_horizon,
        "objective_family": (
            "r2offline" if r2_checkpoint else "legacy_seqmem"
        ),
        "r2_latent_normalize": r2_latent_normalize,
        "checkpoint_n_actions": int(
            checkpoint.get("metadata", {}).get("n_actions", -1)
        ),
        "dataset_n_actions": int(dataset.metadata.n_actions),
        "diagnostics_enabled": bool(args.diagnostics),
        "diagnostic_windows": diagnostic_windows,
        "eval_elapsed_seconds": elapsed_seconds,
        "eval_windows_per_second": windows / max(elapsed_seconds, 1e-12),
        "eval_predictions_per_second": prediction_count
        / max(elapsed_seconds, 1e-12),
    }
    row.update(probe_info)
    row["legacy_decoded_metrics_include_padding"] = True
    row["corrected_decoded_metrics_exclude_padding"] = True
    row["corrected_probe_trained_on_meaningful_features"] = True
    row["blocker_fixed_rollout_semantics"] = True
    row["main_rollout_uses_predicted_soft_presence"] = True
    row["corrected_aggregation"] = (
        "global_numerator_divided_by_global_valid_feature_count"
    )
    row["event_conditioned_evaluation"] = True
    row["event_change_threshold"] = float(args.change_threshold)
    row["event_window_threshold"] = float(args.event_threshold)
    row["event_attack_action_min"] = int(args.attack_action_min)

    corrected_metrics = finalize_exact_regression_statistics(
        corrected_sums,
        corrected_prefixes,
        list(args.thresholds),
    )
    for key, value in corrected_metrics.items():
        row[f"eval_{key}"] = value

    # Event-conditioned model-vs-persistence comparisons.
    event_groups = (
        "changed_dynamic",
        "unchanged_dynamic",
        "event_window_dynamic",
        "no_event_window_dynamic",
        "action_event_dynamic",
        "no_action_event_dynamic",
    )
    for group in event_groups:
        native_key = f"event_native_{group}_mae"
        persistence_key = f"event_persistence_{group}_mae"
        native_value = corrected_metrics.get(native_key)
        persistence_value = corrected_metrics.get(persistence_key)
        if native_value is None or persistence_value is None:
            continue
        row[
            f"eval_event_native_vs_persistence_{group}_mae_improvement"
        ] = persistence_value - native_value
        row[
            f"eval_event_native_vs_persistence_{group}_mae_relative_improvement"
        ] = 1.0 - native_value / max(persistence_value, 1e-12)

        for step in range(1, rollout_horizon + 1):
            native_h = corrected_metrics.get(
                f"event_native_{group}_h{step}_mae"
            )
            persistence_h = corrected_metrics.get(
                f"event_persistence_{group}_h{step}_mae"
            )
            if native_h is None or persistence_h is None:
                continue
            row[
                f"eval_event_native_vs_persistence_{group}_h{step}_mae_relative_improvement"
            ] = 1.0 - native_h / max(persistence_h, 1e-12)

    hidden_dynamic_count = float(
        corrected_metrics.get(
            "corrected_native_hidden_dynamic_count", 0.0
        )
    )
    row["explicit_visibility_mask_version"] = int(
        getattr(dataset, "explicit_visibility_mask_version", 0)
    )
    row["eval_hidden_dynamic_count_verified"] = hidden_dynamic_count
    row["eval_hidden_metrics_available"] = bool(
        hidden_dynamic_count > 0.0
    )
    if (
        dataset.enemy_visibility_mask
        and hidden_dynamic_count <= 0.0
    ):
        row["eval_hidden_metrics_status"] = (
            "no_hidden_targets_in_evaluated_subset"
        )
        print(
            "WARNING: no hidden target entities occurred in the evaluated "
            "subset. Overall, probe, action, memory, presence, and rollout "
            "metrics remain valid; hidden-only metrics are unavailable.",
            flush=True,
        )
    else:
        row["eval_hidden_metrics_status"] = "available"

    native_mae = corrected_metrics.get(
        "corrected_native_dynamic_mae"
    )
    diagnostic_native_mae = corrected_metrics.get(
        "corrected_diagnostic_native_dynamic_mae"
    )
    persistence_mae = corrected_metrics.get(
        "corrected_persistence_dynamic_mae"
    )
    zero_mae = corrected_metrics.get(
        "corrected_zero_dynamic_mae"
    )
    shuffled_mae = corrected_metrics.get(
        "corrected_shuffled_action_dynamic_mae"
    )
    zero_memory_mae = corrected_metrics.get(
        "corrected_zero_memory_dynamic_mae"
    )
    oracle_mae = corrected_metrics.get(
        "corrected_oracle_mask_dynamic_mae"
    )

    if native_mae is not None and persistence_mae is not None:
        row[
            "eval_corrected_native_vs_persistence_dynamic_mae_improvement"
        ] = persistence_mae - native_mae
        row[
            "eval_corrected_native_vs_persistence_dynamic_mae_relative_improvement"
        ] = 1.0 - native_mae / max(persistence_mae, 1e-12)

    if native_mae is not None and zero_mae is not None:
        row[
            "eval_corrected_native_vs_zero_dynamic_mae_improvement"
        ] = zero_mae - native_mae

    if diagnostic_native_mae is not None and shuffled_mae is not None:
        row[
            "eval_corrected_wrong_action_dynamic_mae_penalty"
        ] = shuffled_mae - diagnostic_native_mae

    if diagnostic_native_mae is not None and zero_memory_mae is not None:
        row[
            "eval_corrected_memory_dynamic_mae_gain"
        ] = zero_memory_mae - diagnostic_native_mae

    if diagnostic_native_mae is not None and oracle_mae is not None:
        row[
            "eval_corrected_oracle_mask_dynamic_mae_advantage"
        ] = diagnostic_native_mae - oracle_mae

    for key, value in sums.items():
        denominator = diagnostic_windows if key.startswith("diagnostic_") else windows
        row[f"eval_{key}"] = value / max(denominator, 1)

    return row


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_out = Path(args.summary_out) if args.summary_out else out_dir / "eval_seqmem_combined_summary.csv"
    summary_jsonl = summary_out.with_suffix(".jsonl")

    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    checkpoints = [Path(p) for p in args.checkpoint]
    for p in checkpoints:
        if not p.exists():
            raise SystemExit(f"Missing checkpoint: {p}")

    print(f"Evaluating {len(checkpoints)} checkpoint(s) split={args.split} device={device}", flush=True)

    rows = []
    for checkpoint_path in checkpoints:
        print(f"evaluating {checkpoint_path}", flush=True)
        row = evaluate_one_checkpoint(checkpoint_path, args, device, amp_enabled)
        rows.append(row)

        stem = checkpoint_path.parent.name + "_" + checkpoint_path.stem
        per_json = out_dir / f"eval_{stem}.json"
        per_csv = out_dir / f"eval_{stem}.csv"
        per_json.write_text(json.dumps(row, indent=2) + "\n")
        write_summary(per_csv, [row])
        with summary_jsonl.open("a") as f:
            f.write(json.dumps(row) + "\n")

        keys = [
            "eval_metric_latent_mse",
            "eval_metric_latent_error_growth_ratio",
            "eval_corrected_native_dynamic_mae",
            "eval_corrected_native_dynamic_target_std_normalized_rmse",
            "eval_corrected_native_dynamic_coord_acc_0p05",
            "eval_corrected_native_dynamic_entity_acc_0p05",
            "eval_corrected_native_static_mae",
            "eval_corrected_native_vs_persistence_dynamic_mae_relative_improvement",
            "eval_corrected_native_enemy_dynamic_mae",
            "eval_corrected_native_hidden_dynamic_mae",
            "eval_metric_presence_f1",
            "eval_metric_variance_normalized_latent_mse",
            "eval_metric_target_latent_std_mean",
            "eval_metric_pred_to_target_std_ratio",
            "eval_probe_reconstruction_mse",
            "eval_probe_rollout_mse",
            "eval_probe_rollout_gap_mse",
            "eval_probe_rollout_acc_0p05",
            "eval_probe_accuracy_drop_0p05",
            "eval_diagnostic_wrong_action_latent_penalty",
            "eval_diagnostic_memory_latent_gain",
            "eval_diagnostic_oracle_mask_latent_advantage",
        ]
        msg = " ".join(f"{k}={row[k]:.6f}" for k in keys if k in row and isinstance(row[k], float))
        print(f"done saved_epoch={row['checkpoint_saved_epoch']} global_step={row['global_step']} {msg}", flush=True)
        print(f"wrote {per_json}", flush=True)
        print(f"wrote {per_csv}", flush=True)

    write_summary(summary_out, rows)
    print(f"wrote summary {summary_out}", flush=True)
    print(f"wrote summary {summary_jsonl}", flush=True)


if __name__ == "__main__":
    main()
