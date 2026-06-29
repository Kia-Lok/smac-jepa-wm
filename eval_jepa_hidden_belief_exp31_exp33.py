from __future__ import annotations

"""Targeted hidden-state evaluation for Exp31--Exp33.

The evaluator imports :mod:`eval_jepa_exp31_exp33`, so checkpoint/model/memory
construction and rollout chronology remain identical to the standard evaluator.
It supports both the ordinary Exp31/32 GRU and the anchored Exp33 memory.

Main diagnostics:
- natural previously-seen hidden enemies (optional),
- controlled contiguous enemy occlusion,
- full-memory versus zero-history-memory,
- masked versus unmasked observation,
- native decoder versus independent frozen probe,
- pure last-seen persistence,
- error by time hidden and requested span length,
- reappearance accuracy,
- hidden entity presence/death accuracy,
- changed-versus-unchanged hidden dynamic coordinates.
"""

import argparse
import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

import eval_jepa_exp31_exp33_anchored as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate recurrent hidden belief under natural and controlled "
            "enemy occlusion. Uses the exact model/memory/rollout code from "
            "eval_jepa_exp31_exp33.py."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--probe-dir", required=True)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)

    parser.add_argument("--eval-rollout-horizon", type=int, default=5)
    parser.add_argument("--target-mode", choices=["full"], default="full")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1],
    )
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=0.01,
        help=(
            "A hidden dynamic coordinate is treated as changed when its "
            "absolute difference from the last observed value exceeds this."
        ),
    )

    parser.add_argument(
        "--natural-hidden-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--natural-hidden-target-entity-times",
        type=int,
        default=3000,
    )
    parser.add_argument(
        "--natural-hidden-max-scan-batches",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--controlled-occlusion-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--controlled-occlusion-max-batches",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--controlled-occlusion-target-entity-times",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--controlled-occlusion-spans",
        type=int,
        nargs="+",
        default=[1, 3, 5],
    )
    parser.add_argument("--controlled-occlusion-seed", type=int, default=123)
    parser.add_argument(
        "--controlled-prefer-reappearance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--controlled-include-death-spans",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow a selected enemy to become absent during the hidden span. "
            "State MAE is scored only while present, while presence metrics "
            "also score the absent/dead steps."
        ),
    )
    return parser.parse_args()


def safe_torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def strided_subset(dataset, *, maximum_items: int) -> Subset:
    total = len(dataset)
    if total <= 0:
        raise RuntimeError("Evaluation dataset is empty")
    count = min(total, max(int(maximum_items), 1))
    if count >= total:
        indices = list(range(total))
    else:
        indices = (
            torch.linspace(0, total - 1, steps=count, dtype=torch.float64)
            .round()
            .long()
            .unique_consecutive()
            .tolist()
        )
    return Subset(dataset, indices)


def gather_rollout_targets(
    sequence: torch.Tensor,
    rollout_window: int,
    rollout_horizon: int,
) -> torch.Tensor:
    """Gather sequence targets at ``start + horizon_step + 1``."""
    p = int(rollout_window)
    h = int(rollout_horizon)
    index = (
        torch.arange(p, device=sequence.device)[:, None]
        + torch.arange(h, device=sequence.device)[None, :]
        + 1
    )
    if index.numel() == 0 or int(index.max().item()) >= sequence.shape[1]:
        raise RuntimeError(
            "Rollout target index exceeds sequence length: "
            f"max_index={int(index.max().item()) if index.numel() else -1} "
            f"sequence_length={sequence.shape[1]}"
        )
    return sequence[:, index]


def last_seen_before_sequence(
    full_entity_seq: torch.Tensor,
    observation_mask_seq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Last observed state, has-seen flag, and hidden age before each time."""
    batch, time_steps, entities, _ = full_entity_seq.shape
    last_state = torch.zeros_like(full_entity_seq[:, 0])
    last_index = torch.full(
        (batch, entities),
        -1,
        dtype=torch.long,
        device=full_entity_seq.device,
    )

    state_before: list[torch.Tensor] = []
    has_before: list[torch.Tensor] = []
    age_before: list[torch.Tensor] = []

    for step in range(time_steps):
        has_seen = last_index >= 0
        state_before.append(last_state)
        has_before.append(has_seen)
        age_before.append(
            torch.where(
                has_seen,
                torch.full_like(last_index, step) - last_index,
                torch.full_like(last_index, -1),
            )
        )

        observed = observation_mask_seq[:, step].bool()
        last_state = torch.where(
            observed.unsqueeze(-1),
            full_entity_seq[:, step],
            last_state,
        )
        last_index = torch.where(
            observed,
            torch.full_like(last_index, step),
            last_index,
        )

    return (
        torch.stack(state_before, dim=1),
        torch.stack(has_before, dim=1),
        torch.stack(age_before, dim=1),
    )


def enemy_slot_mask(
    shape: torch.Size,
    max_agents: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    mask[..., int(max_agents) :] = True
    return mask


def position_feature_mask(
    dataset,
    enemy_dynamic_mask: torch.Tensor,
) -> torch.Tensor:
    result = torch.zeros_like(enemy_dynamic_mask)
    for raw_index in getattr(dataset, "xy_indices", (2, 3)):
        index = int(raw_index)
        if 0 <= index < result.shape[-1]:
            result[..., index] = enemy_dynamic_mask[..., index]
    return result


def add_regression_groups(
    accumulator: dict[str, float],
    prefixes: set[str],
    *,
    prefix: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    dynamic_mask: torch.Tensor,
    position_mask: torch.Tensor,
    age: torch.Tensor | None,
    thresholds: list[float],
) -> None:
    groups = {
        "dynamic": dynamic_mask,
        "position": position_mask,
        "non_position": (
            dynamic_mask.bool() & ~position_mask.bool()
        ).to(dynamic_mask.dtype),
    }
    for group_name, mask in groups.items():
        base.add_exact_regression_statistics(
            accumulator,
            prefixes,
            f"{prefix}_{group_name}",
            prediction,
            target,
            mask,
            thresholds,
        )

    for step in range(prediction.shape[2]):
        sl = slice(step, step + 1)
        base.add_exact_regression_statistics(
            accumulator,
            prefixes,
            f"{prefix}_dynamic_h{step + 1}",
            prediction[:, :, sl],
            target[:, :, sl],
            dynamic_mask[:, :, sl],
            thresholds,
        )

    if age is None:
        return
    age_groups = {
        "age1": age == 1,
        "age2": age == 2,
        "age3_5": (age >= 3) & (age <= 5),
        "age6_10": (age >= 6) & (age <= 10),
        "age11_plus": age >= 11,
    }
    for age_name, entity_mask in age_groups.items():
        feature_mask = dynamic_mask * entity_mask.unsqueeze(-1)
        base.add_exact_regression_statistics(
            accumulator,
            prefixes,
            f"{prefix}_{age_name}_dynamic",
            prediction,
            target,
            feature_mask,
            thresholds,
        )
        base.add_exact_regression_statistics(
            accumulator,
            prefixes,
            f"{prefix}_{age_name}_position",
            prediction,
            target,
            position_mask * entity_mask.unsqueeze(-1),
            thresholds,
        )


def add_changed_unchanged_groups(
    accumulator: dict[str, float],
    prefixes: set[str],
    *,
    prefix: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    last_seen: torch.Tensor,
    dynamic_mask: torch.Tensor,
    change_threshold: float,
    thresholds: list[float],
) -> None:
    delta = (target.float() - last_seen.float()).abs()
    changed = dynamic_mask.bool() & (delta > float(change_threshold))
    unchanged = dynamic_mask.bool() & ~changed
    base.add_exact_regression_statistics(
        accumulator,
        prefixes,
        f"{prefix}_changed_dynamic",
        prediction,
        target,
        changed.to(dynamic_mask.dtype),
        thresholds,
    )
    base.add_exact_regression_statistics(
        accumulator,
        prefixes,
        f"{prefix}_unchanged_dynamic",
        prediction,
        target,
        unchanged.to(dynamic_mask.dtype),
        thresholds,
    )


def add_presence_counts(
    accumulator: dict[str, float],
    prefixes: set[str],
    *,
    prefix: str,
    logits: torch.Tensor,
    target_presence: torch.Tensor,
    entity_mask: torch.Tensor,
    threshold: float,
) -> None:
    valid = entity_mask.bool()
    if int(valid.sum().item()) == 0:
        return
    prediction = torch.sigmoid(logits.float()) >= float(threshold)
    target = target_presence >= 0.5
    values = {
        "tp": (prediction & target & valid).sum(),
        "tn": ((~prediction) & (~target) & valid).sum(),
        "fp": (prediction & (~target) & valid).sum(),
        "fn": ((~prediction) & target & valid).sum(),
        "count": valid.sum(),
    }
    prefixes.add(prefix)
    for name, value in values.items():
        key = f"{prefix}__{name}"
        accumulator[key] = accumulator.get(key, 0.0) + float(
            value.detach().double().cpu()
        )


def finalize_presence_counts(
    accumulator: dict[str, float],
    prefixes: set[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for prefix in sorted(prefixes):
        tp = accumulator.get(f"{prefix}__tp", 0.0)
        tn = accumulator.get(f"{prefix}__tn", 0.0)
        fp = accumulator.get(f"{prefix}__fp", 0.0)
        fn = accumulator.get(f"{prefix}__fn", 0.0)
        count = accumulator.get(f"{prefix}__count", 0.0)
        if count <= 0:
            continue
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        result[f"{prefix}_count"] = count
        result[f"{prefix}_accuracy"] = (tp + tn) / count
        result[f"{prefix}_precision"] = precision
        result[f"{prefix}_recall"] = recall
        result[f"{prefix}_f1"] = f1
    return result


def load_existing_probe(
    *,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    model,
    dataset,
    probe_dir: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    cfg = base.get_config(checkpoint)
    sample = dataset[0]
    target_sample = sample.get("target_entity_seq", sample["entity_seq"])
    target_dim = int(target_sample.shape[-1])
    latent_dim = int(cfg["latent_dim"])
    probe, module_name, signature = base.clone_fresh_native_decoder(
        model,
        latent_dim=latent_dim,
        target_dim=target_dim,
        device=device,
    )
    probe_path = probe_dir / (
        f"{checkpoint_path.parent.name}_{checkpoint_path.stem}"
        "_meaningful_features_v2_probe_decoder.pt"
    )
    if not probe_path.is_file():
        raise FileNotFoundError(
            "The hidden evaluator reuses the independent probe trained by the "
            f"standard evaluator, but it was not found: {probe_path}"
        )
    saved = safe_torch_load(probe_path, device)
    if "probe_decoder_state" not in saved:
        raise RuntimeError(
            f"Probe file is missing probe_decoder_state: {probe_path}"
        )
    probe.load_state_dict(saved["probe_decoder_state"], strict=True)
    probe.eval()
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    return probe, {
        "probe_decoder_path": str(probe_path),
        "probe_decoder_loaded_existing": True,
        "probe_decoder_module_name": module_name,
        "probe_decoder_linear_signature": signature,
    }


def action_space_audit(
    *,
    checkpoint: dict[str, Any],
    dataset,
    memory_module,
) -> dict[str, Any]:
    metadata = checkpoint.get("metadata", {})
    sample = dataset[0]
    tensor_width = int(sample["action_seq"].shape[-1])
    split_n_actions = int(dataset.metadata.n_actions)
    dataset_max_actions = int(dataset.metadata.max_actions)
    checkpoint_n_actions = int(metadata.get("n_actions", -1))
    checkpoint_max_actions = int(
        metadata.get("max_actions", checkpoint_n_actions)
    )
    memory_n_actions = int(getattr(memory_module, "n_actions", tensor_width))

    if tensor_width != dataset_max_actions:
        raise RuntimeError(
            "Dataset action tensor width does not equal dataset.max_actions: "
            f"tensor={tensor_width} max_actions={dataset_max_actions}"
        )
    if tensor_width != memory_n_actions:
        raise RuntimeError(
            "Action tensor width does not equal memory action width: "
            f"tensor={tensor_width} memory={memory_n_actions}"
        )
    if checkpoint_max_actions > 0 and tensor_width != checkpoint_max_actions:
        raise RuntimeError(
            "Action tensor width does not equal checkpoint max_actions: "
            f"tensor={tensor_width} checkpoint={checkpoint_max_actions}"
        )
    if split_n_actions > tensor_width:
        raise RuntimeError(
            "Split-local n_actions exceeds padded action width: "
            f"split={split_n_actions} width={tensor_width}"
        )
    return {
        "action_space_audit_status": (
            "split_local_actions_padded_to_global_width"
            if split_n_actions < tensor_width
            else "exact_width_match"
        ),
        "checkpoint_n_actions": checkpoint_n_actions,
        "checkpoint_max_actions": checkpoint_max_actions,
        "dataset_split_n_actions": split_n_actions,
        "dataset_max_actions": dataset_max_actions,
        "batch_action_tensor_width": tensor_width,
        "memory_module_action_width": memory_n_actions,
    }


def relevant_natural_hidden_on_cpu(
    batch: dict[str, torch.Tensor],
    *,
    rollout_window: int,
    rollout_horizon: int,
    max_agents: int,
) -> bool:
    observation = batch["observation_mask_seq"].bool()
    target = batch.get(
        "target_entity_mask_seq", batch["observation_mask_seq"]
    ).bool()
    slot = batch["entity_slot_mask_seq"].bool()
    hidden = target & ~observation & slot
    hidden[..., : int(max_agents)] = False
    hidden_rollout = gather_rollout_targets(
        hidden, rollout_window, rollout_horizon
    )
    valid = gather_rollout_targets(
        batch["state_mask"].bool(), rollout_window, rollout_horizon
    )
    return bool((hidden_rollout & valid.unsqueeze(-1)).any().item())


def natural_hidden_pass(
    *,
    model,
    memory_module,
    probe,
    dataset,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    rollout_window: int,
    rollout_horizon: int,
    r2_latent_normalize: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    maximum_items = (
        int(args.natural_hidden_max_scan_batches) * int(args.batch_size)
    )
    loader = DataLoader(
        strided_subset(dataset, maximum_items=maximum_items),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    regression_sums: dict[str, float] = {}
    regression_prefixes: set[str] = set()
    presence_sums: dict[str, float] = {}
    presence_prefixes: set[str] = set()

    scanned_batches = 0
    evaluated_batches = 0
    scored_hidden_entity_times = 0
    scored_presence_entity_times = 0
    scored_reappearance_entity_times = 0
    elapsed = 0.0
    max_agents = int(dataset.metadata.max_agents)

    for cpu_batch in loader:
        if scanned_batches >= int(args.natural_hidden_max_scan_batches):
            break
        scanned_batches += 1
        if not relevant_natural_hidden_on_cpu(
            cpu_batch,
            rollout_window=rollout_window,
            rollout_horizon=rollout_horizon,
            max_agents=max_agents,
        ):
            continue

        batch = base.to_device(cpu_batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad(), autocast_context(device, amp_enabled):
            outputs = base.rollout_outputs(
                model,
                memory_module,
                batch,
                rollout_window,
                rollout_horizon,
                args.target_mode,
                action_mode="correct",
                zero_history_memory=False,
                mask_mode="predicted",
                presence_threshold=args.presence_threshold,
                r2_latent_normalize=r2_latent_normalize,
            )
            zero_outputs = base.rollout_outputs(
                model,
                memory_module,
                batch,
                rollout_window,
                rollout_horizon,
                args.target_mode,
                action_mode="correct",
                zero_history_memory=True,
                mask_mode="predicted",
                presence_threshold=args.presence_threshold,
                r2_latent_normalize=r2_latent_normalize,
            )
            probe_rollout, _ = base.decode_with_probe(outputs, probe)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        evaluated_batches += 1

        full_sequence = batch.get("target_entity_seq", batch["entity_seq"])
        last_seen_before, has_seen_before, age_before = (
            last_seen_before_sequence(
                full_sequence, batch["observation_mask_seq"]
            )
        )
        last_seen_rollout = gather_rollout_targets(
            last_seen_before, rollout_window, rollout_horizon
        )
        has_seen_rollout = gather_rollout_targets(
            has_seen_before, rollout_window, rollout_horizon
        )
        age_rollout = gather_rollout_targets(
            age_before, rollout_window, rollout_horizon
        )

        feature_masks = base.build_rollout_feature_masks(
            dataset, batch, outputs
        )
        hidden_seen_entity = (
            outputs["target_entity_mask"].bool()
            & ~outputs["observed_entity_mask"].bool()
            & has_seen_rollout.bool()
            & outputs["entity_slot_mask"].bool()
            & outputs["valid_mask"].bool().unsqueeze(-1)
        )
        enemy_slots = enemy_slot_mask(
            outputs["target_entity_mask"].shape,
            max_agents,
            device=device,
        )
        hidden_seen_entity &= enemy_slots
        hidden_dynamic = (
            feature_masks["enemy_dynamic"]
            * hidden_seen_entity.unsqueeze(-1)
        )
        position = position_feature_mask(dataset, hidden_dynamic)
        target = outputs["target_entity"]

        for prefix, prediction in (
            ("natural_hidden_native", outputs["decoded"]),
            ("natural_hidden_zero_memory", zero_outputs["decoded"]),
            ("natural_hidden_probe_rollout", probe_rollout),
            ("natural_hidden_last_seen", last_seen_rollout),
        ):
            add_regression_groups(
                regression_sums,
                regression_prefixes,
                prefix=prefix,
                prediction=prediction,
                target=target,
                dynamic_mask=hidden_dynamic,
                position_mask=position,
                age=age_rollout,
                thresholds=list(args.thresholds),
            )
            add_changed_unchanged_groups(
                regression_sums,
                regression_prefixes,
                prefix=prefix,
                prediction=prediction,
                target=target,
                last_seen=last_seen_rollout,
                dynamic_mask=hidden_dynamic,
                change_threshold=args.change_threshold,
                thresholds=list(args.thresholds),
            )

        belief_presence_selector = (
            enemy_slots
            & has_seen_rollout.bool()
            & ~outputs["observed_entity_mask"].bool()
            & outputs["entity_slot_mask"].bool()
            & outputs["valid_mask"].bool().unsqueeze(-1)
        )
        for prefix, logits in (
            ("natural_hidden_native_presence", outputs["presence_logits"]),
            (
                "natural_hidden_zero_memory_presence",
                zero_outputs["presence_logits"],
            ),
        ):
            add_presence_counts(
                presence_sums,
                presence_prefixes,
                prefix=prefix,
                logits=logits,
                target_presence=outputs["target_entity_mask"],
                entity_mask=belief_presence_selector,
                threshold=args.presence_threshold,
            )

        observation = batch["observation_mask_seq"].bool()
        reappearance_seq = observation & has_seen_before.bool() & (
            age_before >= 2
        )
        reappearance_rollout = gather_rollout_targets(
            reappearance_seq, rollout_window, rollout_horizon
        )
        reappearance_entity = (
            reappearance_rollout
            & outputs["target_entity_mask"].bool()
            & outputs["entity_slot_mask"].bool()
            & outputs["valid_mask"].bool().unsqueeze(-1)
            & enemy_slots
        )
        reappearance_dynamic = (
            feature_masks["enemy_dynamic"]
            * reappearance_entity.unsqueeze(-1)
        )
        reappearance_position = position_feature_mask(
            dataset, reappearance_dynamic
        )
        for prefix, prediction in (
            ("natural_reappearance_native", outputs["decoded"]),
            ("natural_reappearance_zero_memory", zero_outputs["decoded"]),
            ("natural_reappearance_probe_rollout", probe_rollout),
        ):
            add_regression_groups(
                regression_sums,
                regression_prefixes,
                prefix=prefix,
                prediction=prediction,
                target=target,
                dynamic_mask=reappearance_dynamic,
                position_mask=reappearance_position,
                age=None,
                thresholds=list(args.thresholds),
            )

        scored_hidden_entity_times += int(hidden_seen_entity.sum().item())
        scored_presence_entity_times += int(
            belief_presence_selector.sum().item()
        )
        scored_reappearance_entity_times += int(
            reappearance_entity.sum().item()
        )
        if scored_hidden_entity_times >= int(
            args.natural_hidden_target_entity_times
        ):
            break

    metrics = base.finalize_exact_regression_statistics(
        regression_sums, regression_prefixes, list(args.thresholds)
    )
    metrics.update(
        finalize_presence_counts(presence_sums, presence_prefixes)
    )
    native = metrics.get("natural_hidden_native_dynamic_mae")
    zero = metrics.get("natural_hidden_zero_memory_dynamic_mae")
    last_seen = metrics.get("natural_hidden_last_seen_dynamic_mae")
    if native is not None and zero is not None:
        metrics["natural_hidden_memory_gain_dynamic_mae"] = zero - native
    if native is not None and last_seen is not None:
        metrics[
            "natural_hidden_vs_last_seen_dynamic_mae_improvement"
        ] = last_seen - native

    return metrics, {
        "natural_hidden_scanned_batches": scanned_batches,
        "natural_hidden_evaluated_batches": evaluated_batches,
        "natural_hidden_scored_entity_times": scored_hidden_entity_times,
        "natural_hidden_presence_entity_times": scored_presence_entity_times,
        "natural_hidden_reappearance_entity_times": (
            scored_reappearance_entity_times
        ),
        "natural_hidden_elapsed_seconds": elapsed,
        "natural_hidden_status": (
            "available"
            if scored_hidden_entity_times > 0
            else "no_previously_seen_hidden_enemy_targets_found"
        ),
    }


def controlled_occlusion_schedule(
    batch: dict[str, torch.Tensor],
    *,
    max_agents: int,
    spans: list[int],
    seed: int,
    prefer_reappearance: bool,
    include_death_spans: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[dict[str, int | bool]],
]:
    """Select at most one enemy and one contiguous span per sample."""
    observation = batch["observation_mask_seq"].bool()
    target = batch.get(
        "target_entity_mask_seq", batch["observation_mask_seq"]
    ).bool()
    slot = batch["entity_slot_mask_seq"].bool()
    state_valid = batch["state_mask"].bool()

    batch_size, time_steps, entities = observation.shape
    selected = torch.zeros_like(observation)
    reappearance = torch.zeros_like(observation)
    age = torch.full_like(observation, -1, dtype=torch.long)
    span_length = torch.zeros_like(observation, dtype=torch.long)
    records: list[dict[str, int | bool]] = []

    cleaned_spans = sorted({int(v) for v in spans if int(v) >= 1})
    if not cleaned_spans:
        raise ValueError("At least one positive occlusion span is required")

    episode_index = batch.get("episode_index")
    episode_ids = (
        list(range(batch_size))
        if episode_index is None
        else [int(v) for v in episode_index.reshape(-1).tolist()]
    )

    for sample_index in range(batch_size):
        desired = cleaned_spans[
            (episode_ids[sample_index] + int(seed)) % len(cleaned_spans)
        ]
        span_order = [desired] + [
            value
            for value in sorted(cleaned_spans, reverse=True)
            if value != desired
        ]
        chosen: tuple[int, int, int, bool] | None = None

        reappearance_preferences = (
            [True, False] if prefer_reappearance else [False]
        )
        for require_reappearance in reappearance_preferences:
            candidates: list[tuple[int, int, int]] = []
            for length in span_order:
                final_start = (
                    time_steps - length - 1
                    if require_reappearance
                    else time_steps - length
                )
                if final_start < 1:
                    continue
                for entity in range(int(max_agents), entities):
                    for start in range(1, final_start + 1):
                        stop = start + length
                        # The entity must be visible and present immediately
                        # before controlled hiding begins.
                        if not bool(
                            observation[sample_index, start - 1, entity]
                            & target[sample_index, start - 1, entity]
                            & slot[sample_index, start - 1, entity]
                            & state_valid[sample_index, start - 1]
                        ):
                            continue
                        if not bool(
                            (
                                slot[sample_index, start:stop, entity]
                                & state_valid[sample_index, start:stop]
                            ).all()
                        ):
                            continue
                        target_during = target[
                            sample_index, start:stop, entity
                        ]
                        if include_death_spans:
                            # Require at least one present hidden target so the
                            # span contains a state-belief case. Later absent
                            # steps are still useful for presence/death scoring.
                            if not bool(target_during.any()):
                                continue
                        elif not bool(target_during.all()):
                            continue
                        if require_reappearance and not bool(
                            observation[sample_index, stop, entity]
                            & target[sample_index, stop, entity]
                            & slot[sample_index, stop, entity]
                            & state_valid[sample_index, stop]
                        ):
                            continue
                        candidates.append((entity, start, length))

            if candidates:
                choice_index = (
                    episode_ids[sample_index]
                    + int(seed)
                    + sample_index * 997
                ) % len(candidates)
                entity, start, length = candidates[choice_index]
                chosen = (entity, start, length, require_reappearance)
                break

        if chosen is None:
            continue
        entity, start, length, has_reappearance = chosen
        stop = start + length
        selected[sample_index, start:stop, entity] = True
        age[sample_index, start:stop, entity] = torch.arange(
            1, length + 1, dtype=torch.long
        )
        span_length[sample_index, start:stop, entity] = length
        if has_reappearance:
            reappearance[sample_index, stop, entity] = True
        records.append(
            {
                "sample_index": sample_index,
                "episode_index": episode_ids[sample_index],
                "entity_index": entity,
                "start": start,
                "length": length,
                "has_reappearance": bool(has_reappearance),
                "contains_absent_step": bool(
                    (~target[sample_index, start:stop, entity]).any()
                ),
            }
        )

    return selected, reappearance, age, span_length, records


def make_masked_batch(
    batch: dict[str, torch.Tensor],
    selected: torch.Tensor,
) -> dict[str, torch.Tensor]:
    result = dict(batch)
    result["observation_mask_seq"] = batch["observation_mask_seq"].clone()
    result["observation_mask_seq"][selected] = 0
    # Second leakage barrier: remove the token values as well as the mask.
    result["entity_seq"] = batch["entity_seq"].clone()
    result["entity_seq"][selected.unsqueeze(-1).expand_as(result["entity_seq"])] = 0
    return result


def controlled_occlusion_pass(
    *,
    model,
    memory_module,
    probe,
    dataset,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    rollout_window: int,
    rollout_horizon: int,
    r2_latent_normalize: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    maximum_items = (
        int(args.controlled_occlusion_max_batches) * int(args.batch_size)
    )
    loader = DataLoader(
        strided_subset(dataset, maximum_items=maximum_items),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    regression_sums: dict[str, float] = {}
    regression_prefixes: set[str] = set()
    presence_sums: dict[str, float] = {}
    presence_prefixes: set[str] = set()

    evaluated_batches = 0
    selected_spans = 0
    selected_with_reappearance = 0
    selected_with_absence = 0
    scored_entity_times = 0
    scored_presence_entity_times = 0
    scored_reappearance_entity_times = 0
    elapsed = 0.0
    selected_span_histogram: dict[str, int] = {}
    max_agents = int(dataset.metadata.max_agents)

    for cpu_batch in loader:
        if evaluated_batches >= int(args.controlled_occlusion_max_batches):
            break
        (
            selected_cpu,
            reappearance_cpu,
            controlled_age_cpu,
            span_length_cpu,
            records,
        ) = controlled_occlusion_schedule(
            cpu_batch,
            max_agents=max_agents,
            spans=list(args.controlled_occlusion_spans),
            seed=int(args.controlled_occlusion_seed),
            prefer_reappearance=bool(args.controlled_prefer_reappearance),
            include_death_spans=bool(args.controlled_include_death_spans),
        )
        if not records:
            continue

        for record in records:
            key = str(int(record["length"]))
            selected_span_histogram[key] = (
                selected_span_histogram.get(key, 0) + 1
            )
            selected_with_reappearance += int(
                bool(record["has_reappearance"])
            )
            selected_with_absence += int(
                bool(record["contains_absent_step"])
            )
        selected_spans += len(records)

        unmasked_batch = base.to_device(cpu_batch, device)
        selected = selected_cpu.to(device)
        reappearance = reappearance_cpu.to(device)
        controlled_age = controlled_age_cpu.to(device)
        span_length = span_length_cpu.to(device)
        masked_batch = make_masked_batch(unmasked_batch, selected)

        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad(), autocast_context(device, amp_enabled):
            masked_outputs = base.rollout_outputs(
                model,
                memory_module,
                masked_batch,
                rollout_window,
                rollout_horizon,
                args.target_mode,
                action_mode="correct",
                zero_history_memory=False,
                mask_mode="predicted",
                presence_threshold=args.presence_threshold,
                r2_latent_normalize=r2_latent_normalize,
            )
            zero_outputs = base.rollout_outputs(
                model,
                memory_module,
                masked_batch,
                rollout_window,
                rollout_horizon,
                args.target_mode,
                action_mode="correct",
                zero_history_memory=True,
                mask_mode="predicted",
                presence_threshold=args.presence_threshold,
                r2_latent_normalize=r2_latent_normalize,
            )
            unmasked_outputs = base.rollout_outputs(
                model,
                memory_module,
                unmasked_batch,
                rollout_window,
                rollout_horizon,
                args.target_mode,
                action_mode="correct",
                zero_history_memory=False,
                mask_mode="predicted",
                presence_threshold=args.presence_threshold,
                r2_latent_normalize=r2_latent_normalize,
            )
            masked_probe, _ = base.decode_with_probe(masked_outputs, probe)
            unmasked_probe, _ = base.decode_with_probe(unmasked_outputs, probe)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        evaluated_batches += 1

        full_sequence = masked_batch.get(
            "target_entity_seq", masked_batch["entity_seq"]
        )
        last_seen_before, _, _ = last_seen_before_sequence(
            full_sequence, masked_batch["observation_mask_seq"]
        )
        last_seen_rollout = gather_rollout_targets(
            last_seen_before, rollout_window, rollout_horizon
        )
        selected_rollout = gather_rollout_targets(
            selected, rollout_window, rollout_horizon
        )
        age_rollout = gather_rollout_targets(
            controlled_age, rollout_window, rollout_horizon
        )
        span_rollout = gather_rollout_targets(
            span_length, rollout_window, rollout_horizon
        )
        reappearance_rollout = gather_rollout_targets(
            reappearance, rollout_window, rollout_horizon
        )

        feature_masks = base.build_rollout_feature_masks(
            dataset, masked_batch, masked_outputs
        )
        # State reconstruction is meaningful only while the enemy exists.
        selected_entity = (
            selected_rollout.bool()
            & masked_outputs["target_entity_mask"].bool()
            & masked_outputs["entity_slot_mask"].bool()
            & masked_outputs["valid_mask"].bool().unsqueeze(-1)
        )
        # Presence/death evaluation includes every selected slot-time, including
        # target-absent steps after a hidden death.
        selected_presence_entity = (
            selected_rollout.bool()
            & masked_outputs["entity_slot_mask"].bool()
            & masked_outputs["valid_mask"].bool().unsqueeze(-1)
        )
        dynamic = (
            feature_masks["enemy_dynamic"]
            * selected_entity.unsqueeze(-1)
        )
        position = position_feature_mask(dataset, dynamic)
        target = masked_outputs["target_entity"]

        families = (
            ("controlled_masked_native", masked_outputs["decoded"]),
            ("controlled_zero_memory", zero_outputs["decoded"]),
            ("controlled_unmasked_native", unmasked_outputs["decoded"]),
            ("controlled_masked_probe_rollout", masked_probe),
            ("controlled_unmasked_probe_rollout", unmasked_probe),
            ("controlled_last_seen", last_seen_rollout),
        )
        for prefix, prediction in families:
            add_regression_groups(
                regression_sums,
                regression_prefixes,
                prefix=prefix,
                prediction=prediction,
                target=target,
                dynamic_mask=dynamic,
                position_mask=position,
                age=age_rollout,
                thresholds=list(args.thresholds),
            )
            add_changed_unchanged_groups(
                regression_sums,
                regression_prefixes,
                prefix=prefix,
                prediction=prediction,
                target=target,
                last_seen=last_seen_rollout,
                dynamic_mask=dynamic,
                change_threshold=args.change_threshold,
                thresholds=list(args.thresholds),
            )
            for length in sorted(
                {int(value) for value in args.controlled_occlusion_spans}
            ):
                length_dynamic = dynamic * (
                    span_rollout == length
                ).unsqueeze(-1)
                base.add_exact_regression_statistics(
                    regression_sums,
                    regression_prefixes,
                    f"{prefix}_span{length}_dynamic",
                    prediction,
                    target,
                    length_dynamic,
                    list(args.thresholds),
                )

        for prefix, logits in (
            (
                "controlled_masked_native_presence",
                masked_outputs["presence_logits"],
            ),
            (
                "controlled_zero_memory_presence",
                zero_outputs["presence_logits"],
            ),
            (
                "controlled_unmasked_native_presence",
                unmasked_outputs["presence_logits"],
            ),
        ):
            add_presence_counts(
                presence_sums,
                presence_prefixes,
                prefix=prefix,
                logits=logits,
                target_presence=masked_outputs["target_entity_mask"],
                entity_mask=selected_presence_entity,
                threshold=args.presence_threshold,
            )

        reappearance_entity = (
            reappearance_rollout.bool()
            & masked_outputs["target_entity_mask"].bool()
            & masked_outputs["entity_slot_mask"].bool()
            & masked_outputs["valid_mask"].bool().unsqueeze(-1)
        )
        reappearance_dynamic = (
            feature_masks["enemy_dynamic"]
            * reappearance_entity.unsqueeze(-1)
        )
        reappearance_position = position_feature_mask(
            dataset, reappearance_dynamic
        )
        for prefix, prediction in (
            (
                "controlled_reappearance_masked_native",
                masked_outputs["decoded"],
            ),
            (
                "controlled_reappearance_zero_memory",
                zero_outputs["decoded"],
            ),
            (
                "controlled_reappearance_unmasked_native",
                unmasked_outputs["decoded"],
            ),
            (
                "controlled_reappearance_masked_probe",
                masked_probe,
            ),
        ):
            add_regression_groups(
                regression_sums,
                regression_prefixes,
                prefix=prefix,
                prediction=prediction,
                target=target,
                dynamic_mask=reappearance_dynamic,
                position_mask=reappearance_position,
                age=None,
                thresholds=list(args.thresholds),
            )

        scored_entity_times += int(selected_entity.sum().item())
        scored_presence_entity_times += int(
            selected_presence_entity.sum().item()
        )
        scored_reappearance_entity_times += int(
            reappearance_entity.sum().item()
        )
        if scored_entity_times >= int(
            args.controlled_occlusion_target_entity_times
        ):
            break

    metrics = base.finalize_exact_regression_statistics(
        regression_sums, regression_prefixes, list(args.thresholds)
    )
    metrics.update(
        finalize_presence_counts(presence_sums, presence_prefixes)
    )
    masked = metrics.get("controlled_masked_native_dynamic_mae")
    zero = metrics.get("controlled_zero_memory_dynamic_mae")
    unmasked = metrics.get("controlled_unmasked_native_dynamic_mae")
    last_seen = metrics.get("controlled_last_seen_dynamic_mae")
    if masked is not None and zero is not None:
        metrics["controlled_memory_gain_dynamic_mae"] = zero - masked
    if masked is not None and unmasked is not None:
        metrics["controlled_occlusion_cost_dynamic_mae"] = masked - unmasked
    if masked is not None and last_seen is not None:
        metrics[
            "controlled_vs_last_seen_dynamic_mae_improvement"
        ] = last_seen - masked

    # The decisive Exp33 comparison: improve over persistence on coordinates
    # that actually changed while remaining close on unchanged coordinates.
    native_changed = metrics.get(
        "controlled_masked_native_changed_dynamic_mae"
    )
    last_changed = metrics.get(
        "controlled_last_seen_changed_dynamic_mae"
    )
    if native_changed is not None and last_changed is not None:
        metrics[
            "controlled_changed_vs_last_seen_mae_improvement"
        ] = last_changed - native_changed
    native_unchanged = metrics.get(
        "controlled_masked_native_unchanged_dynamic_mae"
    )
    last_unchanged = metrics.get(
        "controlled_last_seen_unchanged_dynamic_mae"
    )
    if native_unchanged is not None and last_unchanged is not None:
        metrics[
            "controlled_unchanged_cost_vs_last_seen_mae"
        ] = native_unchanged - last_unchanged

    return metrics, {
        "controlled_evaluated_batches": evaluated_batches,
        "controlled_selected_spans": selected_spans,
        "controlled_selected_with_reappearance": selected_with_reappearance,
        "controlled_selected_with_absence": selected_with_absence,
        "controlled_scored_entity_times": scored_entity_times,
        "controlled_presence_entity_times": scored_presence_entity_times,
        "controlled_reappearance_entity_times": (
            scored_reappearance_entity_times
        ),
        "controlled_span_histogram": selected_span_histogram,
        "controlled_elapsed_seconds": elapsed,
        "controlled_include_death_spans": bool(
            args.controlled_include_death_spans
        ),
        "controlled_status": (
            "available"
            if scored_entity_times > 0
            else "no_valid_visible_enemy_spans_found"
        ),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = safe_torch_load(checkpoint_path, device)
    cfg = base.get_config(checkpoint)
    rollout_window = int(cfg.get("rollout_window", 20))
    rollout_horizon = int(args.eval_rollout_horizon)
    r2_latent_normalize = bool(cfg.get("r2_latent_normalize", False))

    dataset = base.build_dataset(
        args.manifest,
        args.split,
        cfg,
        "sequential",
        None,
        None,
        None,
        rollout_horizon,
    )
    model = base.build_model(checkpoint, dataset, device)
    memory_module = base.build_memory_module(checkpoint, dataset, device)
    model.eval()
    memory_module.eval()

    probe, probe_info = load_existing_probe(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        model=model,
        dataset=dataset,
        probe_dir=Path(args.probe_dir),
        device=device,
    )
    row: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_dir": str(checkpoint_path.parent),
        "checkpoint_saved_epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "eval_rollout_window": rollout_window,
        "eval_rollout_horizon": rollout_horizon,
        "eval_split": args.split,
        "eval_dataset_segments": len(dataset),
        "hidden_belief_eval_version": 4,
        "change_threshold": float(args.change_threshold),
        "natural_hidden_eval_enabled": bool(args.natural_hidden_eval),
        "controlled_occlusion_eval_enabled": bool(
            args.controlled_occlusion_eval
        ),
        "controlled_occlusion_spans": [
            int(value) for value in args.controlled_occlusion_spans
        ],
        "anchor_gate_forced_zero": bool(
            getattr(memory_module, "force_gate_zero", False)
        ),
    }
    row.update(probe_info)
    row.update(
        action_space_audit(
            checkpoint=checkpoint,
            dataset=dataset,
            memory_module=memory_module,
        )
    )

    amp_enabled = bool(args.amp and device.type == "cuda")
    if args.natural_hidden_eval:
        metrics, metadata = natural_hidden_pass(
            model=model,
            memory_module=memory_module,
            probe=probe,
            dataset=dataset,
            args=args,
            device=device,
            amp_enabled=amp_enabled,
            rollout_window=rollout_window,
            rollout_horizon=rollout_horizon,
            r2_latent_normalize=r2_latent_normalize,
        )
        row.update(metrics)
        row.update(metadata)

    if args.controlled_occlusion_eval:
        metrics, metadata = controlled_occlusion_pass(
            model=model,
            memory_module=memory_module,
            probe=probe,
            dataset=dataset,
            args=args,
            device=device,
            amp_enabled=amp_enabled,
            rollout_window=rollout_window,
            rollout_horizon=rollout_horizon,
            r2_latent_normalize=r2_latent_normalize,
        )
        row.update(metrics)
        row.update(metadata)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def validate_args(args: argparse.Namespace) -> None:
    if args.eval_rollout_horizon < 1:
        raise SystemExit("--eval-rollout-horizon must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.natural_hidden_max_scan_batches < 1:
        raise SystemExit("--natural-hidden-max-scan-batches must be positive")
    if args.controlled_occlusion_max_batches < 1:
        raise SystemExit(
            "--controlled-occlusion-max-batches must be positive"
        )
    if args.natural_hidden_target_entity_times < 1:
        raise SystemExit(
            "--natural-hidden-target-entity-times must be positive"
        )
    if args.controlled_occlusion_target_entity_times < 1:
        raise SystemExit(
            "--controlled-occlusion-target-entity-times must be positive"
        )
    if any(int(value) < 1 for value in args.controlled_occlusion_spans):
        raise SystemExit(
            "--controlled-occlusion-spans must all be positive"
        )
    if args.change_threshold < 0:
        raise SystemExit("--change-threshold must be non-negative")
    if not args.natural_hidden_eval and not args.controlled_occlusion_eval:
        raise SystemExit(
            "At least one of natural or controlled hidden evaluation must be enabled"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = Path(args.probe_dir)
    if not probe_dir.is_dir():
        raise SystemExit(f"Probe directory does not exist: {probe_dir}")

    device = base.resolve_device(args.device)
    print(
        f"hidden_belief_eval device={device} "
        f"amp={bool(args.amp and device.type == 'cuda')} "
        f"checkpoints={len(args.checkpoint)}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    for raw_checkpoint in args.checkpoint:
        checkpoint_path = Path(raw_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        print(
            "=" * 72,
            "\nEvaluating hidden belief:",
            checkpoint_path,
            flush=True,
        )
        row = evaluate_checkpoint(checkpoint_path, args, device)
        rows.append(row)
        output_path = out_dir / (
            f"hidden_belief_{checkpoint_path.parent.name}_"
            f"{checkpoint_path.stem}.json"
        )
        output_path.write_text(json.dumps(row, indent=2, sort_keys=True))
        print(f"wrote {output_path}", flush=True)

    summary_jsonl = out_dir / "hidden_belief_summary.jsonl"
    with summary_jsonl.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary_csv = out_dir / "hidden_belief_summary.csv"
    write_csv(summary_csv, rows)
    print(f"wrote {summary_jsonl}", flush=True)
    print(f"wrote {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
