from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from smac_jepa.data import SMACJEPADataset, load_manifest, load_manifest_all
from smac_jepa.data.markov_rollout_visibility_dataset import VisibilityMarkovRolloutSMACJEPADataset
from smac_jepa.jepa import SMACJEPA
from smac_jepa.modules.rollout_memory import EntityRolloutGRUMemory

try:
    from smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments import (
        ActionConditionedEntityRolloutGRUMemory,
        markov_rollout_rnn_losses,
    )
except Exception as exc:
    raise SystemExit(
        "Could not import smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments. "
        "First copy train_markov_rollout_rnn_visibility_seqmem_experiments.py into smac_jepa/. "
        f"Original error: {exc}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate selected RNN seqmem checkpoints with both latent losses and decoded/presence metrics."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-batches", type=int, default=None)

    parser.add_argument("--window-mode", default="sequential", choices=["sequential", "random"])
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--target-mode", choices=["full", "observed"], default=None)
    parser.add_argument("--enemy-visibility-mask", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enemy-sight-range", type=float, default=None)

    parser.add_argument("--sigreg-weight", type=float, default=None)
    parser.add_argument("--decoder-weight", type=float, default=None)
    parser.add_argument("--presence-weight", type=float, default=None)
    parser.add_argument("--one-step-weight", type=float, default=None)
    parser.add_argument("--td-lambda", type=float, default=None)

    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    return parser.parse_args()


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


def build_dataset(
    manifest: str,
    split: str,
    resolved_config: dict[str, Any],
    window_mode: str,
    samples_per_epoch: int | None,
    enemy_visibility_mask: bool | None,
    enemy_sight_range: float | None,
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
        rollout_horizon=int(resolved_config.get("rollout_horizon", 5)),
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


def build_memory_module(checkpoint: dict[str, Any], dataset: VisibilityMarkovRolloutSMACJEPADataset, device: torch.device) -> torch.nn.Module:
    cfg = get_config(checkpoint)
    latent_dim = int(cfg["latent_dim"])
    memory_dim = int(cfg.get("rollout_memory_dim", 128))
    hidden_dim = cfg.get("rollout_memory_hidden_dim", None)
    residual = not bool(cfg.get("rollout_memory_no_residual", False))
    action_conditioned = bool(cfg.get("action_conditioned_memory", False))

    if action_conditioned:
        # IMPORTANT:
        # Use the action dimension saved in the checkpoint first.
        # The eval dataset metadata may differ if the manifest/cap metadata is rebuilt
        # differently from the training run. Exp02/Exp04 checkpoints can otherwise fail
        # with action_proj size mismatch, e.g. checkpoint n_actions=198 vs eval dataset=188.
        metadata = checkpoint.get("metadata", {})
        n_actions = int(
            cfg.get(
                "n_actions",
                metadata.get("n_actions", dataset.metadata.n_actions),
            )
        )
        memory_module = ActionConditionedEntityRolloutGRUMemory(
            latent_dim=latent_dim,
            memory_dim=memory_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            residual=residual,
        ).to(device)
    else:
        memory_module = EntityRolloutGRUMemory(
            latent_dim=latent_dim,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            residual=residual,
        ).to(device)

    if "memory_module_state" not in checkpoint:
        raise RuntimeError("Checkpoint does not contain memory_module_state; not a valid seqmem RNN checkpoint.")
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
) -> dict[str, torch.Tensor]:
    entity_seq = batch["entity_seq"]
    entity_mask_seq = batch["entity_mask_seq"]
    action_seq = batch["action_seq"]
    action_mask_seq = batch["action_mask_seq"]
    state_mask = batch["state_mask"]
    static_condition = batch.get("static_condition")

    if target_mode == "observed":
        target_entity_seq = entity_seq
        target_entity_mask_seq = entity_mask_seq
    elif target_mode == "full":
        target_entity_seq = batch.get("target_entity_seq", entity_seq)
        target_entity_mask_seq = batch.get("target_entity_mask_seq", entity_mask_seq)
    else:
        raise ValueError(f"Unknown target_mode={target_mode}")

    bsz = entity_seq.shape[0]
    p = int(rollout_window)
    h = int(rollout_horizon)

    input_latents = model.encoder(entity_seq, entity_mask_seq)
    target_latents = model.encoder(target_entity_seq, target_entity_mask_seq)
    _, _, entities, latent_dim = input_latents.shape

    main_memory = memory_module.initial_memory(
        bsz,
        entities,
        device=entity_seq.device,
        dtype=input_latents.dtype,
    )

    pred_by_start = []
    target_latent_by_start = []
    target_entity_by_start = []
    target_mask_by_start = []
    slot_mask_by_start = []
    valid_by_start = []

    static_flat = static_condition if static_condition is not None else None

    for start_idx in range(p):
        z_start = input_latents[:, start_idx]
        start_entity_mask = entity_mask_seq[:, start_idx]

        rollout_memory = main_memory
        z = z_start
        current_entity_mask = start_entity_mask

        pred_steps = []
        target_latent_steps = []
        target_entity_steps = []
        target_mask_steps = []
        slot_mask_steps = []
        valid_steps = []

        for step in range(h):
            action_idx = start_idx + step
            target_idx = start_idx + step + 1

            action_h = action_seq[:, action_idx : action_idx + 1]
            action_mask_h = action_mask_seq[:, action_idx : action_idx + 1]
            valid_h = state_mask[:, target_idx]

            timestep_mask_h = torch.ones((bsz, 1), device=entity_seq.device, dtype=entity_seq.dtype)
            entity_mask_h = current_entity_mask.unsqueeze(1)

            z_conditioned = memory_module.condition(z, rollout_memory, current_entity_mask)
            pred_h = model.predictor(
                z_conditioned.unsqueeze(1),
                action_h,
                action_mask_h,
                timestep_mask_h,
                entity_mask_h,
                static_flat,
            )[:, 0]
            pred_h = pred_h * current_entity_mask.unsqueeze(-1)

            target_mask_h = target_entity_mask_seq[:, target_idx]

            pred_steps.append(pred_h)
            target_latent_steps.append(target_latents[:, target_idx])
            target_entity_steps.append(target_entity_seq[:, target_idx])
            target_mask_steps.append(target_mask_h)
            slot_mask_steps.append(batch["entity_slot_mask_seq"][:, target_idx])
            valid_steps.append(valid_h)

            if getattr(memory_module, "uses_action", False):
                rollout_memory = memory_module.update(
                    pred_h,
                    rollout_memory,
                    target_mask_h,
                    action=action_h[:, 0],
                    action_mask=action_mask_h[:, 0],
                )
            else:
                rollout_memory = memory_module.update(pred_h, rollout_memory, target_mask_h)

            z = pred_h
            current_entity_mask = target_mask_h

        pred_by_start.append(torch.stack(pred_steps, dim=1))
        target_latent_by_start.append(torch.stack(target_latent_steps, dim=1))
        target_entity_by_start.append(torch.stack(target_entity_steps, dim=1))
        target_mask_by_start.append(torch.stack(target_mask_steps, dim=1))
        slot_mask_by_start.append(torch.stack(slot_mask_steps, dim=1))
        valid_by_start.append(torch.stack(valid_steps, dim=1))

        real_action_h = action_seq[:, start_idx]
        real_action_mask_h = action_mask_seq[:, start_idx]
        if getattr(memory_module, "uses_action", False):
            main_memory = memory_module.update(
                z_start,
                main_memory,
                start_entity_mask,
                action=real_action_h,
                action_mask=real_action_mask_h,
            )
        else:
            main_memory = memory_module.update(z_start, main_memory, start_entity_mask)

    pred_latent = torch.stack(pred_by_start, dim=1)
    target_latent = torch.stack(target_latent_by_start, dim=1)
    target_entity = torch.stack(target_entity_by_start, dim=1)
    target_entity_mask = torch.stack(target_mask_by_start, dim=1)
    entity_slot_mask = torch.stack(slot_mask_by_start, dim=1)
    valid_mask = torch.stack(valid_by_start, dim=1)

    decoded = model.decode_entities(pred_latent.reshape(bsz * p * h, entities, latent_dim))
    decoded = decoded.reshape(bsz, p, h, entities, -1)

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


def compute_decoded_metrics(outputs: dict[str, torch.Tensor], thresholds: list[float], presence_threshold: float) -> dict[str, torch.Tensor]:
    pred_latent = outputs["pred_latent"]
    target_latent = outputs["target_latent"]
    decoded = outputs["decoded"]
    target_entity = outputs["target_entity"]
    presence_logits = outputs["presence_logits"]
    target_entity_mask = outputs["target_entity_mask"]
    entity_slot_mask = outputs["entity_slot_mask"]
    valid_mask = outputs["valid_mask"]

    entity_valid = target_entity_mask * valid_mask.unsqueeze(-1)
    entity_valid_feat = entity_valid.unsqueeze(-1)
    slot_valid = entity_slot_mask * valid_mask.unsqueeze(-1)

    metrics: dict[str, torch.Tensor] = {}
    metrics["latent_mse"] = mse(pred_latent, target_latent, entity_valid_feat)
    metrics["decoded_mse"] = mse(decoded, target_entity, entity_valid_feat)
    metrics["decoded_mae"] = mae(decoded, target_entity, entity_valid_feat)

    for threshold in thresholds:
        key = threshold_key(threshold)
        metrics[f"decoded_acc_{key}"] = threshold_acc(decoded, target_entity, entity_valid_feat, threshold)

    metrics.update(presence_metrics(presence_logits, target_entity_mask, slot_valid, presence_threshold))

    horizon = decoded.shape[2]
    for step in range(horizon):
        name = f"h{step + 1}"
        sl = slice(step, step + 1)
        step_entity_valid_feat = entity_valid[:, :, sl].unsqueeze(-1)
        step_slot_valid = slot_valid[:, :, sl]

        metrics[f"latent_mse_{name}"] = mse(pred_latent[:, :, sl], target_latent[:, :, sl], step_entity_valid_feat)
        metrics[f"decoded_mse_{name}"] = mse(decoded[:, :, sl], target_entity[:, :, sl], step_entity_valid_feat)
        metrics[f"decoded_mae_{name}"] = mae(decoded[:, :, sl], target_entity[:, :, sl], step_entity_valid_feat)

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
        for k, v in step_presence.items():
            metrics[f"{k}_{name}"] = v

    return metrics


@torch.no_grad()
def evaluate_one_checkpoint(checkpoint_path: Path, args: argparse.Namespace, device: torch.device, amp_enabled: bool) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = get_config(checkpoint)

    dataset = build_dataset(
        args.manifest,
        args.split,
        cfg,
        args.window_mode,
        args.samples_per_epoch,
        args.enemy_visibility_mask,
        args.enemy_sight_range,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(checkpoint, dataset, device)
    memory_module = build_memory_module(checkpoint, dataset, device)

    target_mode = str(args.target_mode or cfg.get("target_mode", "full"))
    sigreg_weight = float(args.sigreg_weight if args.sigreg_weight is not None else cfg.get("sigreg_weight", 0.01))
    decoder_weight = float(args.decoder_weight if args.decoder_weight is not None else cfg.get("decoder_weight", 0.01))
    presence_weight = float(args.presence_weight if args.presence_weight is not None else cfg.get("presence_weight", 0.01))
    one_step_weight = float(args.one_step_weight if args.one_step_weight is not None else cfg.get("one_step_weight", 0.0))
    td_lambda = float(args.td_lambda if args.td_lambda is not None else cfg.get("td_lambda", 0.9))
    rollout_window = int(cfg.get("rollout_window", 20))
    rollout_horizon = int(cfg.get("rollout_horizon", 5))

    sums: dict[str, float] = {}
    batches = 0
    windows = 0

    autocast_context = torch.cuda.amp.autocast(enabled=amp_enabled) if device.type == "cuda" else nullcontext()

    for batch in loader:
        if args.max_batches is not None and batches >= args.max_batches:
            break

        batch = to_device(batch, device)
        with autocast_context:
            losses = markov_rollout_rnn_losses(
                model,
                memory_module,
                batch,
                rollout_window=rollout_window,
                rollout_horizon=rollout_horizon,
                temporal_loss_mode=str(cfg.get("temporal_loss", "lambda")),
                td_lambda=td_lambda,
                flat_decay_start=cfg.get("flat_decay_start", None),
                flat_decay_final_weight=float(cfg.get("flat_decay_final_weight", 0.5)),
                sigreg_weight=sigreg_weight,
                decoder_weight=decoder_weight,
                presence_weight=presence_weight,
                one_step_weight=one_step_weight,
                target_mode=target_mode,
                detach_rollout_targets=bool(cfg.get("detach_rollout_targets", False)),
                unweighted_aux_losses=bool(cfg.get("unweighted_aux_losses", False)),
            )
            outputs = rollout_outputs(model, memory_module, batch, rollout_window, rollout_horizon, target_mode)
            decoded_metrics = compute_decoded_metrics(outputs, list(args.thresholds), float(args.presence_threshold))

        bsz = next(iter(batch.values())).shape[0]
        batches += 1
        windows += bsz

        for key, value in losses.items():
            sums[f"loss_{key}"] = sums.get(f"loss_{key}", 0.0) + float(value.detach().cpu()) * bsz
        for key, value in decoded_metrics.items():
            sums[f"metric_{key}"] = sums.get(f"metric_{key}", 0.0) + float(value.detach().cpu()) * bsz

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
        "action_conditioned_memory": bool(cfg.get("action_conditioned_memory", False)),
        "one_step_weight": one_step_weight,
        "sigreg_weight": sigreg_weight,
        "decoder_weight": decoder_weight,
        "presence_weight": presence_weight,
        "td_lambda": td_lambda,
        "enemy_visibility_mask": bool(dataset.enemy_visibility_mask),
        "enemy_sight_range": float(dataset.enemy_sight_range),
        "rollout_window": rollout_window,
        "rollout_horizon": rollout_horizon,
        "checkpoint_n_actions": int(checkpoint.get("metadata", {}).get("n_actions", -1)),
        "dataset_n_actions": int(dataset.metadata.n_actions),
    }

    for key, value in sums.items():
        row[f"eval_{key}"] = value / windows

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
            "eval_loss_pred_loss",
            "eval_loss_pred_loss_uniform",
            "eval_metric_decoded_mse",
            "eval_metric_decoded_mae",
            "eval_metric_decoded_acc_0p01",
            "eval_metric_decoded_acc_0p05",
            "eval_metric_decoded_acc_0p1",
            "eval_metric_presence_acc",
            "eval_metric_presence_f1",
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
