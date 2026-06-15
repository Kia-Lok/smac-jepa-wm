from __future__ import annotations

"""
Train SMAC-JEPA with true prefix-rollout latent prediction loss.

This script is intentionally self-contained:
- no separate rollout_loss.py required
- no jepa.py patch required for the first implementation
In order not to mess up the folder more, I just self-contain any changes needed within the script (Requires debugging to ensure it works)

It directly uses the existing SMACJEPA internals:
    model.encoder
    model.predictor
    model.decode_entities
    model.predict_presence

Rollout idea:
    z0 = encode(s0)
    z1_hat = predictor([z0], [a0])[-1]
    z2_hat = predictor([z0, z1_hat], [a0, a1])[-1]
    z3_hat = predictor([z0, z1_hat, z2_hat], [a0, a1, a2])[-1]
    ...
Note that predictor has always been taking a sequence of states (This case predicted states vs ground truth in the rest of train scripts) 
Targets:
    target_latent[:, 0] = encode(s1)
    target_latent[:, 1] = encode(s2)
    target_latent[:, 2] = encode(s3)
    ...
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from smac_jepa.config import TrainConfig
from smac_jepa.data import SMACJEPADataset, load_manifest_all
from smac_jepa.jepa import SMACJEPA, entity_prediction_metrics
from smac_jepa.modules import sigreg_loss
from smac_jepa.presets import MODEL_PRESETS
from smac_jepa.train import load_data_paths_from_args, resolve_device, resolved_arch, to_device
from smac_jepa.utils import set_seed
from smac_jepa.utils.logging import LossLogger
from smac_jepa.utils.plots import write_svg_line_plot

try:
    import wandb
except ImportError:
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train entity-token SMAC-JEPA with recursive prefix-rollout temporal loss"
    )

    # Same core args as train.py / train_nstep_temporal.py.
    parser.add_argument("--manifest", default=None, help="Entity dataset split manifest")
    parser.add_argument("--data-dir", default=None, help="Directory containing .npz files to auto-split")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--split", default="train")

    parser.add_argument("--model-size", default="default", choices=sorted(MODEL_PRESETS))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)

    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--action-dim", type=int)
    parser.add_argument("--context-len", type=int, default=16)

    parser.add_argument("--window-mode", choices=["sequential", "random"], default="random")
    parser.add_argument("--window-len", type=int, default=16)
    parser.add_argument("--samples-per-epoch", type=int)

    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--encoder-layers", type=int)
    parser.add_argument("--action-layers", type=int)
    parser.add_argument("--predictor-layers", type=int)
    parser.add_argument("--max-context-len", type=int, default=32)

    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--decoder-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Prefix-rollout-specific args.
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=4,
        help=(
            "Number of recursive rollout steps to train on. "
            "Use a value <= window/context length. Start small, e.g. 2/4/8."
        ), #Aka how long the window for training. Right now bounded to context length since autoregressive requires sequence for conditioning variable. This raises a problem with conditioning variable in general actually and requires fixing from train.py first.
    )
    parser.add_argument(
        "--temporal-loss",
        choices=["uniform", "lambda", "flat-decay"],
        default="lambda",
        help="How to weight rollout horizons. Default: lambda",
    ) #Shouldn't be changed
    parser.add_argument(
        "--td-lambda",
        "--temporal-lambda",
        dest="td_lambda",
        type=float,
        default=0.9,
        help="Lambda for --temporal-loss lambda. Horizon h uses td_lambda**h. Default: 0.9",
    )
    parser.add_argument(
        "--flat-decay-start",
        type=int,
        default=None,
        help=(
            "For --temporal-loss flat-decay, number of initial horizons with weight 1. "
            "Default: half of rollout horizon."
        ),
    )
    parser.add_argument(
        "--flat-decay-final-weight",
        type=float,
        default=0.5,
        help="For --temporal-loss flat-decay, final horizon weight. Default: 0.5",
    ) #The floor of the weight (May want to set lower?)
    parser.add_argument(
        "--detach-rollout-targets",
        action="store_true",
        help=(
            "Detach target_latent in rollout pred loss. Default false to match current SMACJEPA.loss behavior."
        ),
    )
    parser.add_argument(
        "--unweighted-aux-losses",
        action="store_true",
        help=(
            "If set, decoded/presence losses are averaged uniformly over rollout horizons. "
            "By default they use the same temporal weights as pred_loss."
        ),
    )

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)

    # W&B args.
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="SMAC-JEPA-losses", help="W&B project name")
    parser.add_argument("--wandb-entity", default="kialok-nus", help="W&B username or team/entity")
    parser.add_argument("--wandb-name", default=None, help="W&B run name")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])

    return parser.parse_args()

#Just c
def temporal_time_weights(
    length: int,
    *,
    mode: str,
    td_lambda: float,
    flat_decay_start: int | None,
    flat_decay_final_weight: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return horizon weights of shape [length]."""
    if length < 1:
        raise ValueError("length must be at least 1")

    if mode == "uniform":
        return torch.ones(length, device=device, dtype=dtype)

    if mode == "lambda":
        steps = torch.arange(length, device=device, dtype=dtype)
        return torch.as_tensor(td_lambda, device=device, dtype=dtype).pow(steps)

    if mode == "flat-decay":
        weights = torch.ones(length, device=device, dtype=dtype)
        if length == 1:
            return weights

        start = flat_decay_start
        if start is None:
            start = max(1, length // 2)

        start = max(0, min(start, length))
        final = float(flat_decay_final_weight)

        if start < length:
            decay_len = length - start
            weights[start:] = torch.linspace(
                1.0,
                final,
                decay_len,
                device=device,
                dtype=dtype,
            )
        return weights

    raise ValueError(f"Unknown temporal loss mode: {mode}")


def _weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted masked MSE.

    pred/target: [B, H, E, D]
    mask:        [B, H, E, 1]
    weights:     [H]
    """
    weight_view = weights.view(1, -1, 1, 1)
    weighted_mask = mask * weight_view
    denom = weighted_mask.sum().clamp_min(1.0) * pred.shape[-1]
    return ((pred - target).pow(2) * weighted_mask).sum() / denom


def _weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted masked BCE.

    logits/target: [B, H, E]
    mask:          [B, H, E]
    weights:       [H]
    """
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight_view = weights.view(1, -1, 1)
    weighted_mask = mask * weight_view
    return (raw * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


def prefix_rollout_forward(
    model: SMACJEPA,
    batch: dict[str, torch.Tensor],
    *,
    rollout_horizon: int | None,
) -> dict[str, torch.Tensor]:
    """
    Perform recursive prefix rollout.

    Dataset alignment assumed:
        entity_t[:, h]       = s_h
        target_entity[:, h]  = s_{h+1}
        action_t[:, h]       = action from s_h to s_{h+1}

    Prefix rollout:
        z0_hat is real encode(s0)
        z1_hat = predictor([z0], [a0])[-1]
        z2_hat = predictor([z0, z1_hat], [a0, a1])[-1]
        ...
    """
    current_latent = model.encoder(batch["entity_t"], batch["entity_mask"])
    target_latent = model.encoder(batch["target_entity"], batch["target_entity_mask"])

    batch_size, steps, entities, latent_dim = current_latent.shape
    horizon = steps if rollout_horizon is None else min(int(rollout_horizon), steps)
    if horizon < 1:
        raise ValueError("rollout_horizon must be at least 1")

    # Start from the real encoded current state s0.
    latent_prefix = current_latent[:, :1]
    pred_latents: list[torch.Tensor] = []

    static_condition = batch.get("static_condition")

    for h in range(horizon):
        prefix_len = h + 1

        action_prefix = batch["action_t"][:, :prefix_len]
        action_mask_prefix = batch["action_mask"][:, :prefix_len]
        timestep_mask_prefix = batch["mask"][:, :prefix_len]

        # Prefix masks:
        #   latent_prefix[0] is real s0 -> use entity_mask[:, 0]
        #   later prefix latents are predicted z1..zh -> use target masks s1..sh
        if h == 0:
            entity_mask_prefix = batch["entity_mask"][:, :1]
        else:
            entity_mask_prefix = torch.cat(
                [
                    batch["entity_mask"][:, :1],
                    batch["target_entity_mask"][:, :h],
                ],
                dim=1,
            )

        pred_prefix = model.predictor(
            latent_prefix,
            action_prefix,
            action_mask_prefix,
            timestep_mask_prefix,
            entity_mask_prefix,
            static_condition,
        )

        # Final position predicts the next latent after the current prefix.
        z_next = pred_prefix[:, -1]

        # Mask invalid / padded target slots before feeding them into later prefixes.
        target_step_mask = (
            batch["target_entity_mask"][:, h].unsqueeze(-1)
            * batch["mask"][:, h].view(batch_size, 1, 1)
        )
        z_next = z_next * target_step_mask

        pred_latents.append(z_next)
        latent_prefix = torch.cat([latent_prefix, z_next.unsqueeze(1)], dim=1)

    rollout_pred_latent = torch.stack(pred_latents, dim=1)
    target_latent = target_latent[:, :horizon]

    target_entity = batch["target_entity"][:, :horizon]
    target_entity_mask = batch["target_entity_mask"][:, :horizon]
    timestep_mask = batch["mask"][:, :horizon]
    entity_slot_mask = batch.get("entity_slot_mask")
    if entity_slot_mask is None:
        entity_slot_mask = target_entity_mask * timestep_mask.unsqueeze(-1)
    else:
        entity_slot_mask = entity_slot_mask[:, :horizon]

    decoded_target = model.decode_entities(rollout_pred_latent)
    presence_logits = model.predict_presence(rollout_pred_latent)

    current_latent_mask = batch["entity_mask"][:, :horizon] * batch["mask"][:, :horizon].unsqueeze(-1)
    target_latent_mask = target_entity_mask * timestep_mask.unsqueeze(-1)

    return {
        "rollout_pred_latent": rollout_pred_latent,
        # Also expose as pred_latent so entity_prediction_metrics can work unchanged.
        "pred_latent": rollout_pred_latent,
        "target_latent": target_latent,
        "reg_latent": torch.cat([current_latent[:, :horizon], target_latent], dim=1),
        "reg_mask": torch.cat([current_latent_mask, target_latent_mask], dim=1),
        "decoded_target": decoded_target,
        "presence_logits": presence_logits,
        "target_entity": target_entity,
        "target_entity_mask": target_entity_mask,
        "entity_slot_mask": entity_slot_mask,
        "mask": timestep_mask,
        "current_entity_mask": batch["entity_mask"][:, :horizon],
        "active_rollout_horizon": horizon,
    }


def rollout_jepa_losses(
    model: SMACJEPA,
    batch: dict[str, torch.Tensor],
    *,
    sigreg_weight: float,
    decoder_weight: float,
    rollout_horizon: int | None,
    temporal_loss_mode: str,
    td_lambda: float,
    flat_decay_start: int | None,
    flat_decay_final_weight: float,
    detach_rollout_targets: bool,
    unweighted_aux_losses: bool,
) -> dict[str, torch.Tensor]:
    """Compute recursive prefix-rollout JEPA loss."""
    out = prefix_rollout_forward(model, batch, rollout_horizon=rollout_horizon)

    pred = out["rollout_pred_latent"]
    target = out["target_latent"].detach() if detach_rollout_targets else out["target_latent"]

    horizon = pred.shape[1]
    weights = temporal_time_weights(
        horizon,
        mode=temporal_loss_mode,
        td_lambda=td_lambda,
        flat_decay_start=flat_decay_start,
        flat_decay_final_weight=flat_decay_final_weight,
        device=pred.device,
        dtype=pred.dtype,
    )

    uniform_weights = torch.ones_like(weights)

    mask = out["target_entity_mask"].unsqueeze(-1) * out["mask"].unsqueeze(-1).unsqueeze(-1)

    pred_loss = _weighted_mse(pred, target, mask, weights)
    pred_loss_uniform = _weighted_mse(pred, target, mask, uniform_weights)

    decoded_weights = uniform_weights if unweighted_aux_losses else weights
    decoded_loss = _weighted_mse(
        out["decoded_target"],
        out["target_entity"],
        mask,
        decoded_weights,
    )

    presence_weights = uniform_weights if unweighted_aux_losses else weights
    presence_loss = _weighted_bce_with_logits(
        out["presence_logits"],
        out["target_entity_mask"],
        out["entity_slot_mask"],
        presence_weights,
    )

    reg_loss = sigreg_loss(out["reg_latent"], out["reg_mask"])

    total_loss = (
        pred_loss
        + sigreg_weight * reg_loss
        + decoder_weight * decoded_loss
        + presence_loss
    )

    losses: dict[str, torch.Tensor] = {
        "total_loss": total_loss,
        "pred_loss": pred_loss,
        "pred_loss_uniform": pred_loss_uniform.detach(),
        "sigreg_loss": reg_loss,
        "decoded_loss": decoded_loss,
        "presence_loss": presence_loss,
        "temporal_weight_sum": weights.sum().detach(),
        "active_rollout_horizon": torch.tensor(
            horizon,
            device=pred.device,
            dtype=pred.dtype,
        ),
    }

    # Per-horizon diagnostic losses. These are detached logging values.
    with torch.no_grad():
        for h in range(horizon):
            h_mask = mask[:, h : h + 1]
            h_loss = _weighted_mse(
                pred[:, h : h + 1],
                target[:, h : h + 1],
                h_mask,
                torch.ones(1, device=pred.device, dtype=pred.dtype),
            )
            losses[f"pred_loss_h{h + 1}"] = h_loss.detach()

        # Decode/presence metrics use rollout predictions.
        losses.update(entity_prediction_metrics(out))

    return losses


def main() -> None:
    args = parse_args()

    wandb_fields = {
        "wandb",
        "wandb_project",
        "wandb_entity",
        "wandb_name",
        "wandb_mode",
    }
    rollout_fields = {
        "rollout_horizon",
        "temporal_loss",
        "td_lambda",
        "flat_decay_start",
        "flat_decay_final_weight",
        "detach_rollout_targets",
        "unweighted_aux_losses",
    }

    config_args = vars(args).copy()
    for key in wandb_fields | rollout_fields:
        config_args.pop(key)

    config = TrainConfig(**config_args)
    arch = resolved_arch(config)

    window_len = config.window_len or config.context_len
    if window_len > config.max_context_len:
        raise SystemExit(
            f"window length {window_len} exceeds --max-context-len {config.max_context_len}"
        )
    if args.rollout_horizon > window_len:
        raise SystemExit(
            f"--rollout-horizon {args.rollout_horizon} exceeds window length {window_len}. "
            "Use a smaller rollout horizon or increase --context-len/--window-len."
        )

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = resolve_device(config.device)
    amp_enabled = bool(config.amp and device.type == "cuda")

    data_paths = load_data_paths_from_args(config)
    cap_paths = load_manifest_all(config.manifest) if config.manifest is not None else data_paths

    cap_dataset = SMACJEPADataset(cap_paths, context_len=1, mode="entity")
    cap_metadata = cap_dataset.metadata

    dataset = SMACJEPADataset(
        data_paths,
        context_len=config.context_len,
        mode="entity",
        window_mode=config.window_mode,
        window_len=window_len,
        samples_per_epoch=config.samples_per_epoch,
        seed=config.seed,
        max_agents=cap_metadata.max_agents,
        max_enemies=cap_metadata.max_enemies,
        max_actions=cap_metadata.max_actions,
        token_dim=cap_metadata.token_dim,
        dynamic_token_dim=cap_metadata.dynamic_token_dim,
        static_dim=cap_metadata.static_dim,
        entity_static_feat_size=cap_metadata.entity_static_feat_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(arch["batch_size"]),
        shuffle=True,
        num_workers=config.num_workers,
    )

    model = SMACJEPA(
        state_dim=dataset.metadata.state_dim,
        n_agents=dataset.metadata.n_agents,
        n_actions=dataset.metadata.n_actions,
        latent_dim=int(arch["latent_dim"]),
        hidden_dim=int(arch["hidden_dim"]),
        action_dim=int(arch["action_dim"]),
        num_heads=int(arch["num_heads"]),
        mode=dataset.metadata.mode,
        max_agents=dataset.metadata.max_agents,
        max_enemies=dataset.metadata.max_enemies,
        max_actions=dataset.metadata.max_actions,
        token_dim=dataset.metadata.token_dim,
        static_dim=dataset.metadata.static_dim,
        decoder_weight=config.decoder_weight,
        encoder_layers=int(arch["encoder_layers"]),
        action_layers=int(arch["action_layers"]),
        predictor_layers=int(arch["predictor_layers"]),
        max_context_len=config.max_context_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(arch["lr"]))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    global_step = 0

    if config.resume:
        checkpoint = torch.load(config.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint and amp_enabled:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))

    saved_config = vars(args) | arch | {
        "context_len": window_len,
        "window_len": window_len,
        "resolved_device": device.type,
        "amp_enabled": amp_enabled,
        "dataset_len": len(dataset),
        "training_regime": "prefix_rollout_temporal",
    }
    (out_dir / "config.json").write_text(json.dumps(saved_config, indent=2) + "\n")

    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise SystemExit(
                "W&B logging requested with --wandb, but wandb is not installed. "
                "Install it with: uv pip install wandb"
            )
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or out_dir.name,
            config=saved_config,
            mode=args.wandb_mode,
            dir=str(out_dir),
        )
        wandb_run.watch(model, log=None)

    def save_checkpoint(epoch_to_save: int, checkpoint_path: Path) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "metadata": {
                    "state_dim": dataset.metadata.state_dim,
                    "n_agents": dataset.metadata.n_agents,
                    "n_actions": dataset.metadata.n_actions,
                    "n_enemies": dataset.metadata.n_enemies,
                    "ally_state_feat_size": dataset.metadata.ally_state_feat_size,
                    "enemy_state_feat_size": dataset.metadata.enemy_state_feat_size,
                    "ally_has_shields": dataset.metadata.ally_has_shields,
                    "enemy_has_shields": dataset.metadata.enemy_has_shields,
                    "num_unit_types": dataset.metadata.num_unit_types,
                    "max_agents": dataset.metadata.max_agents,
                    "max_enemies": dataset.metadata.max_enemies,
                    "max_actions": dataset.metadata.max_actions,
                    "token_dim": dataset.metadata.token_dim,
                    "dynamic_token_dim": dataset.metadata.dynamic_token_dim,
                    "static_dim": dataset.metadata.static_dim,
                    "entity_static_feat_size": dataset.metadata.entity_static_feat_size,
                    "mode": dataset.metadata.mode,
                },
                "config": vars(args),
                "resolved_config": saved_config,
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": epoch_to_save,
                "global_step": global_step,
            },
            checkpoint_path,
        )

    logger = LossLogger(out_dir, "loss_log")
    epoch_logger = LossLogger(out_dir, "epoch_loss")

    step_rows: list[dict[str, float | int]] = []
    epoch_rows: list[dict[str, float | int]] = []

    model.train()

    print(
        "prefix_rollout "
        f"horizon={args.rollout_horizon} "
        f"temporal_loss={args.temporal_loss} "
        f"td_lambda={args.td_lambda} "
        f"window_len={window_len} "
        f"window_mode={config.window_mode} "
        f"samples_per_epoch={config.samples_per_epoch}",
        flush=True,
    )

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_sums: dict[str, float] = {}
        epoch_batches = 0

        for batch in loader:
            global_step += 1
            epoch_batches += 1

            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            autocast_context = (
                torch.cuda.amp.autocast(enabled=amp_enabled)
                if device.type == "cuda"
                else nullcontext()
            )

            with autocast_context:
                losses = rollout_jepa_losses(
                    model,
                    batch,
                    sigreg_weight=config.sigreg_weight,
                    decoder_weight=config.decoder_weight,
                    rollout_horizon=args.rollout_horizon,
                    temporal_loss_mode=args.temporal_loss,
                    td_lambda=args.td_lambda,
                    flat_decay_start=args.flat_decay_start,
                    flat_decay_final_weight=args.flat_decay_final_weight,
                    detach_rollout_targets=args.detach_rollout_targets,
                    unweighted_aux_losses=args.unweighted_aux_losses,
                )

            scaler.scale(losses["total_loss"]).backward()

            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            row: dict[str, float | int] = {
                "epoch": epoch,
                "step": global_step,
            }
            for key, value in losses.items():
                row[key] = float(value.detach().cpu())

            logger.log(row)
            step_rows.append(row)

            if wandb_run is not None:
                log_dict = {
                    "train/epoch": epoch,
                    "train/total_loss": row.get("total_loss"),
                    "train/pred_loss": row.get("pred_loss"),
                    "train/pred_loss_uniform": row.get("pred_loss_uniform"),
                    "train/sigreg_loss": row.get("sigreg_loss"),
                    "train/decoded_loss": row.get("decoded_loss"),
                    "train/presence_loss": row.get("presence_loss"),
                    "train/active_rollout_horizon": row.get("active_rollout_horizon"),
                    "train/temporal_weight_sum": row.get("temporal_weight_sum"),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                for key, value in row.items():
                    if key.startswith("pred_loss_h"):
                        log_dict[f"train/{key}"] = value
                wandb_run.log(log_dict, step=global_step)

            for key, value in row.items():
                if key in {"epoch", "step"}:
                    continue
                epoch_sums[key] = epoch_sums.get(key, 0.0) + float(value)

            if global_step == 1 or global_step % config.log_every == 0:
                print(
                    "epoch={epoch} step={step} "
                    "total_loss={total_loss:.6f} pred_loss={pred_loss:.6f} "
                    "pred_uniform={pred_loss_uniform:.6f} sigreg_loss={sigreg_loss:.6f} "
                    "decoded_loss={decoded_loss:.6f} presence_loss={presence_loss:.6f}".format(
                        **row
                    ),
                    flush=True,
                )

        if epoch_batches == 0:
            raise RuntimeError(
                f"Epoch {epoch} finished with 0 batches; refusing to save a misleading checkpoint."
            )

        epoch_row: dict[str, float | int] = {
            "epoch": epoch,
            "step": global_step,
        }
        for key, value in epoch_sums.items():
            epoch_row[key] = value / max(epoch_batches, 1)

        epoch_logger.log(epoch_row)
        epoch_rows.append(epoch_row)

        if wandb_run is not None:
            log_dict = {
                "epoch/epoch": epoch,
                "epoch/total_loss": epoch_row.get("total_loss"),
                "epoch/pred_loss": epoch_row.get("pred_loss"),
                "epoch/pred_loss_uniform": epoch_row.get("pred_loss_uniform"),
                "epoch/sigreg_loss": epoch_row.get("sigreg_loss"),
                "epoch/decoded_loss": epoch_row.get("decoded_loss"),
                "epoch/presence_loss": epoch_row.get("presence_loss"),
                "epoch/active_rollout_horizon": epoch_row.get("active_rollout_horizon"),
                "epoch/temporal_weight_sum": epoch_row.get("temporal_weight_sum"),
            }
            for key, value in epoch_row.items():
                if key.startswith("pred_loss_h"):
                    log_dict[f"epoch/{key}"] = value
            wandb_run.log(log_dict, step=global_step)

        print(
            "epoch_summary epoch={epoch} step={step} "
            "total_loss={total_loss:.6f} pred_loss={pred_loss:.6f} "
            "pred_uniform={pred_loss_uniform:.6f} sigreg_loss={sigreg_loss:.6f} "
            "decoded_loss={decoded_loss:.6f} presence_loss={presence_loss:.6f}".format(
                **epoch_row
            ),
            flush=True,
        )

        epoch_checkpoint_path = out_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        save_checkpoint(epoch, epoch_checkpoint_path)
        save_checkpoint(epoch, out_dir / "checkpoint.pt")

        print(
            f"saved_checkpoint {epoch_checkpoint_path} and {out_dir / 'checkpoint.pt'}",
            flush=True,
        )

        write_svg_line_plot(
            epoch_rows,
            "epoch",
            "total_loss",
            "Average Total Loss Per Epoch",
            out_dir / "loss_by_epoch.svg",
        )
        write_svg_line_plot(
            epoch_rows,
            "epoch",
            "pred_loss",
            "Average Prefix-Rollout Prediction Loss Per Epoch",
            out_dir / "pred_loss_by_epoch.svg",
        )
        write_svg_line_plot(
            step_rows,
            "step",
            "pred_loss",
            "Prefix-Rollout Prediction Loss Per Training Step",
            out_dir / "pred_loss_by_step.svg",
        )

        print(
            "wrote_plots "
            f"{out_dir / 'loss_by_epoch.svg'} "
            f"{out_dir / 'pred_loss_by_epoch.svg'} "
            f"{out_dir / 'pred_loss_by_step.svg'}",
            flush=True,
        )

    if wandb_run is not None:
        wandb_run.save(str(out_dir / "config.json"))
        wandb_run.save(str(out_dir / "checkpoint.pt"))
        wandb_run.finish()


if __name__ == "__main__":
    main()
