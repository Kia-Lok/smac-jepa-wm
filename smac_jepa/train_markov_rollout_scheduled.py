from __future__ import annotations

"""
Train SMAC-JEPA with Markov recursive rollout loss + optional TD-lambda/SIGReg schedules.

Parallel training script. It does not modify train.py.

Idea:
    Treat the latent itself as the recurrent state.
    Each predictor call gets only the latest latent and current action.

For rollout_window=p and rollout_horizon=n:
    sample a valid segment containing p+n transitions

For every start i in 0..p-1:
    z0 = encode(s_i)
    z1_hat = predictor(z0, a_i)
    z2_hat = predictor(z1_hat, a_{i+1})
    ...
    z_n_hat = predictor(z_{n-1}_hat, a_{i+n-1})

Loss:
    compare z_h_hat to encode(s_{i+h}) for h=1..n
    TD-lambda weighting over h emphasizes earlier rollout steps.

Vectorization:
    [B, p, ...] is flattened into [B*p, ...].
    Rollout uses n predictor calls per batch, not p*n calls.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from smac_jepa.data import SMACJEPADataset, load_manifest, load_manifest_all
from smac_jepa.jepa import SMACJEPA
from smac_jepa.data.markov_rollout_dataset import MarkovRolloutSMACJEPADataset
from smac_jepa.modules import sigreg_loss
from smac_jepa.presets import MODEL_PRESETS, get_model_preset
from smac_jepa.utils import set_seed
from smac_jepa.utils.logging import LossLogger
from smac_jepa.utils.plots import write_svg_line_plot

try:
    import wandb
except ImportError:
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SMAC-JEPA with Markov recursive rollout loss")

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
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--encoder-layers", type=int)
    parser.add_argument("--action-layers", type=int)
    parser.add_argument("--predictor-layers", type=int)
    parser.add_argument("--max-context-len", type=int, default=32)

    parser.add_argument("--rollout-window", type=int, default=20, help="Number of start points p per sampled segment.")
    parser.add_argument("--rollout-horizon", type=int, default=5, help="Number of recursive rollout steps n from each start point.")
    parser.add_argument("--window-mode", choices=["sequential", "random"], default="random")
    parser.add_argument("--samples-per-epoch", type=int, default=None)

    parser.add_argument("--temporal-loss", choices=["uniform", "lambda", "flat-decay"], default="lambda")
    parser.add_argument("--td-lambda", "--temporal-lambda", dest="td_lambda", type=float, default=0.9)
    parser.add_argument(
        "--td-lambda-start",
        type=float,
        default=None,
        help="Optional TD-lambda schedule start value. If omitted, use --td-lambda as a fixed value.",
    )
    parser.add_argument(
        "--td-lambda-end",
        type=float,
        default=None,
        help="Optional TD-lambda schedule end value. Defaults to --td-lambda when schedule is enabled.",
    )
    parser.add_argument(
        "--td-lambda-warmup-epochs",
        type=int,
        default=0,
        help="Linearly warm TD-lambda from start to end over this many epochs. 0 disables schedule.",
    )
    parser.add_argument("--flat-decay-start", type=int, default=None)
    parser.add_argument("--flat-decay-final-weight", type=float, default=0.5)
    parser.add_argument("--detach-rollout-targets", action="store_true")
    parser.add_argument("--unweighted-aux-losses", action="store_true")

    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument(
        "--sigreg-weight-start",
        type=float,
        default=None,
        help="Optional SIGReg weight schedule start value. If omitted, use --sigreg-weight as a fixed value.",
    )
    parser.add_argument(
        "--sigreg-weight-end",
        type=float,
        default=None,
        help="Optional SIGReg weight schedule end value. Defaults to --sigreg-weight when schedule is enabled.",
    )
    parser.add_argument(
        "--sigreg-warmup-epochs",
        type=int,
        default=0,
        help="Linearly warm SIGReg weight from start to end over this many epochs. 0 disables schedule.",
    )
    parser.add_argument("--decoder-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="SMAC-JEPA-losses")
    parser.add_argument("--wandb-entity", default="kialok-nus")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])

    return parser.parse_args()


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def resolved_arch_from_args(args: argparse.Namespace) -> dict[str, int | float]:
    preset = get_model_preset(args.model_size)
    return {
        "latent_dim": args.latent_dim or preset.latent_dim,
        "hidden_dim": args.hidden_dim or preset.hidden_dim,
        "action_dim": args.action_dim or preset.action_dim,
        "num_heads": args.num_heads or preset.num_heads,
        "encoder_layers": args.encoder_layers or preset.encoder_layers,
        "action_layers": args.action_layers or preset.action_layers,
        "predictor_layers": args.predictor_layers or preset.predictor_layers,
        "batch_size": args.batch_size or preset.batch_size,
        "lr": args.lr or preset.lr,
    }


def load_data_paths_from_args(args: argparse.Namespace) -> list[str]:
    if args.manifest is not None:
        return [str(path) for path in load_manifest(args.manifest, args.split)]

    if args.data_dir is None:
        raise SystemExit("Either --manifest or --data-dir must be provided.")

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.npz"))
    if len(files) < 2:
        raise SystemExit(f"Need at least 2 .npz files in {data_dir}, found {len(files)}.")

    rng = random.Random(args.seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    eval_count = max(1, round(len(files) * args.eval_fraction))
    eval_files = sorted(shuffled[:eval_count])
    train_files = sorted(shuffled[eval_count:])

    if args.split == "train":
        selected = train_files
    elif args.split in {"eval", "test"}:
        selected = eval_files
    else:
        raise SystemExit(f"Unknown split: {args.split}. Use train or eval.")

    print(
        f"Auto-split from {data_dir}: "
        f"total={len(files)} train={len(train_files)} eval={len(eval_files)} "
        f"using split={args.split}",
        flush=True,
    )
    return [str(path) for path in selected]

#Creates the weight tensor to multiply to scale the losses
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
    if length < 1:
        raise ValueError("length must be >= 1")
    if mode == "uniform":
        return torch.ones(length, device=device, dtype=dtype)
    if mode == "lambda":
        steps = torch.arange(length, device=device, dtype=dtype)
        return torch.as_tensor(td_lambda, device=device, dtype=dtype).pow(steps)
    if mode == "flat-decay":
        weights = torch.ones(length, device=device, dtype=dtype)
        if length == 1:
            return weights
        start = flat_decay_start if flat_decay_start is not None else max(1, length // 2)
        start = max(0, min(start, length))
        if start < length:
            weights[start:] = torch.linspace(
                1.0,
                float(flat_decay_final_weight),
                length - start,
                device=device,
                dtype=dtype,
            )
        return weights
    raise ValueError(f"Unknown temporal loss mode: {mode}")

#Computes rollout latent prediction loss and decoded reconstruction loss. Effectively MSE(predicted future latent, target future latent) but with TD-Lambda horizon weighing
def weighted_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    pred/target: [B, P, H, E, D]
    mask:        [B, P, H, E, 1]
    weights:     [H]
    """
    w = weights.view(1, 1, -1, 1, 1)
    weighted_mask = mask * w
    denom = weighted_mask.sum().clamp_min(1.0) * pred.shape[-1]
    return ((pred - target).pow(2) * weighted_mask).sum() / denom

#Computes presnece loss, which is binary
def weighted_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    logits/target: [B, P, H, E]
    mask:          [B, P, H, E]
    weights:       [H]
    """
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    w = weights.view(1, 1, -1, 1)
    weighted_mask = mask * w
    return (raw * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


def scheduled_value(
    *,
    epoch: int,
    fixed_value: float,
    start_value: float | None,
    end_value: float | None,
    warmup_epochs: int,
) -> float:
    """
    Linear schedule helper.

    If warmup_epochs <= 0 or start_value is None:
        returns fixed_value.

    Otherwise:
        epoch 1 starts at start_value.
        epoch >= warmup_epochs reaches end_value.

    Example:
        start=0.5, end=0.9, warmup_epochs=5

        epoch 1: 0.5
        epoch 2: 0.6
        epoch 3: 0.7
        epoch 4: 0.8
        epoch 5+: 0.9
    """
    if warmup_epochs <= 0 or start_value is None:
        return float(fixed_value)

    end = float(fixed_value if end_value is None else end_value)
    start = float(start_value)

    if warmup_epochs <= 1:
        return end

    progress = (epoch - 1) / float(warmup_epochs - 1)
    progress = max(0.0, min(1.0, progress))
    return start + progress * (end - start)

#core training objective. Takes one batch of sampled rollout segments (20), recursively predicts future latents and compare to target latent. Then compute decoded and presence loss before returning a dictionary of losses for backprop
def markov_rollout_losses(
    model: SMACJEPA,
    batch: dict[str, torch.Tensor],
    *,
    rollout_window: int,
    rollout_horizon: int,
    temporal_loss_mode: str,
    td_lambda: float,
    flat_decay_start: int | None,
    flat_decay_final_weight: float,
    sigreg_weight: float,
    decoder_weight: float,
    detach_rollout_targets: bool,
    unweighted_aux_losses: bool,
) -> dict[str, torch.Tensor]:
    
    entity_seq = batch["entity_seq"]
    entity_mask_seq = batch["entity_mask_seq"]
    action_seq = batch["action_seq"]
    action_mask_seq = batch["action_mask_seq"]
    state_mask = batch["state_mask"]
    static_condition = batch.get("static_condition")

    bsz = entity_seq.shape[0] #Batch size
    p = int(rollout_window) #Window
    h = int(rollout_horizon) #Horizon

    true_latents = model.encoder(entity_seq, entity_mask_seq)  # [B, P+H+1, E, D] #Converts all true entity states into latent space
    _, _, entities, latent_dim = true_latents.shape

    z = true_latents[:, :p].reshape(bsz * p, entities, latent_dim) #Initial rollout states

    static_flat = None
    if static_condition is not None:
        static_flat = static_condition[:, None, :].expand(bsz, p, -1).reshape(bsz * p, -1)

    #Collect rollout horizon
    pred_steps: list[torch.Tensor] = []
    target_steps: list[torch.Tensor] = []
    target_entity_steps: list[torch.Tensor] = []
    target_entity_mask_steps: list[torch.Tensor] = []
    slot_mask_steps: list[torch.Tensor] = []
    valid_steps: list[torch.Tensor] = []

    #For step within the horizon
    for step in range(h):
        action_h = action_seq[:, step : step + p]
        action_mask_h = action_mask_seq[:, step : step + p]
        valid_h = state_mask[:, step + 1 : step + p + 1]

        action_h = action_h.reshape(bsz * p, 1, *action_h.shape[2:])
        action_mask_h = action_mask_h.reshape(bsz * p, 1, action_mask_h.shape[-1])
        timestep_mask_h = torch.ones((bsz * p, 1), device=entity_seq.device, dtype=entity_seq.dtype)

        target_mask_h = entity_mask_seq[:, step + 1 : step + p + 1]
        entity_mask_h = target_mask_h.reshape(bsz * p, 1, entities)

        pred_h = model.predictor(
            z.unsqueeze(1),
            action_h,
            action_mask_h,
            timestep_mask_h,
            entity_mask_h,
            static_flat,
        )[:, 0]

        valid_entity_h = entity_mask_h[:, 0].unsqueeze(-1)
        pred_h = pred_h * valid_entity_h

        target_h = true_latents[:, step + 1 : step + p + 1]
        target_entity_h = entity_seq[:, step + 1 : step + p + 1]
        target_entity_mask_h = entity_mask_seq[:, step + 1 : step + p + 1]
        slot_mask_h = batch["entity_slot_mask_seq"][:, step + 1 : step + p + 1]

        pred_steps.append(pred_h.reshape(bsz, p, entities, latent_dim))
        target_steps.append(target_h)
        target_entity_steps.append(target_entity_h)
        target_entity_mask_steps.append(target_entity_mask_h)
        slot_mask_steps.append(slot_mask_h)
        valid_steps.append(valid_h)

        z = pred_h

    pred_latent = torch.stack(pred_steps, dim=2)
    target_latent = torch.stack(target_steps, dim=2)
    target_entity = torch.stack(target_entity_steps, dim=2)
    target_entity_mask = torch.stack(target_entity_mask_steps, dim=2)
    entity_slot_mask = torch.stack(slot_mask_steps, dim=2)
    valid_mask = torch.stack(valid_steps, dim=2)

    target_for_pred = target_latent.detach() if detach_rollout_targets else target_latent

    #Horizon weights creation
    weights = temporal_time_weights(
        h,
        mode=temporal_loss_mode,
        td_lambda=td_lambda,
        flat_decay_start=flat_decay_start,
        flat_decay_final_weight=flat_decay_final_weight,
        device=pred_latent.device,
        dtype=pred_latent.dtype,
    )
    uniform_weights = torch.ones_like(weights) #Pred loss uniform is assuming no td-lambda (idk why gpt want me log)
    mask = target_entity_mask.unsqueeze(-1) * valid_mask.unsqueeze(-1).unsqueeze(-1)

    pred_loss = weighted_mse(pred_latent, target_for_pred, mask, weights)
    pred_loss_uniform = weighted_mse(pred_latent, target_for_pred, mask, uniform_weights)

    decoded = model.decode_entities(pred_latent.reshape(bsz * p * h, entities, latent_dim))
    decoded = decoded.reshape(bsz, p, h, entities, -1)
    aux_weights = uniform_weights if unweighted_aux_losses else weights
    decoded_loss = weighted_mse(decoded, target_entity, mask, aux_weights)

    presence_logits = model.predict_presence(pred_latent.reshape(bsz * p * h, entities, latent_dim))
    presence_logits = presence_logits.reshape(bsz, p, h, entities)
    presence_mask = entity_slot_mask * valid_mask.unsqueeze(-1)
    presence_loss = weighted_bce(presence_logits, target_entity_mask, presence_mask, aux_weights)

    reg_loss = sigreg_loss(true_latents, entity_mask_seq)

    total_loss = pred_loss + sigreg_weight * reg_loss + decoder_weight * decoded_loss + presence_loss

    losses: dict[str, torch.Tensor] = {
        "total_loss": total_loss,
        "pred_loss": pred_loss,
        "pred_loss_uniform": pred_loss_uniform.detach(),
        "sigreg_loss": reg_loss,
        "decoded_loss": decoded_loss,
        "presence_loss": presence_loss,
        "temporal_weight_sum": weights.sum().detach(),
        "rollout_window": torch.tensor(float(p), device=pred_latent.device),
        "rollout_horizon": torch.tensor(float(h), device=pred_latent.device),
    }

    with torch.no_grad():
        for step in range(h):
            step_loss = weighted_mse(
                pred_latent[:, :, step : step + 1],
                target_for_pred[:, :, step : step + 1],
                mask[:, :, step : step + 1],
                torch.ones(1, device=pred_latent.device, dtype=pred_latent.dtype),
            )
            losses[f"pred_loss_h{step + 1}"] = step_loss.detach()

    return losses


def main() -> None:
    args = parse_args()
    arch = resolved_arch_from_args(args)

    if args.rollout_window < 1 or args.rollout_horizon < 1:
        raise SystemExit("--rollout-window and --rollout-horizon must be >= 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    data_paths = load_data_paths_from_args(args)
    cap_paths = load_manifest_all(args.manifest) if args.manifest is not None else data_paths

    cap_dataset = SMACJEPADataset(cap_paths, context_len=1, mode="entity")
    cap_metadata = cap_dataset.metadata

    dataset = MarkovRolloutSMACJEPADataset(
        data_paths,
        rollout_window=args.rollout_window,
        rollout_horizon=args.rollout_horizon,
        mode="entity",
        window_mode=args.window_mode,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
        max_agents=cap_metadata.max_agents,
        max_enemies=cap_metadata.max_enemies,
        max_actions=cap_metadata.max_actions,
        token_dim=cap_metadata.token_dim,
        dynamic_token_dim=cap_metadata.dynamic_token_dim,
        static_dim=cap_metadata.static_dim,
        entity_static_feat_size=cap_metadata.entity_static_feat_size,
    )

    loader = DataLoader(dataset, batch_size=int(arch["batch_size"]), shuffle=True, num_workers=args.num_workers)

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
        decoder_weight=args.decoder_weight,
        encoder_layers=int(arch["encoder_layers"]),
        action_layers=int(arch["action_layers"]),
        predictor_layers=int(arch["predictor_layers"]),
        max_context_len=args.max_context_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(arch["lr"]))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    global_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint and amp_enabled:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))

    saved_config = vars(args) | arch | {
        "resolved_device": device.type,
        "amp_enabled": amp_enabled,
        "dataset_len": len(dataset),
        "training_regime": "markov_rollout_scheduled",
        "segment_action_len": args.rollout_window + args.rollout_horizon,
        "segment_state_len": args.rollout_window + args.rollout_horizon + 1,
    }
    (out_dir / "config.json").write_text(json.dumps(saved_config, indent=2) + "\n")

    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise SystemExit("W&B logging requested with --wandb, but wandb is not installed. Install it with: uv pip install wandb")
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
        "markov_rollout "
        f"p={args.rollout_window} n={args.rollout_horizon} "
        f"temporal_loss={args.temporal_loss} td_lambda={args.td_lambda} "
        f"td_lambda_start={args.td_lambda_start} td_lambda_end={args.td_lambda_end} "
        f"td_lambda_warmup_epochs={args.td_lambda_warmup_epochs} "
        f"sigreg_weight={args.sigreg_weight} sigreg_weight_start={args.sigreg_weight_start} "
        f"sigreg_weight_end={args.sigreg_weight_end} sigreg_warmup_epochs={args.sigreg_warmup_epochs} "
        f"window_mode={args.window_mode} samples_per_epoch={args.samples_per_epoch}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        active_td_lambda = scheduled_value(
            epoch=epoch,
            fixed_value=args.td_lambda,
            start_value=args.td_lambda_start,
            end_value=args.td_lambda_end,
            warmup_epochs=args.td_lambda_warmup_epochs,
        )
        active_sigreg_weight = scheduled_value(
            epoch=epoch,
            fixed_value=args.sigreg_weight,
            start_value=args.sigreg_weight_start,
            end_value=args.sigreg_weight_end,
            warmup_epochs=args.sigreg_warmup_epochs,
        )

        print(
            f"epoch_schedule epoch={epoch} "
            f"active_td_lambda={active_td_lambda:.6f} "
            f"active_sigreg_weight={active_sigreg_weight:.6f}",
            flush=True,
        )

        epoch_sums: dict[str, float] = {}
        epoch_batches = 0

        for batch in loader:
            global_step += 1
            epoch_batches += 1
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            autocast_context = torch.cuda.amp.autocast(enabled=amp_enabled) if device.type == "cuda" else nullcontext()
            with autocast_context:
                losses = markov_rollout_losses(
                    model,
                    batch,
                    rollout_window=args.rollout_window,
                    rollout_horizon=args.rollout_horizon,
                    temporal_loss_mode=args.temporal_loss,
                    td_lambda=active_td_lambda,
                    flat_decay_start=args.flat_decay_start,
                    flat_decay_final_weight=args.flat_decay_final_weight,
                    sigreg_weight=active_sigreg_weight,
                    decoder_weight=args.decoder_weight,
                    detach_rollout_targets=args.detach_rollout_targets,
                    unweighted_aux_losses=args.unweighted_aux_losses,
                )

            scaler.scale(losses["total_loss"]).backward() #Problematic code for n-rollouts
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            row: dict[str, float | int] = {
                "epoch": epoch,
                "step": global_step,
                "active_td_lambda": active_td_lambda,
                "active_sigreg_weight": active_sigreg_weight,
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
                    "train/temporal_weight_sum": row.get("temporal_weight_sum"),
                    "train/active_td_lambda": row.get("active_td_lambda"),
                    "train/active_sigreg_weight": row.get("active_sigreg_weight"),
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

            if global_step == 1 or global_step % args.log_every == 0:
                print(
                    "epoch={epoch} step={step} total_loss={total_loss:.6f} "
                    "pred_loss={pred_loss:.6f} pred_uniform={pred_loss_uniform:.6f} "
                    "sigreg_loss={sigreg_loss:.6f} decoded_loss={decoded_loss:.6f} "
                    "presence_loss={presence_loss:.6f}".format(**row),
                    flush=True,
                )

        if epoch_batches == 0:
            raise RuntimeError(f"Epoch {epoch} finished with 0 batches; refusing to save checkpoint.")

        epoch_row: dict[str, float | int] = {"epoch": epoch, "step": global_step}
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
                "epoch/temporal_weight_sum": epoch_row.get("temporal_weight_sum"),
                "epoch/active_td_lambda": epoch_row.get("active_td_lambda"),
                "epoch/active_sigreg_weight": epoch_row.get("active_sigreg_weight"),
            }
            for key, value in epoch_row.items():
                if key.startswith("pred_loss_h"):
                    log_dict[f"epoch/{key}"] = value
            wandb_run.log(log_dict, step=global_step)

        print(
            "epoch_summary epoch={epoch} step={step} total_loss={total_loss:.6f} "
            "pred_loss={pred_loss:.6f} pred_uniform={pred_loss_uniform:.6f} "
            "sigreg_loss={sigreg_loss:.6f} decoded_loss={decoded_loss:.6f} "
            "presence_loss={presence_loss:.6f}".format(**epoch_row),
            flush=True,
        )

        epoch_checkpoint_path = out_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        save_checkpoint(epoch, epoch_checkpoint_path)
        save_checkpoint(epoch, out_dir / "checkpoint.pt")
        print(f"saved_checkpoint {epoch_checkpoint_path} and {out_dir / 'checkpoint.pt'}", flush=True)

        write_svg_line_plot(epoch_rows, "epoch", "total_loss", "Average Total Loss Per Epoch", out_dir / "loss_by_epoch.svg")
        write_svg_line_plot(epoch_rows, "epoch", "pred_loss", "Average Markov Rollout Prediction Loss Per Epoch", out_dir / "pred_loss_by_epoch.svg")
        write_svg_line_plot(step_rows, "step", "pred_loss", "Markov Rollout Prediction Loss Per Training Step", out_dir / "pred_loss_by_step.svg")

    if wandb_run is not None:
        wandb_run.save(str(out_dir / "config.json"))
        wandb_run.save(str(out_dir / "checkpoint.pt"))
        wandb_run.finish()


if __name__ == "__main__":
    main()
