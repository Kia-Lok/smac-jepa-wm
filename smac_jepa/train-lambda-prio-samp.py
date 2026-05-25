from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

from smac_jepa.config import TrainConfig
from smac_jepa.data import SMACJEPADataset, load_manifest, load_manifest_all
from smac_jepa.jepa import SMACJEPA
from smac_jepa.presets import MODEL_PRESETS, get_model_preset
from smac_jepa.utils import set_seed
from smac_jepa.utils.logging import LossLogger
from smac_jepa.utils.plots import write_svg_line_plot

try:
    import wandb
except ImportError:
    wandb = None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train entity-token SMAC-JEPA with lambda-weighted temporal loss")
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
    parser.add_argument("--context-len", type=int, default=4)
    parser.add_argument("--window-mode", choices=["sequential", "random"], default="sequential")
    parser.add_argument("--window-len", type=int)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--encoder-layers", type=int)
    parser.add_argument("--action-layers", type=int)
    parser.add_argument("--predictor-layers", type=int)
    parser.add_argument("--max-context-len", type=int, default=32)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--decoder-weight", type=float, default=1.0)
    parser.add_argument(
        "--td-lambda",
        "--temporal-lambda",
        dest="td_lambda",
        type=float,
        default=0.9,
        help=(
            "Lambda for TD(lambda)-inspired temporal weighting over the "
            "prediction window. Horizon 0 has weight 1, horizon 1 has "
            "weight lambda, horizon 2 has lambda^2, etc. Default: 0.9"
        ),
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="SMAC-JEPA-losses", help="W&B project name")
    parser.add_argument("--wandb-entity", default="kialok-nus", help="W&B username or team/entity")
    parser.add_argument("--wandb-name", default=None, help="W&B run name")
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


def resolved_arch(config: TrainConfig) -> dict[str, int | float]:
    preset = get_model_preset(config.model_size)
    return {
        "latent_dim": config.latent_dim or preset.latent_dim,
        "hidden_dim": config.hidden_dim or preset.hidden_dim,
        "action_dim": config.action_dim or preset.action_dim,
        "num_heads": config.num_heads or preset.num_heads,
        "encoder_layers": config.encoder_layers or preset.encoder_layers,
        "action_layers": config.action_layers or preset.action_layers,
        "predictor_layers": config.predictor_layers or preset.predictor_layers,
        "batch_size": config.batch_size or preset.batch_size,
        "lr": config.lr or preset.lr,
    }

def load_data_paths_from_args(config: TrainConfig) -> list[str]:
    if config.manifest is not None:
        return load_manifest(config.manifest, config.split)

    if config.data_dir is None:
        raise SystemExit("Either --manifest or --data-dir must be provided.")

    data_dir = Path(config.data_dir)
    files = sorted(data_dir.glob("*.npz"))

    if len(files) < 2:
        raise SystemExit(f"Need at least 2 .npz files in {data_dir}, found {len(files)}.")

    rng = random.Random(config.seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    eval_count = max(1, round(len(files) * config.eval_fraction))
    eval_files = sorted(shuffled[:eval_count])
    train_files = sorted(shuffled[eval_count:])

    if config.split == "train":
        selected = train_files
    elif config.split in {"eval", "test"}:
        selected = eval_files
    else:
        raise SystemExit(f"Unknown split: {config.split}. Use train or eval.")

    print(
        f"Auto-split from {data_dir}: "
        f"total={len(files)} train={len(train_files)} eval={len(eval_files)} "
        f"using split={config.split}",
        flush=True,
    )

    return [str(path) for path in selected]


def _lambda_time_weights(
    length: int,
    td_lambda: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return [1, lambda, lambda^2, ...] for a prediction window."""
    if not 0.0 <= td_lambda <= 1.0:
        raise ValueError(f"--td-lambda must be in [0, 1], got {td_lambda}")
    steps = torch.arange(length, device=device, dtype=dtype)
    return torch.pow(torch.as_tensor(td_lambda, device=device, dtype=dtype), steps)


def _expand_like_time_weights(weights: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Shape [T] weights into [1, T, 1, ...] so they broadcast over target."""
    if target.ndim < 2:
        return torch.ones_like(target)
    return weights.view(1, weights.shape[0], *([1] * (target.ndim - 2)))


def _expand_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast a [B, T]-style mask over all non-time feature dimensions."""
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(-1)
    return mask.to(dtype=target.dtype)


def temporal_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    td_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Lambda-weighted window MSE.

    This keeps the same scale as a normal masked mean: when td_lambda=1.0,
    this reduces to the ordinary uniformly averaged prediction MSE. For
    td_lambda<1.0, timestep 0 has weight 1, timestep 1 has lambda, timestep 2
    has lambda^2, etc. This works for both sequential and random windows
    because the weighting is relative to the sampled window start.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes must match, got {pred.shape} vs {target.shape}")

    if pred.ndim < 2:
        # Degenerate case: no explicit window dimension.
        loss = (pred - target).pow(2).mean()
        one = torch.ones((), device=pred.device, dtype=pred.dtype)
        return loss, loss, one

    window_steps = pred.shape[1]
    weights = _lambda_time_weights(window_steps, td_lambda, pred.device, pred.dtype)
    weight_view = _expand_like_time_weights(weights, pred)
    squared = (pred - target).pow(2)

    if mask is None:
        weighted = squared * weight_view
        denom = weight_view.expand_as(squared).sum().clamp_min(1.0)
        lambda_loss = weighted.sum() / denom
        uniform_loss = squared.mean()
    else:
        mask_view = _expand_mask(mask, squared)
        weighted_mask = mask_view * weight_view
        lambda_loss = (squared * weighted_mask).sum() / weighted_mask.expand_as(squared).sum().clamp_min(1.0)
        uniform_loss = (squared * mask_view).sum() / mask_view.expand_as(squared).sum().clamp_min(1.0)

    return lambda_loss, uniform_loss, weights.sum()


def lambda_jepa_losses(
    model: SMACJEPA,
    batch: dict[str, torch.Tensor],
    sigreg_weight: float,
    td_lambda: float,
) -> dict[str, torch.Tensor]:
    """
    Compute a TD(lambda)-inspired temporal prediction loss while preserving any
    non-prediction losses already implemented by model.loss.

    The existing model.loss is still used for sigreg/decoder/etc. terms. We
    replace only pred_loss with the lambda-weighted version computed from
    model.forward(...)["pred_latent"], ["target_latent"], and optional ["mask"].
    """
    base_losses = model.loss(batch, sigreg_weight=sigreg_weight)
    out = model.forward(batch)

    if "pred_latent" not in out or "target_latent" not in out:
        raise KeyError(
            "model.forward(batch) must return 'pred_latent' and 'target_latent' "
            "to use train-lambda.py"
        )

    lambda_pred_loss, uniform_pred_loss, lambda_weight_sum = temporal_weighted_mse(
        out["pred_latent"],
        out["target_latent"],
        out.get("mask"),
        td_lambda,
    )

    # Preserve everything besides the original uniformly averaged pred_loss.
    # If model.loss contains decoded_loss, sigreg_loss, or future extra terms,
    # they remain part of total_loss with gradients.
    base_total = base_losses["total_loss"]
    base_pred = base_losses.get("pred_loss", uniform_pred_loss)
    extra_losses = base_total - base_pred
    total_loss = lambda_pred_loss + extra_losses

    losses = dict(base_losses)
    losses["total_loss"] = total_loss
    losses["pred_loss"] = lambda_pred_loss
    losses["pred_loss_uniform"] = uniform_pred_loss.detach()
    losses["lambda_weight_sum"] = lambda_weight_sum.detach()

    # Keep printing/logging robust for model versions without a decoder.
    if "decoded_loss" not in losses:
        losses["decoded_loss"] = torch.zeros((), device=total_loss.device, dtype=total_loss.dtype)

    return losses



def main() -> None:
    args = parse_args()

    wandb_enabled = args.wandb
    wandb_project = args.wandb_project
    wandb_entity = args.wandb_entity
    wandb_name = args.wandb_name
    wandb_mode = args.wandb_mode

    config_args = vars(args).copy()
    for key in ["wandb", "wandb_project", "wandb_entity", "wandb_name", "wandb_mode", "td_lambda"]:
        config_args.pop(key)

    config = TrainConfig(**config_args)
    arch = resolved_arch(config)
    window_len = config.window_len or config.context_len
    if window_len > config.max_context_len:
        raise SystemExit(
            f"window length {window_len} exceeds --max-context-len {config.max_context_len}"
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

    #Creates model using metadata from as well as optimizer and tracker
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
        "td_lambda": args.td_lambda,
    }
    (out_dir / "config.json").write_text(json.dumps(saved_config, indent=2) + "\n")
    wandb_run = None
    if wandb_enabled:
        if wandb is None:
            raise SystemExit(
                "W&B logging requested with --wandb, but wandb is not installed. "
                "Install it with: uv pip install wandb"
            )

        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name or out_dir.name,
            config=saved_config,
            mode=wandb_mode,
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
    #Training loop for specified number of epoch
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_sums: dict[str, float] = {}
        epoch_batches = 0
        for batch in loader:
            global_step += 1 #Hm okay so step is for batches specifically in an epoch.
            epoch_batches += 1
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.cuda.amp.autocast(enabled=amp_enabled)
                if device.type == "cuda"
                else nullcontext()
            )

            with autocast_context:
                losses = lambda_jepa_losses(
                    model,
                    batch,
                    sigreg_weight=config.sigreg_weight,
                    td_lambda=args.td_lambda,
                )
            scaler.scale(losses["total_loss"]).backward()
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            row = {
                "epoch": epoch,
                "step": global_step,
            }
            for key, value in losses.items():
                row[key] = float(value.detach().cpu())
            logger.log(row)
            step_rows.append(row) #step_row stores loss for each step
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/epoch": epoch,
                        "train/total_loss": row.get("total_loss"),
                        "train/pred_loss": row.get("pred_loss"),
                        "train/sigreg_loss": row.get("sigreg_loss"),
                        "train/decoded_loss": row.get("decoded_loss"),
                        "train/pred_loss_uniform": row.get("pred_loss_uniform"),
                        "train/lambda_weight_sum": row.get("lambda_weight_sum"),
                        "train/td_lambda": args.td_lambda,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )
            
            
            for key, value in row.items():
                if key in {"epoch", "step"}:
                    continue
                epoch_sums[key] = epoch_sums.get(key, 0.0) + float(value)
            if global_step == 1 or global_step % config.log_every == 0:
                print(
                    "epoch={epoch} step={step} total_loss={total_loss:.6f} "
                    "pred_loss={pred_loss:.6f} sigreg_loss={sigreg_loss:.6f} "
                    "decoded_loss={decoded_loss:.6f}".format(**row),
                    flush=True,
                )
        epoch_row = {
            "epoch": epoch,
            "step": global_step,
        }
        for key, value in epoch_sums.items():
            epoch_row[key] = value / max(epoch_batches, 1)
        epoch_logger.log(epoch_row)
        epoch_rows.append(epoch_row) #Stores loss for the entire epoch (After all batches of it ran)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch/epoch": epoch,
                    "epoch/total_loss": epoch_row.get("total_loss"),
                    "epoch/pred_loss": epoch_row.get("pred_loss"),
                    "epoch/sigreg_loss": epoch_row.get("sigreg_loss"),
                    "epoch/decoded_loss": epoch_row.get("decoded_loss"),
                    "epoch/pred_loss_uniform": epoch_row.get("pred_loss_uniform"),
                    "epoch/lambda_weight_sum": epoch_row.get("lambda_weight_sum"),
                    "epoch/td_lambda": args.td_lambda,
                },
                step=global_step,
            )
        
        print(
            "epoch_summary epoch={epoch} step={step} total_loss={total_loss:.6f} "
            "pred_loss={pred_loss:.6f} sigreg_loss={sigreg_loss:.6f} "
            "decoded_loss={decoded_loss:.6f}".format(**epoch_row),
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
        "Average Prediction Loss Per Epoch",
        out_dir / "pred_loss_by_epoch.svg",
    )
    write_svg_line_plot(
        step_rows,
        "step",
        "pred_loss",
        "Prediction Loss Per Training Step",
        out_dir / "pred_loss_by_step.svg",
    )
    print(
        "wrote_plots "
        f"{out_dir / 'loss_by_epoch.svg'} "
        f"{out_dir / 'pred_loss_by_epoch.svg'} "
        f"{out_dir / 'pred_loss_by_step.svg'}",
        flush=True,
    )

    # checkpoint.pt is already updated after every epoch.
    
    if wandb_run is not None:
        wandb_run.save(str(out_dir / "config.json"))
        wandb_run.save(str(out_dir / "checkpoint.pt"))
        wandb_run.finish()


if __name__ == "__main__":
    main()
    
