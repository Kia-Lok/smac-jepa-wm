from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from smac_jepa.config import TrainConfig
from smac_jepa.data import SMACJEPADataset, load_manifest_all
from smac_jepa.jepa import SMACJEPA
from smac_jepa.presets import MODEL_PRESETS
from smac_jepa.temporal_loss import temporal_jepa_losses
from smac_jepa.train import load_data_paths_from_args, resolve_device, resolved_arch, to_device
from smac_jepa.utils import set_seed
from smac_jepa.utils.logging import LossLogger
from smac_jepa.utils.plots import write_svg_line_plot

try:
    import wandb
except ImportError:
    wandb = None
"""
GENERAL IDEA
n-step temporal prediction (Note: only affects prediction loss). All aditional args from train.py directly affects how pred loss is computed. Otherwise, no change from train.py. Meant to see if multistep prediction will help. Question is how many steps (Should ideally be >= 20 if possible)
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train entity-token SMAC-JEPA with clean n-step temporal prediction loss."
    )
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
        help="Lambda for --temporal-loss lambda. Horizon h uses td_lambda**h. Default: 0.9",
    )
    parser.add_argument(
        "--temporal-loss", #How much each prediction horizon contributes
        choices=["uniform", "lambda", "flat-decay"],
        default="uniform",
        help="How to weight prediction horizons inside the n-step window. Default: uniform",
    )
    parser.add_argument(
        "--loss-horizon",
        type=int,
        default=None,
        help="Optional number of leading prediction horizons to include. Defaults to the full window.",
    )
    parser.add_argument(
        "--flat-decay-start",
        type=int,
        default=None,
        help="For --temporal-loss flat-decay, number of initial horizons with weight 1.",
    )
    parser.add_argument(
        "--flat-decay-final-weight",
        type=float,
        default=0.5,
        help="For --temporal-loss flat-decay, final horizon weight. Default: 0.5",
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


def main() -> None:
    args = parse_args()

    wandb_enabled = args.wandb
    wandb_project = args.wandb_project
    wandb_entity = args.wandb_entity
    wandb_name = args.wandb_name
    wandb_mode = args.wandb_mode

    config_args = vars(args).copy()
    for key in [
        "wandb",
        "wandb_project",
        "wandb_entity",
        "wandb_name",
        "wandb_mode",
        "td_lambda",
        "temporal_loss",
        "loss_horizon",
        "flat_decay_start",
        "flat_decay_final_weight",
    ]:
        config_args.pop(key)

    config = TrainConfig(**config_args)
    arch = resolved_arch(config)
    window_len = config.window_len or config.context_len
    if window_len > config.max_context_len:
        raise SystemExit(f"window length {window_len} exceeds --max-context-len {config.max_context_len}")

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
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_sums: dict[str, float] = {}
        epoch_batches = 0

        for batch in loader:
            global_step += 1
            epoch_batches += 1
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            autocast_context = torch.cuda.amp.autocast(enabled=amp_enabled) if device.type == "cuda" else nullcontext()

            with autocast_context:
                losses = temporal_jepa_losses(
                    model,
                    batch,
                    sigreg_weight=config.sigreg_weight,
                    temporal_loss_mode=args.temporal_loss,
                    td_lambda=args.td_lambda,
                    loss_horizon=args.loss_horizon,
                    flat_decay_start=args.flat_decay_start,
                    flat_decay_final_weight=args.flat_decay_final_weight,
                )

            scaler.scale(losses["total_loss"]).backward()
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            row: dict[str, float | int] = {"epoch": epoch, "step": global_step}
            for key, value in losses.items():
                if key == "pred_loss_per_sample":
                    continue
                row[key] = float(value.detach().cpu())
            logger.log(row)
            step_rows.append(row)

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/epoch": epoch,
                        "train/total_loss": row.get("total_loss"),
                        "train/pred_loss": row.get("pred_loss"),
                        "train/sigreg_loss": row.get("sigreg_loss"),
                        "train/decoded_loss": row.get("decoded_loss"),
                        "train/presence_loss": row.get("presence_loss"),
                        "train/pred_loss_uniform": row.get("pred_loss_uniform"),
                        "train/temporal_weight_sum": row.get("temporal_weight_sum"),
                        "train/active_loss_horizon": row.get("active_loss_horizon"),
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

        if epoch_batches == 0:
            raise RuntimeError(f"Epoch {epoch} finished with 0 batches; refusing to save a misleading checkpoint.")

        epoch_row: dict[str, float | int] = {"epoch": epoch, "step": global_step}
        for key, value in epoch_sums.items():
            epoch_row[key] = value / max(epoch_batches, 1)
        epoch_logger.log(epoch_row)
        epoch_rows.append(epoch_row)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch/epoch": epoch,
                    "epoch/total_loss": epoch_row.get("total_loss"),
                    "epoch/pred_loss": epoch_row.get("pred_loss"),
                    "epoch/sigreg_loss": epoch_row.get("sigreg_loss"),
                    "epoch/decoded_loss": epoch_row.get("decoded_loss"),
                    "epoch/presence_loss": epoch_row.get("presence_loss"),
                    "epoch/pred_loss_uniform": epoch_row.get("pred_loss_uniform"),
                    "epoch/temporal_weight_sum": epoch_row.get("temporal_weight_sum"),
                    "epoch/active_loss_horizon": epoch_row.get("active_loss_horizon"),
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

    if wandb_run is not None:
        wandb_run.save(str(out_dir / "config.json"))
        wandb_run.save(str(out_dir / "checkpoint.pt"))
        wandb_run.finish()


if __name__ == "__main__":
    main()
