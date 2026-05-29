from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler

from smac_jepa.config import TrainConfig
from smac_jepa.data import SMACJEPADataset, load_manifest, load_manifest_all
from smac_jepa.jepa import SMACJEPA, entity_prediction_metrics
from smac_jepa.modules import sigreg_loss
from smac_jepa.presets import MODEL_PRESETS, get_model_preset
from smac_jepa.utils import set_seed
from smac_jepa.utils.logging import LossLogger
from smac_jepa.utils.plots import write_svg_line_plot

try:
    import wandb
except ImportError:
    wandb = None


class IndexedDataset(Dataset):
    """
    Thin wrapper around the existing dataset.

    The base SMACJEPADataset does not need to know about priority sampling.
    This wrapper adds a stable sample_index so we can update priority_scores[idx]
    after seeing the per-sample loss.
    """

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = dict(self.base[idx])
        item["sample_index"] = torch.tensor(idx, dtype=torch.long)
        return item

    @property
    def metadata(self):
        return self.base.metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SMAC-JEPA with sample-level priority replay"
    )

    # Same core args as train.py.
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
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)

    # New sample-level priority args.
    parser.add_argument("--sample-prioritized", action="store_true")
    parser.add_argument("--priority-alpha", type=float, default=0.6)
    parser.add_argument("--priority-uniform-mix", type=float, default=0.5)
    parser.add_argument("--priority-ema-beta", type=float, default=0.9)
    parser.add_argument("--priority-warmup-epochs", type=int, default=1)
    parser.add_argument("--priority-eps", type=float, default=1e-6)
    parser.add_argument("--priority-max-multiplier", type=float, default=10.0)
    parser.add_argument(
        "--priority-score",
        default="pred_loss",
        choices=["pred_loss", "decoded_loss", "total_loss"],
        help="Which per-sample loss updates the priority table.",
    )

    # W&B args.
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


def make_priority_weights(
    scores: torch.Tensor,
    *,
    alpha: float,
    uniform_mix: float,
    eps: float,
    max_multiplier: float,
) -> torch.Tensor:
    """
    Convert per-sample difficulty scores into sampler weights.

    Important:
    - We keep a uniform mixture so the model does not only replay hard samples.
    - We clamp normalized scores so outliers do not dominate.
    - WeightedRandomSampler only needs relative weights, but we return a
      probability-like vector for easier logging/debugging.
    """
    if scores.numel() == 0:
        raise ValueError("Empty priority score tensor.")

    safe = scores.detach().float().cpu()
    safe = torch.where(torch.isfinite(safe), safe, torch.ones_like(safe))
    safe = safe.clamp_min(eps)

    mean = safe.mean().clamp_min(eps)
    safe = (safe / mean).clamp(min=eps, max=max_multiplier)

    priority = safe.pow(alpha)
    priority_prob = priority / priority.sum().clamp_min(eps)

    n = safe.numel()
    uniform_prob = torch.full_like(priority_prob, 1.0 / n)

    mixed = uniform_mix * uniform_prob + (1.0 - uniform_mix) * priority_prob
    return mixed.clamp_min(eps)


def loss_with_per_sample_scores(
    model: SMACJEPA,
    batch: dict[str, torch.Tensor],
    *,
    sigreg_weight: float,
    decoder_weight: float,
    priority_score: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Same overall loss structure as SMACJEPA.loss(), but also returns a
    per-sample score for the priority table.

    This intentionally scores samples using prediction loss by default, because
    prediction difficulty is the cleanest signal for world-model learning.
    """
    out = model.forward(batch)

    mask = out["target_entity_mask"].unsqueeze(-1) * out["mask"].unsqueeze(-1).unsqueeze(-1)

    # Scalar pred loss, matching the model's existing reduction style.
    pred_denom = mask.sum().clamp_min(1.0) * out["pred_latent"].shape[-1]
    pred_loss = ((out["pred_latent"] - out["target_latent"]).pow(2) * mask).sum() / pred_denom

    # Per-sample pred loss for priority.
    per_sample_pred_num = ((out["pred_latent"] - out["target_latent"]).pow(2) * mask).sum(
        dim=tuple(range(1, mask.ndim))
    )
    per_sample_pred_den = mask.sum(dim=tuple(range(1, mask.ndim))).clamp_min(1.0) * out[
        "pred_latent"
    ].shape[-1]
    per_sample_pred_loss = per_sample_pred_num / per_sample_pred_den

    reg_loss = sigreg_loss(out["reg_latent"], out["reg_mask"])

    entity_denom = mask.sum().clamp_min(1.0) * out["target_entity"].shape[-1]
    decoded_loss = ((out["decoded_target"] - out["target_entity"]).pow(2) * mask).sum() / entity_denom

    per_sample_decoded_num = ((out["decoded_target"] - out["target_entity"]).pow(2) * mask).sum(
        dim=tuple(range(1, mask.ndim))
    )
    per_sample_decoded_den = mask.sum(dim=tuple(range(1, mask.ndim))).clamp_min(1.0) * out[
        "target_entity"
    ].shape[-1]
    per_sample_decoded_loss = per_sample_decoded_num / per_sample_decoded_den

    slot_mask = out["entity_slot_mask"]
    presence_target = out["target_entity_mask"]
    presence_loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
        out["presence_logits"],
        presence_target,
        reduction="none",
    )
    presence_loss = (presence_loss_raw * slot_mask).sum() / slot_mask.sum().clamp_min(1.0)

    per_sample_presence_num = (presence_loss_raw * slot_mask).sum(dim=tuple(range(1, slot_mask.ndim)))
    per_sample_presence_den = slot_mask.sum(dim=tuple(range(1, slot_mask.ndim))).clamp_min(1.0)
    per_sample_presence_loss = per_sample_presence_num / per_sample_presence_den

    total_loss = pred_loss + sigreg_weight * reg_loss + decoder_weight * decoded_loss + presence_loss

    # Per-sample total excludes sigreg because sigreg is batch-level/global.
    per_sample_total_loss = (
        per_sample_pred_loss
        + decoder_weight * per_sample_decoded_loss
        + per_sample_presence_loss
    )

    losses = {
        "total_loss": total_loss,
        "pred_loss": pred_loss,
        "sigreg_loss": reg_loss,
        "decoded_loss": decoded_loss,
        "presence_loss": presence_loss,
    }

    with torch.no_grad():
        losses.update(entity_prediction_metrics(out))

    if priority_score == "pred_loss":
        sample_scores = per_sample_pred_loss
    elif priority_score == "decoded_loss":
        sample_scores = per_sample_decoded_loss
    elif priority_score == "total_loss":
        sample_scores = per_sample_total_loss
    else:
        raise ValueError(f"Unknown priority_score: {priority_score}")

    return losses, sample_scores.detach()


def main() -> None:
    args = parse_args()

    priority_fields = {
        "sample_prioritized",
        "priority_alpha",
        "priority_uniform_mix",
        "priority_ema_beta",
        "priority_warmup_epochs",
        "priority_eps",
        "priority_max_multiplier",
        "priority_score",
    }
    wandb_fields = {
        "wandb",
        "wandb_project",
        "wandb_entity",
        "wandb_name",
        "wandb_mode",
    }

    config_args = vars(args).copy()
    for key in priority_fields | wandb_fields:
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

    base_dataset = SMACJEPADataset(
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
    dataset = IndexedDataset(base_dataset)

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

    priority_scores = torch.ones(len(dataset), dtype=torch.float32)
    priority_seen = torch.zeros(len(dataset), dtype=torch.bool)

    start_epoch = 1
    global_step = 0

    if config.resume:
        checkpoint = torch.load(config.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint and amp_enabled:
            scaler.load_state_dict(checkpoint["scaler_state"])

        if "priority_scores" in checkpoint and len(checkpoint["priority_scores"]) == len(dataset):
            priority_scores = checkpoint["priority_scores"].float().cpu()
            priority_seen = checkpoint.get(
                "priority_seen",
                torch.ones(len(dataset), dtype=torch.bool),
            ).bool().cpu()
            print("loaded_priority_scores_from_checkpoint", flush=True)
        else:
            print("priority_scores not found or wrong shape; starting fresh", flush=True)

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))

    saved_config = vars(args) | arch | {
        "context_len": window_len,
        "window_len": window_len,
        "resolved_device": device.type,
        "amp_enabled": amp_enabled,
        "dataset_len": len(dataset),
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
                "priority_scores": priority_scores,
                "priority_seen": priority_seen,
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
        "sample_priority "
        f"enabled={args.sample_prioritized} "
        f"alpha={args.priority_alpha} "
        f"uniform_mix={args.priority_uniform_mix} "
        f"ema_beta={args.priority_ema_beta} "
        f"warmup_epochs={args.priority_warmup_epochs} "
        f"score={args.priority_score}",
        flush=True,
    )

    for epoch in range(start_epoch, config.epochs + 1):
        use_priority = (
            args.sample_prioritized
            and epoch > args.priority_warmup_epochs
            and priority_seen.any().item()
        )

        if use_priority:
            weights = make_priority_weights(
                priority_scores,
                alpha=args.priority_alpha,
                uniform_mix=args.priority_uniform_mix,
                eps=args.priority_eps,
                max_multiplier=args.priority_max_multiplier,
            )
            sampler = WeightedRandomSampler(
                weights=weights.double(),
                num_samples=len(dataset),
                replacement=True,
            )
            shuffle = False
            sampler_mode = "priority"
        else:
            sampler = RandomSampler(dataset)
            shuffle = False
            weights = torch.full((len(dataset),), 1.0 / max(len(dataset), 1))
            sampler_mode = "uniform"

        loader = DataLoader(
            dataset,
            batch_size=int(arch["batch_size"]),
            sampler=sampler,
            shuffle=shuffle,
            num_workers=config.num_workers,
        )

        epoch_sums: dict[str, float] = {}
        epoch_batches = 0
        repeated_indices = 0
        sampled_indices_this_epoch: set[int] = set()

        for batch in loader:
            global_step += 1
            epoch_batches += 1

            batch = to_device(batch, device)
            sample_indices = batch["sample_index"].detach().cpu().long()

            for idx in sample_indices.tolist():
                if idx in sampled_indices_this_epoch:
                    repeated_indices += 1
                sampled_indices_this_epoch.add(idx)

            optimizer.zero_grad(set_to_none=True)

            autocast_context = (
                torch.cuda.amp.autocast(enabled=amp_enabled)
                if device.type == "cuda"
                else nullcontext()
            )

            with autocast_context:
                losses, sample_scores = loss_with_per_sample_scores(
                    model,
                    batch,
                    sigreg_weight=config.sigreg_weight,
                    decoder_weight=config.decoder_weight,
                    priority_score=args.priority_score,
                )

            scaler.scale(losses["total_loss"]).backward()

            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            # Update priority table after the batch.
            sample_scores_cpu = sample_scores.detach().float().cpu()
            old_scores = priority_scores[sample_indices]
            was_seen = priority_seen[sample_indices]

            updated = torch.where(
                was_seen,
                args.priority_ema_beta * old_scores
                + (1.0 - args.priority_ema_beta) * sample_scores_cpu,
                sample_scores_cpu,
            )

            priority_scores[sample_indices] = updated
            priority_seen[sample_indices] = True

            row: dict[str, float | int | str] = {
                "epoch": epoch,
                "step": global_step,
                "sampler_mode": sampler_mode,
            }

            for key, value in losses.items():
                row[key] = float(value.detach().cpu())

            row["priority_batch_score_mean"] = float(sample_scores_cpu.mean())
            row["priority_batch_score_max"] = float(sample_scores_cpu.max())

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
                        "train/priority_batch_score_mean": row.get("priority_batch_score_mean"),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

            for key, value in row.items():
                if key in {"epoch", "step", "sampler_mode"}:
                    continue
                epoch_sums[key] = epoch_sums.get(key, 0.0) + float(value)

            if global_step == 1 or global_step % config.log_every == 0:
                print(
                    "epoch={epoch} step={step} mode={sampler_mode} "
                    "total_loss={total_loss:.6f} pred_loss={pred_loss:.6f} "
                    "sigreg_loss={sigreg_loss:.6f} decoded_loss={decoded_loss:.6f} "
                    "priority_mean={priority_batch_score_mean:.6f}".format(**row),
                    flush=True,
                )

        seen_scores = priority_scores[priority_seen]
        if seen_scores.numel() == 0:
            priority_mean = 1.0
            priority_max = 1.0
            priority_seen_frac = 0.0
        else:
            priority_mean = float(seen_scores.mean())
            priority_max = float(seen_scores.max())
            priority_seen_frac = float(priority_seen.float().mean())

        epoch_row: dict[str, float | int | str] = {
            "epoch": epoch,
            "step": global_step,
            "sampler_mode": sampler_mode,
            "priority_score_mean": priority_mean,
            "priority_score_max": priority_max,
            "priority_seen_frac": priority_seen_frac,
            "priority_repeated_indices": repeated_indices,
            "priority_unique_indices": len(sampled_indices_this_epoch),
        }

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
                    "epoch/priority_score_mean": priority_mean,
                    "epoch/priority_score_max": priority_max,
                    "epoch/priority_seen_frac": priority_seen_frac,
                    "epoch/priority_unique_indices": len(sampled_indices_this_epoch),
                    "epoch/priority_repeated_indices": repeated_indices,
                },
                step=global_step,
            )

        print(
            "epoch_summary epoch={epoch} step={step} mode={sampler_mode} "
            "total_loss={total_loss:.6f} pred_loss={pred_loss:.6f} "
            "sigreg_loss={sigreg_loss:.6f} decoded_loss={decoded_loss:.6f} "
            "priority_score_mean={priority_score_mean:.6f} "
            "priority_score_max={priority_score_max:.6f} "
            "priority_seen_frac={priority_seen_frac:.3f} "
            "unique_indices={priority_unique_indices} repeated_indices={priority_repeated_indices}".format(
                **epoch_row
            ),
            flush=True,
        )

        priority_stats_path = out_dir / f"priority_stats_epoch_{epoch:03d}.json"
        priority_stats_path.write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "sampler_mode": sampler_mode,
                    "priority_score_mean": priority_mean,
                    "priority_score_max": priority_max,
                    "priority_seen_frac": priority_seen_frac,
                    "unique_indices": len(sampled_indices_this_epoch),
                    "repeated_indices": repeated_indices,
                    "dataset_len": len(dataset),
                },
                indent=2,
            )
            + "\n"
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
