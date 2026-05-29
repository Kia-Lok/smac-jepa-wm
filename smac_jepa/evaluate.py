from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from smac_jepa.data import SMACJEPADataset, load_manifest
from smac_jepa.decoder import format_entity_predictions
from smac_jepa.jepa import SMACJEPA, entity_prediction_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate entity-token SMAC-JEPA")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-decode-samples", type=int, default=8)
    parser.add_argument("--decode-sample-out")
    parser.add_argument("--context-len", type=int, help="Override eval window length")
    parser.add_argument("--rollout-horizons", default="", help="Comma-separated rollout horizons")
    parser.add_argument("--per-config-out", help="Optional JSON path for per-dataset metrics")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("resolved_config", checkpoint["config"])
    metadata = checkpoint["metadata"]
    data_paths = load_manifest(args.manifest, args.split)
    rollout_horizons = parse_horizons(args.rollout_horizons)
    eval_context_len = int(args.context_len or config["context_len"])
    if rollout_horizons:
        eval_context_len = max(eval_context_len, max(rollout_horizons))
    if eval_context_len > int(config.get("max_context_len", 32)):
        raise SystemExit(
            f"eval context length {eval_context_len} exceeds checkpoint max_context_len={config.get('max_context_len', 32)}"
        )
    dataset = SMACJEPADataset(
        data_paths,
        context_len=eval_context_len,
        mode="entity",
        max_agents=metadata["max_agents"],
        max_enemies=metadata["max_enemies"],
        max_actions=metadata["max_actions"],
        token_dim=metadata["token_dim"],
        dynamic_token_dim=metadata.get("dynamic_token_dim"),
        static_dim=metadata.get("static_dim", 0),
        entity_static_feat_size=metadata.get("entity_static_feat_size", 0),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = SMACJEPA(
        state_dim=metadata["state_dim"],
        n_agents=metadata["n_agents"],
        n_actions=metadata["n_actions"],
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        action_dim=int(config["action_dim"]),
        num_heads=int(config["num_heads"]),
        mode="entity",
        max_agents=metadata.get("max_agents", metadata["n_agents"]),
        max_enemies=metadata.get("max_enemies", 0),
        max_actions=metadata.get("max_actions", metadata["n_actions"]),
        token_dim=metadata.get("token_dim", metadata["state_dim"]),
        static_dim=int(metadata.get("static_dim", 0)),
        decoder_weight=float(config.get("decoder_weight", 1.0)),
        encoder_layers=int(config.get("encoder_layers", 1)),
        action_layers=int(config.get("action_layers", 1)),
        predictor_layers=int(config.get("predictor_layers", 1)),
        max_context_len=int(config.get("max_context_len", 32)),
        static_conditioning=str(
            metadata.get("static_conditioning", config.get("static_conditioning", "action"))
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    total_loss = 0.0
    total_count = 0.0
    metric_sums: dict[str, float] = {}
    metric_batches = 0
    rollout_sums: dict[str, float] = {}
    rollout_counts: dict[str, int] = {}
    decoded_samples: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model(batch)
            mask = out["target_entity_mask"].unsqueeze(-1) * out["mask"].unsqueeze(-1).unsqueeze(-1)
            squared = (out["pred_latent"] - out["target_latent"]).pow(2) * mask
            total_loss += float(squared.sum().cpu())
            total_count += float(mask.sum().cpu()) * out["pred_latent"].shape[-1]
            metric_batches += 1
            for key, value in entity_prediction_metrics(out).items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value.cpu())
            if rollout_horizons:
                rollout_metrics = rollout_evaluate_batch(model, batch, metadata, rollout_horizons)
                for key, value in rollout_metrics.items():
                    rollout_sums[key] = rollout_sums.get(key, 0.0) + value
                    rollout_counts[key] = rollout_counts.get(key, 0) + 1
            if len(decoded_samples) < args.num_decode_samples:
                decoded_samples.extend(
                    collect_decoded_samples(
                        out,
                        metadata,
                        remaining=args.num_decode_samples - len(decoded_samples),
                    )
                )

    metrics = {
        "next_state_embedding_mse": total_loss / max(total_count, 1.0),
        "num_windows": len(dataset),
    }
    for key, value in metric_sums.items():
        metrics[key] = value / max(metric_batches, 1)
    for key, value in rollout_sums.items():
        metrics[key] = value / max(rollout_counts.get(key, 0), 1)
    if args.per_config_out:
        rows = []
        for path in data_paths:
            item_dataset = SMACJEPADataset(
                [path],
                context_len=eval_context_len,
                mode="entity",
                max_agents=metadata["max_agents"],
                max_enemies=metadata["max_enemies"],
                max_actions=metadata["max_actions"],
                token_dim=metadata["token_dim"],
                dynamic_token_dim=metadata.get("dynamic_token_dim"),
                static_dim=metadata.get("static_dim", 0),
                entity_static_feat_size=metadata.get("entity_static_feat_size", 0),
            )
            row = evaluate_dataset(model, item_dataset, metadata, args.batch_size, args.num_workers, rollout_horizons, device)
            row["path"] = str(path)
            row["scenario"] = item_dataset.scenarios[0] if item_dataset.scenarios else path.stem
            row["n_agents"] = item_dataset.metadata.n_agents
            row["n_enemies"] = item_dataset.metadata.n_enemies
            rows.append(row)
        per_config_path = Path(args.per_config_out)
        per_config_path.parent.mkdir(parents=True, exist_ok=True)
        per_config_path.write_text(json.dumps(rows, indent=2) + "\n")
        metrics["per_config_path"] = str(per_config_path)
    if args.decode_sample_out:
        sample_path = Path(args.decode_sample_out)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(json.dumps(decoded_samples, indent=2) + "\n")
        metrics["decoded_samples_path"] = str(sample_path)
    else:
        metrics["decoded_samples"] = decoded_samples
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def evaluate_dataset(
    model: SMACJEPA,
    dataset: SMACJEPADataset,
    metadata: dict[str, object],
    batch_size: int,
    num_workers: int,
    rollout_horizons: list[int],
    device: torch.device,
) -> dict[str, float | int]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    total_loss = 0.0
    total_count = 0.0
    metric_sums: dict[str, float] = {}
    metric_batches = 0
    rollout_sums: dict[str, float] = {}
    rollout_counts: dict[str, int] = {}
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model(batch)
            mask = out["target_entity_mask"].unsqueeze(-1) * out["mask"].unsqueeze(-1).unsqueeze(-1)
            total_loss += float((((out["pred_latent"] - out["target_latent"]).pow(2) * mask).sum()).cpu())
            total_count += float(mask.sum().cpu()) * out["pred_latent"].shape[-1]
            metric_batches += 1
            for key, value in entity_prediction_metrics(out).items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value.cpu())
            if rollout_horizons:
                for key, value in rollout_evaluate_batch(model, batch, metadata, rollout_horizons).items():
                    rollout_sums[key] = rollout_sums.get(key, 0.0) + value
                    rollout_counts[key] = rollout_counts.get(key, 0) + 1
    metrics: dict[str, float | int] = {
        "next_state_embedding_mse": total_loss / max(total_count, 1.0),
        "num_windows": len(dataset),
    }
    for key, value in metric_sums.items():
        metrics[key] = value / max(metric_batches, 1)
    for key, value in rollout_sums.items():
        metrics[key] = value / max(rollout_counts.get(key, 0), 1)
    return metrics


def collect_decoded_samples(
    out: dict[str, torch.Tensor],
    metadata: dict[str, object],
    remaining: int,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    decoded = out["decoded_target"].detach().cpu()
    presence = torch.sigmoid(out["presence_logits"]).detach().cpu() if "presence_logits" in out else None
    target = out["target_entity"].detach().cpu()
    token_mask = out["target_entity_mask"].detach().cpu()
    timestep_mask = out["mask"].detach().cpu()
    batch_size, steps = timestep_mask.shape
    for batch_idx in range(batch_size):
        for step_idx in range(steps):
            if len(samples) >= remaining:
                return samples
            if float(timestep_mask[batch_idx, step_idx]) <= 0:
                continue
            pred_step = decoded[batch_idx, step_idx]
            target_step = target[batch_idx, step_idx]
            mask_step = token_mask[batch_idx, step_idx]
            samples.append(
                {
                    "batch_index": batch_idx,
                    "step_index": step_idx,
                    "prediction": format_entity_predictions(
                        pred_step,
                        metadata,
                        mask_step,
                        presence[batch_idx, step_idx] if presence is not None else None,
                    ),
                    "target": format_entity_predictions(target_step, metadata, mask_step),
                    "absolute_error_mean": float(
                        ((pred_step - target_step).abs() * mask_step.unsqueeze(-1)).sum()
                        / (mask_step.sum().clamp_min(1.0) * target_step.shape[-1])
                    ),
                }
            )
    return samples


def parse_horizons(value: str) -> list[int]:
    if not value.strip():
        return []
    horizons = sorted({int(part) for part in value.split(",") if part.strip()})
    if any(horizon < 1 for horizon in horizons):
        raise ValueError("rollout horizons must be >= 1")
    return horizons


def rollout_evaluate_batch(
    model: SMACJEPA,
    batch: dict[str, torch.Tensor],
    metadata: dict[str, object],
    horizons: list[int],
) -> dict[str, float]:
    max_horizon = min(max(horizons), batch["action_t"].shape[1])
    current_tokens = batch["entity_t"][:, :1]
    current_mask = batch["entity_mask"][:, :1]
    static_tail = current_tokens[:, :, :, int(metadata.get("dynamic_token_dim", metadata["token_dim"])) :]
    results: dict[str, float] = {}
    for step in range(max_horizon):
        timestep_mask = batch["mask"][:, step : step + 1]
        latent = model.encoder(current_tokens, current_mask)
        pred_latent = model.predictor(
            latent,
            batch["action_t"][:, step : step + 1],
            batch["action_mask"][:, step : step + 1],
            timestep_mask,
            current_mask,
            batch.get("static_condition"),
        )
        decoded = model.decode_entities(pred_latent)
        presence_scores = torch.sigmoid(model.predict_presence(pred_latent))
        if static_tail.shape[-1]:
            decoded = decoded.clone()
            decoded[:, :, :, -static_tail.shape[-1] :] = static_tail
        current_tokens = decoded.detach()
        slot_mask = batch.get("entity_slot_mask", batch["target_entity_mask"])[:, step : step + 1]
        current_mask = ((presence_scores >= 0.5).to(decoded.dtype) * slot_mask).detach()
        horizon = step + 1
        if horizon not in horizons:
            continue
        target = batch["target_entity"][:, step : step + 1]
        target_mask = batch["target_entity_mask"][:, step : step + 1].unsqueeze(-1)
        valid_mask = target_mask * timestep_mask.unsqueeze(-1).unsqueeze(-1)
        denom = (valid_mask.sum() * target.shape[-1]).clamp_min(1.0)
        err = (decoded - target).abs() * valid_mask
        sq_err = (decoded - target).pow(2) * valid_mask
        prefix = f"rollout_h{horizon}"
        results[f"{prefix}_decoded_mae"] = float((err.sum() / denom).detach().cpu())
        results[f"{prefix}_decoded_mse"] = float((sq_err.sum() / denom).detach().cpu())
        target_presence = batch["target_entity_mask"][:, step : step + 1]
        presence_valid = slot_mask * timestep_mask.unsqueeze(-1)
        presence_pred = (presence_scores >= 0.5).to(target_presence.dtype)
        presence_correct = (presence_pred == target_presence).to(target_presence.dtype) * presence_valid
        results[f"{prefix}_presence_acc"] = float(
            (presence_correct.sum() / presence_valid.sum().clamp_min(1.0)).detach().cpu()
        )
        if target.shape[-1] >= 4:
            hp_mask = valid_mask[..., 0:1]
            xy_mask = valid_mask.expand_as(target)[..., 2:4]
            results[f"{prefix}_hp_mae"] = float(
                ((decoded[..., 0:1] - target[..., 0:1]).abs() * hp_mask).sum()
                .div(hp_mask.sum().clamp_min(1.0))
                .detach()
                .cpu()
            )
            results[f"{prefix}_xy_mae"] = float(
                ((decoded[..., 2:4] - target[..., 2:4]).abs() * xy_mask).sum()
                .div(xy_mask.sum().clamp_min(1.0))
                .detach()
                .cpu()
            )
    return results


if __name__ == "__main__":
    main()
