from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from smac_jepa.data import SMACJEPADataset, load_manifest
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
    dataset = SMACJEPADataset(
        data_paths,
        context_len=int(config["context_len"]),
        mode="entity",
        max_agents=metadata["max_agents"],
        max_enemies=metadata["max_enemies"],
        max_actions=metadata["max_actions"],
        token_dim=metadata["token_dim"],
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
        decoder_weight=float(config.get("decoder_weight", 1.0)),
        encoder_layers=int(config.get("encoder_layers", 1)),
        action_layers=int(config.get("action_layers", 1)),
        predictor_layers=int(config.get("predictor_layers", 1)),
        max_context_len=int(config.get("max_context_len", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    total_loss = 0.0
    total_count = 0.0
    metric_sums: dict[str, float] = {}
    metric_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model(batch)
            mask = out["mask"].unsqueeze(-1)
            squared = (out["pred_latent"] - out["target_latent"]).pow(2) * mask
            total_loss += float(squared.sum().cpu())
            total_count += float(mask.sum().cpu()) * out["pred_latent"].shape[-1]
            metric_batches += 1
            for key, value in entity_prediction_metrics(out).items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value.cpu())

    metrics = {
        "next_state_embedding_mse": total_loss / max(total_count, 1.0),
        "num_windows": len(dataset),
    }
    for key, value in metric_sums.items():
        metrics[key] = value / max(metric_batches, 1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
