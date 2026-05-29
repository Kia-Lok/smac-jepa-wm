from __future__ import annotations

import argparse
import csv
import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from smac_jepa.data import SMACJEPADataset, load_manifest, load_manifest_all
from smac_jepa.jepa import SMACJEPA

try:
    import wandb
except ImportError:
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SMAC-JEPA checkpoints on a manifest split.")
    parser.add_argument("--manifest", required=True, help="Dataset split manifest, e.g. splits/generated_seed2.json")
    parser.add_argument("--split", default="eval", help="Split to evaluate on, usually eval/test")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory containing checkpoint_epoch_*.pt")
    parser.add_argument("--checkpoint", action="append", default=None, help="Specific checkpoint path. Can be repeated.")
    parser.add_argument("--pattern", default="checkpoint_epoch_*.pt", help="Glob pattern inside --checkpoint-dir")
    parser.add_argument("--out-dir", default=None, help="Directory for per-checkpoint eval outputs. Default: <checkpoint-dir>/eval_results")
    parser.add_argument("--summary-out", default=None, help="Summary CSV path. Default: <out-dir>/eval_summary.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-batches", type=int, default=None, help="Optional limit for quick smoke tests")
    parser.add_argument(
        "--enemy-visibility-mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override checkpoint setting for visibility masking. If omitted, uses the "
            "value saved in the checkpoint resolved_config/config, defaulting to False."
        ),
    )
    parser.add_argument(
        "--enemy-sight-range",
        type=float,
        default=None,
        help="Override checkpoint setting for SMACLite sight range. If omitted, defaults to saved value or 9.0.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log eval metrics to W&B")
    parser.add_argument("--wandb-project", default="SMAC-JEPA-losses")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default="eval-checkpoints")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
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


def checkpoint_epoch_num(path: Path) -> int:
    # checkpoint_epoch_003.pt -> 3
    match = re.search(r"checkpoint_epoch_(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1))
    return -1


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    epoch = checkpoint_epoch_num(path)
    if epoch >= 0:
        return (epoch, path.name)
    return (10**9, path.name)


def find_checkpoints(args: argparse.Namespace) -> list[Path]:
    checkpoints: list[Path] = []

    if args.checkpoint:
        checkpoints.extend(Path(p) for p in args.checkpoint)

    if args.checkpoint_dir:
        checkpoints.extend(sorted(Path(args.checkpoint_dir).glob(args.pattern), key=checkpoint_sort_key))

    seen = set()
    unique = []
    for ckpt in checkpoints:
        ckpt = ckpt.resolve()
        if ckpt not in seen:
            unique.append(ckpt)
            seen.add(ckpt)

    if not unique:
        raise SystemExit("No checkpoints found. Use --checkpoint-dir or --checkpoint.")

    missing = [str(p) for p in unique if not p.exists()]
    if missing:
        raise SystemExit(f"Missing checkpoint(s): {missing}")

    return unique


def build_dataset(
    manifest: str,
    split: str,
    resolved_config: dict[str, Any],
    enemy_visibility_mask: bool | None = None,
    enemy_sight_range: float | None = None,
) -> SMACJEPADataset:
    data_paths = load_manifest(manifest, split)
    cap_paths = load_manifest_all(manifest)
    cap_dataset = SMACJEPADataset(cap_paths, context_len=1, mode="entity")
    cap_metadata = cap_dataset.metadata

    context_len = int(resolved_config.get("context_len", resolved_config.get("window_len", 4)))
    window_len = int(resolved_config.get("window_len", context_len))
    window_mode = str(resolved_config.get("window_mode", "sequential"))
    resolved_enemy_visibility_mask = (
        bool(resolved_config.get("enemy_visibility_mask", False))
        if enemy_visibility_mask is None
        else bool(enemy_visibility_mask)
    )
    resolved_enemy_sight_range = (
        float(resolved_config.get("enemy_sight_range", 9.0))
        if enemy_sight_range is None
        else float(enemy_sight_range)
    )

    return SMACJEPADataset(
        data_paths,
        context_len=context_len,
        mode="entity",
        window_mode=window_mode,
        window_len=window_len,
        samples_per_epoch=None,
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


def build_model(checkpoint: dict[str, Any], dataset: SMACJEPADataset, device: torch.device) -> SMACJEPA:
    resolved_config = checkpoint.get("resolved_config", checkpoint.get("config", {}))
    metadata = checkpoint["metadata"]

    model = SMACJEPA(
        state_dim=metadata.get("state_dim", dataset.metadata.state_dim),
        n_agents=metadata.get("n_agents", dataset.metadata.n_agents),
        n_actions=metadata.get("n_actions", dataset.metadata.n_actions),
        latent_dim=int(resolved_config["latent_dim"]),
        hidden_dim=int(resolved_config["hidden_dim"]),
        action_dim=int(resolved_config["action_dim"]),
        num_heads=int(resolved_config["num_heads"]),
        mode=metadata.get("mode", dataset.metadata.mode),
        max_agents=metadata.get("max_agents", dataset.metadata.max_agents),
        max_enemies=metadata.get("max_enemies", dataset.metadata.max_enemies),
        max_actions=metadata.get("max_actions", dataset.metadata.max_actions),
        token_dim=metadata.get("token_dim", dataset.metadata.token_dim),
        static_dim=metadata.get("static_dim", dataset.metadata.static_dim),
        decoder_weight=float(resolved_config.get("decoder_weight", 1.0)),
        encoder_layers=int(resolved_config["encoder_layers"]),
        action_layers=int(resolved_config["action_layers"]),
        predictor_layers=int(resolved_config["predictor_layers"]),
        max_context_len=int(resolved_config.get("max_context_len", 32)),
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_one_checkpoint(
    checkpoint_path: Path,
    manifest: str,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None,
    enemy_visibility_mask: bool | None = None,
    enemy_sight_range: float | None = None,
) -> dict[str, float | int | str]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    resolved_config = checkpoint.get("resolved_config", checkpoint.get("config", {}))

    dataset = build_dataset(
        manifest,
        split,
        resolved_config,
        enemy_visibility_mask=enemy_visibility_mask,
        enemy_sight_range=enemy_sight_range,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_model(checkpoint, dataset, device)

    sums: dict[str, float] = {}
    batches = 0
    windows = 0

    autocast_context = (
        torch.cuda.amp.autocast(enabled=amp_enabled)
        if device.type == "cuda"
        else nullcontext()
    )

    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break

        batch = to_device(batch, device)
        with autocast_context:
            losses = model.loss(
                batch,
                sigreg_weight=float(resolved_config.get("sigreg_weight", 0.09)),
            )

        current_batch_size = next(iter(batch.values())).shape[0]
        batches += 1
        windows += int(current_batch_size)

        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * current_batch_size

    if windows == 0:
        raise RuntimeError(f"No eval windows produced for {checkpoint_path}")

    epoch_from_name = checkpoint_epoch_num(checkpoint_path)
    epoch_from_ckpt = int(checkpoint.get("epoch", -1))
    epoch = epoch_from_name if epoch_from_name >= 0 else epoch_from_ckpt

    row: dict[str, float | int | str] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_epoch_num": epoch_from_name,
        "epoch": epoch,
        "checkpoint_saved_epoch": epoch_from_ckpt,
        "global_step": int(checkpoint.get("global_step", -1)),
        "enemy_visibility_mask": bool(dataset.enemy_visibility_mask),
        "enemy_sight_range": float(dataset.enemy_sight_range),
        "eval_batches": batches,
        "eval_windows": windows,
    }

    for key, value in sums.items():
        row[f"eval_{key}"] = value / windows

    return row


def write_single_row_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    checkpoints = find_checkpoints(args)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif args.checkpoint_dir is not None:
        out_dir = Path(args.checkpoint_dir) / "eval_results"
    else:
        out_dir = checkpoints[0].parent / "eval_results"

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_out is not None:
        summary_out = Path(args.summary_out)
    else:
        summary_out = out_dir / "eval_summary.csv"

    summary_jsonl = summary_out.with_suffix(".jsonl")

    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise SystemExit("W&B requested but wandb is not installed. Install with: uv pip install wandb")

        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={
                "manifest": args.manifest,
                "split": args.split,
                "checkpoint_dir": args.checkpoint_dir,
                "pattern": args.pattern,
                "batch_size": args.batch_size,
                "device": device.type,
                "amp_enabled": amp_enabled,
                "max_batches": args.max_batches,
                "enemy_visibility_mask": args.enemy_visibility_mask,
                "enemy_sight_range": args.enemy_sight_range,
            },
        )

    rows: list[dict[str, Any]] = []

    print(f"Evaluating {len(checkpoints)} checkpoint(s) on split={args.split} device={device}", flush=True)
    for checkpoint_path in checkpoints:
        print(f"evaluating {checkpoint_path}", flush=True)
        row = evaluate_one_checkpoint(
            checkpoint_path=checkpoint_path,
            manifest=args.manifest,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            amp_enabled=amp_enabled,
            max_batches=args.max_batches,
            enemy_visibility_mask=args.enemy_visibility_mask,
            enemy_sight_range=args.enemy_sight_range,
        )
        rows.append(row)

        epoch = int(row["epoch"])
        epoch_tag = f"epoch_{epoch:03d}" if epoch >= 0 else checkpoint_path.stem
        per_csv = out_dir / f"eval_{epoch_tag}.csv"
        per_json = out_dir / f"eval_{epoch_tag}.json"

        write_single_row_csv(per_csv, row)
        per_json.write_text(json.dumps(row, indent=2) + "\n")

        with summary_jsonl.open("a") as f:
            f.write(json.dumps(row) + "\n")

        if wandb_run is not None:
            wandb_payload = {
                key.replace("eval_", "eval/"): value
                for key, value in row.items()
                if key.startswith("eval_") and isinstance(value, (int, float))
            }
            wandb_payload["eval/epoch"] = epoch
            wandb_payload["eval/global_step"] = row.get("global_step", -1)
            wandb_run.log(wandb_payload, step=epoch if epoch >= 0 else len(rows))

        metric_str = " ".join(
            f"{key}={value:.6f}"
            for key, value in row.items()
            if key.startswith("eval_") and isinstance(value, float)
        )
        print(
            f"done checkpoint={row['checkpoint_name']} epoch={row['epoch']} "
            f"global_step={row['global_step']} {metric_str}",
            flush=True,
        )
        print(f"wrote {per_csv}", flush=True)
        print(f"wrote {per_json}", flush=True)

    write_summary_csv(summary_out, rows)

    print(f"wrote summary {summary_out}", flush=True)
    print(f"wrote summary {summary_jsonl}", flush=True)

    if wandb_run is not None:
        wandb_run.save(str(summary_out))
        wandb_run.save(str(summary_jsonl))
        for path in sorted(out_dir.glob("eval_epoch_*.*")):
            wandb_run.save(str(path))
        wandb_run.finish()


if __name__ == "__main__":
    main()
