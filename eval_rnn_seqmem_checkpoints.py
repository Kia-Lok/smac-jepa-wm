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

# This imports the exact loss function and action-conditioned memory class
# used by the exp1-6 training script.
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
        description="Evaluate specific RNN seqmem visibility checkpoints on rollout eval loss."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Specific checkpoint path. Can be repeated to compare selected checkpoints.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-batches", type=int, default=None)

    # Evaluation behavior. Defaults are loaded from checkpoint resolved_config.
    parser.add_argument("--window-mode", default="sequential", choices=["sequential", "random"])
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--target-mode", choices=["full", "observed"], default=None)
    parser.add_argument("--enemy-visibility-mask", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enemy-sight-range", type=float, default=None)

    # Loss-weight overrides. If omitted, use checkpoint config.
    parser.add_argument("--sigreg-weight", type=float, default=None)
    parser.add_argument("--decoder-weight", type=float, default=None)
    parser.add_argument("--presence-weight", type=float, default=None)
    parser.add_argument("--one-step-weight", type=float, default=None)
    parser.add_argument("--td-lambda", type=float, default=None)
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


def build_dataset(
    *,
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


def build_model(
    *,
    checkpoint: dict[str, Any],
    dataset: VisibilityMarkovRolloutSMACJEPADataset,
    device: torch.device,
) -> SMACJEPA:
    resolved_config = get_config(checkpoint)
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


def build_memory_module(
    *,
    checkpoint: dict[str, Any],
    dataset: VisibilityMarkovRolloutSMACJEPADataset,
    device: torch.device,
) -> torch.nn.Module:
    resolved_config = get_config(checkpoint)

    latent_dim = int(resolved_config["latent_dim"])
    memory_dim = int(resolved_config.get("rollout_memory_dim", 128))
    hidden_dim = resolved_config.get("rollout_memory_hidden_dim", None)
    residual = not bool(resolved_config.get("rollout_memory_no_residual", False))
    action_conditioned = bool(resolved_config.get("action_conditioned_memory", False))

    if action_conditioned:
        memory_module = ActionConditionedEntityRolloutGRUMemory(
            latent_dim=latent_dim,
            memory_dim=memory_dim,
            n_actions=dataset.metadata.n_actions,
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
        raise RuntimeError(
            "Checkpoint does not contain memory_module_state. "
            "This is not a valid RNN seqmem checkpoint."
        )

    memory_module.load_state_dict(checkpoint["memory_module_state"])
    memory_module.eval()
    return memory_module


@torch.no_grad()
def evaluate_one_checkpoint(
    *,
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    resolved_config = get_config(checkpoint)

    dataset = build_dataset(
        manifest=args.manifest,
        split=args.split,
        resolved_config=resolved_config,
        window_mode=args.window_mode,
        samples_per_epoch=args.samples_per_epoch,
        enemy_visibility_mask=args.enemy_visibility_mask,
        enemy_sight_range=args.enemy_sight_range,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(checkpoint=checkpoint, dataset=dataset, device=device)
    memory_module = build_memory_module(checkpoint=checkpoint, dataset=dataset, device=device)

    target_mode = str(args.target_mode or resolved_config.get("target_mode", "full"))
    sigreg_weight = float(args.sigreg_weight if args.sigreg_weight is not None else resolved_config.get("sigreg_weight", 0.01))
    decoder_weight = float(args.decoder_weight if args.decoder_weight is not None else resolved_config.get("decoder_weight", 0.01))
    presence_weight = float(args.presence_weight if args.presence_weight is not None else resolved_config.get("presence_weight", 0.01))
    one_step_weight = float(args.one_step_weight if args.one_step_weight is not None else resolved_config.get("one_step_weight", 0.0))
    td_lambda = float(args.td_lambda if args.td_lambda is not None else resolved_config.get("td_lambda", 0.9))

    sums: dict[str, float] = {}
    batches = 0
    windows = 0

    autocast_context = (
        torch.cuda.amp.autocast(enabled=amp_enabled)
        if device.type == "cuda"
        else nullcontext()
    )

    for batch in loader:
        if args.max_batches is not None and batches >= args.max_batches:
            break

        batch = to_device(batch, device)
        with autocast_context:
            losses = markov_rollout_rnn_losses(
                model,
                memory_module,
                batch,
                rollout_window=int(resolved_config.get("rollout_window", 20)),
                rollout_horizon=int(resolved_config.get("rollout_horizon", 5)),
                temporal_loss_mode=str(resolved_config.get("temporal_loss", "lambda")),
                td_lambda=td_lambda,
                flat_decay_start=resolved_config.get("flat_decay_start", None),
                flat_decay_final_weight=float(resolved_config.get("flat_decay_final_weight", 0.5)),
                sigreg_weight=sigreg_weight,
                decoder_weight=decoder_weight,
                presence_weight=presence_weight,
                one_step_weight=one_step_weight,
                target_mode=target_mode,
                detach_rollout_targets=bool(resolved_config.get("detach_rollout_targets", False)),
                unweighted_aux_losses=bool(resolved_config.get("unweighted_aux_losses", False)),
            )

        bsz = next(iter(batch.values())).shape[0]
        batches += 1
        windows += int(bsz)

        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * bsz

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
        "action_conditioned_memory": bool(resolved_config.get("action_conditioned_memory", False)),
        "one_step_weight": one_step_weight,
        "sigreg_weight": sigreg_weight,
        "decoder_weight": decoder_weight,
        "presence_weight": presence_weight,
        "td_lambda": td_lambda,
        "enemy_visibility_mask": bool(dataset.enemy_visibility_mask),
        "enemy_sight_range": float(dataset.enemy_sight_range),
        "rollout_window": int(resolved_config.get("rollout_window", 20)),
        "rollout_horizon": int(resolved_config.get("rollout_horizon", 5)),
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

    summary_out = Path(args.summary_out) if args.summary_out else out_dir / "eval_seqmem_summary.csv"
    summary_jsonl = summary_out.with_suffix(".jsonl")

    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    checkpoints = [Path(p) for p in args.checkpoint]
    for p in checkpoints:
        if not p.exists():
            raise SystemExit(f"Missing checkpoint: {p}")

    print(
        f"Evaluating {len(checkpoints)} selected checkpoint(s) "
        f"split={args.split} device={device} window_mode={args.window_mode}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        print(f"evaluating {checkpoint_path}", flush=True)
        row = evaluate_one_checkpoint(
            checkpoint_path=checkpoint_path,
            args=args,
            device=device,
            amp_enabled=amp_enabled,
        )
        rows.append(row)

        stem = checkpoint_path.parent.name + "_" + checkpoint_path.stem
        per_json = out_dir / f"eval_{stem}.json"
        per_csv = out_dir / f"eval_{stem}.csv"
        per_json.write_text(json.dumps(row, indent=2) + "\n")
        write_summary(per_csv, [row])

        with summary_jsonl.open("a") as f:
            f.write(json.dumps(row) + "\n")

        metric_str = " ".join(
            f"{key}={value:.6f}"
            for key, value in row.items()
            if key.startswith("eval_") and isinstance(value, float)
        )
        print(
            f"done checkpoint={row['checkpoint_dir']} saved_epoch={row['checkpoint_saved_epoch']} "
            f"global_step={row['global_step']} {metric_str}",
            flush=True,
        )
        print(f"wrote {per_json}", flush=True)
        print(f"wrote {per_csv}", flush=True)

    write_summary(summary_out, rows)
    print(f"wrote summary {summary_out}", flush=True)
    print(f"wrote summary {summary_jsonl}", flush=True)


if __name__ == "__main__":
    main()
