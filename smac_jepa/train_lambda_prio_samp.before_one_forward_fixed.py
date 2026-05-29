from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

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


class CategoryDataset(Dataset):
    """Wrap a dataset so every item carries a numeric category id."""

    def __init__(self, dataset: Dataset, category_id: int) -> None:
        self.dataset = dataset
        self.category_id = int(category_id)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.dataset[index]
        if not isinstance(item, dict):
            raise TypeError(f"Expected dataset item to be a dict, got {type(item)!r}")
        item = dict(item)
        item["__category_id__"] = torch.tensor(self.category_id, dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train entity-token SMAC-JEPA with lambda temporal loss and optional category-prioritized sampling"
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
        help=(
            "Lambda for TD(lambda)-inspired temporal weighting over the prediction window. "
            "Horizon 0 has weight 1, horizon 1 has lambda, horizon 2 has lambda^2, etc. Default: 0.9"
        ),
    )

    # Category-prioritized sampling args.
    parser.add_argument(
        "--category-prioritized",
        action="store_true",
        help="Enable loss-aware category-prioritized sampling over .npz files.",
    )
    parser.add_argument(
        "--category-prefixes-json",
        default=None,
        help="JSON mapping category names to filename prefixes, e.g. {'cat': ['prefix_']}.",
    )
    # Compatibility aliases for lazy-transfer style commands.
    parser.add_argument(
        "--category-prefixes",
        dest="category_prefixes_json",
        default=None,
        help="Alias for --category-prefixes-json.",
    )
    parser.add_argument(
        "--category-priority-alpha",
        type=float,
        default=0.7,
        help="Exponent applied to EMA category losses before converting them to probabilities. Default: 0.7",
    )
    parser.add_argument(
        "--priority-alpha",
        dest="category_priority_alpha",
        type=float,
        help="Alias for --category-priority-alpha.",
    )
    parser.add_argument(
        "--category-priority-ema",
        type=float,
        default=0.9,
        help="EMA coefficient for category losses. Default: 0.9",
    )
    parser.add_argument(
        "--priority-ema",
        dest="category_priority_ema",
        type=float,
        help="Alias for --category-priority-ema.",
    )
    parser.add_argument(
        "--category-uniform-mix",
        type=float,
        default=0.2,
        help="Mix this much uniform sampling into loss-prioritized probabilities. Default: 0.2",
    )
    parser.add_argument(
        "--category-min-prob",
        type=float,
        default=0.02,
        help="Minimum sampling probability per category. Must satisfy n_categories * min_prob < 1. Default: 0.02",
    )
    parser.add_argument(
        "--priority-min-prob",
        dest="category_min_prob",
        type=float,
        help="Alias for --category-min-prob.",
    )
    parser.add_argument(
        "--priority-min-samples",
        type=int,
        default=1,
        help="Accepted for lazy-transfer compatibility. This sampler uses probability floors, so this is informational only.",
    )
    parser.add_argument(
        "--priority-metric",
        choices=["pred_loss"],
        default="pred_loss",
        help="Accepted for lazy-transfer compatibility. This compatible sampler prioritizes by per-sample prediction loss.",
    )
    parser.add_argument(
        "--allow-all-uncategorized",
        action="store_true",
        help="Allow priority mode to continue even if no file matches any named prefix; otherwise this is treated as a configuration error.",
    )
    parser.add_argument(
        "--category-warmup-epochs",
        type=int,
        default=1,
        help="Number of initial epochs with uniform/shuffled sampling before prioritization starts. Default: 1",
    )
    parser.add_argument(
        "--uncategorized-category-name",
        default="uncategorized",
        help="Category name for files that do not match any prefix. Default: uncategorized",
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
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


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
        f"Auto-split from {data_dir}: total={len(files)} train={len(train_files)} "
        f"eval={len(eval_files)} using split={config.split}",
        flush=True,
    )

    return [str(path) for path in selected]


def load_category_prefixes(path: str | None, uncategorized_name: str = "uncategorized") -> dict[str, list[str]]:
    if path is None:
        raise SystemExit("--category-prioritized requires --category-prefixes-json/--category-prefixes")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise SystemExit("Category prefixes JSON must be a non-empty object/dict.")

    out: dict[str, list[str]] = {}
    seen_prefixes: dict[str, str] = {}
    for category, prefixes in data.items():
        if not isinstance(category, str) or not category.strip():
            raise SystemExit(f"Invalid category name in JSON: {category!r}")
        if category == uncategorized_name:
            raise SystemExit(
                f"Category name {category!r} is reserved for unmatched files. "
                "Use --uncategorized-category-name to choose a different fallback name."
            )
        if not isinstance(prefixes, list) or not prefixes:
            raise SystemExit(f"Category {category!r} must map to a non-empty list of prefixes.")
        clean_prefixes: list[str] = []
        for prefix in prefixes:
            if not isinstance(prefix, str) or not prefix:
                raise SystemExit(f"Invalid prefix {prefix!r} in category {category!r}")
            if prefix in seen_prefixes:
                raise SystemExit(
                    f"Duplicate prefix {prefix!r} appears in both "
                    f"{seen_prefixes[prefix]!r} and {category!r}."
                )
            seen_prefixes[prefix] = category
            clean_prefixes.append(prefix)
        out[category] = clean_prefixes
    return out


def warn_and_validate_category_matches(
    paths_by_category: dict[str, list[str]],
    category_prefixes: dict[str, list[str]],
    uncategorized_name: str,
    allow_all_uncategorized: bool,
) -> None:
    zero_match = [name for name in category_prefixes if name not in paths_by_category]
    if zero_match:
        print("warning_zero_match_categories " + ",".join(zero_match), flush=True)

    real_matched = sum(
        len(paths_by_category.get(name, []))
        for name in category_prefixes
    )
    uncategorized_count = len(paths_by_category.get(uncategorized_name, []))

    if real_matched == 0 and not allow_all_uncategorized:
        raise SystemExit(
            "No files matched any configured category prefix. "
            f"All {uncategorized_count} files would be assigned to {uncategorized_name!r}, "
            "so category-prioritized sampling would be meaningless. "
            "Fix the prefix JSON, regenerate category prefixes from the manifest, "
            "or pass --allow-all-uncategorized to continue anyway."
        )

    if real_matched == 0:
        print(
            f"warning_all_uncategorized count={uncategorized_count}; priority sampling will behave almost like uniform sampling",
            flush=True,
        )


def categorize_paths(
    paths: list[str],
    category_prefixes: dict[str, list[str]],
    uncategorized_name: str,
) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {category: [] for category in category_prefixes}
    categories.setdefault(uncategorized_name, [])

    for path in paths:
        filename = Path(path).name
        matches: list[tuple[int, str]] = []
        for category, prefixes in category_prefixes.items():
            for prefix in prefixes:
                if filename.startswith(prefix):
                    matches.append((len(prefix), category))
        if matches:
            # Longest prefix wins. This is safer when prefixes overlap.
            _, category = max(matches, key=lambda item: item[0])
        else:
            category = uncategorized_name
        categories.setdefault(category, []).append(path)

    return {category: files for category, files in categories.items() if files}


def _lambda_time_weights(length: int, td_lambda: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not 0.0 <= td_lambda <= 1.0:
        raise ValueError(f"--td-lambda must be in [0, 1], got {td_lambda}")
    steps = torch.arange(length, device=device, dtype=dtype)
    return torch.pow(torch.as_tensor(td_lambda, device=device, dtype=dtype), steps)


def _expand_like_time_weights(weights: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim < 2:
        return torch.ones_like(target)
    return weights.view(1, weights.shape[0], *([1] * (target.ndim - 2)))


def _expand_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(-1)
    return mask.to(dtype=target.dtype)


def _reduce_per_sample(x: torch.Tensor) -> torch.Tensor:
    if x.ndim <= 1:
        return x
    dims = tuple(range(1, x.ndim))
    return x.sum(dim=dims)


def temporal_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    td_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        lambda_loss: scalar weighted mean over batch/window/features
        uniform_loss: scalar ordinary masked mean
        lambda_weight_sum: scalar sum of horizon weights
        per_sample_lambda_loss: [B] loss for category tracking
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes must match, got {pred.shape} vs {target.shape}")

    if pred.ndim < 2:
        per_sample = (pred - target).pow(2).view(pred.shape[0], -1).mean(dim=1) if pred.ndim > 0 else (pred - target).pow(2).view(1)
        loss = per_sample.mean()
        one = torch.ones((), device=pred.device, dtype=pred.dtype)
        return loss, loss, one, per_sample.detach()

    window_steps = pred.shape[1]
    weights = _lambda_time_weights(window_steps, td_lambda, pred.device, pred.dtype)
    weight_view = _expand_like_time_weights(weights, pred)
    squared = (pred - target).pow(2)

    if mask is None:
        weighted_mask = weight_view.expand_as(squared)
        uniform_mask = torch.ones_like(squared)
    else:
        mask_view = _expand_mask(mask, squared)
        weighted_mask = (mask_view * weight_view).expand_as(squared)
        uniform_mask = mask_view.expand_as(squared)

    weighted_num_per_sample = _reduce_per_sample(squared * weighted_mask)
    weighted_den_per_sample = _reduce_per_sample(weighted_mask).clamp_min(1.0)
    per_sample_lambda_loss = weighted_num_per_sample / weighted_den_per_sample
    lambda_loss = per_sample_lambda_loss.mean()

    uniform_num = (squared * uniform_mask).sum()
    uniform_den = uniform_mask.sum().clamp_min(1.0)
    uniform_loss = uniform_num / uniform_den

    return lambda_loss, uniform_loss, weights.sum(), per_sample_lambda_loss.detach()


def lambda_jepa_losses(
    model: SMACJEPA,
    batch: dict[str, torch.Tensor],
    sigreg_weight: float,
    td_lambda: float,
) -> dict[str, torch.Tensor]:
    """
    Compute a TD(lambda)-inspired temporal prediction loss while preserving
    non-prediction losses implemented by model.loss.
    """
    base_losses = model.loss(batch, sigreg_weight=sigreg_weight)
    out = model.forward(batch)

    if "pred_latent" not in out or "target_latent" not in out:
        raise KeyError("model.forward(batch) must return 'pred_latent' and 'target_latent'.")

    lambda_pred_loss, uniform_pred_loss, lambda_weight_sum, pred_loss_per_sample = temporal_weighted_mse(
        out["pred_latent"],
        out["target_latent"],
        out.get("mask"),
        td_lambda,
    )

    base_total = base_losses["total_loss"]
    base_pred = base_losses.get("pred_loss", uniform_pred_loss)
    extra_losses = base_total - base_pred
    total_loss = lambda_pred_loss + extra_losses

    losses = dict(base_losses)
    losses["total_loss"] = total_loss
    losses["pred_loss"] = lambda_pred_loss
    losses["pred_loss_uniform"] = uniform_pred_loss.detach()
    losses["lambda_weight_sum"] = lambda_weight_sum.detach()
    losses["pred_loss_per_sample"] = pred_loss_per_sample

    if "decoded_loss" not in losses:
        losses["decoded_loss"] = torch.zeros((), device=total_loss.device, dtype=total_loss.dtype)

    return losses


def build_smac_dataset(
    paths: list[str],
    config: TrainConfig,
    window_len: int,
    cap_metadata: Any,
    samples_per_epoch: int | None = None,
) -> SMACJEPADataset:
    return SMACJEPADataset(
        paths,
        context_len=config.context_len,
        mode="entity",
        window_mode=config.window_mode,
        window_len=window_len,
        samples_per_epoch=samples_per_epoch,
        seed=config.seed,
        max_agents=cap_metadata.max_agents,
        max_enemies=cap_metadata.max_enemies,
        max_actions=cap_metadata.max_actions,
        token_dim=cap_metadata.token_dim,
        dynamic_token_dim=cap_metadata.dynamic_token_dim,
        static_dim=cap_metadata.static_dim,
        entity_static_feat_size=cap_metadata.entity_static_feat_size,
    )


def make_category_concat_dataset(
    paths_by_category: dict[str, list[str]],
    config: TrainConfig,
    window_len: int,
    cap_metadata: Any,
    total_paths: int,
) -> tuple[ConcatDataset, list[str], dict[str, int]]:
    category_names = list(paths_by_category.keys())
    wrapped_datasets: list[Dataset] = []
    category_lengths: dict[str, int] = {}

    for category_id, category_name in enumerate(category_names):
        paths = paths_by_category[category_name]
        category_samples_per_epoch = None
        if config.window_mode == "random" and config.samples_per_epoch is not None:
            # Preserve the global samples_per_epoch approximately across all category datasets.
            share = len(paths) / max(total_paths, 1)
            category_samples_per_epoch = max(1, round(config.samples_per_epoch * share))

        ds = build_smac_dataset(
            paths,
            config,
            window_len,
            cap_metadata,
            samples_per_epoch=category_samples_per_epoch,
        )
        wrapped = CategoryDataset(ds, category_id)
        wrapped_len = len(wrapped)
        if wrapped_len <= 0:
            raise RuntimeError(
                f"Category {category_name!r} has {len(paths)} file(s) but produced 0 trainable windows/items. "
                "Check context_len/window_len and the underlying .npz lengths."
            )
        wrapped_datasets.append(wrapped)
        category_lengths[category_name] = wrapped_len

    if not wrapped_datasets:
        raise RuntimeError("Priority sampling produced no category datasets.")
    concat = ConcatDataset(wrapped_datasets)
    if len(concat) <= 0:
        raise RuntimeError("Priority sampling produced an empty ConcatDataset.")
    return concat, category_names, category_lengths


def compute_category_probs(
    category_names: list[str],
    ema_losses: dict[str, float],
    alpha: float,
    uniform_mix: float,
    min_prob: float,
) -> dict[str, float]:
    if not category_names:
        return {}
    if alpha < 0:
        raise ValueError("--category-priority-alpha must be >= 0")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("--category-uniform-mix must be in [0, 1]")
    n = len(category_names)
    if min_prob < 0:
        raise ValueError("--category-min-prob must be >= 0")
    if n * min_prob >= 1.0:
        raise ValueError(
            f"--category-min-prob={min_prob} is too high for {n} categories; "
            f"need n_categories * min_prob < 1. Try {0.5 / n:.4f} or lower."
        )

    losses = [max(float(ema_losses.get(name, 1.0)), 1e-12) for name in category_names]
    scores = [loss ** alpha for loss in losses]
    score_sum = sum(scores)
    loss_probs = [score / score_sum for score in scores]
    uniform_prob = 1.0 / n
    mixed = [(1.0 - uniform_mix) * p + uniform_mix * uniform_prob for p in loss_probs]

    # Hard probability floor while preserving sum=1.
    if min_prob > 0:
        remaining_mass = 1.0 - n * min_prob
        mixed_sum = sum(mixed)
        mixed = [min_prob + remaining_mass * (p / mixed_sum) for p in mixed]

    if not all(torch.isfinite(torch.tensor(mixed, dtype=torch.double)).tolist()):
        raise RuntimeError(f"Non-finite category probabilities computed: {mixed}")
    prob_sum = sum(mixed)
    if prob_sum <= 0:
        raise RuntimeError(f"Category probabilities sum to non-positive value: {mixed}")
    # Normalize away tiny float drift.
    mixed = [float(p / prob_sum) for p in mixed]
    return {name: float(prob) for name, prob in zip(category_names, mixed)}


def make_prioritized_loader(
    concat_dataset: ConcatDataset,
    category_names: list[str],
    category_lengths: dict[str, int],
    category_probs: dict[str, float],
    batch_size: int,
    num_workers: int,
    seed: int,
    epoch: int,
) -> DataLoader:
    sample_weights: list[float] = []
    for category_name in category_names:
        length = category_lengths[category_name]
        prob = category_probs[category_name]
        per_sample_weight = prob / max(length, 1)
        sample_weights.extend([per_sample_weight] * length)

    if len(sample_weights) != len(concat_dataset):
        raise RuntimeError(f"sample weight length {len(sample_weights)} != dataset length {len(concat_dataset)}")
    if len(sample_weights) == 0:
        raise RuntimeError("Priority sampler received 0 samples.")
    weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    if not torch.isfinite(weights_tensor).all():
        raise RuntimeError("Priority sampler weights contain NaN/Inf values.")
    if float(weights_tensor.sum()) <= 0.0:
        raise RuntimeError("Priority sampler weights sum to 0; cannot sample.")

    generator = torch.Generator()
    generator.manual_seed(seed + epoch * 1009)
    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(concat_dataset),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        concat_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )


def make_uniform_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def update_category_tracking(
    category_ids: torch.Tensor,
    per_sample_losses: torch.Tensor,
    category_names: list[str],
    sums: dict[str, float],
    counts: dict[str, int],
) -> None:
    category_ids_cpu = category_ids.detach().cpu().long()
    losses_cpu = per_sample_losses.detach().cpu().float()
    for category_id, loss in zip(category_ids_cpu.tolist(), losses_cpu.tolist()):
        if 0 <= category_id < len(category_names):
            name = category_names[category_id]
            sums[name] = sums.get(name, 0.0) + float(loss)
            counts[name] = counts.get(name, 0) + 1


def write_category_state(
    path: Path,
    epoch: int,
    category_names: list[str],
    category_lengths: dict[str, int],
    epoch_losses: dict[str, float],
    ema_losses: dict[str, float],
    category_probs_next: dict[str, float],
) -> None:
    payload = {
        "epoch": epoch,
        "categories": {
            name: {
                "num_items": category_lengths.get(name, 0),
                "epoch_pred_loss": epoch_losses.get(name),
                "ema_pred_loss": ema_losses.get(name),
                "sample_prob_next": category_probs_next.get(name),
            }
            for name in category_names
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.category_prioritized and args.split != "train":
        print(
            "warning: --category-prioritized is usually intended for --split train; "
            f"current split={args.split!r}",
            flush=True,
        )
    if not 0.0 <= args.category_priority_ema <= 1.0:
        raise SystemExit("--category-priority-ema must be in [0, 1]")

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
        "category_prioritized",
        "category_prefixes_json",
        "category_priority_alpha",
        "category_priority_ema",
        "category_uniform_mix",
        "category_min_prob",
        "category_warmup_epochs",
        "uncategorized_category_name",
        "priority_min_samples",
        "priority_metric",
        "allow_all_uncategorized",
    ]:
        config_args.pop(key)

    config = TrainConfig(**config_args)
    arch = resolved_arch(config)
    batch_size = int(arch["batch_size"])
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

    category_enabled = bool(args.category_prioritized or args.category_prefixes_json is not None)
    if args.category_prefixes_json is not None and not args.category_prioritized:
        print("category_prioritized enabled because category prefixes were provided", flush=True)
    category_names: list[str] = []
    category_lengths: dict[str, int] = {}
    category_concat_dataset: ConcatDataset | None = None
    category_probs: dict[str, float] = {}
    ema_category_losses: dict[str, float] = {}

    if category_enabled:
        category_prefixes = load_category_prefixes(args.category_prefixes_json, args.uncategorized_category_name)
        paths_by_category = categorize_paths(data_paths, category_prefixes, args.uncategorized_category_name)
        warn_and_validate_category_matches(
            paths_by_category,
            category_prefixes,
            args.uncategorized_category_name,
            args.allow_all_uncategorized,
        )
        if not paths_by_category:
            raise SystemExit("No category datasets were created. Check your data paths and category prefix JSON.")
        category_concat_dataset, category_names, category_lengths = make_category_concat_dataset(
            paths_by_category,
            config,
            window_len,
            cap_metadata,
            total_paths=len(data_paths),
        )
        category_probs = {name: 1.0 / len(category_names) for name in category_names}

        print(f"category_prioritized enabled total_items={len(category_concat_dataset)} categories={len(category_names)}", flush=True)
        for name in category_names:
            print(
                f"  category={name} files={len(paths_by_category[name])} items={category_lengths[name]} "
                f"initial_prob={category_probs[name]:.6f}",
                flush=True,
            )
    else:
        dataset = build_smac_dataset(
            data_paths,
            config,
            window_len,
            cap_metadata,
            samples_per_epoch=config.samples_per_epoch,
        )
        loader = make_uniform_loader(dataset, batch_size, config.num_workers)

    model = SMACJEPA(
        state_dim=cap_metadata.state_dim,
        n_agents=cap_metadata.n_agents,
        n_actions=cap_metadata.n_actions,
        latent_dim=int(arch["latent_dim"]),
        hidden_dim=int(arch["hidden_dim"]),
        action_dim=int(arch["action_dim"]),
        num_heads=int(arch["num_heads"]),
        mode=cap_metadata.mode,
        max_agents=cap_metadata.max_agents,
        max_enemies=cap_metadata.max_enemies,
        max_actions=cap_metadata.max_actions,
        token_dim=cap_metadata.token_dim,
        static_dim=cap_metadata.static_dim,
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
        "category_names": category_names,
        "category_lengths": category_lengths,
    }
    (out_dir / "config.json").write_text(json.dumps(saved_config, indent=2) + "\n", encoding="utf-8")

    wandb_run = None
    if wandb_enabled:
        if wandb is None:
            raise SystemExit(
                "W&B logging requested with --wandb, but wandb is not installed. Install it with: uv pip install wandb"
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
                    "state_dim": cap_metadata.state_dim,
                    "n_agents": cap_metadata.n_agents,
                    "n_actions": cap_metadata.n_actions,
                    "n_enemies": cap_metadata.n_enemies,
                    "ally_state_feat_size": cap_metadata.ally_state_feat_size,
                    "enemy_state_feat_size": cap_metadata.enemy_state_feat_size,
                    "ally_has_shields": cap_metadata.ally_has_shields,
                    "enemy_has_shields": cap_metadata.enemy_has_shields,
                    "num_unit_types": cap_metadata.num_unit_types,
                    "max_agents": cap_metadata.max_agents,
                    "max_enemies": cap_metadata.max_enemies,
                    "max_actions": cap_metadata.max_actions,
                    "token_dim": cap_metadata.token_dim,
                    "dynamic_token_dim": cap_metadata.dynamic_token_dim,
                    "static_dim": cap_metadata.static_dim,
                    "entity_static_feat_size": cap_metadata.entity_static_feat_size,
                    "mode": cap_metadata.mode,
                },
                "config": vars(args),
                "resolved_config": saved_config,
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": epoch_to_save,
                "global_step": global_step,
                "category_probs": category_probs,
                "ema_category_losses": ema_category_losses,
            },
            checkpoint_path,
        )

    logger = LossLogger(out_dir, "loss_log")
    epoch_logger = LossLogger(out_dir, "epoch_loss")

    step_rows: list[dict[str, float | int]] = []
    epoch_rows: list[dict[str, float | int]] = []
    model.train()

    for epoch in range(start_epoch, config.epochs + 1):
        if category_enabled:
            assert category_concat_dataset is not None
            if epoch <= args.category_warmup_epochs:
                active_loader = make_uniform_loader(category_concat_dataset, batch_size, config.num_workers)
                sampling_mode = "uniform_warmup"
            else:
                active_loader = make_prioritized_loader(
                    category_concat_dataset,
                    category_names,
                    category_lengths,
                    category_probs,
                    batch_size,
                    config.num_workers,
                    config.seed,
                    epoch,
                )
                sampling_mode = "category_prioritized"
        else:
            active_loader = loader
            sampling_mode = "uniform"

        try:
            active_loader_len = len(active_loader)
        except TypeError:
            active_loader_len = -1
        if active_loader_len == 0:
            raise RuntimeError(f"Epoch {epoch} active_loader has 0 batches; refusing to continue with no training.")
        print(
            f"epoch_start epoch={epoch} sampling_mode={sampling_mode} batches={active_loader_len}",
            flush=True,
        )
        epoch_sums: dict[str, float] = {}
        epoch_batches = 0
        category_loss_sums: dict[str, float] = {}
        category_loss_counts: dict[str, int] = {}
        category_sample_counts: dict[str, int] = {}

        for batch in active_loader:
            global_step += 1
            epoch_batches += 1
            batch = to_device(batch, device)
            category_ids = batch.pop("__category_id__", None)

            optimizer.zero_grad(set_to_none=True)
            autocast_context = torch.cuda.amp.autocast(enabled=amp_enabled) if device.type == "cuda" else nullcontext()

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

            if category_enabled:
                if category_ids is None:
                    raise RuntimeError(
                        "Category-prioritized training expected '__category_id__' in each batch, "
                        "but it was missing. Check CategoryDataset wrapping."
                    )
                if "pred_loss_per_sample" not in losses:
                    raise RuntimeError(
                        "Category-prioritized training needs 'pred_loss_per_sample' from lambda_jepa_losses."
                    )
                update_category_tracking(
                    category_ids,
                    losses["pred_loss_per_sample"],
                    category_names,
                    category_loss_sums,
                    category_loss_counts,
                )
                for cid in category_ids.detach().cpu().long().tolist():
                    if 0 <= cid < len(category_names):
                        cname = category_names[cid]
                        category_sample_counts[cname] = category_sample_counts.get(cname, 0) + 1

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

        if epoch_batches == 0:
            raise RuntimeError(f"Epoch {epoch} finished with 0 batches; refusing to save a misleading checkpoint.")

        epoch_row: dict[str, float | int] = {"epoch": epoch, "step": global_step}
        for key, value in epoch_sums.items():
            epoch_row[key] = value / max(epoch_batches, 1)

        epoch_category_losses: dict[str, float] = {}
        if category_enabled:
            for name in category_names:
                count = category_loss_counts.get(name, 0)
                if count > 0:
                    epoch_category_losses[name] = category_loss_sums[name] / count
                    if name not in ema_category_losses:
                        ema_category_losses[name] = epoch_category_losses[name]
                    else:
                        beta = args.category_priority_ema
                        ema_category_losses[name] = beta * ema_category_losses[name] + (1.0 - beta) * epoch_category_losses[name]

            category_probs = compute_category_probs(
                category_names,
                ema_category_losses,
                alpha=args.category_priority_alpha,
                uniform_mix=args.category_uniform_mix,
                min_prob=args.category_min_prob,
            )

            if not category_loss_counts:
                raise RuntimeError(
                    f"Epoch {epoch} completed with category priority enabled but no category losses were tracked."
                )

            for name in category_names:
                safe_name = name.replace("/", "_")
                epoch_row[f"category_{safe_name}_sample_count"] = category_sample_counts.get(name, 0)
                if name in epoch_category_losses:
                    epoch_row[f"category_{safe_name}_pred_loss"] = epoch_category_losses[name]
                if name in ema_category_losses:
                    epoch_row[f"category_{safe_name}_ema_pred_loss"] = ema_category_losses[name]
                epoch_row[f"category_{safe_name}_sample_prob_next"] = category_probs[name]

            write_category_state(
                out_dir / "category_sampling_state.json",
                epoch,
                category_names,
                category_lengths,
                epoch_category_losses,
                ema_category_losses,
                category_probs,
            )

            if wandb_run is not None:
                wandb_payload = {}
                for name in category_names:
                    if name in epoch_category_losses:
                        wandb_payload[f"category/{name}/pred_loss"] = epoch_category_losses[name]
                    if name in ema_category_losses:
                        wandb_payload[f"category/{name}/ema_pred_loss"] = ema_category_losses[name]
                    wandb_payload[f"category/{name}/sample_prob_next"] = category_probs[name]
                wandb_run.log(wandb_payload, step=global_step)

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
        if category_enabled:
            print("category_sample_counts " + json.dumps(category_sample_counts, sort_keys=True), flush=True)
            print("category_sampling_next " + json.dumps(category_probs, sort_keys=True), flush=True)

        epoch_checkpoint_path = out_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        save_checkpoint(epoch, epoch_checkpoint_path)
        save_checkpoint(epoch, out_dir / "checkpoint.pt")
        print(f"saved_checkpoint {epoch_checkpoint_path} and {out_dir / 'checkpoint.pt'}", flush=True)

    write_svg_line_plot(epoch_rows, "epoch", "total_loss", "Average Total Loss Per Epoch", out_dir / "loss_by_epoch.svg")
    write_svg_line_plot(epoch_rows, "epoch", "pred_loss", "Average Prediction Loss Per Epoch", out_dir / "pred_loss_by_epoch.svg")
    write_svg_line_plot(step_rows, "step", "pred_loss", "Prediction Loss Per Training Step", out_dir / "pred_loss_by_step.svg")
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
        if category_enabled:
            wandb_run.save(str(out_dir / "category_sampling_state.json"))
        wandb_run.finish()


if __name__ == "__main__":
    main()
