from __future__ import annotations

"""
Train SMAC-JEPA with Markov recursive rollout + sequential RNN memory + enemy visibility masking.

Parallel script. Does not modify train_markov_rollout.py.

Assumes:
    smac_jepa/data/markov_rollout_visibility_dataset.py
    smac_jepa/modules/rollout_memory.py

Run as:
    python -m smac_jepa.train_markov_rollout_rnn_visibility_mask

Rollout with memory:
    z0 = encode(s_t)
    m0 = zeros

    z0_mem = memory.condition(z0, m0)
    z1_hat = predictor(z0_mem, a_t)
    m1 = memory.update(z1_hat, m0)

    z1_mem = memory.condition(z1_hat, m1)
    z2_hat = predictor(z1_mem, a_{t+1})
    m2 = memory.update(z2_hat, m1)

    ...

The memory module is trained through the normal rollout losses because its outputs
affect predictor inputs. No separate RNN-specific loss is needed.
"""

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from smac_jepa.data import SMACJEPADataset, load_manifest, load_manifest_all
from smac_jepa.data.markov_rollout_visibility_dataset import VisibilityMarkovRolloutSMACJEPADataset
from smac_jepa.jepa import SMACJEPA
from smac_jepa.modules import sigreg_loss
from smac_jepa.modules.rollout_memory import EntityRolloutGRUMemory
from smac_jepa.presets import MODEL_PRESETS, get_model_preset
from smac_jepa.utils import set_seed
from smac_jepa.utils.logging import LossLogger
from smac_jepa.utils.plots import write_svg_line_plot

try:
    import wandb
except ImportError:
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SMAC-JEPA seqmem with R2-Dreamer-style Barlow redundancy reduction")

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

    parser.add_argument("--rollout-window", type=int, default=20)
    parser.add_argument("--rollout-horizon", type=int, default=5)
    parser.add_argument("--window-mode", choices=["sequential", "random"], default="random")
    parser.add_argument("--samples-per-epoch", type=int, default=None)

    parser.add_argument(
        "--enemy-visibility-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Mask enemy entity tokens in rollout input observations when they are not visible "
            "from any alive ally. Targets remain full-state. Default: enabled."
        ),
    )
    parser.add_argument(
        "--enemy-sight-range",
        type=float,
        default=9.0,
        help="Ally sight range in SMACLite map units for enemy visibility masking. Default: 9.0.",
    )

    parser.add_argument("--temporal-loss", choices=["uniform", "lambda", "flat-decay"], default="lambda")
    parser.add_argument("--td-lambda", "--temporal-lambda", dest="td_lambda", type=float, default=0.9)
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
    parser.add_argument(
        "--barlow-weight",
        type=float,
        default=0.0,
        help=(
            "Scale beta_BT applied to the R2-Dreamer-style Barlow loss. "
            "The paper uses 0.05."
        ),
    )
    parser.add_argument(
        "--barlow-lambda",
        type=float,
        default=5e-4,
        help=(
            "Alpha applied to off-diagonal cross-correlation terms. "
            "The R2-Dreamer paper uses 5e-4."
        ),
    )
    parser.add_argument(
        "--barlow-mode",
        choices=["global", "per-horizon"],
        default="global",
        help=(
            "global: compute one cross-correlation matrix over all valid "
            "batch/start/horizon/entity samples, matching R2-Dreamer's B*T treatment. "
            "per-horizon: compute one matrix per rollout horizon and combine using "
            "the temporal weights."
        ),
    )
    parser.add_argument(
        "--barlow-target-ratio",
        type=float,
        default=0.30,
        help=(
            "Maximum desired weighted Barlow contribution relative to pred_loss. "
            "For example 0.30 means weighted_barlow_loss <= 0.30 * pred_loss. "
            "The actual coefficient is adapted per batch and never exceeds --barlow-weight."
        ),
    )
    parser.add_argument(
        "--sigreg-target-ratio",
        type=float,
        default=0.0,
        help=(
            "Maximum desired weighted SIGReg contribution relative to pred_loss. "
            "The actual coefficient is adapted per batch and never exceeds --sigreg-weight."
        ),
    )
    parser.add_argument(
        "--regularizer-ratio-warmup-epochs",
        type=int,
        default=2,
        help=(
            "Linearly warm relative regularizer budgets during the first N epochs. "
            "Epoch 1 uses 1/N of the target ratios and epoch N reaches the full ratios."
        ),
    )
    parser.add_argument(
        "--regularizer-reference-floor",
        type=float,
        default=0.05,
        help=(
            "Detached lower bound used only for scale-restoring regularizers "
            "(SIGReg and variance floor). This prevents them from vanishing "
            "when raw latent MSE becomes tiny through latent-scale shrinkage. "
            "Barlow remains capped against live pred_loss."
        ),
    )
    parser.add_argument(
        "--variance-floor-weight",
        type=float,
        default=0.0,
        help="Maximum coefficient for the latent standard-deviation floor.",
    )
    parser.add_argument(
        "--variance-floor-target-ratio",
        type=float,
        default=0.0,
        help=(
            "Desired variance-floor contribution relative to "
            "max(pred_loss, regularizer_reference_floor)."
        ),
    )
    parser.add_argument(
        "--latent-std-floor",
        type=float,
        default=None,
        help=(
            "Explicit latent standard-deviation floor. When omitted, it is "
            "calibrated before training as --latent-std-floor-ratio times the "
            "mean per-dimension standard deviation of the initialized encoder."
        ),
    )
    parser.add_argument(
        "--latent-std-floor-ratio",
        type=float,
        default=0.8,
        help="Automatic variance-floor ratio relative to the pre-training encoder scale.",
    )
    parser.add_argument(
        "--scale-calibration-batches",
        type=int,
        default=20,
        help="Number of training batches used to measure the initial encoder scale.",
    )
    parser.add_argument(
        "--sigreg-knots",
        type=int,
        default=17,
        help="Number of characteristic-function knots used by stable SIGReg.",
    )
    parser.add_argument(
        "--sigreg-num-proj",
        type=int,
        default=1024,
        help=(
            "Number of random SIGReg projections. This preserves the repository "
            "default while processing them in small float32 chunks."
        ),
    )
    parser.add_argument(
        "--sigreg-proj-chunk",
        type=int,
        default=64,
        help="Number of SIGReg projections processed simultaneously in float32.",
    )
    parser.add_argument(
        "--sigreg-max-samples",
        type=int,
        default=0,
        help="Maximum valid latent samples used by SIGReg per batch; 0 uses every valid sample.",
    )
    parser.add_argument(
        "--health-check-after-steps",
        type=int,
        default=100,
        help="Start enforcing latent-scale health checks after this many optimizer steps.",
    )
    parser.add_argument(
        "--abort-target-std-ratio",
        type=float,
        default=0.6,
        help=(
            "Abort when target latent std falls below this fraction of its "
            "pre-training calibrated value after the health-check warmup."
        ),
    )
    parser.add_argument(
        "--abort-pred-target-std-ratio",
        type=float,
        default=0.05,
        help=(
            "Abort when predicted latent std falls below this fraction of target "
            "latent std after the health-check warmup."
        ),
    )
    parser.add_argument("--decoder-weight", type=float, default=1.0)
    parser.add_argument("--presence-weight", type=float, default=1.0)
    parser.add_argument(
        "--action-conditioned-memory",
        action="store_true",
        help="Use action-conditioned GRU memory update, closer to RSSM h_{t+1}=f(h_t,z_t,a_t).",
    )
    parser.add_argument(
        "--one-step-weight",
        type=float,
        default=0.0,
        help="Extra teacher-forced one-step latent prediction loss weight. 0 disables.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["full", "observed"],
        default="full",
        help=(
            "full: predict full-state targets from masked inputs. "
            "observed: predict masked/observable targets for easier visibility diagnostic."
        ),
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # RNN memory args.
    parser.add_argument("--rollout-memory-dim", type=int, default=128)
    parser.add_argument("--rollout-memory-hidden-dim", type=int, default=None)
    parser.add_argument(
        "--rollout-memory-no-residual",
        action="store_true",
        help="Disable residual z + memory correction. Default uses residual.",
    )

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
        start = flat_decay_start
        if start is None:
            start = max(1, length // 2)
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


def weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    pred/target: [B, P, H, E, D]
    mask:        [B, P, H, E, 1]
    weights:     [H]
    """
    w = weights.view(1, 1, -1, 1, 1)
    weighted_mask = mask * w
    denom = weighted_mask.sum().clamp_min(1.0) * pred.shape[-1]
    return ((pred - target).pow(2) * weighted_mask).sum() / denom


def weighted_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
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


def pooled_action_context(
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    *,
    n_actions: int,
) -> torch.Tensor:
    """
    Convert per-agent actions into one batch-level action context.

    Expected common cases:
        actions [B, A]       integer action ids
        actions [B, A, C]    one-hot or float action features

    Returns:
        [B, n_actions]
    """
    if actions.dim() == 3:
        # Usually [B, agents, n_actions] one-hot/int/bool.
        x = actions.float()
        if x.shape[-1] != n_actions:
            if x.shape[-1] > n_actions:
                x = x[..., :n_actions]
            else:
                pad = torch.zeros(*x.shape[:-1], n_actions - x.shape[-1], device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=-1)
    elif actions.dim() == 2:
        # Usually [B, agents] integer ids.
        ids = actions.long().clamp(min=0, max=n_actions - 1)
        x = F.one_hot(ids, num_classes=n_actions).float()
    else:
        raise ValueError(f"Unsupported action shape for memory update: {tuple(actions.shape)}")

    if action_mask is not None:
        mask = action_mask.float()
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        x = x * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return x.sum(dim=1) / denom

    return x.mean(dim=1)


class ActionConditionedEntityRolloutGRUMemory(nn.Module):
    """
    Entity-wise GRU memory that updates from both latent state and pooled joint action.

    This is closer to RSSM-style deterministic memory:
        h_{t+1} = f(h_t, z_t, a_t)

    It intentionally keeps the same condition/update interface style as
    EntityRolloutGRUMemory, but update additionally accepts action/action_mask.
    """

    uses_action = True

    def __init__(
        self,
        *,
        latent_dim: int,
        memory_dim: int,
        n_actions: int,
        hidden_dim: int | None = None,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.memory_dim = int(memory_dim)
        self.n_actions = int(n_actions)
        self.residual = bool(residual)

        hidden = int(hidden_dim or max(latent_dim, memory_dim))
        self.action_proj = nn.Sequential(
            nn.Linear(self.n_actions, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, memory_dim),
        )
        self.gru = nn.GRUCell(latent_dim + memory_dim, memory_dim)
        self.condition_net = nn.Sequential(
            nn.Linear(latent_dim + memory_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def initial_memory(
        self,
        batch_size: int,
        entities: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, entities, self.memory_dim, device=device, dtype=dtype)

    def condition(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        entity_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        correction = self.condition_net(torch.cat([z, memory], dim=-1))
        out = z + correction if self.residual else correction
        if entity_mask is not None:
            out = out * entity_mask.unsqueeze(-1)
        return out

    def update(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        entity_mask: torch.Tensor | None = None,
        *,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, entities, _ = z.shape

        if action is None:
            action_ctx = torch.zeros(bsz, self.n_actions, device=z.device, dtype=z.dtype)
        else:
            action_ctx = pooled_action_context(action, action_mask, n_actions=self.n_actions).to(dtype=z.dtype)

        action_emb = self.action_proj(action_ctx).unsqueeze(1).expand(-1, entities, -1)
        gru_in = torch.cat([z, action_emb], dim=-1)

        new_memory = self.gru(
            gru_in.reshape(bsz * entities, -1),
            memory.reshape(bsz * entities, -1),
        ).reshape(bsz, entities, self.memory_dim)

        if entity_mask is not None:
            keep = entity_mask.unsqueeze(-1).to(dtype=torch.bool)
            new_memory = torch.where(keep, new_memory, memory)

        return new_memory



def stable_sigreg_loss(
    latents: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    knots: int,
    num_proj: int,
    proj_chunk: int,
    max_samples: int,
) -> torch.Tensor:
    """
    Numerically stable implementation of the repository's SIGReg statistic.

    Differences from the original functional wrapper:
    - always computes in float32, even under AMP;
    - processes random projections in chunks;
    - optionally subsamples valid latent vectors to bound memory;
    - raises on non-finite output instead of allowing 0 * inf -> NaN.
    """
    if knots < 2:
        raise ValueError("sigreg knots must be at least 2")
    if num_proj < 1:
        raise ValueError("sigreg num_proj must be at least 1")
    if proj_chunk < 1:
        raise ValueError("sigreg proj_chunk must be at least 1")

    if mask is not None:
        valid = mask.reshape(-1) > 0
        z = latents.reshape(-1, latents.shape[-1])[valid]
    else:
        z = latents.reshape(-1, latents.shape[-1])

    if z.shape[0] < 2:
        return latents.float().sum() * 0.0

    z = z.float()
    if max_samples > 0 and z.shape[0] > max_samples:
        indices = torch.randperm(z.shape[0], device=z.device)[:max_samples]
        z = z[indices]

    sample_count = int(z.shape[0])
    latent_dim = int(z.shape[-1])

    t = torch.linspace(0.0, 3.0, knots, device=z.device, dtype=torch.float32)
    dt = 3.0 / float(knots - 1)
    trapezoid = torch.full((knots,), 2.0 * dt, device=z.device)
    trapezoid[[0, -1]] = dt
    phi = torch.exp(-0.5 * t.square())
    weights = trapezoid * phi

    statistic_sum = z.new_zeros(())
    completed = 0

    while completed < num_proj:
        current = min(proj_chunk, num_proj - completed)
        projections = torch.randn(
            latent_dim,
            current,
            device=z.device,
            dtype=torch.float32,
        )
        projections = projections / projections.norm(
            p=2, dim=0, keepdim=True
        ).clamp_min(1e-8)

        projected = z @ projections
        x_t = projected.unsqueeze(-1) * t
        err = (x_t.cos().mean(dim=0) - phi).square()
        err = err + x_t.sin().mean(dim=0).square()
        per_projection = (err @ weights) * float(sample_count)
        statistic_sum = statistic_sum + per_projection.sum()
        completed += current

    result = statistic_sum / float(num_proj)
    if not torch.isfinite(result):
        raise FloatingPointError(
            f"Stable SIGReg became non-finite: {result.detach().cpu().item()}"
        )
    return result


@torch.no_grad()
def calibrate_initial_latent_scale(
    model: SMACJEPA,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
    target_mode: str,
) -> dict[str, float]:
    """
    Measure encoder scale before the first optimizer step.

    The variance floor is tied to this measured scale rather than an arbitrary
    value such as 1.0.
    """
    if max_batches < 1:
        raise ValueError("--scale-calibration-batches must be at least 1")

    was_training = model.training
    model.eval()

    sum_z = None
    sum_z2 = None
    count = 0
    used_batches = 0

    for batch in loader:
        if used_batches >= max_batches:
            break
        used_batches += 1
        batch = to_device(batch, device)

        input_entity = batch["entity_seq"]
        input_mask = batch["entity_mask_seq"]

        if target_mode == "full":
            target_entity = batch.get("target_entity_seq", input_entity)
            target_mask = batch.get("target_entity_mask_seq", input_mask)
        else:
            target_entity = input_entity
            target_mask = input_mask

        # Encoder weights are float32. Disable AMP for a reliable reference.
        with torch.cuda.amp.autocast(enabled=False) if device.type == "cuda" else nullcontext():
            input_latent = model.encoder(input_entity.float(), input_mask.float()).float()
            target_latent = model.encoder(target_entity.float(), target_mask.float()).float()

        combined = torch.cat([input_latent, target_latent], dim=1)
        combined_mask = torch.cat([input_mask, target_mask], dim=1).bool()
        flat = combined[combined_mask]
        if flat.shape[0] == 0:
            continue

        batch_sum = flat.sum(dim=0)
        batch_sum2 = flat.square().sum(dim=0)
        sum_z = batch_sum if sum_z is None else sum_z + batch_sum
        sum_z2 = batch_sum2 if sum_z2 is None else sum_z2 + batch_sum2
        count += int(flat.shape[0])

    if was_training:
        model.train()

    if count < 2 or sum_z is None or sum_z2 is None:
        raise RuntimeError("Could not calibrate latent scale: fewer than two valid latents")

    mean = sum_z / float(count)
    variance = (sum_z2 / float(count) - mean.square()).clamp_min(0.0)
    std = variance.sqrt()

    return {
        "samples": float(count),
        "batches": float(used_batches),
        "std_mean": float(std.mean().cpu()),
        "std_min": float(std.min().cpu()),
        "std_median": float(std.median().cpu()),
        "variance_mean": float(variance.mean().cpu()),
    }


def assert_finite_loss(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        scalar = value.detach().float().mean().cpu().item()
        raise FloatingPointError(f"{name} became non-finite: {scalar}")


def capped_relative_weight(
    raw_loss: torch.Tensor,
    reference_loss: torch.Tensor,
    *,
    max_weight: float,
    target_ratio: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Return a detached, per-batch coefficient whose weighted scalar contribution
    cannot exceed target_ratio * reference_loss.

    The configured max_weight remains an upper bound. This makes the objective
    insensitive to the raw numerical scale of SIGReg or Barlow while preserving
    their gradients through raw_loss itself.
    """
    if max_weight <= 0.0 or target_ratio <= 0.0:
        return raw_loss.new_zeros(())

    assert_finite_loss("raw regularizer loss", raw_loss)
    assert_finite_loss("regularizer reference loss", reference_loss)

    desired_contribution = float(target_ratio) * reference_loss.detach()
    ratio_weight = desired_contribution / raw_loss.detach().abs().clamp_min(eps)
    return torch.minimum(
        ratio_weight,
        raw_loss.new_tensor(float(max_weight)),
    ).detach()


def regularizer_warmup_factor(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))


def masked_latent_statistics(
    latents: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    std_floor: float,
) -> dict[str, torch.Tensor]:
    """Compute per-dimension latent statistics using only valid samples."""
    zero = latents.sum() * 0.0
    valid = valid_mask.bool()
    if int(valid.sum().item()) < 2:
        return {
            "std_mean": zero.detach(),
            "std_min": zero.detach(),
            "std_max": zero.detach(),
            "variance_mean": zero.detach(),
            "norm_mean": zero.detach(),
            "fraction_std_below_0p1": zero.detach(),
            "fraction_std_below_0p5": zero.detach(),
            "fraction_std_below_floor": zero.detach(),
            "samples": zero.detach(),
        }

    flat = latents[valid].float()
    std = flat.std(dim=0, unbiased=False)
    return {
        "std_mean": std.mean().detach(),
        "std_min": std.min().detach(),
        "std_max": std.max().detach(),
        "variance_mean": std.pow(2).mean().detach(),
        "norm_mean": flat.norm(dim=-1).mean().detach(),
        "fraction_std_below_0p1": (std < 0.1).float().mean().detach(),
        "fraction_std_below_0p5": (std < 0.5).float().mean().detach(),
        "fraction_std_below_floor": (
            std < float(std_floor)
        ).float().mean().detach(),
        "samples": torch.tensor(float(flat.shape[0]), device=latents.device),
    }


def masked_variance_floor_loss(
    latents: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    std_floor: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    VICReg-style variance floor:
        mean_d ReLU(std_floor - std(z_d))
    """
    valid = valid_mask.bool()
    stats = masked_latent_statistics(
        latents,
        valid_mask,
        std_floor=std_floor,
    )
    if int(valid.sum().item()) < 2:
        return latents.sum() * 0.0, stats

    flat = latents[valid].float()
    std = flat.std(dim=0, unbiased=False)
    return torch.relu(float(std_floor) - std).mean(), stats


class R2EntityProjector(nn.Module):
    """
    R2-Dreamer-style linear projector.

    The JEPA analogue of the RSSM state s_t=(h_t,z_t) is:
        concat(predicted_entity_latent, recurrent_entity_memory)

    The projector maps that complete state back to the encoder latent dimension.
    """

    def __init__(self, latent_dim: int, memory_dim: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.memory_dim = int(memory_dim)
        self.linear = nn.Linear(
            self.latent_dim + self.memory_dim,
            self.latent_dim,
            bias=False,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state)


def _barlow_from_flat(
    projected_state: torch.Tensor,
    target_embedding: torch.Tensor,
    *,
    alpha: float,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """
    R2-Dreamer / Barlow-Twins objective on [N, D] paired samples.

    Target embeddings are always stop-gradient. Statistics and correlation are
    computed in float32 for stability under AMP.
    """
    zero = projected_state.sum() * 0.0
    if projected_state.ndim != 2 or target_embedding.ndim != 2:
        raise ValueError(
            "Barlow inputs must be [N, D], got "
            f"{tuple(projected_state.shape)} and {tuple(target_embedding.shape)}"
        )
    if projected_state.shape != target_embedding.shape:
        raise ValueError(
            "Barlow projected and target shapes must match, got "
            f"{tuple(projected_state.shape)} and {tuple(target_embedding.shape)}"
        )
    if projected_state.shape[0] < 2:
        return {
            "loss": zero,
            "raw_loss": zero.detach(),
            "invariance": zero.detach(),
            "redundancy": zero.detach(),
            "diag_mean": zero.detach(),
            "offdiag_rms": zero.detach(),
            "samples": torch.tensor(
                float(projected_state.shape[0]),
                device=projected_state.device,
            ),
        }

    x1 = projected_state.float()
    x2 = target_embedding.detach().float()

    # Match the official implementation: standardize each feature over samples.
    x1 = (x1 - x1.mean(dim=0)) / (x1.std(dim=0) + eps)
    x2 = (x2 - x2.mean(dim=0)) / (x2.std(dim=0) + eps)

    n = x1.shape[0]
    cross_corr = (x1.transpose(0, 1) @ x2) / float(n)

    diagonal = torch.diagonal(cross_corr)
    invariance = (diagonal - 1.0).pow(2).sum()

    offdiag_mask = ~torch.eye(
        cross_corr.shape[0],
        dtype=torch.bool,
        device=cross_corr.device,
    )
    offdiag_values = cross_corr[offdiag_mask]
    redundancy = offdiag_values.pow(2).sum()

    raw_loss = invariance + float(alpha) * redundancy
    # The official objective is a sum over D diagonal dimensions. Dividing by D
    # changes only the global scale, not the invariance/redundancy trade-off, and
    # makes the outer coefficient interpretable across latent dimensions.
    loss = raw_loss / float(cross_corr.shape[0])
    return {
        "loss": loss,
        "raw_loss": raw_loss.detach(),
        "invariance": invariance.detach(),
        "redundancy": redundancy.detach(),
        "diag_mean": diagonal.mean().detach(),
        "offdiag_rms": offdiag_values.pow(2).mean().sqrt().detach(),
        "samples": torch.tensor(float(n), device=cross_corr.device),
    }


def masked_barlow_loss(
    projector: R2EntityProjector,
    imagined_state: torch.Tensor,
    target_embedding: torch.Tensor,
    valid_entity_mask: torch.Tensor,
    *,
    alpha: float,
    mode: str,
    temporal_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    imagined_state:   [B, P, H, E, latent_dim + memory_dim]
    target_embedding: [B, P, H, E, latent_dim]
    valid_entity_mask:[B, P, H, E]

    Padded, absent and invalid-timestep entities never enter the correlation.
    """
    if mode == "global":
        projected = projector(imagined_state)
        valid = valid_entity_mask.bool()
        return _barlow_from_flat(
            projected[valid],
            target_embedding[valid],
            alpha=alpha,
        )

    if mode != "per-horizon":
        raise ValueError(f"Unknown Barlow mode: {mode}")

    horizon = imagined_state.shape[2]
    weighted_loss = imagined_state.sum() * 0.0
    weight_sum = imagined_state.new_tensor(0.0)
    raw_loss_sum = imagined_state.new_tensor(0.0)
    invariance_sum = imagined_state.new_tensor(0.0)
    redundancy_sum = imagined_state.new_tensor(0.0)
    diag_sum = imagined_state.new_tensor(0.0)
    offdiag_rms_sum = imagined_state.new_tensor(0.0)
    sample_sum = imagined_state.new_tensor(0.0)
    valid_horizons = 0

    for step in range(horizon):
        valid = valid_entity_mask[:, :, step].bool()
        if int(valid.sum().item()) < 2:
            continue

        projected = projector(imagined_state[:, :, step])
        stats = _barlow_from_flat(
            projected[valid],
            target_embedding[:, :, step][valid],
            alpha=alpha,
        )
        weight = temporal_weights[step]
        weighted_loss = weighted_loss + weight * stats["loss"]
        weight_sum = weight_sum + weight
        raw_loss_sum = raw_loss_sum + stats["raw_loss"]
        invariance_sum = invariance_sum + stats["invariance"]
        redundancy_sum = redundancy_sum + stats["redundancy"]
        diag_sum = diag_sum + stats["diag_mean"]
        offdiag_rms_sum = offdiag_rms_sum + stats["offdiag_rms"]
        sample_sum = sample_sum + stats["samples"]
        valid_horizons += 1

    if valid_horizons == 0:
        zero = imagined_state.sum() * 0.0
        return {
            "loss": zero,
            "raw_loss": zero.detach(),
            "invariance": zero.detach(),
            "redundancy": zero.detach(),
            "diag_mean": zero.detach(),
            "offdiag_rms": zero.detach(),
            "samples": zero.detach(),
        }

    return {
        "loss": weighted_loss / weight_sum.clamp_min(1e-8),
        "raw_loss": (raw_loss_sum / valid_horizons).detach(),
        "invariance": (invariance_sum / valid_horizons).detach(),
        "redundancy": (redundancy_sum / valid_horizons).detach(),
        "diag_mean": (diag_sum / valid_horizons).detach(),
        "offdiag_rms": (offdiag_rms_sum / valid_horizons).detach(),
        "samples": sample_sum.detach(),
    }



def markov_rollout_rnn_losses(
    model: SMACJEPA,
    memory_module: EntityRolloutGRUMemory,
    barlow_projector: R2EntityProjector,
    batch: dict[str, torch.Tensor],
    *,
    rollout_window: int,
    rollout_horizon: int,
    temporal_loss_mode: str,
    td_lambda: float,
    flat_decay_start: int | None,
    flat_decay_final_weight: float,
    sigreg_weight: float,
    sigreg_target_ratio: float,
    sigreg_knots: int,
    sigreg_num_proj: int,
    sigreg_proj_chunk: int,
    sigreg_max_samples: int,
    variance_floor_weight: float,
    variance_floor_target_ratio: float,
    latent_std_floor: float,
    regularizer_reference_floor: float,
    barlow_weight: float,
    barlow_target_ratio: float,
    barlow_lambda: float,
    barlow_mode: str,
    decoder_weight: float,
    presence_weight: float,
    one_step_weight: float,
    target_mode: str,
    detach_rollout_targets: bool,
    unweighted_aux_losses: bool,
) -> dict[str, torch.Tensor]:
    """
    Compute recursive Markov rollout losses with sequential RNN memory.

    Previous RNN version:
        every rollout start inside the P-window began with zero memory.

    This version:
        a main real-history memory is carried across the P real timesteps.
        For start t, memory contains information from real observed states
        before t inside the sampled window.

    For each start t:
        1. Copy current real-history memory into rollout_memory.
        2. Run H-step imagined rollout from z_t.
        3. Inside the H-step rollout, memory is updated using predicted latents.
        4. After the local rollout, update the main memory using real observed z_t.

    Note:
        EntityRolloutGRUMemory.update currently accepts latent + mask, not action.
        So previous actions still condition the predictor, but are not directly
        written into memory unless the memory module is extended.
    """
    entity_seq = batch["entity_seq"]                       # visibility-masked inputs
    entity_mask_seq = batch["entity_mask_seq"]             # visibility-masked input masks
    if target_mode == "observed":
        # Easier diagnostic: target only the masked/observable future.
        target_entity_seq_full = entity_seq
        target_entity_mask_seq_full = entity_mask_seq
    elif target_mode == "full":
        # Harder belief-state target: predict full future state from masked inputs.
        target_entity_seq_full = batch.get("target_entity_seq", entity_seq)
        target_entity_mask_seq_full = batch.get("target_entity_mask_seq", entity_mask_seq)
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")
    action_seq = batch["action_seq"]
    action_mask_seq = batch["action_mask_seq"]
    state_mask = batch["state_mask"]
    static_condition = batch.get("static_condition")

    bsz = entity_seq.shape[0]
    p = int(rollout_window)
    h = int(rollout_horizon)

    input_latents = model.encoder(entity_seq, entity_mask_seq)  # [B, P+H+1, E, D]
    target_latents = model.encoder(target_entity_seq_full, target_entity_mask_seq_full)
    _, _, entities, latent_dim = input_latents.shape

    main_memory = memory_module.initial_memory(
        bsz,
        entities,
        device=entity_seq.device,
        dtype=input_latents.dtype,
    )

    static_flat = static_condition if static_condition is not None else None

    pred_by_start: list[torch.Tensor] = []
    target_by_start: list[torch.Tensor] = []
    target_entity_by_start: list[torch.Tensor] = []
    target_entity_mask_by_start: list[torch.Tensor] = []
    slot_mask_by_start: list[torch.Tensor] = []
    valid_by_start: list[torch.Tensor] = []
    imagined_state_by_start: list[torch.Tensor] = []
    memory_norms: list[torch.Tensor] = []

    for start_idx in range(p):
        # Real observed latent at the rollout start.
        z_start = input_latents[:, start_idx]               # [B, E, D]
        start_entity_mask = entity_mask_seq[:, start_idx]   # [B, E]

        # Local rollout starts from the current real-history memory.
        # Updates inside this local rollout should not replace the main memory.
        rollout_memory = main_memory
        z = z_start
        current_entity_mask = start_entity_mask

        pred_steps: list[torch.Tensor] = []
        target_steps: list[torch.Tensor] = []
        target_entity_steps: list[torch.Tensor] = []
        target_entity_mask_steps: list[torch.Tensor] = []
        slot_mask_steps: list[torch.Tensor] = []
        valid_steps: list[torch.Tensor] = []
        imagined_state_steps: list[torch.Tensor] = []

        for step in range(h):
            action_idx = start_idx + step
            target_idx = start_idx + step + 1

            action_h = action_seq[:, action_idx : action_idx + 1]             # [B, 1, A, Act]
            action_mask_h = action_mask_seq[:, action_idx : action_idx + 1]   # [B, 1, A]
            valid_h = state_mask[:, target_idx]                               # [B]

            timestep_mask_h = torch.ones(
                (bsz, 1),
                device=entity_seq.device,
                dtype=entity_seq.dtype,
            )

            entity_mask_h = current_entity_mask.unsqueeze(1)                  # [B, 1, E]

            z_conditioned = memory_module.condition(z, rollout_memory, current_entity_mask)

            pred_h = model.predictor(
                z_conditioned.unsqueeze(1),
                action_h,
                action_mask_h,
                timestep_mask_h,
                entity_mask_h,
                static_flat,
            )[:, 0]  # [B, E, D]

            pred_h = pred_h * current_entity_mask.unsqueeze(-1)

            target_mask_h = target_entity_mask_seq_full[:, target_idx]  # [B, E]

            pred_steps.append(pred_h)
            target_steps.append(target_latents[:, target_idx])
            target_entity_steps.append(target_entity_seq_full[:, target_idx])
            target_entity_mask_steps.append(target_mask_h)
            slot_mask_steps.append(batch["entity_slot_mask_seq"][:, target_idx])
            valid_steps.append(valid_h)

            # Imagined rollout memory: future observations are unavailable,
            # so update using predicted latent.
            if getattr(memory_module, "uses_action", False):
                rollout_memory = memory_module.update(
                    pred_h,
                    rollout_memory,
                    target_mask_h,
                    action=action_h[:, 0],
                    action_mask=action_mask_h[:, 0],
                )
            else:
                rollout_memory = memory_module.update(pred_h, rollout_memory, target_mask_h)
            memory_norms.append(rollout_memory.detach().float().norm(dim=-1).mean())

            # R2 analogue of s_t=(h_t,z_t), aligned with target_idx:
            # predicted latent plus the recurrent memory after consuming it.
            imagined_state_steps.append(torch.cat([pred_h, rollout_memory], dim=-1))

            z = pred_h
            current_entity_mask = target_mask_h

        pred_by_start.append(torch.stack(pred_steps, dim=1))                       # [B, H, E, D]
        target_by_start.append(torch.stack(target_steps, dim=1))                   # [B, H, E, D]
        target_entity_by_start.append(torch.stack(target_entity_steps, dim=1))      # [B, H, E, F]
        target_entity_mask_by_start.append(torch.stack(target_entity_mask_steps, dim=1))  # [B, H, E]
        slot_mask_by_start.append(torch.stack(slot_mask_steps, dim=1))             # [B, H, E]
        valid_by_start.append(torch.stack(valid_steps, dim=1))                     # [B, H]
        imagined_state_by_start.append(
            torch.stack(imagined_state_steps, dim=1)
        )                                                                           # [B, H, E, D+M]

        # Real-history memory update: after evaluating rollouts from start_idx,
        # advance the main memory with the real observed latent at start_idx.
        # This means start_idx+1 has memory from real start_idx.
        real_action_h = action_seq[:, start_idx]
        real_action_mask_h = action_mask_seq[:, start_idx]
        if getattr(memory_module, "uses_action", False):
            main_memory = memory_module.update(
                z_start,
                main_memory,
                start_entity_mask,
                action=real_action_h,
                action_mask=real_action_mask_h,
            )
        else:
            main_memory = memory_module.update(z_start, main_memory, start_entity_mask)
        memory_norms.append(main_memory.detach().float().norm(dim=-1).mean())

    pred_latent = torch.stack(pred_by_start, dim=1)                      # [B, P, H, E, D]
    target_latent = torch.stack(target_by_start, dim=1)                  # [B, P, H, E, D]
    target_entity = torch.stack(target_entity_by_start, dim=1)           # [B, P, H, E, F]
    target_entity_mask = torch.stack(target_entity_mask_by_start, dim=1) # [B, P, H, E]
    entity_slot_mask = torch.stack(slot_mask_by_start, dim=1)            # [B, P, H, E]
    valid_mask = torch.stack(valid_by_start, dim=1)                      # [B, P, H]
    imagined_state = torch.stack(imagined_state_by_start, dim=1)           # [B, P, H, E, D+M]

    target_for_pred = target_latent.detach() if detach_rollout_targets else target_latent

    weights = temporal_time_weights(
        h,
        mode=temporal_loss_mode,
        td_lambda=td_lambda,
        flat_decay_start=flat_decay_start,
        flat_decay_final_weight=flat_decay_final_weight,
        device=pred_latent.device,
        dtype=pred_latent.dtype,
    )
    uniform_weights = torch.ones_like(weights)

    mask = target_entity_mask.unsqueeze(-1) * valid_mask.unsqueeze(-1).unsqueeze(-1)

    pred_loss = weighted_mse(pred_latent, target_for_pred, mask, weights)
    pred_loss_uniform = weighted_mse(pred_latent, target_for_pred, mask, uniform_weights)

    one_step_loss = weighted_mse(
        pred_latent[:, :, 0:1],
        target_for_pred[:, :, 0:1],
        mask[:, :, 0:1],
        torch.ones(1, device=pred_latent.device, dtype=pred_latent.dtype),
    )

    decoded = model.decode_entities(pred_latent.reshape(bsz * p * h, entities, latent_dim))
    decoded = decoded.reshape(bsz, p, h, entities, -1)

    aux_weights = uniform_weights if unweighted_aux_losses else weights
    decoded_loss = weighted_mse(decoded, target_entity, mask, aux_weights)

    presence_logits = model.predict_presence(
        pred_latent.reshape(bsz * p * h, entities, latent_dim)
    ).reshape(bsz, p, h, entities)

    presence_mask = entity_slot_mask * valid_mask.unsqueeze(-1)
    presence_loss = weighted_bce(presence_logits, target_entity_mask, presence_mask, aux_weights)

    reg_latents = torch.cat([input_latents, target_latents], dim=1)
    reg_masks = torch.cat([entity_mask_seq, target_entity_mask_seq_full], dim=1)

    if sigreg_weight > 0.0:
        # The surrounding training step may be under autocast. Force full
        # float32 here because the original SIGReg multiplies by sample count
        # and can overflow float16.
        with torch.cuda.amp.autocast(enabled=False) if reg_latents.is_cuda else nullcontext():
            reg_loss = stable_sigreg_loss(
                reg_latents.float(),
                reg_masks,
                knots=sigreg_knots,
                num_proj=sigreg_num_proj,
                proj_chunk=sigreg_proj_chunk,
                max_samples=sigreg_max_samples,
            )
    else:
        reg_loss = pred_loss.float().new_zeros(())

    if variance_floor_weight > 0.0:
        variance_floor_loss, encoder_scale_stats = masked_variance_floor_loss(
            reg_latents,
            reg_masks,
            std_floor=latent_std_floor,
        )
    else:
        variance_floor_loss = pred_loss.new_zeros(())
        encoder_scale_stats = masked_latent_statistics(
            reg_latents,
            reg_masks,
            std_floor=latent_std_floor,
        )

    # Include only real, present, non-padded entity samples.
    barlow_valid_mask = (
        target_entity_mask
        * entity_slot_mask
        * valid_mask.unsqueeze(-1)
    )
    if barlow_weight > 0.0:
        barlow_stats = masked_barlow_loss(
            barlow_projector,
            imagined_state,
            target_latent.detach(),
            barlow_valid_mask,
            alpha=barlow_lambda,
            mode=barlow_mode,
            temporal_weights=weights,
        )
        barlow_loss = barlow_stats["loss"]
    else:
        zero = pred_loss.new_zeros(())
        barlow_stats = {
            "loss": zero,
            "raw_loss": zero,
            "invariance": zero,
            "redundancy": zero,
            "diag_mean": zero,
            "offdiag_rms": zero,
            "samples": zero,
        }
        barlow_loss = zero

    # Barlow remains strictly tied to live prediction loss, so it cannot
    # dominate the objective numerically.
    effective_barlow_weight = capped_relative_weight(
        barlow_loss,
        pred_loss,
        max_weight=barlow_weight,
        target_ratio=barlow_target_ratio,
    )

    # SIGReg and the explicit variance floor are scale-restoring terms.
    # Their reference must not vanish when latent scale shrinks.
    scale_guard_reference = torch.maximum(
        pred_loss.detach(),
        pred_loss.new_tensor(float(regularizer_reference_floor)),
    )
    effective_sigreg_weight = capped_relative_weight(
        reg_loss,
        scale_guard_reference,
        max_weight=sigreg_weight,
        target_ratio=sigreg_target_ratio,
    )
    effective_variance_floor_weight = capped_relative_weight(
        variance_floor_loss,
        scale_guard_reference,
        max_weight=variance_floor_weight,
        target_ratio=variance_floor_target_ratio,
    )

    assert_finite_loss("prediction loss", pred_loss)
    assert_finite_loss("one-step loss", one_step_loss)
    assert_finite_loss("Barlow loss", barlow_loss)
    assert_finite_loss("SIGReg loss", reg_loss)
    assert_finite_loss("variance-floor loss", variance_floor_loss)
    assert_finite_loss("decoded loss", decoded_loss)
    assert_finite_loss("presence loss", presence_loss)

    weighted_sigreg_loss = effective_sigreg_weight * reg_loss
    weighted_barlow_loss = effective_barlow_weight * barlow_loss
    weighted_variance_floor_loss = (
        effective_variance_floor_weight * variance_floor_loss
    )

    total_loss = (
        pred_loss
        + one_step_weight * one_step_loss
        + weighted_sigreg_loss
        + weighted_barlow_loss
        + weighted_variance_floor_loss
        + decoder_weight * decoded_loss
        + presence_weight * presence_loss
    )
    assert_finite_loss("total loss", total_loss)

    pred_denom = pred_loss.detach().abs().clamp_min(1e-12)
    regularization_contribution = (
        weighted_sigreg_loss.detach()
        + weighted_barlow_loss.detach()
        + weighted_variance_floor_loss.detach()
    )

    target_scale_stats = masked_latent_statistics(
        target_latent,
        barlow_valid_mask,
        std_floor=latent_std_floor,
    )
    pred_scale_stats = masked_latent_statistics(
        pred_latent,
        barlow_valid_mask,
        std_floor=latent_std_floor,
    )
    target_variance_denom = target_scale_stats[
        "variance_mean"
    ].clamp_min(1e-12)

    losses: dict[str, torch.Tensor] = {
        "total_loss": total_loss,
        "pred_loss": pred_loss,
        "pred_loss_uniform": pred_loss_uniform.detach(),
        "one_step_loss": one_step_loss.detach(),
        "weighted_one_step_loss": (one_step_weight * one_step_loss).detach(),
        "sigreg_loss": reg_loss,
        "effective_sigreg_weight": effective_sigreg_weight,
        "weighted_sigreg_loss": weighted_sigreg_loss.detach(),
        "sigreg_to_pred_ratio": weighted_sigreg_loss.detach() / pred_denom,
        "variance_floor_loss": variance_floor_loss,
        "effective_variance_floor_weight": effective_variance_floor_weight,
        "weighted_variance_floor_loss": weighted_variance_floor_loss.detach(),
        "variance_floor_to_pred_ratio": (
            weighted_variance_floor_loss.detach() / pred_denom
        ),
        "scale_guard_reference": scale_guard_reference.detach(),
        "barlow_loss": barlow_loss,
        "barlow_raw_loss": barlow_stats["raw_loss"],
        "effective_barlow_weight": effective_barlow_weight,
        "weighted_barlow_loss": weighted_barlow_loss.detach(),
        "barlow_to_pred_ratio": weighted_barlow_loss.detach() / pred_denom,
        "regularization_to_pred_ratio": regularization_contribution / pred_denom,
        "prediction_fraction_pred_plus_regularizers": (
            pred_loss.detach()
            / (pred_loss.detach() + regularization_contribution).clamp_min(1e-12)
        ),
        "variance_normalized_pred_loss": (
            pred_loss.detach() / target_variance_denom
        ),
        "encoder_latent_std_mean": encoder_scale_stats["std_mean"],
        "encoder_latent_std_min": encoder_scale_stats["std_min"],
        "encoder_latent_std_max": encoder_scale_stats["std_max"],
        "encoder_latent_variance_mean": encoder_scale_stats["variance_mean"],
        "encoder_latent_norm_mean": encoder_scale_stats["norm_mean"],
        "encoder_fraction_std_below_0p1": (
            encoder_scale_stats["fraction_std_below_0p1"]
        ),
        "encoder_fraction_std_below_0p5": (
            encoder_scale_stats["fraction_std_below_0p5"]
        ),
        "encoder_fraction_std_below_floor": (
            encoder_scale_stats["fraction_std_below_floor"]
        ),
        "target_latent_std_mean": target_scale_stats["std_mean"],
        "target_latent_std_min": target_scale_stats["std_min"],
        "target_latent_variance_mean": target_scale_stats["variance_mean"],
        "target_latent_norm_mean": target_scale_stats["norm_mean"],
        "target_fraction_std_below_0p1": (
            target_scale_stats["fraction_std_below_0p1"]
        ),
        "target_fraction_std_below_0p5": (
            target_scale_stats["fraction_std_below_0p5"]
        ),
        "pred_latent_std_mean": pred_scale_stats["std_mean"],
        "pred_latent_std_min": pred_scale_stats["std_min"],
        "pred_latent_variance_mean": pred_scale_stats["variance_mean"],
        "pred_latent_norm_mean": pred_scale_stats["norm_mean"],
        "pred_fraction_std_below_0p1": (
            pred_scale_stats["fraction_std_below_0p1"]
        ),
        "pred_fraction_std_below_0p5": (
            pred_scale_stats["fraction_std_below_0p5"]
        ),
        "pred_to_target_std_ratio": (
            pred_scale_stats["std_mean"]
            / target_scale_stats["std_mean"].clamp_min(1e-12)
        ),
        "barlow_invariance_loss": barlow_stats["invariance"],
        "barlow_redundancy_loss": barlow_stats["redundancy"],
        "barlow_diag_mean": barlow_stats["diag_mean"],
        "barlow_offdiag_rms": barlow_stats["offdiag_rms"],
        "barlow_valid_samples": barlow_stats["samples"],
        "decoded_loss": decoded_loss,
        "presence_loss": presence_loss,
        "weighted_presence_loss": (presence_weight * presence_loss).detach(),
        "temporal_weight_sum": weights.sum().detach(),
        "rollout_window": torch.tensor(float(p), device=pred_latent.device),
        "rollout_horizon": torch.tensor(float(h), device=pred_latent.device),
        "memory_norm_mean": torch.stack(memory_norms).mean() if memory_norms else torch.tensor(0.0, device=pred_latent.device),
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

    dataset = VisibilityMarkovRolloutSMACJEPADataset(
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
        enemy_visibility_mask=args.enemy_visibility_mask,
        enemy_sight_range=args.enemy_sight_range,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(arch["batch_size"]),
        shuffle=True,
        num_workers=args.num_workers,
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
        decoder_weight=args.decoder_weight,
        encoder_layers=int(arch["encoder_layers"]),
        action_layers=int(arch["action_layers"]),
        predictor_layers=int(arch["predictor_layers"]),
        max_context_len=args.max_context_len,
    ).to(device)

    if args.action_conditioned_memory:
        memory_module = ActionConditionedEntityRolloutGRUMemory(
            latent_dim=int(arch["latent_dim"]),
            memory_dim=args.rollout_memory_dim,
            n_actions=dataset.metadata.n_actions,
            hidden_dim=args.rollout_memory_hidden_dim,
            residual=not args.rollout_memory_no_residual,
        ).to(device)
    else:
        memory_module = EntityRolloutGRUMemory(
            latent_dim=int(arch["latent_dim"]),
            memory_dim=args.rollout_memory_dim,
            hidden_dim=args.rollout_memory_hidden_dim,
            residual=not args.rollout_memory_no_residual,
        ).to(device)

    barlow_projector = R2EntityProjector(
        latent_dim=int(arch["latent_dim"]),
        memory_dim=args.rollout_memory_dim,
    ).to(device)

    trainable_parameters = (
        list(model.parameters())
        + list(memory_module.parameters())
        + list(barlow_projector.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(arch["lr"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    global_step = 0
    checkpoint = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "memory_module_state" in checkpoint:
            memory_module.load_state_dict(checkpoint["memory_module_state"])
        else:
            print("memory_module_state not found in checkpoint; starting memory module fresh", flush=True)
        if "barlow_projector_state" in checkpoint:
            barlow_projector.load_state_dict(checkpoint["barlow_projector_state"])
        else:
            print("barlow_projector_state not found in checkpoint; starting projector fresh", flush=True)
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint and amp_enabled:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))

    resume_config = (
        checkpoint.get("resolved_config", checkpoint.get("config", {}))
        if checkpoint is not None
        else {}
    )

    if args.latent_std_floor is not None:
        resolved_latent_std_floor = float(args.latent_std_floor)
        initial_scale_stats = {
            "samples": float("nan"),
            "batches": 0.0,
            "std_mean": resolved_latent_std_floor
            / max(float(args.latent_std_floor_ratio), 1e-8),
            "std_min": float("nan"),
            "std_median": float("nan"),
            "variance_mean": float("nan"),
        }
    elif "resolved_latent_std_floor" in resume_config:
        resolved_latent_std_floor = float(
            resume_config["resolved_latent_std_floor"]
        )
        initial_scale_stats = {
            "samples": float(resume_config.get("scale_calibration_samples", float("nan"))),
            "batches": float(resume_config.get("scale_calibration_batches_used", 0)),
            "std_mean": float(resume_config.get("initial_latent_std_mean", float("nan"))),
            "std_min": float(resume_config.get("initial_latent_std_min", float("nan"))),
            "std_median": float(resume_config.get("initial_latent_std_median", float("nan"))),
            "variance_mean": float(resume_config.get("initial_latent_variance_mean", float("nan"))),
        }
    else:
        initial_scale_stats = calibrate_initial_latent_scale(
            model,
            loader,
            device=device,
            max_batches=args.scale_calibration_batches,
            target_mode=args.target_mode,
        )
        resolved_latent_std_floor = (
            float(args.latent_std_floor_ratio)
            * initial_scale_stats["std_mean"]
        )

    initial_latent_std_mean = float(initial_scale_stats["std_mean"])
    if not math.isfinite(initial_latent_std_mean) or initial_latent_std_mean <= 0:
        raise RuntimeError(
            f"Invalid calibrated initial latent std: {initial_latent_std_mean}"
        )
    if not math.isfinite(resolved_latent_std_floor) or resolved_latent_std_floor <= 0:
        raise RuntimeError(
            f"Invalid resolved latent std floor: {resolved_latent_std_floor}"
        )

    print(
        "latent_scale_calibration "
        f"batches={int(initial_scale_stats['batches'])} "
        f"samples={int(initial_scale_stats['samples']) if math.isfinite(initial_scale_stats['samples']) else -1} "
        f"initial_std_mean={initial_latent_std_mean:.6f} "
        f"initial_std_min={initial_scale_stats['std_min']:.6f} "
        f"resolved_std_floor={resolved_latent_std_floor:.6f}",
        flush=True,
    )

    saved_config = vars(args) | arch | {
        "resolved_device": device.type,
        "amp_enabled": amp_enabled,
        "dataset_len": len(dataset),
        "training_regime": "markov_rollout_rnn_seqmem_barlow_plausible_v2",
        "resolved_latent_std_floor": resolved_latent_std_floor,
        "initial_latent_std_mean": initial_latent_std_mean,
        "initial_latent_std_min": initial_scale_stats["std_min"],
        "initial_latent_std_median": initial_scale_stats["std_median"],
        "initial_latent_variance_mean": initial_scale_stats["variance_mean"],
        "scale_calibration_samples": initial_scale_stats["samples"],
        "scale_calibration_batches_used": initial_scale_stats["batches"],
        "enemy_visibility_mask": args.enemy_visibility_mask,
        "enemy_sight_range": args.enemy_sight_range,
        "action_conditioned_memory": args.action_conditioned_memory,
        "one_step_weight": args.one_step_weight,
        "barlow_weight": args.barlow_weight,
        "barlow_target_ratio": args.barlow_target_ratio,
        "sigreg_target_ratio": args.sigreg_target_ratio,
        "variance_floor_weight": args.variance_floor_weight,
        "variance_floor_target_ratio": args.variance_floor_target_ratio,
        "latent_std_floor": args.latent_std_floor,
        "regularizer_reference_floor": args.regularizer_reference_floor,
        "regularizer_ratio_warmup_epochs": args.regularizer_ratio_warmup_epochs,
        "barlow_lambda": args.barlow_lambda,
        "barlow_mode": args.barlow_mode,
        "barlow_projector_input_dim": int(arch["latent_dim"]) + args.rollout_memory_dim,
        "barlow_projector_output_dim": int(arch["latent_dim"]),
        "target_mode": args.target_mode,
        "segment_action_len": args.rollout_window + args.rollout_horizon,
        "segment_state_len": args.rollout_window + args.rollout_horizon + 1,
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
        wandb_run.watch(memory_module, log=None)
        wandb_run.watch(barlow_projector, log=None)

    def save_checkpoint(epoch_to_save: int, checkpoint_path: Path) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "memory_module_state": memory_module.state_dict(),
                "barlow_projector_state": barlow_projector.state_dict(),
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
    memory_module.train()
    barlow_projector.train()

    print(
        "markov_rollout_rnn_visibility_seqmem_barlow_plausible_v2 "
        f"p={args.rollout_window} n={args.rollout_horizon} "
        f"memory_dim={args.rollout_memory_dim} "
        f"enemy_visibility_mask={args.enemy_visibility_mask} sight_range={args.enemy_sight_range} "
        f"temporal_loss={args.temporal_loss} td_lambda={args.td_lambda} "
        f"sigreg_weight={args.sigreg_weight} sigreg_weight_start={args.sigreg_weight_start} "
        f"sigreg_weight_end={args.sigreg_weight_end} sigreg_warmup_epochs={args.sigreg_warmup_epochs} "
        f"barlow_weight_max={args.barlow_weight} "
        f"barlow_target_ratio={args.barlow_target_ratio} "
        f"sigreg_target_ratio={args.sigreg_target_ratio} "
        f"variance_floor_weight_max={args.variance_floor_weight} "
        f"variance_floor_target_ratio={args.variance_floor_target_ratio} "
        f"latent_std_floor={resolved_latent_std_floor:.6f} "
        f"initial_latent_std_mean={initial_latent_std_mean:.6f} "
        f"regularizer_reference_floor={args.regularizer_reference_floor} "
        f"regularizer_ratio_warmup_epochs={args.regularizer_ratio_warmup_epochs} "
        f"barlow_lambda={args.barlow_lambda} "
        f"barlow_mode={args.barlow_mode} "
        f"decoder_weight={args.decoder_weight} presence_weight={args.presence_weight} "
        f"action_conditioned_memory={args.action_conditioned_memory} "
        f"one_step_weight={args.one_step_weight} target_mode={args.target_mode} "
        f"window_mode={args.window_mode} samples_per_epoch={args.samples_per_epoch}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        active_sigreg_weight = scheduled_value(
            epoch=epoch,
            fixed_value=args.sigreg_weight,
            start_value=args.sigreg_weight_start,
            end_value=args.sigreg_weight_end,
            warmup_epochs=args.sigreg_warmup_epochs,
        )

        ratio_warmup = regularizer_warmup_factor(
            epoch,
            args.regularizer_ratio_warmup_epochs,
        )
        active_barlow_target_ratio = args.barlow_target_ratio * ratio_warmup
        active_sigreg_target_ratio = args.sigreg_target_ratio * ratio_warmup
        active_variance_floor_target_ratio = (
            args.variance_floor_target_ratio * ratio_warmup
        )

        print(
            f"epoch_schedule epoch={epoch} "
            f"active_sigreg_max_weight={active_sigreg_weight:.8f} "
            f"active_barlow_target_ratio={active_barlow_target_ratio:.4f} "
            f"active_sigreg_target_ratio={active_sigreg_target_ratio:.4f} "
            f"active_variance_floor_target_ratio="
            f"{active_variance_floor_target_ratio:.4f}",
            flush=True,
        )

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
                losses = markov_rollout_rnn_losses(
                    model,
                    memory_module,
                    barlow_projector,
                    batch,
                    rollout_window=args.rollout_window,
                    rollout_horizon=args.rollout_horizon,
                    temporal_loss_mode=args.temporal_loss,
                    td_lambda=args.td_lambda,
                    flat_decay_start=args.flat_decay_start,
                    flat_decay_final_weight=args.flat_decay_final_weight,
                    sigreg_weight=active_sigreg_weight,
                    sigreg_target_ratio=active_sigreg_target_ratio,
                    sigreg_knots=args.sigreg_knots,
                    sigreg_num_proj=args.sigreg_num_proj,
                    sigreg_proj_chunk=args.sigreg_proj_chunk,
                    sigreg_max_samples=args.sigreg_max_samples,
                    variance_floor_weight=args.variance_floor_weight,
                    variance_floor_target_ratio=(
                        active_variance_floor_target_ratio
                    ),
                    latent_std_floor=resolved_latent_std_floor,
                    regularizer_reference_floor=(
                        args.regularizer_reference_floor
                    ),
                    barlow_weight=args.barlow_weight,
                    barlow_target_ratio=active_barlow_target_ratio,
                    barlow_lambda=args.barlow_lambda,
                    barlow_mode=args.barlow_mode,
                    decoder_weight=args.decoder_weight,
                    presence_weight=args.presence_weight,
                    one_step_weight=args.one_step_weight,
                    target_mode=args.target_mode,
                    detach_rollout_targets=args.detach_rollout_targets,
                    unweighted_aux_losses=args.unweighted_aux_losses,
                )

            assert_finite_loss("training total loss", losses["total_loss"])
            scaler.scale(losses["total_loss"]).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    args.grad_clip,
                    error_if_nonfinite=True,
                )

            scaler.step(optimizer)
            scaler.update()

            row: dict[str, float | int] = {
                "epoch": epoch,
                "step": global_step,
                "active_sigreg_weight": active_sigreg_weight,
                "active_barlow_weight": args.barlow_weight,
                "active_barlow_target_ratio": active_barlow_target_ratio,
                "active_sigreg_target_ratio": active_sigreg_target_ratio,
                "active_variance_floor_target_ratio": (
                    active_variance_floor_target_ratio
                ),
            }
            for key, value in losses.items():
                row[key] = float(value.detach().cpu())

            row["target_std_to_initial_ratio"] = (
                row["target_latent_std_mean"]
                / max(initial_latent_std_mean, 1e-12)
            )

            nonfinite_metrics = [
                key
                for key, value in row.items()
                if isinstance(value, float) and not math.isfinite(value)
            ]
            if nonfinite_metrics:
                raise FloatingPointError(
                    "Non-finite training metrics at "
                    f"step={global_step}: {nonfinite_metrics}"
                )

            if global_step >= args.health_check_after_steps:
                if (
                    row["target_std_to_initial_ratio"]
                    < args.abort_target_std_ratio
                ):
                    raise RuntimeError(
                        "Aborting due to latent-scale collapse: "
                        f"target_std_to_initial_ratio="
                        f"{row['target_std_to_initial_ratio']:.4f} < "
                        f"{args.abort_target_std_ratio:.4f}"
                    )
                if (
                    row["pred_to_target_std_ratio"]
                    < args.abort_pred_target_std_ratio
                ):
                    raise RuntimeError(
                        "Aborting due to predicted-latent collapse: "
                        f"pred_to_target_std_ratio="
                        f"{row['pred_to_target_std_ratio']:.4f} < "
                        f"{args.abort_pred_target_std_ratio:.4f}"
                    )

            logger.log(row)
            step_rows.append(row)

            if wandb_run is not None:
                log_dict = {
                    "train/epoch": epoch,
                    "train/total_loss": row.get("total_loss"),
                    "train/pred_loss": row.get("pred_loss"),
                    "train/pred_loss_uniform": row.get("pred_loss_uniform"),
                    "train/one_step_loss": row.get("one_step_loss"),
                    "train/weighted_one_step_loss": row.get("weighted_one_step_loss"),
                    "train/sigreg_loss": row.get("sigreg_loss"),
                    "train/weighted_sigreg_loss": row.get("weighted_sigreg_loss"),
                    "train/active_sigreg_weight": row.get("active_sigreg_weight"),
                    "train/barlow_loss": row.get("barlow_loss"),
                    "train/weighted_barlow_loss": row.get("weighted_barlow_loss"),
                    "train/barlow_invariance_loss": row.get("barlow_invariance_loss"),
                    "train/barlow_redundancy_loss": row.get("barlow_redundancy_loss"),
                    "train/barlow_diag_mean": row.get("barlow_diag_mean"),
                    "train/barlow_offdiag_rms": row.get("barlow_offdiag_rms"),
                    "train/barlow_valid_samples": row.get("barlow_valid_samples"),
                    "train/active_barlow_weight": row.get("active_barlow_weight"),
                    "train/effective_barlow_weight": row.get("effective_barlow_weight"),
                    "train/effective_sigreg_weight": row.get("effective_sigreg_weight"),
                    "train/barlow_to_pred_ratio": row.get("barlow_to_pred_ratio"),
                    "train/sigreg_to_pred_ratio": row.get("sigreg_to_pred_ratio"),
                    "train/variance_floor_loss": row.get("variance_floor_loss"),
                    "train/weighted_variance_floor_loss": row.get(
                        "weighted_variance_floor_loss"
                    ),
                    "train/variance_floor_to_pred_ratio": row.get(
                        "variance_floor_to_pred_ratio"
                    ),
                    "train/variance_normalized_pred_loss": row.get(
                        "variance_normalized_pred_loss"
                    ),
                    "train/target_latent_std_mean": row.get(
                        "target_latent_std_mean"
                    ),
                    "train/target_std_to_initial_ratio": row.get(
                        "target_std_to_initial_ratio"
                    ),
                    "train/target_latent_std_min": row.get(
                        "target_latent_std_min"
                    ),
                    "train/pred_latent_std_mean": row.get(
                        "pred_latent_std_mean"
                    ),
                    "train/pred_to_target_std_ratio": row.get(
                        "pred_to_target_std_ratio"
                    ),
                    "train/encoder_fraction_std_below_0p1": row.get(
                        "encoder_fraction_std_below_0p1"
                    ),
                    "train/regularization_to_pred_ratio": row.get("regularization_to_pred_ratio"),
                    "train/prediction_fraction_pred_plus_regularizers": row.get("prediction_fraction_pred_plus_regularizers"),
                    "train/active_barlow_target_ratio": row.get("active_barlow_target_ratio"),
                    "train/active_sigreg_target_ratio": row.get("active_sigreg_target_ratio"),
                    "train/decoded_loss": row.get("decoded_loss"),
                    "train/presence_loss": row.get("presence_loss"),
                    "train/weighted_presence_loss": row.get("weighted_presence_loss"),
                    "train/presence_weight": args.presence_weight,
                    "train/memory_norm_mean": row.get("memory_norm_mean"),
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
                    "epoch={epoch} step={step} "
                    "total_loss={total_loss:.6f} pred_loss={pred_loss:.6f} "
                    "pred_uniform={pred_loss_uniform:.6f} sigreg_loss={sigreg_loss:.6f} "
                    "barlow_loss={barlow_loss:.6f} weighted_barlow={weighted_barlow_loss:.6f} "
                    "barlow_ratio={barlow_to_pred_ratio:.3f} reg_ratio={regularization_to_pred_ratio:.3f} "
                    "barlow_diag={barlow_diag_mean:.4f} barlow_offdiag_rms={barlow_offdiag_rms:.4f} "
                    "var_floor={variance_floor_loss:.6f} "
                    "target_std={target_latent_std_mean:.4f} "
                    "target_std_ratio={target_std_to_initial_ratio:.3f} "
                    "pred_std={pred_latent_std_mean:.4f} "
                    "norm_pred={variance_normalized_pred_loss:.4f} "
                    "decoded_loss={decoded_loss:.6f} presence_loss={presence_loss:.6f} "
                    "memory_norm={memory_norm_mean:.6f}".format(**row),
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
                "epoch/one_step_loss": epoch_row.get("one_step_loss"),
                "epoch/weighted_one_step_loss": epoch_row.get("weighted_one_step_loss"),
                "epoch/sigreg_loss": epoch_row.get("sigreg_loss"),
                "epoch/weighted_sigreg_loss": epoch_row.get("weighted_sigreg_loss"),
                "epoch/active_sigreg_weight": epoch_row.get("active_sigreg_weight"),
                "epoch/barlow_loss": epoch_row.get("barlow_loss"),
                "epoch/weighted_barlow_loss": epoch_row.get("weighted_barlow_loss"),
                "epoch/barlow_invariance_loss": epoch_row.get("barlow_invariance_loss"),
                "epoch/barlow_redundancy_loss": epoch_row.get("barlow_redundancy_loss"),
                "epoch/barlow_diag_mean": epoch_row.get("barlow_diag_mean"),
                "epoch/barlow_offdiag_rms": epoch_row.get("barlow_offdiag_rms"),
                "epoch/barlow_valid_samples": epoch_row.get("barlow_valid_samples"),
                "epoch/active_barlow_weight": epoch_row.get("active_barlow_weight"),
                "epoch/effective_barlow_weight": epoch_row.get("effective_barlow_weight"),
                "epoch/effective_sigreg_weight": epoch_row.get("effective_sigreg_weight"),
                "epoch/barlow_to_pred_ratio": epoch_row.get("barlow_to_pred_ratio"),
                "epoch/sigreg_to_pred_ratio": epoch_row.get("sigreg_to_pred_ratio"),
                "epoch/variance_floor_loss": epoch_row.get("variance_floor_loss"),
                "epoch/weighted_variance_floor_loss": epoch_row.get(
                    "weighted_variance_floor_loss"
                ),
                "epoch/variance_floor_to_pred_ratio": epoch_row.get(
                    "variance_floor_to_pred_ratio"
                ),
                "epoch/variance_normalized_pred_loss": epoch_row.get(
                    "variance_normalized_pred_loss"
                ),
                "epoch/target_latent_std_mean": epoch_row.get(
                    "target_latent_std_mean"
                ),
                "epoch/target_latent_std_min": epoch_row.get(
                    "target_latent_std_min"
                ),
                "epoch/pred_latent_std_mean": epoch_row.get(
                    "pred_latent_std_mean"
                ),
                "epoch/pred_to_target_std_ratio": epoch_row.get(
                    "pred_to_target_std_ratio"
                ),
                "epoch/encoder_fraction_std_below_0p1": epoch_row.get(
                    "encoder_fraction_std_below_0p1"
                ),
                "epoch/regularization_to_pred_ratio": epoch_row.get("regularization_to_pred_ratio"),
                "epoch/prediction_fraction_pred_plus_regularizers": epoch_row.get("prediction_fraction_pred_plus_regularizers"),
                "epoch/decoded_loss": epoch_row.get("decoded_loss"),
                "epoch/presence_loss": epoch_row.get("presence_loss"),
                "epoch/weighted_presence_loss": epoch_row.get("weighted_presence_loss"),
                "epoch/presence_weight": args.presence_weight,
                "epoch/memory_norm_mean": epoch_row.get("memory_norm_mean"),
            }
            for key, value in epoch_row.items():
                if key.startswith("pred_loss_h"):
                    log_dict[f"epoch/{key}"] = value
            wandb_run.log(log_dict, step=global_step)

        print(
            "epoch_summary epoch={epoch} step={step} "
            "total_loss={total_loss:.6f} pred_loss={pred_loss:.6f} "
            "pred_uniform={pred_loss_uniform:.6f} sigreg_loss={sigreg_loss:.6f} "
            "barlow_loss={barlow_loss:.6f} weighted_barlow={weighted_barlow_loss:.6f} "
            "barlow_ratio={barlow_to_pred_ratio:.3f} reg_ratio={regularization_to_pred_ratio:.3f} "
            "barlow_diag={barlow_diag_mean:.4f} barlow_offdiag_rms={barlow_offdiag_rms:.4f} "
            "var_floor={variance_floor_loss:.6f} "
            "target_std={target_latent_std_mean:.4f} "
            "target_std_ratio={target_std_to_initial_ratio:.3f} "
            "pred_std={pred_latent_std_mean:.4f} "
            "norm_pred={variance_normalized_pred_loss:.4f} "
            "decoded_loss={decoded_loss:.6f} presence_loss={presence_loss:.6f} "
            "memory_norm={memory_norm_mean:.6f}".format(**epoch_row),
            flush=True,
        )

        epoch_checkpoint_path = out_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        save_checkpoint(epoch, epoch_checkpoint_path)
        save_checkpoint(epoch, out_dir / "checkpoint.pt")
        print(f"saved_checkpoint {epoch_checkpoint_path} and {out_dir / 'checkpoint.pt'}", flush=True)

        write_svg_line_plot(epoch_rows, "epoch", "total_loss", "Average Total Loss Per Epoch", out_dir / "loss_by_epoch.svg")
        write_svg_line_plot(epoch_rows, "epoch", "pred_loss", "Average Markov Rollout RNN Prediction Loss Per Epoch", out_dir / "pred_loss_by_epoch.svg")
        write_svg_line_plot(step_rows, "step", "pred_loss", "Markov Rollout RNN Prediction Loss Per Training Step", out_dir / "pred_loss_by_step.svg")
        write_svg_line_plot(epoch_rows, "epoch", "barlow_loss", "Average Barlow Loss Per Epoch", out_dir / "barlow_loss_by_epoch.svg")
        write_svg_line_plot(epoch_rows, "epoch", "barlow_diag_mean", "Average Barlow Correlation Diagonal Per Epoch", out_dir / "barlow_diag_by_epoch.svg")
        write_svg_line_plot(epoch_rows, "epoch", "regularization_to_pred_ratio", "Regularization-to-Prediction Contribution Ratio", out_dir / "regularization_to_pred_ratio.svg")

    if wandb_run is not None:
        wandb_run.save(str(out_dir / "config.json"))
        wandb_run.save(str(out_dir / "checkpoint.pt"))
        wandb_run.finish()


if __name__ == "__main__":
    main()
