from __future__ import annotations

"""
Train SMAC-JEPA with Markov recursive rollout + sequential RNN memory + enemy visibility masking.

Parallel script. Does not modify train_markov_rollout.py.

Assumes:
    smac_jepa/data/markov_rollout_visibility_dataset.py
    smac_jepa/modules/rollout_memory.py

Run as:
    python -m smac_jepa.train_jepa_blocker_fixed_experiments

Blocker-fixed rollout semantics:
    belief_t = fuse(observed_z_t, memory_t), without zeroing hidden slots
    z_hat_{t+1} = predictor(belief_t, a_t)
    memory_{t+1} = update(memory_t, z_t, a_t)

The same chronology is used for real-history and imagined transitions. Future
entity presence is predicted by the presence head and is never taken from the
future target mask. Decoder supervision uses only real ally/enemy feature
coordinates, excluding fixed-width padding. Joint actions preserve agent identity.
"""

import argparse
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

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
    parser = argparse.ArgumentParser(description="Train blocker-fixed SMAC-JEPA Exp16-Exp19 ablations")

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

    # R2-Dreamer-inspired offline representation objective.
    parser.add_argument(
        "--r2-dyn-scale",
        type=float,
        default=1.0,
        help="Scale for prior -> stop-gradient posterior dynamics consistency.",
    )
    parser.add_argument(
        "--r2-rep-scale",
        type=float,
        default=0.1,
        help="Scale for stop-gradient prior -> posterior representation consistency.",
    )
    parser.add_argument(
        "--r2-barlow-scale",
        type=float,
        default=0.05,
        help=(
            "Scale for dimension-normalized posterior-state Barlow loss. "
            "The paper uses 0.05 on its unnormalized objective; this "
            "deterministic adaptation divides the raw loss by latent dimension."
        ),
    )
    parser.add_argument(
        "--r2-barlow-lambda",
        type=float,
        default=5e-4,
        help="Off-diagonal redundancy coefficient used by R2-Dreamer.",
    )

    r2_norm_group = parser.add_mutually_exclusive_group()
    r2_norm_group.add_argument(
        "--r2-latent-normalize",
        dest="r2_latent_normalize",
        action="store_true",
        help=(
            "Apply non-affine per-entity LayerNorm to encoder and predictor "
            "latents before recurrence and losses. This removes the raw-MSE "
            "scale-shrinkage loophole."
        ),
    )
    r2_norm_group.add_argument(
        "--no-r2-latent-normalize",
        dest="r2_latent_normalize",
        action="store_false",
    )
    parser.set_defaults(r2_latent_normalize=True)

    parser.add_argument(
        "--r2-sigreg-divide-by-dim",
        action="store_true",
        default=True,
        help=(
            "Divide stable float32 SIGReg by latent dimension before applying "
            "--sigreg-weight. Used only by Exp15."
        ),
    )
    parser.add_argument("--r2-sigreg-knots", type=int, default=17)
    parser.add_argument("--r2-sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--r2-sigreg-proj-chunk", type=int, default=64)
    parser.add_argument(
        "--r2-sigreg-max-samples",
        type=int,
        default=0,
        help="Maximum valid latent vectors used by SIGReg; 0 uses all.",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=1000,
        help="Linear learning-rate warmup for initial representation stability.",
    )

    # Exp17: stable EMA target encoder.
    parser.add_argument(
        "--ema-target-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a frozen exponential-moving-average encoder for dynamics and "
            "Barlow targets. The trainable online full-state encoder still receives "
            "the asymmetric representation loss and SIGReg."
        ),
    )
    parser.add_argument(
        "--ema-momentum",
        type=float,
        default=0.996,
        help="EMA target encoder momentum. Default: 0.996.",
    )

    # Exp18: force recurrent history itself to retain current full-state information.
    parser.add_argument(
        "--memory-barlow-scale",
        type=float,
        default=0.0,
        help=(
            "Scale of memory-only Barlow loss M(h_t) -> full-state target latent. "
            "0 disables the auxiliary. Exp18/19 use 0.01."
        ),
    )

    # Exp19: explicit event-balanced sampling instead of latent-error PER.
    parser.add_argument(
        "--event-balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample an explicit mixture of ordinary and eventful rollout segments.",
    )
    parser.add_argument(
        "--event-fraction",
        type=float,
        default=0.30,
        help="Fraction of samples drawn from the high-event pool. Default: 0.30.",
    )
    parser.add_argument(
        "--event-pool-fraction",
        type=float,
        default=0.20,
        help=(
            "Fraction of all valid segments assigned to the high-event pool "
            "by event-score ranking. Default: 0.20."
        ),
    )
    parser.add_argument(
        "--event-movement-threshold",
        type=float,
        default=0.01,
        help="Per-coordinate XY change threshold for a movement event.",
    )
    parser.add_argument(
        "--event-state-threshold",
        type=float,
        default=1e-3,
        help="Health/other dynamic change threshold for an event.",
    )
    parser.add_argument(
        "--event-attack-action-min",
        type=int,
        default=6,
        help="Minimum SMAC action ID treated as an attack action.",
    )
    parser.add_argument(
        "--event-min-transitions",
        type=int,
        default=1,
        help="Minimum event transitions inside a segment for eventful classification.",
    )

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



def weighted_feature_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    feature_mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    pred/target:  [B, P, H, E, D]
    feature_mask: [B, P, H, E, D]
    weights:      [H]

    Unlike weighted_mse(), the denominator is the exact number of meaningful
    feature coordinates. No fixed token padding contributes to the loss.
    """
    w = weights.view(1, 1, -1, 1, 1)
    weighted_mask = feature_mask.to(dtype=pred.dtype) * w
    denom = weighted_mask.sum().clamp_min(1.0)
    return ((pred - target).pow(2) * weighted_mask).sum() / denom


def add_feature_valid_masks(
    batch: dict[str, torch.Tensor],
    dataset: VisibilityMarkovRolloutSMACJEPADataset,
) -> dict[str, torch.Tensor]:
    """
    Add [B, E, D] masks for real dynamic/static token coordinates.

    The entity token width is capped globally across maps, so ally and enemy
    tokens contain padding whenever their actual feature sizes are smaller than
    the global token dimension. These masks keep that padding out of decoder
    training.
    """
    episode_indices = [
        int(value)
        for value in batch["episode_index"].reshape(-1).tolist()
    ]
    batch_size = len(episode_indices)
    max_agents = int(dataset.metadata.max_agents)
    max_enemies = int(dataset.metadata.max_enemies)
    entities = max_agents + max_enemies
    token_dim = int(dataset.metadata.token_dim)
    static_offset = int(dataset.metadata.dynamic_token_dim)

    dynamic = torch.zeros(batch_size, entities, token_dim, dtype=torch.float32)
    static = torch.zeros_like(dynamic)

    for batch_idx, episode_idx in enumerate(episode_indices):
        meta = dataset.episodes[episode_idx]["metadata"]
        ally_count = min(int(meta.n_agents), max_agents)
        enemy_count = min(int(meta.n_enemies), max_enemies)
        ally_dim = min(int(meta.ally_state_feat_size), token_dim)
        enemy_dim = min(int(meta.enemy_state_feat_size), token_dim)
        static_dim = min(
            int(meta.entity_static_feat_size),
            max(token_dim - static_offset, 0),
        )

        if ally_count > 0 and ally_dim > 0:
            dynamic[batch_idx, :ally_count, :ally_dim] = 1.0

        if enemy_count > 0 and enemy_dim > 0:
            enemy_start = max_agents
            enemy_stop = enemy_start + enemy_count
            dynamic[batch_idx, enemy_start:enemy_stop, :enemy_dim] = 1.0

        if static_dim > 0:
            static_start = static_offset
            static_stop = static_start + static_dim
            if ally_count > 0:
                static[batch_idx, :ally_count, static_start:static_stop] = 1.0
            if enemy_count > 0:
                enemy_start = max_agents
                enemy_stop = enemy_start + enemy_count
                static[batch_idx, enemy_start:enemy_stop, static_start:static_stop] = 1.0

    batch["dynamic_feature_valid_mask"] = dynamic
    batch["static_feature_valid_mask"] = static
    batch["feature_valid_mask"] = (dynamic + static).clamp_max(1.0)
    return batch


def merge_observed_presence(
    predicted_presence: torch.Tensor,
    observed_presence: torch.Tensor,
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Soft OR: any currently observed entity is certainly present, while hidden
    entities retain the model's prior belief. No future target mask is used.
    """
    predicted = predicted_presence.clamp(0.0, 1.0)
    observed = observed_presence.clamp(0.0, 1.0)
    merged = 1.0 - (1.0 - predicted) * (1.0 - observed)
    return merged * slot_mask


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


def ordered_agent_actions(
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    *,
    n_actions: int,
    max_agents: int,
) -> torch.Tensor:
    """Return ordered one-hot joint actions with shape [B, max_agents, C]."""
    if actions.dim() == 3:
        x = actions.float()
        if x.shape[-1] > n_actions:
            x = x[..., :n_actions]
        elif x.shape[-1] < n_actions:
            pad = torch.zeros(
                *x.shape[:-1],
                n_actions - x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=-1)
    elif actions.dim() == 2:
        ids = actions.long().clamp(min=0, max=n_actions - 1)
        x = F.one_hot(ids, num_classes=n_actions).float()
    else:
        raise ValueError(
            f"Unsupported action shape for memory update: {tuple(actions.shape)}"
        )

    if x.shape[1] > max_agents:
        x = x[:, :max_agents]
    elif x.shape[1] < max_agents:
        pad = torch.zeros(
            x.shape[0],
            max_agents - x.shape[1],
            n_actions,
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, pad], dim=1)

    if action_mask is not None:
        mask = action_mask.float()
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        if mask.shape[1] > max_agents:
            mask = mask[:, :max_agents]
        elif mask.shape[1] < max_agents:
            pad = torch.zeros(
                mask.shape[0],
                max_agents - mask.shape[1],
                device=mask.device,
                dtype=mask.dtype,
            )
            mask = torch.cat([mask, pad], dim=1)
        x = x * mask.unsqueeze(-1)

    return x


class BeliefEntityRolloutGRUMemory(nn.Module):
    """Entity memory whose hidden beliefs are not erased by observation masks."""

    uses_action = False
    blocker_fixed_memory = True

    def __init__(
        self,
        *,
        latent_dim: int,
        memory_dim: int,
        hidden_dim: int | None = None,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.memory_dim = int(memory_dim)
        self.residual = bool(residual)
        hidden = int(hidden_dim or max(latent_dim, memory_dim))

        self.gru = nn.GRUCell(self.latent_dim, self.memory_dim)
        self.condition_net = nn.Sequential(
            nn.Linear(self.latent_dim + self.memory_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.latent_dim),
        )

    def initial_memory(
        self,
        batch_size: int,
        entities: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            entities,
            self.memory_dim,
            device=device,
            dtype=dtype,
        )

    def condition(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        belief_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        correction = self.condition_net(torch.cat([z, memory], dim=-1))
        out = z + correction if self.residual else correction
        if belief_gate is not None:
            out = out * belief_gate.clamp(0.0, 1.0).unsqueeze(-1)
        return out

    def update(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        update_gate: torch.Tensor | None = None,
        *,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del action, action_mask
        bsz, entities, _ = z.shape
        candidate = self.gru(
            z.reshape(bsz * entities, -1),
            memory.reshape(bsz * entities, -1),
        ).reshape(bsz, entities, self.memory_dim)

        if update_gate is None:
            return candidate
        gate = update_gate.clamp(0.0, 1.0).unsqueeze(-1).to(candidate.dtype)
        return gate * candidate + (1.0 - gate) * memory


class ActionConditionedEntityRolloutGRUMemory(BeliefEntityRolloutGRUMemory):
    """
    Agent-aware deterministic memory update with cached action contexts.

    This preserves the exact Exp16--Exp19 parameterization and checkpoint keys,
    but avoids repeatedly materializing ordered one-hot joint actions inside
    every overlapping rollout. The first Linear layers are evaluated through
    indexed weight lookup, which is mathematically equivalent to multiplying by
    one-hot vectors.
    """

    uses_action = True
    blocker_fixed_memory = True
    action_identity_preserved = True
    supports_precomputed_action_context = True

    def __init__(
        self,
        *,
        latent_dim: int,
        memory_dim: int,
        n_actions: int,
        max_agents: int,
        hidden_dim: int | None = None,
        residual: bool = True,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            residual=residual,
        )
        self.n_actions = int(n_actions)
        self.max_agents = int(max_agents)

        # Keep these modules and names unchanged for checkpoint/evaluator
        # compatibility. Their first Linear operations are evaluated with
        # indexed lookup rather than dense one-hot matrix multiplication.
        self.own_action_proj = nn.Sequential(
            nn.Linear(self.n_actions, self.memory_dim),
            nn.SiLU(),
            nn.Linear(self.memory_dim, self.memory_dim),
        )
        self.joint_action_proj = nn.Sequential(
            nn.Linear(self.max_agents * self.n_actions, self.memory_dim),
            nn.SiLU(),
            nn.Linear(self.memory_dim, self.memory_dim),
        )
        self.action_fuse = nn.Sequential(
            nn.Linear(2 * self.memory_dim, self.memory_dim),
            nn.SiLU(),
            nn.Linear(self.memory_dim, self.memory_dim),
        )
        self.gru = nn.GRUCell(
            self.latent_dim + self.memory_dim,
            self.memory_dim,
        )

    def _action_indices_and_mask(
        self,
        action: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded ordered action indices and a float validity mask."""
        if action.dim() == 3 and action.shape[-1] == self.n_actions:
            inferred_mask = action.abs().sum(dim=-1) > 0
            indices = action.argmax(dim=-1).long()
        elif action.dim() == 2:
            indices = action.long()
            inferred_mask = torch.ones_like(indices, dtype=torch.bool)
        else:
            raise ValueError(
                "Unsupported action shape for agent-aware memory: "
                f"{tuple(action.shape)}"
            )

        if indices.shape[1] > self.max_agents:
            indices = indices[:, : self.max_agents]
            inferred_mask = inferred_mask[:, : self.max_agents]
        elif indices.shape[1] < self.max_agents:
            pad_agents = self.max_agents - indices.shape[1]
            indices = F.pad(indices, (0, pad_agents), value=0)
            inferred_mask = F.pad(
                inferred_mask,
                (0, pad_agents),
                value=False,
            )

        if action_mask is None:
            valid = inferred_mask
        else:
            valid = action_mask
            if valid.dim() == 3:
                valid = valid.squeeze(-1)
            valid = valid.bool()
            if valid.shape[1] > self.max_agents:
                valid = valid[:, : self.max_agents]
            elif valid.shape[1] < self.max_agents:
                valid = F.pad(
                    valid,
                    (0, self.max_agents - valid.shape[1]),
                    value=False,
                )
            valid = valid & inferred_mask

        return indices.clamp_(0, self.n_actions - 1), valid.float()

    def _project_ordered_actions(
        self,
        indices: torch.Tensor,
        valid: torch.Tensor,
        *,
        entities: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Project ordered actions without constructing [B,A,n_actions] one-hot.

        For a Linear layer, W @ one_hot(a) is exactly the selected column of W.
        The implementation below therefore has the same output and gradients as
        the dense one-hot implementation, up to normal floating-point ordering.
        """
        own_first = self.own_action_proj[0]
        own_second = self.own_action_proj[2]
        joint_first = self.joint_action_proj[0]
        joint_second = self.joint_action_proj[2]

        # [B,A,M]: select the corresponding input column of the own-action
        # Linear weight for every ordered agent.
        own_hidden = F.embedding(
            indices,
            own_first.weight.transpose(0, 1),
        )
        own_hidden = own_hidden * valid.unsqueeze(-1)
        if own_first.bias is not None:
            own_hidden = own_hidden + own_first.bias
        own = own_second(F.silu(own_hidden))

        # The first joint Linear has shape [M, A*N]. Reshape it so each
        # agent/action pair addresses one input column, select the ordered
        # columns, and sum them. This equals dense flattened-one-hot matmul.
        joint_columns = joint_first.weight.reshape(
            self.memory_dim,
            self.max_agents,
            self.n_actions,
        ).permute(1, 2, 0)
        agent_ids = torch.arange(
            self.max_agents,
            device=indices.device,
        ).unsqueeze(0).expand(indices.shape[0], -1)
        selected = joint_columns[agent_ids, indices]
        joint_hidden = (selected * valid.unsqueeze(-1)).sum(dim=1)
        if joint_first.bias is not None:
            joint_hidden = joint_hidden + joint_first.bias
        joint = joint_second(F.silu(joint_hidden))

        joint_expanded = joint.unsqueeze(1).expand(-1, entities, -1)
        ally_slots = min(self.max_agents, entities)
        if entities <= self.max_agents:
            own_per_entity = own[:, :entities]
        else:
            pad = own.new_zeros(
                own.shape[0],
                entities - self.max_agents,
                self.memory_dim,
            )
            own_per_entity = torch.cat([own, pad], dim=1)

        return self.action_fuse(
            torch.cat([own_per_entity, joint_expanded], dim=-1)
        ).to(dtype=dtype)

    def entity_action_context(
        self,
        action: torch.Tensor,
        action_mask: torch.Tensor | None,
        *,
        entities: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        indices, valid = self._action_indices_and_mask(action, action_mask)
        return self._project_ordered_actions(
            indices,
            valid,
            entities=entities,
            dtype=dtype,
        )

    def precompute_action_context_sequence(
        self,
        action_seq: torch.Tensor,
        action_mask_seq: torch.Tensor | None,
        *,
        entities: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute all timestep/entity action contexts once per batch."""
        if action_seq.dim() < 3:
            raise ValueError(
                f"Expected batched action sequence, got {tuple(action_seq.shape)}"
            )
        bsz, timesteps = action_seq.shape[:2]
        flat_action = action_seq.reshape(
            bsz * timesteps,
            *action_seq.shape[2:],
        )
        flat_mask = None
        if action_mask_seq is not None:
            flat_mask = action_mask_seq.reshape(
                bsz * timesteps,
                *action_mask_seq.shape[2:],
            )
        flat_context = self.entity_action_context(
            flat_action,
            flat_mask,
            entities=entities,
            dtype=dtype,
        )
        return flat_context.reshape(
            bsz,
            timesteps,
            entities,
            self.memory_dim,
        )

    def update(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        update_gate: torch.Tensor | None = None,
        *,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        action_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, entities, _ = z.shape
        if action_context is None:
            if action is None:
                action = torch.zeros(
                    bsz,
                    self.max_agents,
                    device=z.device,
                    dtype=torch.long,
                )
                action_mask = torch.zeros(
                    bsz,
                    self.max_agents,
                    device=z.device,
                    dtype=z.dtype,
                )
            action_context = self.entity_action_context(
                action,
                action_mask,
                entities=entities,
                dtype=z.dtype,
            )

        candidate = self.gru(
            torch.cat([z, action_context], dim=-1).reshape(
                bsz * entities, -1
            ),
            memory.reshape(bsz * entities, -1),
        ).reshape(bsz, entities, self.memory_dim)

        if update_gate is None:
            return candidate
        gate = update_gate.clamp(0.0, 1.0).unsqueeze(-1).to(candidate.dtype)
        return gate * candidate + (1.0 - gate) * memory


class R2PosteriorProjector(nn.Module):
    """
    Projects the JEPA analogue of the observed RSSM posterior state to the
    encoder embedding dimension.

    R2-Dreamer uses W[h_t; z_t]. Here:
      h_t ~= real-history entity memory before consuming observation t
      z_t ~= memory-conditioned observed entity latent at t
    """

    def __init__(self, latent_dim: int, memory_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(
            int(latent_dim) + int(memory_dim),
            int(latent_dim),
            bias=False,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state)



class R2MemoryProjector(nn.Module):
    """Project recurrent memory alone into the encoder latent dimension."""

    def __init__(self, memory_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(memory_dim), int(latent_dim), bias=False)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        return self.linear(memory)


@torch.no_grad()
def update_ema_encoder(
    target_encoder: nn.Module,
    online_encoder: nn.Module,
    momentum: float,
) -> None:
    """Update parameters and floating buffers of the EMA target encoder."""
    m = float(momentum)
    if not 0.0 <= m < 1.0:
        raise ValueError(f"EMA momentum must satisfy 0 <= m < 1, got {m}")

    for target_param, online_param in zip(
        target_encoder.parameters(), online_encoder.parameters()
    ):
        target_param.data.mul_(m).add_(online_param.data, alpha=1.0 - m)

    for target_buffer, online_buffer in zip(
        target_encoder.buffers(), online_encoder.buffers()
    ):
        if torch.is_floating_point(target_buffer):
            target_buffer.data.mul_(m).add_(
                online_buffer.data, alpha=1.0 - m
            )
        else:
            target_buffer.data.copy_(online_buffer.data)


class EventBalancedSampler(Sampler[int]):
    """Replacement sampler with a controlled event/ordinary mixture."""

    def __init__(
        self,
        *,
        event_indices: list[int],
        ordinary_indices: list[int],
        num_samples: int,
        event_fraction: float,
        seed: int,
    ) -> None:
        self.event_indices = list(event_indices)
        self.ordinary_indices = list(ordinary_indices)
        self.num_samples = int(num_samples)
        self.event_fraction = float(event_fraction)
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))

        if self.num_samples < 1:
            raise ValueError("EventBalancedSampler num_samples must be >= 1")
        if not 0.0 <= self.event_fraction <= 1.0:
            raise ValueError("event_fraction must be between 0 and 1")
        if not self.event_indices:
            raise ValueError("No eventful segments were detected")
        if not self.ordinary_indices:
            raise ValueError("No ordinary segments were detected")

    def __len__(self) -> int:
        return self.num_samples

    def _draw(self, pool: list[int], count: int) -> list[int]:
        if count <= 0:
            return []
        choices = torch.randint(
            low=0,
            high=len(pool),
            size=(count,),
            generator=self.generator,
        ).tolist()
        return [pool[index] for index in choices]

    def __iter__(self):
        event_count = int(round(self.num_samples * self.event_fraction))
        ordinary_count = self.num_samples - event_count
        indices = self._draw(self.event_indices, event_count)
        indices.extend(self._draw(self.ordinary_indices, ordinary_count))
        permutation = torch.randperm(
            len(indices), generator=self.generator
        ).tolist()
        return iter([indices[index] for index in permutation])


def _action_ids(actions: np.ndarray) -> np.ndarray:
    if actions.ndim >= 3:
        return np.asarray(actions).argmax(axis=-1)
    return np.asarray(actions)


def classify_event_segments(
    dataset: VisibilityMarkovRolloutSMACJEPADataset,
    *,
    movement_threshold: float,
    state_threshold: float,
    attack_action_min: int,
    min_transitions: int,
    pool_fraction: float,
) -> tuple[list[int], list[int], dict[str, int | float]]:
    """Classify dataset.index segments using state/action/visibility events."""
    event_flags_by_episode: list[np.ndarray] = []
    category_totals = {
        "movement": 0,
        "state_change": 0,
        "presence_change": 0,
        "visibility_change": 0,
        "attack_action": 0,
    }

    for episode in dataset.episodes:
        states = np.asarray(episode["states"])
        actions = np.asarray(episode["actions"])
        meta = episode["metadata"]
        transitions = min(actions.shape[0], max(states.shape[0] - 1, 0))
        if transitions <= 0:
            event_flags_by_episode.append(np.zeros(0, dtype=np.int32))
            continue

        ally_size = int(meta.n_agents * meta.ally_state_feat_size)
        enemy_size = int(meta.n_enemies * meta.enemy_state_feat_size)
        dynamic_size = ally_size + enemy_size
        current = states[:transitions, :dynamic_size]
        future = states[1 : transitions + 1, :dynamic_size]
        delta = np.abs(future - current)

        state_change = delta.max(axis=1) > float(state_threshold)

        movement = np.zeros(transitions, dtype=bool)
        x_idx, y_idx = getattr(dataset, "xy_indices", (2, 3))
        if int(meta.ally_state_feat_size) > max(x_idx, y_idx):
            ally_current = current[:, :ally_size].reshape(
                transitions, int(meta.n_agents), int(meta.ally_state_feat_size)
            )
            ally_future = future[:, :ally_size].reshape(
                transitions, int(meta.n_agents), int(meta.ally_state_feat_size)
            )
            movement |= (
                np.abs(
                    ally_future[..., [x_idx, y_idx]]
                    - ally_current[..., [x_idx, y_idx]]
                ).max(axis=(1, 2))
                > float(movement_threshold)
            )
        if int(meta.n_enemies) > 0 and int(meta.enemy_state_feat_size) > max(x_idx, y_idx):
            enemy_current = current[:, ally_size:dynamic_size].reshape(
                transitions, int(meta.n_enemies), int(meta.enemy_state_feat_size)
            )
            enemy_future = future[:, ally_size:dynamic_size].reshape(
                transitions, int(meta.n_enemies), int(meta.enemy_state_feat_size)
            )
            movement |= (
                np.abs(
                    enemy_future[..., [x_idx, y_idx]]
                    - enemy_current[..., [x_idx, y_idx]]
                ).max(axis=(1, 2))
                > float(movement_threshold)
            )

        def entity_presence(flat: np.ndarray) -> np.ndarray:
            pieces = []
            if int(meta.n_agents) > 0:
                allies = flat[:, :ally_size].reshape(
                    transitions, int(meta.n_agents), int(meta.ally_state_feat_size)
                )
                pieces.append(np.abs(allies).sum(axis=-1) > 1e-8)
            if int(meta.n_enemies) > 0:
                enemies = flat[:, ally_size:dynamic_size].reshape(
                    transitions, int(meta.n_enemies), int(meta.enemy_state_feat_size)
                )
                pieces.append(np.abs(enemies).sum(axis=-1) > 1e-8)
            return np.concatenate(pieces, axis=1) if pieces else np.zeros((transitions, 0), dtype=bool)

        presence_change = (
            entity_presence(current) != entity_presence(future)
        ).any(axis=1)

        visibility_change = np.zeros(transitions, dtype=bool)
        if getattr(dataset, "enemy_visibility_mask", False) and int(meta.n_enemies) > 0:
            masked = dataset._apply_enemy_visibility_mask_to_states(
                states[: transitions + 1].copy(),
                meta,
                episode.get("static_condition"),
            )
            masked_enemy = masked[:, ally_size:dynamic_size].reshape(
                transitions + 1, int(meta.n_enemies), int(meta.enemy_state_feat_size)
            )
            visible = np.abs(masked_enemy).sum(axis=-1) > 1e-8
            visibility_change = (visible[1:] != visible[:-1]).any(axis=1)

        ids = _action_ids(actions[:transitions])
        if ids.ndim == 1:
            attack_action = ids >= int(attack_action_min)
        else:
            attack_action = (ids >= int(attack_action_min)).any(axis=1)

        categories = {
            "movement": movement,
            "state_change": state_change,
            "presence_change": presence_change,
            "visibility_change": visibility_change,
            "attack_action": attack_action,
        }
        for name, values in categories.items():
            category_totals[name] += int(values.sum())

        event = np.logical_or.reduce(list(categories.values())).astype(np.int32)
        event_flags_by_episode.append(event)

    minimum = max(int(min_transitions), 1)
    scored_segments: list[tuple[int, int]] = []
    segments_above_minimum = 0

    for dataset_index, (episode_idx, start) in enumerate(dataset.index):
        flags = event_flags_by_episode[episode_idx]
        stop = min(start + dataset.segment_action_len, flags.shape[0])
        event_count = int(flags[start:stop].sum())
        scored_segments.append((event_count, dataset_index))
        if event_count >= minimum:
            segments_above_minimum += 1

    total_segments = len(scored_segments)
    if total_segments < 2:
        raise ValueError("Need at least two valid segments for event balancing")
    fraction = float(pool_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("event_pool_fraction must satisfy 0 < value < 1")

    # Rank by event count. This guarantees both pools even when every 25-step
    # segment contains some movement/action event, avoiding a brittle empty-pool
    # failure from a simple event/no-event threshold.
    scored_segments.sort(key=lambda item: (item[0], item[1]))
    event_pool_size = max(1, min(total_segments - 1, int(round(total_segments * fraction))))
    event_indices = [index for _, index in scored_segments[-event_pool_size:]]
    ordinary_indices = [index for _, index in scored_segments[:-event_pool_size]]

    scores = np.asarray([score for score, _ in scored_segments], dtype=np.float64)
    category_totals["segments_above_minimum"] = segments_above_minimum
    category_totals["event_segments"] = len(event_indices)
    category_totals["ordinary_segments"] = len(ordinary_indices)
    category_totals["total_segments"] = total_segments
    category_totals["event_pool_fraction"] = fraction
    category_totals["event_score_mean"] = float(scores.mean())
    category_totals["event_score_event_pool_mean"] = float(scores[-event_pool_size:].mean())
    category_totals["event_score_ordinary_pool_mean"] = float(scores[:-event_pool_size].mean())
    return event_indices, ordinary_indices, category_totals

def r2_normalize_latent(
    latent: torch.Tensor,
    entity_mask: torch.Tensor,
    *,
    enabled: bool,
) -> torch.Tensor:
    """
    Non-affine LayerNorm fixes each entity-vector scale without introducing
    learnable gain/bias. This prevents both branches from shrinking raw MSE
    while retaining sample-to-sample information for Barlow.
    """
    if enabled:
        normalized = F.layer_norm(
            latent.float(),
            (latent.shape[-1],),
            weight=None,
            bias=None,
            eps=1e-5,
        ).to(dtype=latent.dtype)
    else:
        normalized = latent
    return normalized * entity_mask.unsqueeze(-1)


def r2_barlow_loss(
    projected_state: torch.Tensor,
    target_embedding: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    redundancy_lambda: float,
) -> dict[str, torch.Tensor]:
    """
    R2-Dreamer-style Barlow objective on observed posterior states only.

    Samples are flattened across batch, real sequence time, and valid entity
    slots. The encoder target is stop-gradient. Computation is float32.
    """
    valid = valid_mask.bool()
    x = projected_state[valid].float()
    y = target_embedding.detach()[valid].float()

    if x.shape[0] < 2:
        zero = projected_state.float().sum() * 0.0
        return {
            "raw": zero,
            "normalized": zero,
            "invariance": zero.detach(),
            "redundancy": zero.detach(),
            "diag_mean": zero.detach(),
            "offdiag_rms": zero.detach(),
            "samples": zero.detach(),
        }

    x = (x - x.mean(dim=0)) / x.std(
        dim=0, unbiased=False
    ).clamp_min(1e-4)
    y = (y - y.mean(dim=0)) / y.std(
        dim=0, unbiased=False
    ).clamp_min(1e-4)

    cross_corr = x.transpose(0, 1) @ y / float(x.shape[0])
    diagonal = torch.diagonal(cross_corr)
    invariance = (diagonal - 1.0).square().sum()

    offdiag_mask = ~torch.eye(
        cross_corr.shape[0],
        dtype=torch.bool,
        device=cross_corr.device,
    )
    offdiag = cross_corr[offdiag_mask]
    redundancy = offdiag.square().sum()

    raw = invariance + float(redundancy_lambda) * redundancy
    normalized = raw / float(cross_corr.shape[0])

    if not torch.isfinite(raw):
        raise FloatingPointError(
            f"R2 Barlow loss became non-finite: "
            f"{raw.detach().cpu().item()}"
        )

    return {
        "raw": raw,
        "normalized": normalized,
        "invariance": invariance.detach(),
        "redundancy": redundancy.detach(),
        "diag_mean": diagonal.mean().detach(),
        "offdiag_rms": offdiag.square().mean().sqrt().detach(),
        "samples": torch.tensor(float(x.shape[0]), device=x.device),
    }


def stable_sigreg_loss(
    latents: torch.Tensor,
    masks: torch.Tensor,
    *,
    knots: int,
    num_proj: int,
    projection_chunk: int,
    max_samples: int,
) -> torch.Tensor:
    """
    Float32, chunked implementation of the repository SIGReg statistic.
    This preserves its characteristic-function statistic while avoiding
    float16 overflow under AMP.
    """
    valid = masks.reshape(-1).bool()
    z = latents.reshape(-1, latents.shape[-1])[valid].float()

    if z.shape[0] < 2:
        return latents.float().sum() * 0.0

    if max_samples > 0 and z.shape[0] > max_samples:
        indices = torch.randperm(z.shape[0], device=z.device)[:max_samples]
        z = z[indices]

    sample_count = int(z.shape[0])
    dim = int(z.shape[-1])

    t = torch.linspace(
        0.0, 3.0, int(knots), device=z.device, dtype=torch.float32
    )
    dt = 3.0 / float(int(knots) - 1)
    trapezoid = torch.full_like(t, 2.0 * dt)
    trapezoid[[0, -1]] = dt
    normal_cf = torch.exp(-0.5 * t.square())
    integration_weights = trapezoid * normal_cf

    total = z.new_zeros(())
    completed = 0
    while completed < int(num_proj):
        current = min(int(projection_chunk), int(num_proj) - completed)
        directions = torch.randn(
            dim, current, device=z.device, dtype=torch.float32
        )
        directions = directions / directions.norm(
            dim=0, keepdim=True
        ).clamp_min(1e-8)

        projected = z @ directions
        projected_t = projected.unsqueeze(-1) * t
        error = (
            projected_t.cos().mean(dim=0) - normal_cf
        ).square()
        error = error + projected_t.sin().mean(dim=0).square()
        total = total + (
            (error @ integration_weights) * float(sample_count)
        ).sum()
        completed += current

    result = total / float(num_proj)
    if not torch.isfinite(result):
        raise FloatingPointError(
            f"Stable SIGReg became non-finite: "
            f"{result.detach().cpu().item()}"
        )
    return result


def latent_batch_statistics(
    latent: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    valid = valid_mask.bool()
    flat = latent[valid].float()
    zero = latent.float().sum() * 0.0

    if flat.shape[0] < 2:
        return {
            "std_mean": zero.detach(),
            "std_min": zero.detach(),
            "norm_mean": zero.detach(),
            "fraction_std_below_0p1": zero.detach(),
        }

    std = flat.std(dim=0, unbiased=False)
    return {
        "std_mean": std.mean().detach(),
        "std_min": std.min().detach(),
        "norm_mean": flat.norm(dim=-1).mean().detach(),
        "fraction_std_below_0p1": (
            std < 0.1
        ).float().mean().detach(),
    }


def assert_finite_losses(losses: dict[str, torch.Tensor]) -> None:
    nonfinite = [
        key
        for key, value in losses.items()
        if isinstance(value, torch.Tensor)
        and not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise FloatingPointError(
            f"Non-finite R2-offline losses: {nonfinite}"
        )



def markov_rollout_rnn_losses(
    model: SMACJEPA,
    memory_module: nn.Module,
    r2_projector: R2PosteriorProjector,
    memory_projector: R2MemoryProjector,
    target_encoder: nn.Module | None,
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
    presence_weight: float,
    one_step_weight: float,
    target_mode: str,
    detach_rollout_targets: bool,
    unweighted_aux_losses: bool,
    r2_dyn_scale: float,
    r2_rep_scale: float,
    r2_barlow_scale: float,
    memory_barlow_scale: float,
    r2_barlow_lambda: float,
    r2_latent_normalize: bool,
    r2_sigreg_divide_by_dim: bool,
    r2_sigreg_knots: int,
    r2_sigreg_num_proj: int,
    r2_sigreg_proj_chunk: int,
    r2_sigreg_max_samples: int,
) -> dict[str, torch.Tensor]:
    """Exp15 objective with all five rollout correctness blockers fixed."""
    del detach_rollout_targets

    entity_seq = batch["entity_seq"]
    observation_mask_seq = batch["entity_mask_seq"]

    if target_mode == "observed":
        target_entity_seq_full = entity_seq
        target_entity_mask_seq_full = observation_mask_seq
    elif target_mode == "full":
        target_entity_seq_full = batch.get("target_entity_seq", entity_seq)
        target_entity_mask_seq_full = batch.get(
            "target_entity_mask_seq", observation_mask_seq
        )
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    action_seq = batch["action_seq"]
    action_mask_seq = batch["action_mask_seq"]
    state_mask = batch["state_mask"]
    static_condition = batch.get("static_condition")
    slot_mask_seq = batch["entity_slot_mask_seq"]

    feature_valid = batch["feature_valid_mask"]
    dynamic_feature_valid = batch["dynamic_feature_valid_mask"]
    static_feature_valid = batch["static_feature_valid_mask"]

    bsz = entity_seq.shape[0]
    p = int(rollout_window)
    h = int(rollout_horizon)

    input_latents_raw = model.encoder(entity_seq, observation_mask_seq)
    online_target_latents_raw = model.encoder(
        target_entity_seq_full,
        target_entity_mask_seq_full,
    )

    if target_encoder is not None:
        with torch.no_grad():
            ema_input_latents_raw = target_encoder(
                entity_seq, observation_mask_seq
            )
            consistency_target_latents_raw = target_encoder(
                target_entity_seq_full,
                target_entity_mask_seq_full,
            )
    else:
        # Preserve Exp16 baseline semantics: without EMA, the Barlow target is
        # the same trainable online observed latent used by blocker-fixed Exp15.
        ema_input_latents_raw = input_latents_raw
        consistency_target_latents_raw = online_target_latents_raw

    # The online full-state branch remains trainable through the representation
    # consistency term. The EMA branch is used only as a stable detached target.
    input_latents = r2_normalize_latent(
        input_latents_raw,
        observation_mask_seq,
        enabled=r2_latent_normalize,
    )
    posterior_target_latents = r2_normalize_latent(
        ema_input_latents_raw,
        observation_mask_seq,
        enabled=r2_latent_normalize,
    )
    online_target_latents = r2_normalize_latent(
        online_target_latents_raw,
        target_entity_mask_seq_full,
        enabled=r2_latent_normalize,
    )
    target_latents = r2_normalize_latent(
        consistency_target_latents_raw,
        target_entity_mask_seq_full,
        enabled=r2_latent_normalize,
    )

    _, _, entities, latent_dim = input_latents.shape
    main_memory = memory_module.initial_memory(
        bsz,
        entities,
        device=entity_seq.device,
        dtype=input_latents.dtype,
    )

    # At an arbitrary segment start, the roster/slot structure is known but
    # visibility is not presence. Start with a structural prior and let the
    # learned presence head reduce belief after deaths/disappearances.
    main_presence = slot_mask_seq[:, 0].to(input_latents.dtype)
    static_flat = static_condition if static_condition is not None else None

    # Compute the expensive ordered joint-action representation once for every
    # sequence timestep. The old implementation recomputed it in every one of
    # the P x H overlapping rollout updates.
    action_context_seq: torch.Tensor | None = None
    if getattr(memory_module, "uses_action", False):
        if not hasattr(memory_module, "precompute_action_context_sequence"):
            raise RuntimeError(
                "Action-conditioned memory lacks cached sequence context support"
            )
        action_context_seq = memory_module.precompute_action_context_sequence(
            action_seq,
            action_mask_seq,
            entities=entities,
            dtype=input_latents.dtype,
        )

    static_bp: torch.Tensor | None = None
    if static_condition is not None:
        static_bp = (
            static_condition.unsqueeze(1)
            .expand(-1, p, *static_condition.shape[1:])
            .reshape(bsz * p, *static_condition.shape[1:])
        )

    # The first transition for each start is also the transition needed to
    # advance real-history presence and memory. Compute it once in chronological
    # order, retain the exact original semantics, and reuse it as H1.
    h1_pred: list[torch.Tensor] = []
    h1_raw_pred: list[torch.Tensor] = []
    h1_target: list[torch.Tensor] = []
    h1_rep_target: list[torch.Tensor] = []
    h1_raw_target: list[torch.Tensor] = []
    h1_target_entity: list[torch.Tensor] = []
    h1_target_mask: list[torch.Tensor] = []
    h1_slot_mask: list[torch.Tensor] = []
    h1_valid: list[torch.Tensor] = []
    h1_presence_logits: list[torch.Tensor] = []
    h1_future_presence: list[torch.Tensor] = []
    h1_next_memory: list[torch.Tensor] = []
    memory_norms: list[torch.Tensor] = []

    posterior_state_by_time: list[torch.Tensor] = []
    posterior_embed_by_time: list[torch.Tensor] = []
    posterior_valid_by_time: list[torch.Tensor] = []
    memory_state_by_time: list[torch.Tensor] = []
    memory_target_by_time: list[torch.Tensor] = []
    memory_valid_by_time: list[torch.Tensor] = []

    for start_idx in range(p):
        z_start = input_latents[:, start_idx]
        observed_start = observation_mask_seq[:, start_idx]
        slot_start = slot_mask_seq[:, start_idx]

        main_presence = merge_observed_presence(
            main_presence,
            observed_start,
            slot_start,
        )

        posterior_conditioned = memory_module.condition(
            z_start,
            main_memory,
            main_presence,
        )
        posterior_state_by_time.append(
            torch.cat([main_memory, posterior_conditioned], dim=-1)
        )
        posterior_embed_by_time.append(
            posterior_target_latents[:, start_idx]
        )
        posterior_valid_by_time.append(
            observed_start
            * slot_start
            * state_mask[:, start_idx].unsqueeze(-1)
        )

        if start_idx > 0:
            memory_state_by_time.append(main_memory)
            memory_target_by_time.append(target_latents[:, start_idx])
            memory_valid_by_time.append(
                target_entity_mask_seq_full[:, start_idx]
                * slot_start
                * state_mask[:, start_idx].unsqueeze(-1)
            )

        action_h = action_seq[:, start_idx : start_idx + 1]
        action_mask_h = action_mask_seq[:, start_idx : start_idx + 1]
        target_idx = start_idx + 1
        future_slot_mask = slot_mask_seq[:, target_idx]

        pred_h_raw = model.predictor(
            posterior_conditioned.unsqueeze(1),
            action_h,
            action_mask_h,
            torch.ones(
                (bsz, 1),
                device=entity_seq.device,
                dtype=entity_seq.dtype,
            ),
            slot_start.unsqueeze(1),
            static_flat,
        )[:, 0]
        pred_h_raw = pred_h_raw * future_slot_mask.unsqueeze(-1)
        pred_h = r2_normalize_latent(
            pred_h_raw,
            future_slot_mask,
            enabled=r2_latent_normalize,
        )
        presence_logits_h = model.predict_presence(pred_h)
        future_presence = (
            torch.sigmoid(presence_logits_h.float())
            .to(dtype=pred_h.dtype)
            * future_slot_mask
        )

        if action_context_seq is not None:
            next_memory = memory_module.update(
                z_start,
                main_memory,
                observed_start,
                action_context=action_context_seq[:, start_idx],
            )
        else:
            next_memory = memory_module.update(
                z_start,
                main_memory,
                observed_start,
            )

        h1_pred.append(pred_h)
        h1_raw_pred.append(pred_h_raw)
        h1_target.append(target_latents[:, target_idx])
        h1_rep_target.append(online_target_latents[:, target_idx])
        h1_raw_target.append(consistency_target_latents_raw[:, target_idx])
        h1_target_entity.append(target_entity_seq_full[:, target_idx])
        h1_target_mask.append(target_entity_mask_seq_full[:, target_idx])
        h1_slot_mask.append(future_slot_mask)
        h1_valid.append(state_mask[:, target_idx])
        h1_presence_logits.append(presence_logits_h)
        h1_future_presence.append(future_presence)
        h1_next_memory.append(next_memory)
        memory_norms.append(
            next_memory.detach().float().norm(dim=-1).mean()
        )

        main_memory = next_memory
        main_presence = future_presence

    pred_steps_by_h = [torch.stack(h1_pred, dim=1)]
    raw_pred_steps_by_h = [torch.stack(h1_raw_pred, dim=1)]
    target_steps_by_h = [torch.stack(h1_target, dim=1)]
    rep_target_steps_by_h = [torch.stack(h1_rep_target, dim=1)]
    raw_target_steps_by_h = [torch.stack(h1_raw_target, dim=1)]
    target_entity_steps_by_h = [torch.stack(h1_target_entity, dim=1)]
    target_mask_steps_by_h = [torch.stack(h1_target_mask, dim=1)]
    slot_mask_steps_by_h = [torch.stack(h1_slot_mask, dim=1)]
    valid_steps_by_h = [torch.stack(h1_valid, dim=1)]
    presence_steps_by_h = [torch.stack(h1_presence_logits, dim=1)]

    # All P rollout starts are now one large batch. H2--H5 therefore require
    # H-1 predictor/GRU launches rather than P*(H-1) launches.
    z_roll = pred_steps_by_h[0]
    rollout_memory = torch.stack(h1_next_memory, dim=1)
    current_presence = torch.stack(h1_future_presence, dim=1)
    current_slot_mask = slot_mask_steps_by_h[0]
    current_update_gate = current_presence
    start_indices = torch.arange(p, device=entity_seq.device)

    for step in range(1, h):
        action_indices = start_indices + step
        target_indices = action_indices + 1
        bp = bsz * p

        z_flat = z_roll.reshape(bp, entities, latent_dim)
        memory_flat = rollout_memory.reshape(
            bp,
            entities,
            rollout_memory.shape[-1],
        )
        presence_flat = current_presence.reshape(bp, entities)
        slot_flat = current_slot_mask.reshape(bp, entities)
        update_gate_flat = current_update_gate.reshape(bp, entities)

        z_conditioned = memory_module.condition(
            z_flat,
            memory_flat,
            presence_flat,
        )

        action_selected = action_seq.index_select(1, action_indices)
        action_mask_selected = action_mask_seq.index_select(
            1, action_indices
        )
        action_h_flat = action_selected.reshape(
            bp, *action_selected.shape[2:]
        ).unsqueeze(1)
        action_mask_h_flat = action_mask_selected.reshape(
            bp, *action_mask_selected.shape[2:]
        ).unsqueeze(1)

        pred_raw_flat = model.predictor(
            z_conditioned.unsqueeze(1),
            action_h_flat,
            action_mask_h_flat,
            torch.ones(
                (bp, 1),
                device=entity_seq.device,
                dtype=entity_seq.dtype,
            ),
            slot_flat.unsqueeze(1),
            static_bp,
        )[:, 0]

        future_slot = slot_mask_seq.index_select(1, target_indices)
        future_slot_flat = future_slot.reshape(bp, entities)
        pred_raw_flat = (
            pred_raw_flat * future_slot_flat.unsqueeze(-1)
        )
        pred_flat = r2_normalize_latent(
            pred_raw_flat,
            future_slot_flat,
            enabled=r2_latent_normalize,
        )
        presence_logits_flat = model.predict_presence(pred_flat)
        future_presence_flat = (
            torch.sigmoid(presence_logits_flat.float())
            .to(dtype=pred_flat.dtype)
            * future_slot_flat
        )

        if action_context_seq is not None:
            action_context = action_context_seq.index_select(
                1, action_indices
            ).reshape(
                bp,
                entities,
                action_context_seq.shape[-1],
            )
            next_memory_flat = memory_module.update(
                z_flat,
                memory_flat,
                update_gate_flat,
                action_context=action_context,
            )
        else:
            next_memory_flat = memory_module.update(
                z_flat,
                memory_flat,
                update_gate_flat,
            )

        pred_step = pred_flat.reshape(bsz, p, entities, latent_dim)
        raw_pred_step = pred_raw_flat.reshape(
            bsz, p, entities, latent_dim
        )
        presence_step = presence_logits_flat.reshape(bsz, p, entities)
        future_presence = future_presence_flat.reshape(bsz, p, entities)
        next_memory = next_memory_flat.reshape(
            bsz,
            p,
            entities,
            next_memory_flat.shape[-1],
        )

        pred_steps_by_h.append(pred_step)
        raw_pred_steps_by_h.append(raw_pred_step)
        target_steps_by_h.append(
            target_latents.index_select(1, target_indices)
        )
        rep_target_steps_by_h.append(
            online_target_latents.index_select(1, target_indices)
        )
        raw_target_steps_by_h.append(
            consistency_target_latents_raw.index_select(
                1, target_indices
            )
        )
        target_entity_steps_by_h.append(
            target_entity_seq_full.index_select(1, target_indices)
        )
        target_mask_steps_by_h.append(
            target_entity_mask_seq_full.index_select(1, target_indices)
        )
        slot_mask_steps_by_h.append(future_slot)
        valid_steps_by_h.append(
            state_mask.index_select(1, target_indices)
        )
        presence_steps_by_h.append(presence_step)
        memory_norms.append(
            next_memory_flat.detach().float().norm(dim=-1).mean()
        )

        z_roll = pred_step
        rollout_memory = next_memory
        current_presence = future_presence
        current_slot_mask = future_slot
        current_update_gate = future_presence

    pred_latent = torch.stack(pred_steps_by_h, dim=2)
    raw_pred_latent = torch.stack(raw_pred_steps_by_h, dim=2)
    target_latent = torch.stack(target_steps_by_h, dim=2)
    rep_target_latent = torch.stack(rep_target_steps_by_h, dim=2)
    raw_target_latent = torch.stack(raw_target_steps_by_h, dim=2)
    target_entity = torch.stack(target_entity_steps_by_h, dim=2)
    target_entity_mask = torch.stack(target_mask_steps_by_h, dim=2)
    entity_slot_mask = torch.stack(slot_mask_steps_by_h, dim=2)
    valid_mask = torch.stack(valid_steps_by_h, dim=2)
    presence_logits = torch.stack(presence_steps_by_h, dim=2)

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

    latent_mask = (
        target_entity_mask.unsqueeze(-1)
        * valid_mask.unsqueeze(-1).unsqueeze(-1)
    )

    dyn_loss = weighted_mse(
        pred_latent,
        target_latent.detach(),
        latent_mask,
        weights,
    )
    rep_loss = weighted_mse(
        pred_latent.detach(),
        rep_target_latent,
        latent_mask,
        weights,
    )
    dyn_loss_uniform = weighted_mse(
        pred_latent,
        target_latent.detach(),
        latent_mask,
        uniform_weights,
    )
    raw_pred_loss = weighted_mse(
        raw_pred_latent,
        raw_target_latent.detach(),
        latent_mask,
        weights,
    )
    one_step_dyn_loss = weighted_mse(
        pred_latent[:, :, 0:1],
        target_latent[:, :, 0:1].detach(),
        latent_mask[:, :, 0:1],
        torch.ones(1, device=pred_latent.device, dtype=pred_latent.dtype),
    )

    decoded = model.decode_entities(
        pred_latent.reshape(bsz * p * h, entities, latent_dim)
    ).reshape(bsz, p, h, entities, -1)

    aux_weights = uniform_weights if unweighted_aux_losses else weights
    target_valid = (
        target_entity_mask.unsqueeze(-1)
        * valid_mask.unsqueeze(-1).unsqueeze(-1)
    )
    base_feature_valid = feature_valid[:, None, None].expand_as(decoded)
    base_dynamic_valid = dynamic_feature_valid[:, None, None].expand_as(decoded)
    base_static_valid = static_feature_valid[:, None, None].expand_as(decoded)

    decoded_feature_mask = base_feature_valid * target_valid
    decoded_dynamic_mask = base_dynamic_valid * target_valid
    decoded_static_mask = base_static_valid * target_valid

    decoded_loss = weighted_feature_mse(
        decoded,
        target_entity,
        decoded_feature_mask,
        aux_weights,
    )
    decoded_dynamic_loss = weighted_feature_mse(
        decoded,
        target_entity,
        decoded_dynamic_mask,
        aux_weights,
    )
    decoded_static_loss = weighted_feature_mse(
        decoded,
        target_entity,
        decoded_static_mask,
        aux_weights,
    )

    presence_mask = entity_slot_mask * valid_mask.unsqueeze(-1)
    presence_loss = weighted_bce(
        presence_logits,
        target_entity_mask,
        presence_mask,
        aux_weights,
    )

    posterior_state = torch.stack(posterior_state_by_time, dim=1)
    posterior_embed = torch.stack(posterior_embed_by_time, dim=1)
    posterior_valid = torch.stack(posterior_valid_by_time, dim=1)
    projected_posterior = r2_projector(posterior_state)

    with (
        torch.cuda.amp.autocast(enabled=False)
        if pred_latent.is_cuda
        else nullcontext()
    ):
        barlow_stats = r2_barlow_loss(
            projected_posterior.float(),
            posterior_embed.float(),
            posterior_valid,
            redundancy_lambda=r2_barlow_lambda,
        )

    if memory_state_by_time and float(memory_barlow_scale) > 0.0:
        memory_state = torch.stack(memory_state_by_time, dim=1)
        memory_target = torch.stack(memory_target_by_time, dim=1)
        memory_valid = torch.stack(memory_valid_by_time, dim=1)
        projected_memory = memory_projector(memory_state)
        with (
            torch.cuda.amp.autocast(enabled=False)
            if pred_latent.is_cuda
            else nullcontext()
        ):
            memory_barlow_stats = r2_barlow_loss(
                projected_memory.float(),
                memory_target.float(),
                memory_valid,
                redundancy_lambda=r2_barlow_lambda,
            )
    else:
        zero = pred_latent.float().new_zeros(())
        memory_barlow_stats = {
            "raw": zero,
            "normalized": zero,
            "invariance": zero,
            "redundancy": zero,
            "diag_mean": zero,
            "offdiag_rms": zero,
            "samples": zero,
        }

    if sigreg_weight > 0.0:
        reg_latents = torch.cat(
            [input_latents, online_target_latents], dim=1
        )
        reg_masks = torch.cat(
            [observation_mask_seq, target_entity_mask_seq_full], dim=1
        )
        with (
            torch.cuda.amp.autocast(enabled=False)
            if pred_latent.is_cuda
            else nullcontext()
        ):
            sigreg_raw = stable_sigreg_loss(
                reg_latents.float(),
                reg_masks,
                knots=r2_sigreg_knots,
                num_proj=r2_sigreg_num_proj,
                projection_chunk=r2_sigreg_proj_chunk,
                max_samples=r2_sigreg_max_samples,
            )
        sigreg_objective = (
            sigreg_raw / float(latent_dim)
            if r2_sigreg_divide_by_dim
            else sigreg_raw
        )
    else:
        sigreg_raw = pred_latent.float().new_zeros(())
        sigreg_objective = pred_latent.float().new_zeros(())

    total_loss = (
        float(r2_dyn_scale) * dyn_loss
        + float(r2_rep_scale) * rep_loss
        + float(one_step_weight) * one_step_dyn_loss
        + float(r2_barlow_scale) * barlow_stats["normalized"]
        + float(memory_barlow_scale)
        * memory_barlow_stats["normalized"]
        + float(sigreg_weight) * sigreg_objective
        + float(decoder_weight) * decoded_loss
        + float(presence_weight) * presence_loss
    )

    latent_valid = (
        target_entity_mask
        * entity_slot_mask
        * valid_mask.unsqueeze(-1)
    )
    target_stats = latent_batch_statistics(target_latent, latent_valid)
    pred_stats = latent_batch_statistics(pred_latent, latent_valid)
    raw_target_stats = latent_batch_statistics(raw_target_latent, latent_valid)

    losses: dict[str, torch.Tensor] = {
        "total_loss": total_loss,
        "pred_loss": dyn_loss,
        "pred_loss_uniform": dyn_loss_uniform.detach(),
        "one_step_loss": one_step_dyn_loss.detach(),
        "weighted_one_step_loss": (
            float(one_step_weight) * one_step_dyn_loss
        ).detach(),
        "sigreg_loss": sigreg_raw,
        "r2_dyn_loss": dyn_loss,
        "r2_weighted_dyn_loss": (
            float(r2_dyn_scale) * dyn_loss
        ).detach(),
        "r2_rep_loss": rep_loss,
        "r2_weighted_rep_loss": (
            float(r2_rep_scale) * rep_loss
        ).detach(),
        "r2_barlow_raw": barlow_stats["raw"],
        "r2_barlow_loss": barlow_stats["normalized"],
        "r2_weighted_barlow_loss": (
            float(r2_barlow_scale) * barlow_stats["normalized"]
        ).detach(),
        "r2_barlow_invariance": barlow_stats["invariance"],
        "r2_barlow_redundancy": barlow_stats["redundancy"],
        "r2_barlow_diag_mean": barlow_stats["diag_mean"],
        "r2_barlow_offdiag_rms": barlow_stats["offdiag_rms"],
        "r2_barlow_samples": barlow_stats["samples"],
        "memory_barlow_raw": memory_barlow_stats["raw"],
        "memory_barlow_loss": memory_barlow_stats["normalized"],
        "memory_weighted_barlow_loss": (
            float(memory_barlow_scale)
            * memory_barlow_stats["normalized"]
        ).detach(),
        "memory_barlow_invariance": memory_barlow_stats["invariance"],
        "memory_barlow_redundancy": memory_barlow_stats["redundancy"],
        "memory_barlow_diag_mean": memory_barlow_stats["diag_mean"],
        "memory_barlow_offdiag_rms": memory_barlow_stats["offdiag_rms"],
        "memory_barlow_samples": memory_barlow_stats["samples"],
        "r2_sigreg_objective": sigreg_objective,
        "r2_weighted_sigreg_loss": (
            float(sigreg_weight) * sigreg_objective
        ).detach(),
        "raw_pred_loss_diagnostic": raw_pred_loss.detach(),
        "decoded_loss": decoded_loss,
        "decoded_dynamic_loss": decoded_dynamic_loss.detach(),
        "decoded_static_loss": decoded_static_loss.detach(),
        "presence_loss": presence_loss,
        "weighted_presence_loss": (
            float(presence_weight) * presence_loss
        ).detach(),
        "target_latent_std_mean": target_stats["std_mean"],
        "target_latent_std_min": target_stats["std_min"],
        "target_latent_norm_mean": target_stats["norm_mean"],
        "target_fraction_std_below_0p1": target_stats[
            "fraction_std_below_0p1"
        ],
        "pred_latent_std_mean": pred_stats["std_mean"],
        "pred_latent_std_min": pred_stats["std_min"],
        "pred_latent_norm_mean": pred_stats["norm_mean"],
        "pred_fraction_std_below_0p1": pred_stats[
            "fraction_std_below_0p1"
        ],
        "raw_target_latent_std_mean": raw_target_stats["std_mean"],
        "pred_to_target_std_ratio": (
            pred_stats["std_mean"]
            / target_stats["std_mean"].clamp_min(1e-8)
        ),
        "temporal_weight_sum": weights.sum().detach(),
        "rollout_window": torch.tensor(float(p), device=pred_latent.device),
        "rollout_horizon": torch.tensor(float(h), device=pred_latent.device),
        "memory_norm_mean": (
            torch.stack(memory_norms).mean()
            if memory_norms
            else torch.tensor(0.0, device=pred_latent.device)
        ),
        "predicted_presence_mean": (
            torch.sigmoid(presence_logits.float())
            * presence_mask
        ).sum().detach()
        / presence_mask.sum().clamp_min(1.0),
    }

    with torch.no_grad():
        for step in range(h):
            step_loss = weighted_mse(
                pred_latent[:, :, step : step + 1],
                target_latent[:, :, step : step + 1].detach(),
                latent_mask[:, :, step : step + 1],
                torch.ones(
                    1,
                    device=pred_latent.device,
                    dtype=pred_latent.dtype,
                ),
            )
            losses[f"pred_loss_h{step + 1}"] = step_loss.detach()

            decoded_step = weighted_feature_mse(
                decoded[:, :, step : step + 1],
                target_entity[:, :, step : step + 1],
                decoded_dynamic_mask[:, :, step : step + 1],
                torch.ones(
                    1,
                    device=pred_latent.device,
                    dtype=pred_latent.dtype,
                ),
            )
            losses[f"decoded_dynamic_loss_h{step + 1}"] = (
                decoded_step.detach()
            )

    assert_finite_losses(losses)
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

    dataset_window_mode = (
        "sequential" if args.event_balanced_sampling else args.window_mode
    )
    dataset_samples_per_epoch = (
        None if args.event_balanced_sampling else args.samples_per_epoch
    )

    dataset = VisibilityMarkovRolloutSMACJEPADataset(
        data_paths,
        rollout_window=args.rollout_window,
        rollout_horizon=args.rollout_horizon,
        mode="entity",
        window_mode=dataset_window_mode,
        samples_per_epoch=dataset_samples_per_epoch,
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

    event_sampler = None
    event_sampling_stats: dict[str, int | float] = {}
    if args.event_balanced_sampling:
        if args.samples_per_epoch is None:
            raise SystemExit(
                "--event-balanced-sampling requires --samples-per-epoch"
            )
        event_indices, ordinary_indices, event_counts = classify_event_segments(
            dataset,
            movement_threshold=args.event_movement_threshold,
            state_threshold=args.event_state_threshold,
            attack_action_min=args.event_attack_action_min,
            min_transitions=args.event_min_transitions,
            pool_fraction=args.event_pool_fraction,
        )
        event_sampler = EventBalancedSampler(
            event_indices=event_indices,
            ordinary_indices=ordinary_indices,
            num_samples=args.samples_per_epoch,
            event_fraction=args.event_fraction,
            seed=args.seed,
        )
        event_sampling_stats = event_counts | {
            "requested_event_fraction": float(args.event_fraction),
        }
        print(
            "event_balanced_sampling "
            + " ".join(
                f"{key}={value}"
                for key, value in event_sampling_stats.items()
            ),
            flush=True,
        )

    loader = DataLoader(
        dataset,
        batch_size=int(arch["batch_size"]),
        shuffle=(event_sampler is None),
        sampler=event_sampler,
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
            max_agents=dataset.metadata.max_agents,
            hidden_dim=args.rollout_memory_hidden_dim,
            residual=not args.rollout_memory_no_residual,
        ).to(device)
    else:
        memory_module = BeliefEntityRolloutGRUMemory(
            latent_dim=int(arch["latent_dim"]),
            memory_dim=args.rollout_memory_dim,
            hidden_dim=args.rollout_memory_hidden_dim,
            residual=not args.rollout_memory_no_residual,
        ).to(device)

    r2_projector = R2PosteriorProjector(
        latent_dim=int(arch["latent_dim"]),
        memory_dim=args.rollout_memory_dim,
    ).to(device)
    memory_projector = R2MemoryProjector(
        memory_dim=args.rollout_memory_dim,
        latent_dim=int(arch["latent_dim"]),
    ).to(device)

    target_encoder = None
    if args.ema_target_encoder:
        if not 0.0 <= float(args.ema_momentum) < 1.0:
            raise SystemExit("--ema-momentum must satisfy 0 <= m < 1")
        target_encoder = copy.deepcopy(model.encoder).to(device)
        target_encoder.eval()
        for parameter in target_encoder.parameters():
            parameter.requires_grad_(False)

    trainable_parameters = (
        list(model.parameters())
        + list(memory_module.parameters())
        + list(r2_projector.parameters())
        + list(memory_projector.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(arch["lr"]),
    )

    warmup_steps = max(int(args.lr_warmup_steps), 0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=(
            (
                lambda step: min(
                    1.0,
                    float(step + 1) / float(warmup_steps),
                )
            )
            if warmup_steps > 0
            else (lambda step: 1.0)
        ),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    global_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        checkpoint_config = checkpoint.get(
            "resolved_config", checkpoint.get("config", {})
        )
        if int(checkpoint_config.get("r2_blocker_fix_version", 0)) < 1:
            raise SystemExit(
                "This blocker-fixed trainer cannot resume an original Exp15 "
                "checkpoint. Memory architecture, recurrence chronology, and "
                "mask semantics changed. Start a new run directory instead."
            )
        model.load_state_dict(checkpoint["model_state"])
        if "memory_module_state" in checkpoint:
            memory_module.load_state_dict(checkpoint["memory_module_state"])
        else:
            print("memory_module_state not found in checkpoint; starting memory module fresh", flush=True)
        if "r2_projector_state" in checkpoint:
            r2_projector.load_state_dict(
                checkpoint["r2_projector_state"]
            )
        else:
            print(
                "r2_projector_state not found; starting projector fresh",
                flush=True,
            )
        if "memory_projector_state" in checkpoint:
            memory_projector.load_state_dict(
                checkpoint["memory_projector_state"]
            )
        elif float(args.memory_barlow_scale) > 0.0:
            raise SystemExit(
                "Cannot resume memory-Barlow run without memory_projector_state"
            )
        if args.ema_target_encoder:
            if "target_encoder_state" not in checkpoint:
                raise SystemExit(
                    "Cannot resume EMA run without target_encoder_state"
                )
            assert target_encoder is not None
            target_encoder.load_state_dict(
                checkpoint["target_encoder_state"]
            )
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint and amp_enabled:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))

    saved_config = vars(args) | arch | {
        "resolved_device": device.type,
        "amp_enabled": amp_enabled,
        "dataset_len": len(dataset),
        "training_regime": "markov_rollout_rnn_seqmem_r2offline",
        "enemy_visibility_mask": args.enemy_visibility_mask,
        "enemy_sight_range": args.enemy_sight_range,
        "action_conditioned_memory": args.action_conditioned_memory,
        "one_step_weight": args.one_step_weight,
        "target_mode": args.target_mode,
        "r2_latent_normalize": args.r2_latent_normalize,
        "r2_objective_version": 2,
        "r2_blocker_fix_version": 3,
        "experiment_suite": "exp16_exp19_blocker_fixed_ablations",
        "rollout_implementation": "cached_actions_vectorized_starts_v1",
        "action_context_cached_per_sequence": True,
        "rollout_starts_vectorized_after_h1": True,
        "ema_target_encoder": args.ema_target_encoder,
        "ema_momentum": args.ema_momentum,
        "ema_target_semantics": (
            "ema_dynamics_and_barlow_targets_online_representation_branch"
            if args.ema_target_encoder
            else "disabled"
        ),
        "memory_barlow_scale": args.memory_barlow_scale,
        "memory_barlow_excludes_segment_t0": True,
        "event_balanced_sampling": args.event_balanced_sampling,
        "event_sampling_stats": event_sampling_stats,
        "r2_barlow_on_observed_posterior": True,
        "r2_asymmetric_consistency": True,
        "hidden_belief_not_observation_masked": True,
        "uses_oracle_future_presence_masks": False,
        "consistent_real_imagined_chronology": True,
        "decoder_feature_padding_excluded": True,
        "agent_identity_preserved_in_memory_actions": True,
        "memory_architecture": (
            "agent_aware_ordered_joint_action_v2"
            if args.action_conditioned_memory
            else "belief_entity_gru_v2"
        ),
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
        wandb_run.watch(r2_projector, log=None)
        wandb_run.watch(memory_projector, log=None)

    def save_checkpoint(epoch_to_save: int, checkpoint_path: Path) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "memory_module_state": memory_module.state_dict(),
                "r2_projector_state": r2_projector.state_dict(),
                "memory_projector_state": memory_projector.state_dict(),
                "target_encoder_state": (
                    target_encoder.state_dict()
                    if target_encoder is not None
                    else None
                ),
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
                "scheduler_state": scheduler.state_dict(),
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
    memory_projector.train()
    if target_encoder is not None:
        target_encoder.eval()
    r2_projector.train()

    print(
        "markov_rollout_rnn_visibility_seqmem_r2offline "
        f"p={args.rollout_window} n={args.rollout_horizon} "
        f"memory_dim={args.rollout_memory_dim} "
        f"enemy_visibility_mask={args.enemy_visibility_mask} sight_range={args.enemy_sight_range} "
        f"temporal_loss={args.temporal_loss} td_lambda={args.td_lambda} "
        f"r2_dyn_scale={args.r2_dyn_scale} r2_rep_scale={args.r2_rep_scale} "
        f"r2_barlow_scale={args.r2_barlow_scale} "
        f"r2_barlow_lambda={args.r2_barlow_lambda} "
        f"r2_latent_normalize={args.r2_latent_normalize} "
        f"sigreg_weight={args.sigreg_weight} sigreg_weight_start={args.sigreg_weight_start} "
        f"sigreg_weight_end={args.sigreg_weight_end} sigreg_warmup_epochs={args.sigreg_warmup_epochs} "
        f"decoder_weight={args.decoder_weight} presence_weight={args.presence_weight} "
        f"action_conditioned_memory={args.action_conditioned_memory} "
        f"one_step_weight={args.one_step_weight} target_mode={args.target_mode} "
        f"ema_target_encoder={args.ema_target_encoder} ema_momentum={args.ema_momentum} "
        f"memory_barlow_scale={args.memory_barlow_scale} "
        f"event_balanced_sampling={args.event_balanced_sampling} event_fraction={args.event_fraction} "
        f"event_pool_fraction={args.event_pool_fraction} "
        f"window_mode={args.window_mode} samples_per_epoch={args.samples_per_epoch} "
        "blocker_fixes=hidden_belief,predicted_presence,consistent_chronology,"
        "feature_masked_decoder,agent_aware_actions",
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

        print(
            f"epoch_schedule epoch={epoch} "
            f"active_sigreg_weight={active_sigreg_weight:.6f}",
            flush=True,
        )

        epoch_sums: dict[str, float] = {}
        epoch_batches = 0

        for batch in loader:
            global_step += 1
            epoch_batches += 1

            batch = add_feature_valid_masks(batch, dataset)
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
                    r2_projector,
                    memory_projector,
                    target_encoder,
                    batch,
                    rollout_window=args.rollout_window,
                    rollout_horizon=args.rollout_horizon,
                    temporal_loss_mode=args.temporal_loss,
                    td_lambda=args.td_lambda,
                    flat_decay_start=args.flat_decay_start,
                    flat_decay_final_weight=args.flat_decay_final_weight,
                    sigreg_weight=active_sigreg_weight,
                    decoder_weight=args.decoder_weight,
                    presence_weight=args.presence_weight,
                    one_step_weight=args.one_step_weight,
                    target_mode=args.target_mode,
                    detach_rollout_targets=args.detach_rollout_targets,
                    unweighted_aux_losses=args.unweighted_aux_losses,
                    r2_dyn_scale=args.r2_dyn_scale,
                    r2_rep_scale=args.r2_rep_scale,
                    r2_barlow_scale=args.r2_barlow_scale,
                    memory_barlow_scale=args.memory_barlow_scale,
                    r2_barlow_lambda=args.r2_barlow_lambda,
                    r2_latent_normalize=args.r2_latent_normalize,
                    r2_sigreg_divide_by_dim=(
                        args.r2_sigreg_divide_by_dim
                    ),
                    r2_sigreg_knots=args.r2_sigreg_knots,
                    r2_sigreg_num_proj=args.r2_sigreg_num_proj,
                    r2_sigreg_proj_chunk=args.r2_sigreg_proj_chunk,
                    r2_sigreg_max_samples=args.r2_sigreg_max_samples,
                )

            assert_finite_losses(losses)
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
            scheduler.step()
            if target_encoder is not None:
                update_ema_encoder(
                    target_encoder,
                    model.encoder,
                    args.ema_momentum,
                )

            row: dict[str, float | int] = {
                "epoch": epoch,
                "step": global_step,
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
                    "train/one_step_loss": row.get("one_step_loss"),
                    "train/weighted_one_step_loss": row.get("weighted_one_step_loss"),
                    "train/sigreg_loss": row.get("sigreg_loss"),
                    "train/active_sigreg_weight": row.get("active_sigreg_weight"),
                    "train/decoded_loss": row.get("decoded_loss"),
                    "train/decoded_dynamic_loss": row.get("decoded_dynamic_loss"),
                    "train/decoded_static_loss": row.get("decoded_static_loss"),
                    "train/presence_loss": row.get("presence_loss"),
                    "train/weighted_presence_loss": row.get("weighted_presence_loss"),
                    "train/presence_weight": args.presence_weight,
                    "train/memory_norm_mean": row.get("memory_norm_mean"),
                    "train/memory_barlow_loss": row.get("memory_barlow_loss"),
                    "train/memory_barlow_diag_mean": row.get("memory_barlow_diag_mean"),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                for key, value in row.items():
                    if (
                        key.startswith("pred_loss_h")
                        or key.startswith("decoded_dynamic_loss_h")
                        or key.startswith("r2_")
                        or key.startswith("memory_barlow")
                        or key.startswith("target_latent_")
                        or key.startswith("pred_latent_")
                        or key.startswith("raw_")
                        or key in {
                            "pred_to_target_std_ratio",
                            "target_fraction_std_below_0p1",
                            "pred_fraction_std_below_0p1",
                        }
                    ):
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
                    "pred_uniform={pred_loss_uniform:.6f} "
                    "rep={r2_rep_loss:.6f} "
                    "barlow={r2_barlow_loss:.6f} "
                    "barlow_diag={r2_barlow_diag_mean:.4f} "
                    "barlow_offdiag={r2_barlow_offdiag_rms:.4f} "
                    "sigreg_obj={r2_sigreg_objective:.6f} "
                    "memory_barlow={memory_barlow_loss:.6f} "
                    "target_std={target_latent_std_mean:.4f} "
                    "pred_std={pred_latent_std_mean:.4f} "
                    "decoded_loss={decoded_loss:.6f} decoded_dynamic={decoded_dynamic_loss:.6f} "
                    "presence_loss={presence_loss:.6f} "
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
                "epoch/active_sigreg_weight": epoch_row.get("active_sigreg_weight"),
                "epoch/decoded_loss": epoch_row.get("decoded_loss"),
                "epoch/decoded_dynamic_loss": epoch_row.get("decoded_dynamic_loss"),
                "epoch/decoded_static_loss": epoch_row.get("decoded_static_loss"),
                "epoch/presence_loss": epoch_row.get("presence_loss"),
                "epoch/weighted_presence_loss": epoch_row.get("weighted_presence_loss"),
                "epoch/presence_weight": args.presence_weight,
                "epoch/memory_norm_mean": epoch_row.get("memory_norm_mean"),
                "epoch/memory_barlow_loss": epoch_row.get("memory_barlow_loss"),
                "epoch/memory_barlow_diag_mean": epoch_row.get("memory_barlow_diag_mean"),
            }
            for key, value in epoch_row.items():
                if (
                    key.startswith("pred_loss_h")
                    or key.startswith("r2_")
                    or key.startswith("memory_barlow")
                    or key.startswith("target_latent_")
                    or key.startswith("pred_latent_")
                    or key.startswith("raw_")
                    or key in {
                        "pred_to_target_std_ratio",
                        "target_fraction_std_below_0p1",
                        "pred_fraction_std_below_0p1",
                    }
                ):
                    log_dict[f"epoch/{key}"] = value
            wandb_run.log(log_dict, step=global_step)

        print(
            "epoch_summary epoch={epoch} step={step} "
            "total_loss={total_loss:.6f} pred_loss={pred_loss:.6f} "
            "pred_uniform={pred_loss_uniform:.6f} "
            "rep={r2_rep_loss:.6f} "
            "barlow={r2_barlow_loss:.6f} "
            "barlow_diag={r2_barlow_diag_mean:.4f} "
            "barlow_offdiag={r2_barlow_offdiag_rms:.4f} "
            "sigreg_obj={r2_sigreg_objective:.6f} "
            "memory_barlow={memory_barlow_loss:.6f} "
            "target_std={target_latent_std_mean:.4f} "
            "pred_std={pred_latent_std_mean:.4f} "
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
        write_svg_line_plot(step_rows, "step", "pred_loss", "R2 Offline Dynamics Loss Per Training Step", out_dir / "pred_loss_by_step.svg")
        write_svg_line_plot(epoch_rows, "epoch", "r2_barlow_loss", "R2 Posterior Barlow Loss Per Epoch", out_dir / "r2_barlow_loss_by_epoch.svg")
        write_svg_line_plot(epoch_rows, "epoch", "r2_barlow_diag_mean", "R2 Barlow Diagonal Correlation Per Epoch", out_dir / "r2_barlow_diag_by_epoch.svg")
        if float(args.memory_barlow_scale) > 0.0:
            write_svg_line_plot(epoch_rows, "epoch", "memory_barlow_loss", "Memory-only Barlow Loss Per Epoch", out_dir / "memory_barlow_loss_by_epoch.svg")

    if wandb_run is not None:
        wandb_run.save(str(out_dir / "config.json"))
        wandb_run.save(str(out_dir / "checkpoint.pt"))
        wandb_run.finish()


if __name__ == "__main__":
    main()
