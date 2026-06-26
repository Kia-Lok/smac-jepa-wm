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
    parser = argparse.ArgumentParser(description="Train SMAC-JEPA with Markov rollout + RNN memory")

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
    memory_module: EntityRolloutGRUMemory,
    r2_projector: R2PosteriorProjector,
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
    r2_barlow_lambda: float,
    r2_latent_normalize: bool,
    r2_sigreg_divide_by_dim: bool,
    r2_sigreg_knots: int,
    r2_sigreg_num_proj: int,
    r2_sigreg_proj_chunk: int,
    r2_sigreg_max_samples: int,
) -> dict[str, torch.Tensor]:
    """
    Offline deterministic analogue of the R2-Dreamer world-model objective.

    Faithful structural correspondences:
      * observed posterior state -> detached current observation embedding Barlow
      * asymmetric dynamics and representation consistency via stop-gradient
      * multi-step action-conditioned prior imagination
      * semantic prediction heads remain because offline JEPA lacks R2's
        reward, continuation, value, and policy supervision

    The old symmetric raw latent MSE is not optimized because a trainable
    encoder and predictor can jointly shrink its coordinate scale.
    """
    del detach_rollout_targets

    entity_seq = batch["entity_seq"]
    entity_mask_seq = batch["entity_mask_seq"]

    if target_mode == "observed":
        target_entity_seq_full = entity_seq
        target_entity_mask_seq_full = entity_mask_seq
    elif target_mode == "full":
        target_entity_seq_full = batch.get(
            "target_entity_seq", entity_seq
        )
        target_entity_mask_seq_full = batch.get(
            "target_entity_mask_seq", entity_mask_seq
        )
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    action_seq = batch["action_seq"]
    action_mask_seq = batch["action_mask_seq"]
    state_mask = batch["state_mask"]
    static_condition = batch.get("static_condition")
    entity_slot_mask_seq = batch["entity_slot_mask_seq"]

    bsz = entity_seq.shape[0]
    p = int(rollout_window)
    h = int(rollout_horizon)

    input_latents_raw = model.encoder(
        entity_seq, entity_mask_seq
    )
    target_latents_raw = model.encoder(
        target_entity_seq_full,
        target_entity_mask_seq_full,
    )

    input_latents = r2_normalize_latent(
        input_latents_raw,
        entity_mask_seq,
        enabled=r2_latent_normalize,
    )
    target_latents = r2_normalize_latent(
        target_latents_raw,
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
    static_flat = (
        static_condition if static_condition is not None else None
    )

    pred_by_start: list[torch.Tensor] = []
    raw_pred_by_start: list[torch.Tensor] = []
    target_by_start: list[torch.Tensor] = []
    raw_target_by_start: list[torch.Tensor] = []
    target_entity_by_start: list[torch.Tensor] = []
    target_entity_mask_by_start: list[torch.Tensor] = []
    slot_mask_by_start: list[torch.Tensor] = []
    valid_by_start: list[torch.Tensor] = []
    memory_norms: list[torch.Tensor] = []

    posterior_state_by_time: list[torch.Tensor] = []
    posterior_embed_by_time: list[torch.Tensor] = []
    posterior_valid_by_time: list[torch.Tensor] = []

    for start_idx in range(p):
        z_start = input_latents[:, start_idx]
        start_entity_mask = entity_mask_seq[:, start_idx]

        # R2 analogue of observed posterior s_t=(h_t,z_t).
        posterior_conditioned = memory_module.condition(
            z_start,
            main_memory,
            start_entity_mask,
        )
        posterior_state_by_time.append(
            torch.cat([main_memory, posterior_conditioned], dim=-1)
        )
        posterior_embed_by_time.append(z_start)
        posterior_valid_by_time.append(
            start_entity_mask
            * entity_slot_mask_seq[:, start_idx]
            * state_mask[:, start_idx].unsqueeze(-1)
        )

        rollout_memory = main_memory
        z = z_start
        current_entity_mask = start_entity_mask

        pred_steps: list[torch.Tensor] = []
        raw_pred_steps: list[torch.Tensor] = []
        target_steps: list[torch.Tensor] = []
        raw_target_steps: list[torch.Tensor] = []
        target_entity_steps: list[torch.Tensor] = []
        target_entity_mask_steps: list[torch.Tensor] = []
        slot_mask_steps: list[torch.Tensor] = []
        valid_steps: list[torch.Tensor] = []

        for step in range(h):
            action_idx = start_idx + step
            target_idx = start_idx + step + 1

            action_h = action_seq[:, action_idx : action_idx + 1]
            action_mask_h = action_mask_seq[
                :, action_idx : action_idx + 1
            ]
            valid_h = state_mask[:, target_idx]

            timestep_mask_h = torch.ones(
                (bsz, 1),
                device=entity_seq.device,
                dtype=entity_seq.dtype,
            )
            entity_mask_h = current_entity_mask.unsqueeze(1)

            z_conditioned = memory_module.condition(
                z,
                rollout_memory,
                current_entity_mask,
            )
            pred_h_raw = model.predictor(
                z_conditioned.unsqueeze(1),
                action_h,
                action_mask_h,
                timestep_mask_h,
                entity_mask_h,
                static_flat,
            )[:, 0]
            pred_h_raw = (
                pred_h_raw * current_entity_mask.unsqueeze(-1)
            )
            pred_h = r2_normalize_latent(
                pred_h_raw,
                current_entity_mask,
                enabled=r2_latent_normalize,
            )

            target_mask_h = target_entity_mask_seq_full[:, target_idx]

            pred_steps.append(pred_h)
            raw_pred_steps.append(pred_h_raw)
            target_steps.append(target_latents[:, target_idx])
            raw_target_steps.append(
                target_latents_raw[:, target_idx]
            )
            target_entity_steps.append(
                target_entity_seq_full[:, target_idx]
            )
            target_entity_mask_steps.append(target_mask_h)
            slot_mask_steps.append(
                entity_slot_mask_seq[:, target_idx]
            )
            valid_steps.append(valid_h)

            if getattr(memory_module, "uses_action", False):
                rollout_memory = memory_module.update(
                    pred_h,
                    rollout_memory,
                    target_mask_h,
                    action=action_h[:, 0],
                    action_mask=action_mask_h[:, 0],
                )
            else:
                rollout_memory = memory_module.update(
                    pred_h,
                    rollout_memory,
                    target_mask_h,
                )

            memory_norms.append(
                rollout_memory.detach().float().norm(
                    dim=-1
                ).mean()
            )
            z = pred_h
            current_entity_mask = target_mask_h

        pred_by_start.append(torch.stack(pred_steps, dim=1))
        raw_pred_by_start.append(
            torch.stack(raw_pred_steps, dim=1)
        )
        target_by_start.append(torch.stack(target_steps, dim=1))
        raw_target_by_start.append(
            torch.stack(raw_target_steps, dim=1)
        )
        target_entity_by_start.append(
            torch.stack(target_entity_steps, dim=1)
        )
        target_entity_mask_by_start.append(
            torch.stack(target_entity_mask_steps, dim=1)
        )
        slot_mask_by_start.append(
            torch.stack(slot_mask_steps, dim=1)
        )
        valid_by_start.append(torch.stack(valid_steps, dim=1))

        # RSSM chronology: h_{t+1}=f(h_t,z_t,a_t).
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
            main_memory = memory_module.update(
                z_start,
                main_memory,
                start_entity_mask,
            )
        memory_norms.append(
            main_memory.detach().float().norm(dim=-1).mean()
        )

    pred_latent = torch.stack(pred_by_start, dim=1)
    raw_pred_latent = torch.stack(raw_pred_by_start, dim=1)
    target_latent = torch.stack(target_by_start, dim=1)
    raw_target_latent = torch.stack(raw_target_by_start, dim=1)
    target_entity = torch.stack(target_entity_by_start, dim=1)
    target_entity_mask = torch.stack(
        target_entity_mask_by_start, dim=1
    )
    entity_slot_mask = torch.stack(slot_mask_by_start, dim=1)
    valid_mask = torch.stack(valid_by_start, dim=1)

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

    mask = (
        target_entity_mask.unsqueeze(-1)
        * valid_mask.unsqueeze(-1).unsqueeze(-1)
    )

    # Deterministic analogue of R2/Dreamer KL balancing.
    dyn_loss = weighted_mse(
        pred_latent,
        target_latent.detach(),
        mask,
        weights,
    )
    rep_loss = weighted_mse(
        pred_latent.detach(),
        target_latent,
        mask,
        weights,
    )
    dyn_loss_uniform = weighted_mse(
        pred_latent,
        target_latent.detach(),
        mask,
        uniform_weights,
    )
    raw_pred_loss = weighted_mse(
        raw_pred_latent,
        raw_target_latent.detach(),
        mask,
        weights,
    )

    one_step_dyn_loss = weighted_mse(
        pred_latent[:, :, 0:1],
        target_latent[:, :, 0:1].detach(),
        mask[:, :, 0:1],
        torch.ones(
            1,
            device=pred_latent.device,
            dtype=pred_latent.dtype,
        ),
    )

    decoded = model.decode_entities(
        pred_latent.reshape(
            bsz * p * h, entities, latent_dim
        )
    ).reshape(bsz, p, h, entities, -1)

    aux_weights = (
        uniform_weights if unweighted_aux_losses else weights
    )
    decoded_loss = weighted_mse(
        decoded,
        target_entity,
        mask,
        aux_weights,
    )

    presence_logits = model.predict_presence(
        pred_latent.reshape(
            bsz * p * h, entities, latent_dim
        )
    ).reshape(bsz, p, h, entities)

    presence_mask = (
        entity_slot_mask * valid_mask.unsqueeze(-1)
    )
    presence_loss = weighted_bce(
        presence_logits,
        target_entity_mask,
        presence_mask,
        aux_weights,
    )

    posterior_state = torch.stack(
        posterior_state_by_time, dim=1
    )
    posterior_embed = torch.stack(
        posterior_embed_by_time, dim=1
    )
    posterior_valid = torch.stack(
        posterior_valid_by_time, dim=1
    )
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

    # Exp15 only: weak, stable SIGReg safety term.
    if sigreg_weight > 0.0:
        reg_latents = torch.cat(
            [input_latents, target_latents], dim=1
        )
        reg_masks = torch.cat(
            [
                entity_mask_seq,
                target_entity_mask_seq_full,
            ],
            dim=1,
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
        + float(sigreg_weight) * sigreg_objective
        + float(decoder_weight) * decoded_loss
        + float(presence_weight) * presence_loss
    )

    latent_valid = (
        target_entity_mask
        * entity_slot_mask
        * valid_mask.unsqueeze(-1)
    )
    target_stats = latent_batch_statistics(
        target_latent, latent_valid
    )
    pred_stats = latent_batch_statistics(
        pred_latent, latent_valid
    )
    raw_target_stats = latent_batch_statistics(
        raw_target_latent, latent_valid
    )

    losses: dict[str, torch.Tensor] = {
        "total_loss": total_loss,
        # Legacy aliases retained for old plotting/evaluation utilities.
        "pred_loss": dyn_loss,
        "pred_loss_uniform": dyn_loss_uniform.detach(),
        "one_step_loss": one_step_dyn_loss.detach(),
        "weighted_one_step_loss": (
            float(one_step_weight) * one_step_dyn_loss
        ).detach(),
        "sigreg_loss": sigreg_raw,
        # R2 terms.
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
            float(r2_barlow_scale)
            * barlow_stats["normalized"]
        ).detach(),
        "r2_barlow_invariance": barlow_stats["invariance"],
        "r2_barlow_redundancy": barlow_stats["redundancy"],
        "r2_barlow_diag_mean": barlow_stats["diag_mean"],
        "r2_barlow_offdiag_rms": barlow_stats["offdiag_rms"],
        "r2_barlow_samples": barlow_stats["samples"],
        "r2_sigreg_objective": sigreg_objective,
        "r2_weighted_sigreg_loss": (
            float(sigreg_weight) * sigreg_objective
        ).detach(),
        "raw_pred_loss_diagnostic": raw_pred_loss.detach(),
        "decoded_loss": decoded_loss,
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
        "rollout_window": torch.tensor(
            float(p), device=pred_latent.device
        ),
        "rollout_horizon": torch.tensor(
            float(h), device=pred_latent.device
        ),
        "memory_norm_mean": (
            torch.stack(memory_norms).mean()
            if memory_norms
            else torch.tensor(0.0, device=pred_latent.device)
        ),
    }

    with torch.no_grad():
        for step in range(h):
            step_loss = weighted_mse(
                pred_latent[:, :, step : step + 1],
                target_latent[
                    :, :, step : step + 1
                ].detach(),
                mask[:, :, step : step + 1],
                torch.ones(
                    1,
                    device=pred_latent.device,
                    dtype=pred_latent.dtype,
                ),
            )
            losses[f"pred_loss_h{step + 1}"] = step_loss.detach()

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

    r2_projector = R2PosteriorProjector(
        latent_dim=int(arch["latent_dim"]),
        memory_dim=args.rollout_memory_dim,
    ).to(device)

    trainable_parameters = (
        list(model.parameters())
        + list(memory_module.parameters())
        + list(r2_projector.parameters())
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
        "r2_objective_version": 1,
        "r2_barlow_on_observed_posterior": True,
        "r2_asymmetric_consistency": True,
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

    def save_checkpoint(epoch_to_save: int, checkpoint_path: Path) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "memory_module_state": memory_module.state_dict(),
                "r2_projector_state": r2_projector.state_dict(),
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
                    "train/presence_loss": row.get("presence_loss"),
                    "train/weighted_presence_loss": row.get("weighted_presence_loss"),
                    "train/presence_weight": args.presence_weight,
                    "train/memory_norm_mean": row.get("memory_norm_mean"),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                for key, value in row.items():
                    if (
                        key.startswith("pred_loss_h")
                        or key.startswith("r2_")
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
                    "target_std={target_latent_std_mean:.4f} "
                    "pred_std={pred_latent_std_mean:.4f} "
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
                "epoch/active_sigreg_weight": epoch_row.get("active_sigreg_weight"),
                "epoch/decoded_loss": epoch_row.get("decoded_loss"),
                "epoch/presence_loss": epoch_row.get("presence_loss"),
                "epoch/weighted_presence_loss": epoch_row.get("weighted_presence_loss"),
                "epoch/presence_weight": args.presence_weight,
                "epoch/memory_norm_mean": epoch_row.get("memory_norm_mean"),
            }
            for key, value in epoch_row.items():
                if (
                    key.startswith("pred_loss_h")
                    or key.startswith("r2_")
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

    if wandb_run is not None:
        wandb_run.save(str(out_dir / "config.json"))
        wandb_run.save(str(out_dir / "checkpoint.pt"))
        wandb_run.finish()


if __name__ == "__main__":
    main()
