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
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler

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


class IndexedDataset(Dataset):
    """Adds stable sample_index for priority-table updates."""

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


def make_priority_weights(
    scores: torch.Tensor,
    *,
    alpha: float,
    uniform_mix: float,
    eps: float,
    max_multiplier: float,
) -> torch.Tensor:
    """Convert per-sample difficulty scores into sampler probabilities."""
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

    # Sample-level prioritized replay / PER-style sampling.
    parser.add_argument("--sample-prioritized", action="store_true")
    parser.add_argument("--priority-alpha", type=float, default=0.4)
    parser.add_argument("--priority-uniform-mix", type=float, default=0.7)
    parser.add_argument("--priority-ema-beta", type=float, default=0.95)
    parser.add_argument("--priority-warmup-epochs", type=int, default=2)
    parser.add_argument("--priority-eps", type=float, default=1e-6)
    parser.add_argument("--priority-max-multiplier", type=float, default=10.0)
    parser.add_argument(
        "--priority-score",
        choices=["pred_loss", "decoded_loss", "total_loss"],
        default="pred_loss",
        help="Per-segment score used for sample priority updates.",
    )

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


def weighted_mse_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return one weighted MSE per dataset item. Shape returns [B]."""
    w = weights.view(1, 1, -1, 1, 1)
    weighted_mask = mask * w
    num = ((pred - target).pow(2) * weighted_mask).sum(dim=(1, 2, 3, 4))
    den = weighted_mask.sum(dim=(1, 2, 3, 4)).clamp_min(1.0) * pred.shape[-1]
    return num / den


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


def weighted_bce_per_sample(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    w = weights.view(1, 1, -1, 1)
    weighted_mask = mask * w
    num = (raw * weighted_mask).sum(dim=(1, 2, 3))
    den = weighted_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return num / den


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


def markov_rollout_rnn_losses(
    model: SMACJEPA,
    memory_module: EntityRolloutGRUMemory,
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
    return_sample_scores: bool = False,
    priority_score: str = "pred_loss",
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], torch.Tensor]:
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

            z = pred_h
            current_entity_mask = target_mask_h

        pred_by_start.append(torch.stack(pred_steps, dim=1))                       # [B, H, E, D]
        target_by_start.append(torch.stack(target_steps, dim=1))                   # [B, H, E, D]
        target_entity_by_start.append(torch.stack(target_entity_steps, dim=1))      # [B, H, E, F]
        target_entity_mask_by_start.append(torch.stack(target_entity_mask_steps, dim=1))  # [B, H, E]
        slot_mask_by_start.append(torch.stack(slot_mask_steps, dim=1))             # [B, H, E]
        valid_by_start.append(torch.stack(valid_steps, dim=1))                     # [B, H]

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
    per_sample_pred_loss = weighted_mse_per_sample(pred_latent, target_for_pred, mask, weights)

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
    per_sample_decoded_loss = weighted_mse_per_sample(decoded, target_entity, mask, aux_weights)

    presence_logits = model.predict_presence(
        pred_latent.reshape(bsz * p * h, entities, latent_dim)
    ).reshape(bsz, p, h, entities)

    presence_mask = entity_slot_mask * valid_mask.unsqueeze(-1)
    presence_loss = weighted_bce(presence_logits, target_entity_mask, presence_mask, aux_weights)
    per_sample_presence_loss = weighted_bce_per_sample(
        presence_logits,
        target_entity_mask,
        presence_mask,
        aux_weights,
    )

    reg_latents = torch.cat([input_latents, target_latents], dim=1)
    reg_masks = torch.cat([entity_mask_seq, target_entity_mask_seq_full], dim=1)
    reg_loss = sigreg_loss(reg_latents, reg_masks)

    total_loss = (
        pred_loss
        + one_step_weight * one_step_loss
        + sigreg_weight * reg_loss
        + decoder_weight * decoded_loss
        + presence_weight * presence_loss
    )

    per_sample_total_loss = (
        per_sample_pred_loss
        + decoder_weight * per_sample_decoded_loss
        + presence_weight * per_sample_presence_loss
    )

    losses: dict[str, torch.Tensor] = {
        "total_loss": total_loss,
        "pred_loss": pred_loss,
        "pred_loss_uniform": pred_loss_uniform.detach(),
        "one_step_loss": one_step_loss.detach(),
        "weighted_one_step_loss": (one_step_weight * one_step_loss).detach(),
        "sigreg_loss": reg_loss,
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

    if return_sample_scores:
        if priority_score == "pred_loss":
            sample_scores = per_sample_pred_loss
        elif priority_score == "decoded_loss":
            sample_scores = per_sample_decoded_loss
        elif priority_score == "total_loss":
            sample_scores = per_sample_total_loss
        else:
            raise ValueError(f"Unknown priority_score: {priority_score}")
        return losses, sample_scores.detach()

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

    base_dataset = VisibilityMarkovRolloutSMACJEPADataset(
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

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(memory_module.parameters()),
        lr=float(arch["lr"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    priority_scores = torch.ones(len(dataset), dtype=torch.float32)
    priority_seen = torch.zeros(len(dataset), dtype=torch.bool)

    start_epoch = 1
    global_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "memory_module_state" in checkpoint:
            memory_module.load_state_dict(checkpoint["memory_module_state"])
        else:
            print("memory_module_state not found in checkpoint; starting memory module fresh", flush=True)
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
        elif args.sample_prioritized:
            print("priority_scores not found or wrong shape; starting fresh", flush=True)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))

    saved_config = vars(args) | arch | {
        "resolved_device": device.type,
        "amp_enabled": amp_enabled,
        "dataset_len": len(dataset),
        "training_regime": "markov_rollout_rnn_seqmem_per_experiments",
        "enemy_visibility_mask": args.enemy_visibility_mask,
        "enemy_sight_range": args.enemy_sight_range,
        "action_conditioned_memory": args.action_conditioned_memory,
        "one_step_weight": args.one_step_weight,
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

    def save_checkpoint(epoch_to_save: int, checkpoint_path: Path) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "memory_module_state": memory_module.state_dict(),
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
                "priority_scores": priority_scores,
                "priority_seen": priority_seen,
            },
            checkpoint_path,
        )

    logger = LossLogger(out_dir, "loss_log")
    epoch_logger = LossLogger(out_dir, "epoch_loss")
    step_rows: list[dict[str, float | int]] = []
    epoch_rows: list[dict[str, float | int]] = []

    model.train()
    memory_module.train()

    print(
        "markov_rollout_rnn_visibility_seqmem_experiments "
        f"p={args.rollout_window} n={args.rollout_horizon} "
        f"memory_dim={args.rollout_memory_dim} "
        f"enemy_visibility_mask={args.enemy_visibility_mask} sight_range={args.enemy_sight_range} "
        f"temporal_loss={args.temporal_loss} td_lambda={args.td_lambda} "
        f"sigreg_weight={args.sigreg_weight} sigreg_weight_start={args.sigreg_weight_start} "
        f"sigreg_weight_end={args.sigreg_weight_end} sigreg_warmup_epochs={args.sigreg_warmup_epochs} "
        f"decoder_weight={args.decoder_weight} presence_weight={args.presence_weight} "
        f"action_conditioned_memory={args.action_conditioned_memory} "
        f"one_step_weight={args.one_step_weight} target_mode={args.target_mode} "
        f"window_mode={args.window_mode} samples_per_epoch={args.samples_per_epoch} "
        f"sample_prioritized={args.sample_prioritized} priority_alpha={args.priority_alpha} "
        f"priority_uniform_mix={args.priority_uniform_mix} priority_warmup_epochs={args.priority_warmup_epochs} "
        f"priority_score={args.priority_score}",
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

        use_priority = (
            args.sample_prioritized
            and epoch > args.priority_warmup_epochs
            and priority_seen.any().item()
        )

        if use_priority:
            sampler_weights = make_priority_weights(
                priority_scores,
                alpha=args.priority_alpha,
                uniform_mix=args.priority_uniform_mix,
                eps=args.priority_eps,
                max_multiplier=args.priority_max_multiplier,
            )
            sampler = WeightedRandomSampler(
                weights=sampler_weights.double(),
                num_samples=len(dataset),
                replacement=True,
            )
            sampler_mode = "priority"
        else:
            sampler = RandomSampler(dataset)
            sampler_mode = "uniform"

        loader = DataLoader(
            dataset,
            batch_size=int(arch["batch_size"]),
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
        )

        epoch_sums: dict[str, float] = {}
        epoch_batches = 0
        repeated_indices = 0
        sampled_indices_this_epoch: set[int] = set()

        for batch in loader:
            global_step += 1
            epoch_batches += 1

            sample_indices = batch["sample_index"].detach().cpu().long()
            for idx in sample_indices.tolist():
                if idx in sampled_indices_this_epoch:
                    repeated_indices += 1
                sampled_indices_this_epoch.add(idx)

            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            autocast_context = (
                torch.cuda.amp.autocast(enabled=amp_enabled)
                if device.type == "cuda"
                else nullcontext()
            )

            with autocast_context:
                losses, sample_scores = markov_rollout_rnn_losses(
                    model,
                    memory_module,
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
                    return_sample_scores=True,
                    priority_score=args.priority_score,
                )

            scaler.scale(losses["total_loss"]).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(memory_module.parameters()),
                    args.grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()

            sample_scores_cpu = sample_scores.detach().float().cpu()
            old_scores = priority_scores[sample_indices]
            was_seen = priority_seen[sample_indices]
            updated = torch.where(
                was_seen,
                args.priority_ema_beta * old_scores + (1.0 - args.priority_ema_beta) * sample_scores_cpu,
                sample_scores_cpu,
            )
            priority_scores[sample_indices] = updated
            priority_seen[sample_indices] = True

            row: dict[str, float | int | str] = {
                "epoch": epoch,
                "step": global_step,
                "active_sigreg_weight": active_sigreg_weight,
                "sampler_mode": sampler_mode,
            }
            for key, value in losses.items():
                row[key] = float(value.detach().cpu())
            row["priority_batch_score_mean"] = float(sample_scores_cpu.mean())
            row["priority_batch_score_max"] = float(sample_scores_cpu.max())

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
                    "train/sampler_mode_is_priority": 1 if sampler_mode == "priority" else 0,
                    "train/priority_batch_score_mean": row.get("priority_batch_score_mean"),
                    "train/priority_batch_score_max": row.get("priority_batch_score_max"),
                }
                for key, value in row.items():
                    if key.startswith("pred_loss_h"):
                        log_dict[f"train/{key}"] = value
                wandb_run.log(log_dict, step=global_step)

            for key, value in row.items():
                if key in {"epoch", "step", "sampler_mode"}:
                    continue
                epoch_sums[key] = epoch_sums.get(key, 0.0) + float(value)

            if global_step == 1 or global_step % args.log_every == 0:
                print(
                    "epoch={epoch} step={step} "
                    "total_loss={total_loss:.6f} pred_loss={pred_loss:.6f} "
                    "pred_uniform={pred_loss_uniform:.6f} sigreg_loss={sigreg_loss:.6f} "
                    "decoded_loss={decoded_loss:.6f} presence_loss={presence_loss:.6f} "
                    "memory_norm={memory_norm_mean:.6f}".format(**row),
                    flush=True,
                )

        if epoch_batches == 0:
            raise RuntimeError(f"Epoch {epoch} finished with 0 batches; refusing to save checkpoint.")

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
            "priority_unique_indices": len(sampled_indices_this_epoch),
            "priority_repeated_indices": repeated_indices,
        }
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
                "epoch/sampler_mode_is_priority": 1 if sampler_mode == "priority" else 0,
                "epoch/priority_score_mean": priority_mean,
                "epoch/priority_score_max": priority_max,
                "epoch/priority_seen_frac": priority_seen_frac,
                "epoch/priority_unique_indices": len(sampled_indices_this_epoch),
                "epoch/priority_repeated_indices": repeated_indices,
            }
            for key, value in epoch_row.items():
                if key.startswith("pred_loss_h"):
                    log_dict[f"epoch/{key}"] = value
            wandb_run.log(log_dict, step=global_step)

        print(
            "epoch_summary epoch={epoch} step={step} mode={sampler_mode} "
            "total_loss={total_loss:.6f} pred_loss={pred_loss:.6f} "
            "pred_uniform={pred_loss_uniform:.6f} sigreg_loss={sigreg_loss:.6f} "
            "decoded_loss={decoded_loss:.6f} presence_loss={presence_loss:.6f} "
            "memory_norm={memory_norm_mean:.6f} "
            "priority_mean={priority_score_mean:.6f} priority_max={priority_score_max:.6f} "
            "priority_seen_frac={priority_seen_frac:.3f} unique_indices={priority_unique_indices} "
            "repeated_indices={priority_repeated_indices}".format(**epoch_row),
            flush=True,
        )

        epoch_checkpoint_path = out_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        save_checkpoint(epoch, epoch_checkpoint_path)
        save_checkpoint(epoch, out_dir / "checkpoint.pt")
        print(f"saved_checkpoint {epoch_checkpoint_path} and {out_dir / 'checkpoint.pt'}", flush=True)

        write_svg_line_plot(epoch_rows, "epoch", "total_loss", "Average Total Loss Per Epoch", out_dir / "loss_by_epoch.svg")
        write_svg_line_plot(epoch_rows, "epoch", "pred_loss", "Average Markov Rollout RNN Prediction Loss Per Epoch", out_dir / "pred_loss_by_epoch.svg")
        write_svg_line_plot(step_rows, "step", "pred_loss", "Markov Rollout RNN Prediction Loss Per Training Step", out_dir / "pred_loss_by_step.svg")

    if wandb_run is not None:
        wandb_run.save(str(out_dir / "config.json"))
        wandb_run.save(str(out_dir / "checkpoint.pt"))
        wandb_run.finish()


if __name__ == "__main__":
    main()
