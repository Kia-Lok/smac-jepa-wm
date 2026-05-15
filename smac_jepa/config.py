from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

#Config management but looks hardcoded. Likely need to change to make it variable.
@dataclass
class TrainConfig:
    out_dir: str
    manifest: str | None = None
    data_dir: str | None = None
    eval_fraction: float = 0.2
    split: str = "train"
    model_size: str = "default"
    epochs: int = 5
    batch_size: int | None = None
    lr: float | None = None
    latent_dim: int | None = None
    hidden_dim: int | None = None
    action_dim: int | None = None
    context_len: int = 4
    num_heads: int | None = None
    encoder_layers: int | None = None
    action_layers: int | None = None
    predictor_layers: int | None = None
    max_context_len: int = 32
    sigreg_weight: float = 0.01
    decoder_weight: float = 1.0
    grad_clip: float = 1.0
    device: str = "auto"
    amp: bool = True
    resume: str | None = None
    seed: int = 1
    num_workers: int = 0
    log_every: int = 10

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
