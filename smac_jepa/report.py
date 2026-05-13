from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from smac_jepa.jepa import SMACJEPA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a static SMAC-JEPA run report")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def count_params(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def line_plot(rows: list[dict[str, str]], x_key: str, y_key: str, title: str) -> str:
    if not rows:
        return "<p>No data.</p>"
    values = [(as_float(row, x_key), as_float(row, y_key)) for row in rows]
    xs = [x for x, _ in values]
    ys = [y for _, y in values]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x = min_x + 1.0
    if min_y == max_y:
        max_y = min_y + 1.0

    width, height = 760, 260
    left, right, top, bottom = 62, 24, 28, 46
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        return left + ((x - min_x) / (max_x - min_x)) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - ((y - min_y) / (max_y - min_y)) * plot_h

    points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in values)
    circles = "\n".join(
        f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.4"><title>{x_key}={x:g}, {y_key}={y:.6g}</title></circle>'
        for x, y in values
    )
    return f"""
<figure class="plot">
  <figcaption>{html.escape(title)}</figcaption>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
    <line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" />
    <line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" />
    <text class="tick" x="{left}" y="{height - 14}">{min_x:g}</text>
    <text class="tick end" x="{left + plot_w}" y="{height - 14}">{max_x:g}</text>
    <text class="tick" x="8" y="{top + plot_h}">{min_y:.3g}</text>
    <text class="tick" x="8" y="{top + 4}">{max_y:.3g}</text>
    <polyline class="series" points="{points}" />
    {circles}
  </svg>
</figure>
"""


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row[col]))}</td>" for col in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def embed_svg(path: Path, fallback: str) -> str:
    if path.exists():
        return path.read_text()
    return fallback


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    config = checkpoint["config"]
    metadata = checkpoint["metadata"]
    model = SMACJEPA(
        state_dim=metadata["state_dim"],
        n_agents=metadata["n_agents"],
        n_actions=metadata["n_actions"],
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        action_dim=int(config["action_dim"]),
        num_heads=int(config["num_heads"]),
        mode=metadata.get("mode", "flat"),
        max_agents=metadata.get("max_agents", metadata["n_agents"]),
        max_enemies=metadata.get("max_enemies", 0),
        max_actions=metadata.get("max_actions", metadata["n_actions"]),
        token_dim=metadata.get("token_dim", metadata["state_dim"]),
        decoder_weight=float(config.get("decoder_weight", 1.0)),
    )
    model.load_state_dict(checkpoint["model_state"])

    epoch_rows = read_csv(run_dir / "epoch_loss.csv")
    step_rows = read_csv(run_dir / "loss_log.csv")
    metrics = json.loads((run_dir / "eval_metrics.json").read_text())
    with np.load(args.data, allow_pickle=False) as data:
        dataset_summary = {
            "dataset": args.data,
            "episodes": data["states"].shape[0],
            "max_steps": data["actions"].shape[1],
            "valid_steps": int(data["valid"].sum()),
            "state_dim": int(data["state_dim"]),
            "n_agents": int(data["n_agents"]),
            "n_actions": int(data["n_actions"]),
        }
        if "n_enemies" in data:
            dataset_summary["n_enemies"] = int(data["n_enemies"])

    param_rows = [
        {"component": "State encoder", "params": count_params(model.encoder)},
        {"component": "Predictor", "params": count_params(model.predictor)},
        {"component": "Action encoder", "params": count_params(model.predictor.action_encoder)},
        {"component": "Attention block", "params": count_params(model.predictor.block)},
        {"component": "Total trainable", "params": count_params(model)},
    ]
    epoch_table_rows = [
        {
            "epoch": row["epoch"],
            "avg_total_loss": f"{as_float(row, 'total_loss'):.6f}",
            "avg_pred_loss": f"{as_float(row, 'pred_loss'):.6f}",
            "avg_sigreg_loss": f"{as_float(row, 'sigreg_loss'):.6f}",
        }
        for row in epoch_rows
    ]

    first_step = step_rows[0]
    last_step = step_rows[-1]
    loss_summary_rows = [
        {
            "point": "First batch",
            "step": first_step["step"],
            "total_loss": f"{as_float(first_step, 'total_loss'):.6f}",
            "pred_loss": f"{as_float(first_step, 'pred_loss'):.6f}",
        },
        {
            "point": "Last batch",
            "step": last_step["step"],
            "total_loss": f"{as_float(last_step, 'total_loss'):.6f}",
            "pred_loss": f"{as_float(last_step, 'pred_loss'):.6f}",
        },
    ]

    dataset_rows = [{"field": key, "value": value} for key, value in dataset_summary.items()]
    config_rows = [
        {"field": "context_len", "value": config["context_len"]},
        {"field": "latent_dim", "value": config["latent_dim"]},
        {"field": "hidden_dim", "value": config["hidden_dim"]},
        {"field": "action_dim", "value": config["action_dim"]},
        {"field": "num_heads", "value": config["num_heads"]},
        {"field": "mode", "value": metadata.get("mode", "flat")},
    ]
    metric_rows = [{"metric": key, "value": value} for key, value in metrics.items()]

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SMAC-JEPA 2s3z CPU Report</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1d252c;
      background: #f6f7f9;
    }}
    main {{
      max-width: 1040px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 34px; }}
    h2 {{ margin-top: 32px; font-size: 24px; }}
    p {{ line-height: 1.55; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 22px 0;
    }}
    .metric {{
      background: #ffffff;
      border: 1px solid #dce1e7;
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .metric strong {{ display: block; font-size: 22px; margin-top: 4px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid #dce1e7;
      border-radius: 8px;
      overflow: hidden;
      margin: 12px 0 22px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid #e7ebef;
      font-size: 14px;
    }}
    th {{ background: #eef2f5; }}
    tr:last-child td {{ border-bottom: 0; }}
    .plot {{
      background: #ffffff;
      border: 1px solid #dce1e7;
      border-radius: 8px;
      padding: 14px;
      margin: 14px 0 20px;
    }}
    figcaption {{ font-weight: 700; margin-bottom: 8px; }}
    svg {{ width: 100%; height: auto; display: block; }}
    .axis {{ stroke: #7b8794; stroke-width: 1; }}
    .series {{ fill: none; stroke: #2563eb; stroke-width: 3; }}
    circle {{ fill: #2563eb; }}
    .tick {{ fill: #5f6b76; font-size: 12px; }}
    .end {{ text-anchor: end; }}
    code {{ background: #e8edf2; padding: 2px 5px; border-radius: 4px; }}
    ol {{ background: #ffffff; border: 1px solid #dce1e7; border-radius: 8px; padding: 18px 24px 18px 42px; }}
    li {{ margin: 8px 0; }}
  </style>
</head>
<body>
  <main>
    <h1>SMAC-JEPA 2s3z CPU Report</h1>
    <p>Run directory: <code>{html.escape(str(run_dir))}</code></p>
    <section class="summary">
      <div class="metric">Trainable params<strong>{count_params(model):,}</strong></div>
      <div class="metric">Valid windows<strong>{dataset_summary["valid_steps"]:,}</strong></div>
      <div class="metric">Eval MSE<strong>{metrics["next_state_embedding_mse"]:.6g}</strong></div>
      <div class="metric">Context length<strong>{config["context_len"]}</strong></div>
    </section>

    <h2>Model</h2>
    <p>The encoder maps each SMACLite global-state observation to a latent. The predictor conditions on the sequence of joint actions taken from the previous states in the context window, combines those action-history embeddings with observation latents, and predicts the next latent sequence.</p>
    {table(param_rows, ["component", "params"])}
    {table(config_rows, ["field", "value"])}

    <h2>Dataset</h2>
    {table(dataset_rows, ["field", "value"])}

    <h2>Training Loss</h2>
    {table(loss_summary_rows, ["point", "step", "total_loss", "pred_loss"])}
    {table(epoch_table_rows, ["epoch", "avg_total_loss", "avg_pred_loss", "avg_sigreg_loss"])}
    <figure class="plot"><figcaption>Average Total Loss Per Epoch</figcaption>{embed_svg(run_dir / "loss_by_epoch.svg", line_plot(epoch_rows, "epoch", "total_loss", "Average Total Loss Per Epoch"))}</figure>
    <figure class="plot"><figcaption>Average Prediction Loss Per Epoch</figcaption>{embed_svg(run_dir / "pred_loss_by_epoch.svg", line_plot(epoch_rows, "epoch", "pred_loss", "Average Prediction Loss Per Epoch"))}</figure>
    <figure class="plot"><figcaption>Prediction Loss Per Training Step</figcaption>{embed_svg(run_dir / "pred_loss_by_step.svg", line_plot(step_rows, "step", "pred_loss", "Prediction Loss Per Training Step"))}</figure>

    <h2>Evaluation</h2>
    {table(metric_rows, ["metric", "value"])}
    <h3>Evaluation Process</h3>
    <ol>
      <li>Load <code>checkpoint.pt</code>, including model weights, config, and dataset metadata.</li>
      <li>Rebuild <code>SMACJEPA</code> with the saved dimensions.</li>
      <li>Load the same NPZ data and create valid state-action-next-state windows.</li>
      <li>Encode the observation sequence and target next-state sequence into latent vectors.</li>
      <li>Predict the next latent sequence from observation latents plus action-history conditioning.</li>
      <li>Compute masked mean squared error between predicted and target next latent.</li>
    </ol>
  </main>
</body>
</html>
"""
    out_path = out_dir / "index.html"
    out_path.write_text(html_text)
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
