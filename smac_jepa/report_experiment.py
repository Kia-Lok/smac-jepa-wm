from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SMAC-JEPA experiment report")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns)
            + "</tr>"
        )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def metric_rows(label: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row = {"split": label}
    for key in [
        "num_windows",
        "next_state_embedding_mse",
        "decoded_mae",
        "decoded_mse",
        "decoded_r2",
        "tol_acc_0.01",
        "tol_acc_0.05",
        "tol_acc_0.10",
    ]:
        row[key] = fmt(metrics.get(key, ""))
    return row


def embed(path: Path) -> str:
    return path.read_text() if path.exists() else "<p>Plot unavailable.</p>"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(Path(args.manifest))
    train_metrics = read_json(run_dir / "train_metrics.json")
    eval_metrics = read_json(run_dir / "eval_metrics.json")
    corridor_metrics = read_json(run_dir / "eval_corridor_metrics.json")
    mmm2_metrics = read_json(run_dir / "eval_mmm2_metrics.json")
    config = read_json(run_dir / "config.json")
    epoch_rows = read_csv(run_dir / "epoch_loss.csv")

    scenario_rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            scenario_rows.append(
                {
                    "scenario": path.stem,
                    "split": "eval" if path.stem in manifest["eval_scenarios"] else "train",
                    "episodes": data["states"].shape[0],
                    "valid_steps": int(data["valid"].sum()),
                    "agents": int(data["n_agents"]),
                    "enemies": int(data["n_enemies"]),
                    "actions": int(data["n_actions"]),
                    "step_errors": int(data["step_errors"]) if "step_errors" in data else 0,
                }
            )

    loss_rows = [
        {
            "epoch": row["epoch"],
            "total_loss": f"{float(row['total_loss']):.6f}",
            "pred_loss": f"{float(row['pred_loss']):.6f}",
            "decoded_loss": f"{float(row['decoded_loss']):.6f}",
            "decoded_mae": f"{float(row['decoded_mae'] or 0):.6f}",
            "tol_acc_0.10": f"{float(row['tol_acc_0.10'] or 0):.6f}",
        }
        for row in epoch_rows
    ]
    metric_table_rows = [
        metric_rows("train aggregate", train_metrics),
        metric_rows("held-out aggregate", eval_metrics),
        metric_rows("held-out corridor", corridor_metrics),
        metric_rows("held-out mmm2", mmm2_metrics),
    ]
    config_rows = [
        {"field": key, "value": value}
        for key, value in config.items()
        if key
        in {
            "epochs",
            "model_size",
            "batch_size",
            "context_len",
            "latent_dim",
            "hidden_dim",
            "action_dim",
            "num_heads",
            "encoder_layers",
            "action_layers",
            "predictor_layers",
            "device",
            "resolved_device",
            "amp_enabled",
            "sigreg_weight",
            "decoder_weight",
            "seed",
        }
    ]

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SMAC-JEPA Generalization Report</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17212b; background: #f7f8fa; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 56px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; }}
    h2 {{ margin: 32px 0 12px; font-size: 22px; }}
    p {{ line-height: 1.55; }}
    code {{ background: #e9edf2; padding: 2px 5px; border-radius: 4px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 12px 0 20px; background: white; border: 1px solid #d9e0e7; }}
    th, td {{ padding: 9px 10px; text-align: left; border-bottom: 1px solid #e6ebf0; font-size: 13px; }}
    th {{ background: #eef3f7; font-weight: 700; }}
    tr:last-child td {{ border-bottom: 0; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 20px 0; }}
    .metric {{ background: white; border: 1px solid #d9e0e7; border-radius: 8px; padding: 14px 16px; }}
    .metric span {{ color: #5d6b78; font-size: 13px; }}
    .metric strong {{ display: block; font-size: 24px; margin-top: 5px; }}
    .plot {{ background: white; border: 1px solid #d9e0e7; border-radius: 8px; padding: 14px; margin: 14px 0 18px; }}
    svg {{ width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
<main>
  <h1>SMAC-JEPA Generalization Report</h1>
  <p>Entity-token JEPA trained on 80% of original SMACLite scenarios and evaluated on the held-out 20% split. Run directory: <code>{html.escape(str(run_dir))}</code>.</p>
  <section class="summary">
    <div class="metric"><span>Train scenarios</span><strong>{len(manifest["train_scenarios"])}</strong></div>
    <div class="metric"><span>Held-out scenarios</span><strong>{", ".join(manifest["eval_scenarios"])}</strong></div>
    <div class="metric"><span>Held-out embedding MSE</span><strong>{fmt(eval_metrics["next_state_embedding_mse"])}</strong></div>
    <div class="metric"><span>Held-out tolerance accuracy @0.10</span><strong>{fmt(eval_metrics["tol_acc_0.10"])}</strong></div>
  </section>
  <h2>Experiment Setup</h2>
  <p>The predictor conditions on the full sequence of joint actions in each context window. Entity and action tensors are padded to checkpoint caps and masked so excess agents, enemies, and actions do not participate in attention.</p>
  {table(config_rows, ["field", "value"])}
  <h2>Scenario Coverage</h2>
  {table(scenario_rows, ["scenario", "split", "episodes", "valid_steps", "agents", "enemies", "actions", "step_errors"])}
  <h2>Loss by Epoch</h2>
  {table(loss_rows, ["epoch", "total_loss", "pred_loss", "decoded_loss", "decoded_mae", "tol_acc_0.10"])}
  <div class="plot">{embed(run_dir / "loss_by_epoch.svg")}</div>
  <div class="plot">{embed(run_dir / "pred_loss_by_epoch.svg")}</div>
  <div class="plot">{embed(run_dir / "pred_loss_by_step.svg")}</div>
  <h2>Evaluation Results</h2>
  {table(metric_table_rows, ["split", "num_windows", "next_state_embedding_mse", "decoded_mae", "decoded_mse", "decoded_r2", "tol_acc_0.01", "tol_acc_0.05", "tol_acc_0.10"])}
</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text)
    print(f"Wrote {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
