from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class LossLogger:
    def __init__(self, out_dir: str | Path, filename: str = "loss_log"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / f"{filename}.csv"
        self.jsonl_path = self.out_dir / f"{filename}.jsonl"
        self.fields = [
            "epoch",
            "step",
            "total_loss",
            "pred_loss",
            "sigreg_loss",
            "decoded_loss",
            "decoded_mae",
            "decoded_mse",
            "decoded_r2",
            "tol_acc_0.01",
            "tol_acc_0.05",
            "tol_acc_0.10",
        ]
        with self.csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()
        self.jsonl_path.write_text("")

    def log(self, row: dict[str, Any]) -> None:
        clean = {field: row.get(field, "") for field in self.fields}
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(clean)
        with self.jsonl_path.open("a") as f:
            f.write(json.dumps(clean) + "\n")
