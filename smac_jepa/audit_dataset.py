from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from smac_jepa.data import load_manifest_all
from smac_jepa.data.dataset import load_npz_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SMAC-JEPA dataset compatibility")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_manifest_all(args.manifest)
    report = audit_paths(paths)
    output = json.dumps(report, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output)
    else:
        print(output, end="")


def audit_paths(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("No dataset paths found")
    rows: list[dict[str, Any]] = []
    terrain_counts: dict[str, int] = {}
    missing_static = 0
    total_step_errors = 0
    for path in paths:
        metadata = load_npz_metadata(path)
        with np.load(path, allow_pickle=False) as data:
            valid = data["valid"].astype(bool) if "valid" in data else None
            scenario = str(np.asarray(data["scenario"]).item()) if "scenario" in data else path.stem
            terrain = str(np.asarray(data["terrain_preset"]).item()) if "terrain_preset" in data else "unknown"
            terrain_counts[terrain] = terrain_counts.get(terrain, 0) + 1
            step_errors = int(np.asarray(data["step_errors"]).item()) if "step_errors" in data else 0
            total_step_errors += step_errors
            if metadata.static_dim <= 0 or "static_condition" not in data:
                missing_static += 1
            rows.append(
                {
                    "path": str(path),
                    "scenario": scenario,
                    "n_agents": metadata.n_agents,
                    "n_enemies": metadata.n_enemies,
                    "n_actions": metadata.n_actions,
                    "ally_feat": metadata.ally_state_feat_size,
                    "enemy_feat": metadata.enemy_state_feat_size,
                    "static_dim": metadata.static_dim,
                    "entity_static_feat_size": metadata.entity_static_feat_size,
                    "terrain": terrain,
                    "valid_steps": int(valid.sum()) if valid is not None else None,
                    "step_errors": step_errors,
                }
            )
    return {
        "num_datasets": len(rows),
        "caps": {
            "max_agents": max(row["n_agents"] for row in rows),
            "max_enemies": max(row["n_enemies"] for row in rows),
            "max_actions": max(row["n_actions"] for row in rows),
            "max_ally_feat": max(row["ally_feat"] for row in rows),
            "max_enemy_feat": max(row["enemy_feat"] for row in rows),
            "max_static_dim": max(row["static_dim"] for row in rows),
            "max_entity_static_feat_size": max(row["entity_static_feat_size"] for row in rows),
        },
        "terrain_counts": terrain_counts,
        "missing_static_datasets": missing_static,
        "total_step_errors": total_step_errors,
        "datasets": rows,
    }


if __name__ == "__main__":
    main()
