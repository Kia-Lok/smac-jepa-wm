from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BUILTIN_UNITS = {
    "BANELING",
    "COLOSSUS",
    "MARAUDER",
    "MARINE",
    "MEDIVAC",
    "SPINE_CRAWLER",
    "STALKER",
    "ZEALOT",
    "ZERGLING",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a generated-config SMAC-JEPA probe")
    parser.add_argument("--config-dir", default="configs/generated")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-configs", type=int, default=50)
    parser.add_argument("--eval-configs", type=int, default=10)
    parser.add_argument("--max-agents", type=int, default=50)
    parser.add_argument("--max-enemies", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples-per-epoch", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-len", type=int, default=4)
    parser.add_argument("--rollout-horizons", default="1,2,4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env-key", default="smaclite:smaclite/custom-v0")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.out_dir)
    data_dir = root / "data"
    train_config_dir = root / "configs_train"
    eval_config_dir = root / "configs_eval"
    for directory in (data_dir, train_config_dir, eval_config_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selected = select_configs(args)
    write_selected(root, selected, train_config_dir, eval_config_dir)
    manifest = collect_missing(args, root, data_dir, selected)
    manifest_path = root / "split.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    run([sys.executable, "-m", "smac_jepa.audit_dataset", "--manifest", str(manifest_path), "--out", str(root / "audit.json")])
    if args.skip_train:
        return
    run(
        [
            sys.executable,
            "-m",
            "smac_jepa.train",
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(root / "run"),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--context-len",
            str(args.context_len),
            "--window-mode",
            "random",
            "--window-len",
            str(args.context_len),
            "--samples-per-epoch",
            str(args.samples_per_epoch),
            "--device",
            args.device,
            "--no-amp",
        ]
    )
    checkpoint = root / "run" / "checkpoint.pt"
    for split in ("train", "eval"):
        run(
            [
                sys.executable,
                "-m",
                "smac_jepa.evaluate",
                "--manifest",
                str(manifest_path),
                "--split",
                split,
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(root / f"{split}_eval.json"),
                "--decode-sample-out",
                str(root / f"{split}_decoded.json"),
                "--per-config-out",
                str(root / f"{split}_per_config.json"),
                "--rollout-horizons",
                args.rollout_horizons,
                "--device",
                args.device,
            ]
        )


def select_configs(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    rows = []
    for path in sorted(Path(args.config_dir).glob("*.json")):
        config = json.loads(path.read_text())
        n_agents = int(config.get("num_allied_units", 0))
        n_enemies = int(config.get("num_enemy_units", 0))
        if n_agents > args.max_agents or n_enemies > args.max_enemies:
            continue
        unit_names = {
            unit
            for group in config.get("groups", [])
            for unit in group.get("units", {}).keys()
        }
        if not unit_names.issubset(BUILTIN_UNITS):
            continue
        family = path.name.split("_var_")[0]
        if family == path.name:
            family = path.name.split("_")[0]
        rows.append(
            {
                "path": str(path),
                "stem": path.stem,
                "family": family,
                "terrain": str(config.get("terrain_preset", "unknown")),
                "n_agents": n_agents,
                "n_enemies": n_enemies,
            }
        )
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["family"], row["terrain"])].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: (row["n_agents"] + row["n_enemies"], row["stem"]))
    selected = []
    while len(selected) < args.train_configs + args.eval_configs:
        progressed = False
        for key in sorted(buckets):
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                progressed = True
                if len(selected) >= args.train_configs + args.eval_configs:
                    break
        if not progressed:
            break
    if len(selected) < args.train_configs + args.eval_configs:
        raise SystemExit("Not enough valid generated configs after filtering")
    train, eval_rows = [], []
    for idx, row in enumerate(selected):
        (eval_rows if idx % 6 == 5 else train).append(row)
    return {"train": train[: args.train_configs], "eval": eval_rows[: args.eval_configs]}


def write_selected(
    root: Path,
    selected: dict[str, list[dict[str, Any]]],
    train_config_dir: Path,
    eval_config_dir: Path,
) -> None:
    for directory in (train_config_dir, eval_config_dir):
        for old in directory.glob("*.json"):
            old.unlink()
    for split, directory in (("train", train_config_dir), ("eval", eval_config_dir)):
        for row in selected[split]:
            shutil.copy2(row["path"], directory / Path(row["path"]).name)
    summary = {
        **selected,
        "train_families": dict(Counter(row["family"] for row in selected["train"])),
        "eval_families": dict(Counter(row["family"] for row in selected["eval"])),
        "train_terrains": dict(Counter(row["terrain"] for row in selected["train"])),
        "eval_terrains": dict(Counter(row["terrain"] for row in selected["eval"])),
    }
    (root / "selected_configs.json").write_text(json.dumps(summary, indent=2) + "\n")


def collect_missing(
    args: argparse.Namespace,
    root: Path,
    data_dir: Path,
    selected: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    collector = Path("simulator/collect_smaclite_data.py")
    manifest = {
        "preset": "generalization_probe",
        "seed": args.seed,
        "split_unit": "configuration",
        "datasets": {"train": [], "eval": []},
        "failed": [],
    }
    counter = 0
    for split in ("train", "eval"):
        config_dir = root / ("configs_train" if split == "train" else "configs_eval")
        for row in selected[split]:
            counter += 1
            out = data_dir / f"{row['stem']}.npz"
            if not out.exists():
                try:
                    run(
                        [
                            sys.executable,
                            str(collector),
                            "--env-key",
                            args.env_key,
                            "--scenario-name",
                            row["stem"],
                            "--map-file",
                            str(config_dir / f"{row['stem']}.json"),
                            "--episodes",
                            str(args.episodes),
                            "--max-steps",
                            str(args.max_steps),
                            "--out",
                            str(out),
                            "--seed",
                            str(args.seed + counter),
                        ]
                    )
                except subprocess.CalledProcessError as exc:
                    manifest["failed"].append([split, row["stem"], exc.returncode])
                    continue
            manifest["datasets"][split].append(str(Path("data") / out.name))
    if len(manifest["datasets"]["train"]) != args.train_configs or len(manifest["datasets"]["eval"]) != args.eval_configs:
        raise SystemExit(f"Collection incomplete: {manifest}")
    return manifest


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
