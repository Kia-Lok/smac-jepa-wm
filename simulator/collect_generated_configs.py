from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect SMACLite datasets for a directory of generated JSON configs"
    )
    parser.add_argument("--config-dir", required=True, help="Directory containing generated JSON maps")
    parser.add_argument("--out-dir", default="data/generated")
    parser.add_argument("--manifest-out", default="splits/generated_seed1.json")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--env-key", default="smaclite:smaclite/custom-v0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = sorted(config_dir.glob("*.json"))
    if not configs:
        raise SystemExit(f"No JSON configs found in {config_dir}")

    collector = Path(__file__).with_name("collect_smaclite_data.py")
    for idx, config_path in enumerate(configs):
        scenario = config_path.stem
        out_path = out_dir / f"{scenario}.npz"
        seed = args.seed + idx
        subprocess.run(
            [
                sys.executable,
                str(collector),
                "--env-key",
                args.env_key,
                "--scenario-name",
                scenario,
                "--map-file",
                str(config_path),
                "--episodes",
                str(args.episodes),
                "--max-steps",
                str(args.max_steps),
                "--out",
                str(out_path),
                "--seed",
                str(seed),
            ],
            check=True,
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "smac_jepa.splits",
            "--preset",
            "generated",
            "--data-dir",
            str(out_dir),
            "--out",
            args.manifest_out,
            "--eval-fraction",
            str(args.eval_fraction),
            "--seed",
            str(args.seed),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
