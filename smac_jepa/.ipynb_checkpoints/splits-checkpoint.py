from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


ORIGINAL_SCENARIOS = [
    "10m_vs_11m",
    "27m_vs_30m",
    "2c_vs_64zg",
    "2s3z",
    "2s_vs_1sc",
    "3s5z",
    "3s5z_vs_3s6z",
    "3s_vs_5z",
    "bane_vs_bane",
    "corridor",
    "mmm",
    "mmm2",
]

DEFAULT_EVAL_SCENARIOS = ["corridor", "mmm2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write SMACLite train/eval split manifests")
    parser.add_argument("--preset", default="original", choices=["original", "generated"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-dir", default="data/original")
    parser.add_argument("--eval-scenarios", nargs="+", default=DEFAULT_EVAL_SCENARIOS)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out = Path(args.out)
    out_parent = out.parent if out.parent != Path("") else Path(".")

    def relative_path(path: Path) -> str:
        if path.is_absolute():
            return str(path)
        return os.path.relpath(path, out_parent)

    if args.preset == "original":
        scenarios = ORIGINAL_SCENARIOS
        eval_scenarios = [name for name in args.eval_scenarios if name in scenarios]
        if len(eval_scenarios) != len(args.eval_scenarios):
            missing = sorted(set(args.eval_scenarios) - set(eval_scenarios))
            raise SystemExit(f"Unknown eval scenarios: {missing}")
        train_scenarios = [name for name in scenarios if name not in set(eval_scenarios)]
        train_paths = [relative_path(data_dir / f"{name}.npz") for name in train_scenarios]
        eval_paths = [relative_path(data_dir / f"{name}.npz") for name in eval_scenarios]
    else:
        files = sorted(data_dir.glob("*.npz"))
        if len(files) < 2:
            raise SystemExit(f"Generated split needs at least 2 NPZ files in {data_dir}")
        rng = random.Random(args.seed)
        shuffled = files[:]
        rng.shuffle(shuffled)
        eval_count = max(1, round(len(files) * args.eval_fraction))
        eval_files = sorted(shuffled[:eval_count])
        train_files = sorted(shuffled[eval_count:])
        train_scenarios = [path.stem for path in train_files]
        eval_scenarios = [path.stem for path in eval_files]
        train_paths = [relative_path(path) for path in train_files]
        eval_paths = [relative_path(path) for path in eval_files]

    manifest = {
        "preset": args.preset,
        "seed": args.seed,
        "split_unit": "configuration",
        "train_scenarios": train_scenarios,
        "eval_scenarios": eval_scenarios,
        "datasets": {
            "train": train_paths,
            "eval": eval_paths,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
