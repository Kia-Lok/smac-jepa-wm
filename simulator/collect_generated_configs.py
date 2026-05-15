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
    manifest_out = Path(args.manifest_out)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    configs = sorted(config_dir.glob("*.json"))

    if not configs:
        raise SystemExit(f"No JSON configs found in {config_dir}")

    collector = Path(__file__).with_name("collect_smaclite_data.py")

    successful: list[Path] = []
    failed: list[Path] = []

    print(f"Found {len(configs)} configs in {config_dir}")
    print(f"Output directory: {out_dir}")
    print()

    for idx, config_path in enumerate(configs):
        scenario = config_path.stem
        out_path = out_dir / f"{scenario}.npz"
        seed = args.seed + idx

        cmd = [
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
        ]

        print("=" * 80)
        print(f"[{idx + 1}/{len(configs)}] Collecting: {config_path.name}")
        print(f"Scenario: {scenario}")
        print(f"Seed: {seed}")
        print("=" * 80)

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print()
            print("=" * 80)
            print(f"[SKIP] Failed config: {config_path}")
            print(f"[SKIP] Exit code: {e.returncode}")
            print("[SKIP] Continuing with next config.")
            print("=" * 80)
            print()

            failed.append(config_path)

            # Remove possibly corrupted or partially written file.
            if out_path.exists():
                try:
                    out_path.unlink()
                    print(f"[CLEANUP] Removed partial output: {out_path}")
                except OSError as cleanup_error:
                    print(f"[WARNING] Could not remove partial output {out_path}: {cleanup_error}")

            continue

        if out_path.exists():
            successful.append(out_path)
            print(f"[OK] Saved: {out_path}")
        else:
            failed.append(config_path)
            print(f"[SKIP] Collector exited successfully but output file was missing: {out_path}")

        print()

    print("=" * 80)
    print("Collection summary")
    print("=" * 80)
    print(f"Total configs: {len(configs)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed/skipped: {len(failed)}")
    print()

    if failed:
        failed_path = out_dir / "failed_configs.txt"
        failed_path.write_text("\n".join(str(path) for path in failed), encoding="utf-8")
        print(f"Failed config list written to: {failed_path}")

    if not successful:
        raise SystemExit(
            "No configs were collected successfully, so no manifest will be generated."
        )

    print()
    print("Generating manifest from successful .npz files only...")
    print()

    # smac_jepa.splits scans the output directory for existing .npz files.
    # Since failed partial files are removed above, the manifest should only include valid outputs.
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
            str(manifest_out),
            "--eval-fraction",
            str(args.eval_fraction),
            "--seed",
            str(args.seed),
        ],
        check=True,
    )

    print()
    print("=" * 80)
    print("Done")
    print("=" * 80)
    print(f"Manifest written to: {manifest_out}")
    print(f"Successful files: {len(successful)}")
    print(f"Skipped configs: {len(failed)}")


if __name__ == "__main__":
    main()