#!/usr/bin/env python3
"""
Generate a category-prefix JSON file for train-lambda.py.

Usage examples:

  python make_category_prefixes.py \
    --out category_prefixes.json

  python make_category_prefixes.py \
    --out configs/category_prefixes_v1.json \
    --overwrite

Edit CATEGORY_PREFIXES below, then run this script.

The output JSON format is:

{
  "category_name": ["prefix_1", "prefix_2"],
  "another_category": ["prefix_3"]
}

A .npz file belongs to a category if its filename starts with one of the
prefixes for that category.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------
# EDIT THIS DICTIONARY MANUALLY
# ---------------------------------------------------------------------
# Replace these example categories/prefixes with your actual config prefixes.
#
# Example:
# If your generated files look like:
#   marine_small_001.npz
#   marine_small_002.npz
#   stalker_large_001.npz
#
# Then you might write:
#   "marine_small": ["marine_small_"],
#   "stalker_large": ["stalker_large_"],
#
CATEGORY_PREFIXES: dict[str, list[str]] = {
    "small_army": [
        "small_",
        "tiny_",
    ],
    "large_army": [
        "large_",
        "big_",
    ],
    "mixed_units": [
        "mixed_",
        "generated_mixed_",
    ],
    "hard_maps": [
        "hard_",
        "choke_",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a category-prefix JSON file for category-prioritized SMAC-JEPA training."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON path, e.g. category_prefixes.json or configs/category_prefixes_v1.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level. Default: 2",
    )
    return parser.parse_args()


def validate_category_prefixes(category_prefixes: dict[str, list[str]]) -> None:
    if not isinstance(category_prefixes, dict) or not category_prefixes:
        raise ValueError("CATEGORY_PREFIXES must be a non-empty dictionary.")

    seen_prefixes: dict[str, str] = {}

    for category, prefixes in category_prefixes.items():
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"Invalid category name: {category!r}")

        if not isinstance(prefixes, list) or not prefixes:
            raise ValueError(
                f"Category {category!r} must map to a non-empty list of prefixes."
            )

        for prefix in prefixes:
            if not isinstance(prefix, str) or not prefix:
                raise ValueError(
                    f"Invalid prefix {prefix!r} in category {category!r}. "
                    "Prefixes must be non-empty strings."
                )

            if prefix in seen_prefixes:
                raise ValueError(
                    f"Duplicate prefix {prefix!r} appears in both "
                    f"{seen_prefixes[prefix]!r} and {category!r}."
                )

            seen_prefixes[prefix] = category


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)

    validate_category_prefixes(CATEGORY_PREFIXES)

    if out_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Output file already exists: {out_path}\n"
            "Use --overwrite if you want to replace it."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(CATEGORY_PREFIXES, indent=args.indent, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    total_prefixes = sum(len(prefixes) for prefixes in CATEGORY_PREFIXES.values())
    print(f"Wrote category-prefix JSON to: {out_path}")
    print(f"Categories: {len(CATEGORY_PREFIXES)}")
    print(f"Prefixes: {total_prefixes}")

    for category, prefixes in CATEGORY_PREFIXES.items():
        print(f"  {category}: {prefixes}")


if __name__ == "__main__":
    main()
