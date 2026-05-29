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
    "Anti Swarm": [
        "anti_swarm",
        "hs_colossus_anti_swarm"
    ],
    "Mirrors": [
        "balanced_mirrors",
        "v5_mirror",
        "bm_three_group_mirror",
        "large_capped_mirror"
    ],
    "Swarm": [
        "v5_swarm",
        "hs_split_swarm",
        "hs_stalker_kite"
    ],
    "Ravine": [
        "tv_ravine"
    ],
    "10m vs 11m": [
        "10m_vs_11m",
        "sa_10m_vs_11m"
    ],
    "27m vs 30m": [
        "27m_vs_30m"
    ],
    "2c vs 64zg": [
        "2c_vs_64zg"
    ],
    "2s vs 1sc": [
        "2s_vs_1sc"
    ],
    "2s3z": [
        "2s3z"
    ],
    "3s vs 5z": [
        "3s_vs_5z"
    ],
    "3s5z": [
        "3s5z"
    ],
    "3s5z vs 3s6z": [
        "3s5z_vs_3s6z"
    ],
    "bane vs bane": [
        "bane_vs_bane"
    ],
    "bm": [
        "bm_0",
        "bm_split_2v2"
    ],
    "corridor": [
        "corridor"
    ],
    "bio support": [
        "bm_bio_support"
    ],
    "colossus guard": [
        "bm_colossus_guard"
    ],
    "large capped marine": [
        "bm_large_capped_marine"
    ],
    "marine small": [
        "bm_marine_small"
    ],
    "stalker zealot": [
        "bm_stalker_zealot"
    ],
    "zerg bane skirmish": [
        "bm_zerg_bane_skirm"
    ],
    "heavy swarm": [
        "heavy_swarm"
    ],
    "medivac swarm": [
        "hs_bio_medivac_swarm"
    ],
    "close quarters bane": [
        "hs_close_quarters_bane"
    ],
    "marine holdout": [
        "hs_marine_holdout"
    ],
    "multiwave enemy": [
        "hs_multiwave_enemy"
    ],
    "zealot vs zerglings": [
        "hs_zealot_vs_zerglings"
    ],
    "hsm": [
        "hsm"
    ],
    "asym": [
        "large_capped_asym",
        "mc_support_asymmetry"
    ],
    "low count micro": [
        "low_count_micro"
    ],
    "marine roach cross": [
        "marine_roach_cross"
    ],
    "bio vs protoss": [
        "mc_bio_vs_protoss",
        "v5_mix_bio_vs_protoss"
    ],
    "colossus vs bio": [
        "mc_colossus_vs_bio"
    ],
    "large mixed cap": [
        "mc_large_mixed_cap"
    ],
    "low count roles": [
        "mc_low_count_roles"
    ],
    "protoss vs zerg": [
        "mc_protoss_vs_zerg"
    ],
    "zerg pressure": [
        "mc_zerg_pressure"
    ],
    "medium mixed roles": [
        "medium_mixed_roles"
    ],
    "medivac support": [
        "medivac_support"
    ],
    "mixed composition": [
        "mixed_composition"
    ],
    "mmm": [
        "mmmm",
        "mmm2"
    ],
    "muc": [
        "muc"
    ],
    "quantity vs quality": [
        "quality_vs_quantity"
    ],
    "baneling threat": [
        "sa_baneling_threat"
    ],
    "bio plus medivac": [
        "bio_plus_medivac"
    ],
    "enemy support": [
        "sa_enemy_support",
    ],
    "few strong vs many weak": [
        "sa_few_strong_vs_many_weak",
        "sa_many_weak_vs_few_strong"
    ],
    "large capped asym": [
        "sa_large_capped_asym"
    ],
    "slight advantage": [
        "seaa"
    ],
    "small zerg skirmish": [
        "small_zerg_skirmish"
    ],
    "stalker kite test": [
        "stalker_kite_test"
    ],
    "terrain stress": [
        "terrain_stress"
    ],
    "terrain variants": [
        "terrain_variants"
    ],
    "ravine crossfire": [
        "tv_ravine_crossfire"
    ],
    "pentagon diagonal": [
        "tv_pentagon_diagonal",
    ],
    "octagon surround": [
        "tv_octagon_surround"
    ],
    "narrow split": [
        "tv_narrow_split"
    ],
    "corridor choke": [
        "tv_corridor_choke"
    ],
    "corner to corner": [
        "tv_corner_to_corner"
    ],
    "backline protection": [
        "tv_backline_protection"
    ],
    "green open field": [
        "tv_all_green_open"
    ],
    "tv": [
        "tv_0",
        "tv_100"
    ],
    "baneling threat": [
        "v5_adv_baneling_threat"
    ],
    "close scramble": [
        "v5_adv_close_scramble"
    ],
    "colossus anchor": [
        "v5_adv_colossus_anchor"
    ],
    "corner push": [
        "v5_adv_corner_push"
    ],
    "exact 50": [
        "v5_adv_exact50"
    ],
    "medivac kite": [
        "v5_adv_medivac_kite"
    ],
    "small left fast": [
        "v5_adv_small_left_fast"
    ],
    "small right ranged": [
        "v5_adv_small_right_ranged"
    ],
    "split ally": [
        "v5_adv_split_ally"
    ],
    "split enemy": [
        "v5_adv_split_enemy"
    ],
    "vertical clash": [
        "vertical clash"
    ],
    "corner push": [
        "v5_adv_corner_push"
    ],
    "exact 50": [
        "v5_adv_exact50",
        "v5_terrain_exact50"
    ],
    "medivac kite": [
        "v5_adv_medivac_kite"
    ],
    "small left fast": [
        "v5_adv_small_left_fast"
    ],
    "small right ranged": [
        "v5_adv_small_right_ranged"
    ],
    "corner push": [
        "v5_adv_corner_push"
    ],
    "mix support": [
        "v5_mix"
    ],
    "swarm bio funnel": [
        "v5_swarm_bio_funnel"
    ],
    "swarm center hold": [
        "v5_swarm_center_hold"
    ],
    "swarm colossus": [
        "v5_swarm_colossus"
    ],
    "swarm small pincer": [
        "v5_swarm_small"
    ],
    "mix support": [
        "v5_mix"
    ],
    "swarm bio funnel": [
        "v5_swarm_bio_funnel"
    ],
    "swarm center hold": [
        "v5_swarm_center_hold"
    ],
    "swarm colossus": [
        "v5_swarm_colossus"
    ],
    "terrain wide": [
        "v5_terrain_wide"
    ],
    "terrain sparse": [
        "v5_terrain_sparse"
    ],
    "terrain column": [
        "v5_terrain_right",
        "v5_terrain_left"
    ],
    "terain enemy arc": [
        "v5_terrain_enemy_arc"
    ],
    "terrain corner": [
        "v5_terrain_corner"
    ],
    "terrain close offset": [
        "v5_terrain_close_offset"
    ],
    "terrain center fort": [
        "v5_terrain_center_fort"
    ],
    "terrain_box_split": [
        "v5_terrain_box_split"
    ],
    "terrain bottomup": [
        "v5_terrain_bottomup"
    ],
    "swarm zealot wall": [
        "v5_swarm_zealot_wall"
    ],
    "swarm vertical wave": [
        "v5_swarm_vertical_wave"
    ],
    "swarm three wave": [
        "v5_swarm_three_wave"
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
