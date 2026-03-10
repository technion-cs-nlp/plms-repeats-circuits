import argparse
from functools import reduce
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from plms_repeats_circuits.utils.counterfactuals_config import METHOD_PATTERNS, find_file_for_method


def find_files_by_methods(input_dir: Path, methods: List[str]) -> Dict[str, Path]:
    """Find main CSV files for each method using find_file_for_method."""
    csv_dict = {}
    for method in methods:
        if method not in METHOD_PATTERNS:
            print(f"Warning: Unknown method '{method}'. Skipping.")
            print(f"  Available: {', '.join(sorted(METHOD_PATTERNS.keys()))}")
            continue
        csv_file = find_file_for_method(method, input_dir)
        if csv_file:
            csv_dict[method] = csv_file
            print(f"Found {method}: {csv_file.name}")
        else:
            print(f"Warning: No file for method '{method}'")
    return csv_dict


# Columns to merge from evaluate output into the filtered dataset (not all result columns)
MERGE_COLS = [
    "masked_position",
    "corrupted_positions",
    "replacements",
    "corrupted_sequence",
    "masked_repeat_example",
    "corrupted_repeat_example",
    "is_corrupt_changed_argmax",
]


def get_common_changed_keys(csv_dict: Dict[str, Path]) -> Set[Tuple]:
    """Load evaluate outputs, find (rep_id, repeat_key) where is_corrupt_changed_argmax==1, return intersection."""
    changed_keys = {}
    for method, path in csv_dict.items():
        if not path.exists():
            print(f"  Skipping {method}: file not found")
            continue
        df = pd.read_csv(path)
        if "is_corrupt_changed_argmax" not in df.columns:
            print(f"  Skipping {method}: missing is_corrupt_changed_argmax")
            continue
        changed = df[df["is_corrupt_changed_argmax"] == 1]
        keys = set(zip(changed["rep_id"], changed["repeat_key"]))
        changed_keys[method] = keys
        print(f"  {method}: {len(keys)} changed")
    if not changed_keys:
        return set()
    return reduce(set.intersection, changed_keys.values())


def create_filtered_datasets(
    csv_dict: Dict[str, Path],
    common_keys: Set[Tuple],
    source: pd.DataFrame,
    output_dir: Path,
) -> None:
    """For each method, merge MERGE_COLS from that method's evaluate output into
    the source rows common to all methods, and save as a separate file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source = source.copy()
    source["__key__"] = list(zip(source["rep_id"], source["repeat_key"]))
    filtered_base = source[source["__key__"].isin(common_keys)].drop(columns="__key__").copy()

    for method, path in csv_dict.items():
        eval_df = pd.read_csv(path)
        cols_to_merge = [c for c in MERGE_COLS if c in eval_df.columns]
        if cols_to_merge:
            eval_sub = eval_df[["rep_id", "repeat_key"] + cols_to_merge]
            out_df = filtered_base.merge(eval_sub, on=["rep_id", "repeat_key"], how="left")
        else:
            out_df = filtered_base.copy()

        output_path = output_dir / path.name
        out_df.to_csv(output_path, index=False)
        print(f"  Saved {method}: {len(out_df)} rows → {output_path.name}")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Filter to positions where all methods changed argmax; merge with source."
    )
    parser.add_argument("--source_dataset", type=Path, required=True, help="Source dataset CSV")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory of evaluate output CSVs")
    parser.add_argument("--methods", nargs="+", required=True, help="Method names (e.g. mask blosum50)")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for filtered CSVs")
    parsed = parser.parse_args(args)

    if not parsed.input_dir.exists():
        print(f"Error: Input dir not found: {parsed.input_dir}")
        return
    if not parsed.source_dataset.exists():
        print(f"Error: Source dataset not found: {parsed.source_dataset}")
        return

    csv_dict = find_files_by_methods(parsed.input_dir, parsed.methods)
    if not csv_dict:
        print("Error: No method files found")
        return

    common_keys = get_common_changed_keys(csv_dict)
    print(f"\nCommon changed positions: {len(common_keys)}")

    if not common_keys:
        print("No common positions; nothing to save.")
        return

    source = pd.read_csv(parsed.source_dataset)
    if not all(c in source.columns for c in ["rep_id", "repeat_key"]):
        print("Error: Source must have rep_id and repeat_key")
        return

    print(f"Creating filtered datasets in: {parsed.output_dir}")
    create_filtered_datasets(csv_dict, common_keys, source, parsed.output_dir)
    print(f"\nDone! Created {len(csv_dict)} filtered datasets")


if __name__ == "__main__":
    main()
