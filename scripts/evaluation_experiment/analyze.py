import pandas as pd
import argparse
from pathlib import Path
from utils import (
    parse_to_list_column,
    add_approximate_repeat_columns,
    compute_stats,
    KEY_COLS,
)


def _print_excluded_approximate_proteins(approximate_all):
    """Print how many approximate protein identifiers are not in near_sub or near_indel. (Side func, delete later)"""
    all_keys = set(approximate_all[KEY_COLS].drop_duplicates().apply(tuple, axis=1))
    near_sub_df = approximate_all[
        (approximate_all["is_aligned_matching_identical"] == True)
        & (approximate_all["is_near_sub"] == True)
        & (approximate_all["indels_count"] == 0)
    ]
    near_indel_df = approximate_all[
        (approximate_all["is_aligned_matching_identical"] == True)
        & (approximate_all["is_near_indel"] == True)
        & (approximate_all["indels_count"] > 0)
    ]
    near_sub_keys = set(near_sub_df[KEY_COLS].drop_duplicates().apply(tuple, axis=1))
    near_indel_keys = set(near_indel_df[KEY_COLS].drop_duplicates().apply(tuple, axis=1))
    excluded = all_keys - near_sub_keys - near_indel_keys

    # Categorize why excluded
    has_aligned = set(
        approximate_all[approximate_all["is_aligned_matching_identical"] == True][KEY_COLS]
        .drop_duplicates().apply(tuple, axis=1)
    )
    indels_eq_0 = set(
        approximate_all[approximate_all["indels_count"] == 0][KEY_COLS]
        .drop_duplicates().apply(tuple, axis=1)
    )
    indels_gt_0 = set(
        approximate_all[approximate_all["indels_count"] > 0][KEY_COLS]
        .drop_duplicates().apply(tuple, axis=1)
    )
    has_near_sub_pos = set(near_sub_df[KEY_COLS].drop_duplicates().apply(tuple, axis=1))
    has_near_indel_pos = set(near_indel_df[KEY_COLS].drop_duplicates().apply(tuple, axis=1))
    has_near_indel_any = set(
        approximate_all[approximate_all["is_near_indel"] == True][KEY_COLS]
        .drop_duplicates().apply(tuple, axis=1)
    )

    reason_counts = {
        "indels==0, no aligned_identical at all": len(excluded & indels_eq_0 - has_aligned),
        "indels==0, has aligned_identical but none is_near_sub": len(excluded & indels_eq_0 & has_aligned - has_near_sub_pos),
        "indels>0, no aligned_identical at all": len(excluded & indels_gt_0 - has_aligned),
        "indels>0, positions near indel but none identical": len(excluded & indels_gt_0 & has_near_indel_any - has_near_indel_pos),
        "indels>0, has aligned_identical but no positions near indel": len(excluded & indels_gt_0 & has_aligned - has_near_indel_any),
    }

    print(f"Approximate: {len(excluded)} protein (cluster_id+rep_id+repeat_key) identifiers "
          f"excluded from both near_sub and near_indel (total: {len(all_keys)})")
    for reason, n in reason_counts.items():
        if n > 0:
            print(f"  - {reason}: {n}")


def create_baseline_task_df(df):
    """Prepare baseline predictions to match the format of repeat predictions."""
    return df.rename(columns={"is_correct": "accuracy"})


def _task_stats(df, scope):
    """Get per-protein stats for create_stats_df."""
    return compute_stats(df, scope=scope)[KEY_COLS + ["accuracy"]]


def create_stats_df(dfs, acc_col="accuracy"):
    """Compute accuracy and accuracy std for each task."""
    rows = []
    for name, df in dfs.items():
        rows.append({
            "Task": name,
            "N": len(df),
            "Accuracy": df[acc_col].mean(),
            "Accuracy_std": df[acc_col].std(),
        })
    return pd.DataFrame(rows)


def build_task_comparison(approximate_all, identical_all, synthetic_all, baseline):
    """Build the task comparison table."""
    dfs = {
        "Rand. Identical": _task_stats(synthetic_all, scope="all"),
        "Nat. Identical": _task_stats(identical_all, scope="all"),
        "Sub-Adjacent": _task_stats(approximate_all, scope="near_sub"),
        "Indel-Adjacent": _task_stats(approximate_all, scope="near_indel"),
        "Baseline": baseline,
    }

    return create_stats_df(dfs).round(5)


def main(args=None):
    parser = argparse.ArgumentParser(description="Compute evaluation performance statistics")
    parser.add_argument("--approximate_source", required=True, help="Source dataset CSV for approximate repeats")
    parser.add_argument("--approximate_all", required=True, help="predictions_all.csv for approximate")
    parser.add_argument("--identical_all", required=True, help="predictions_all.csv for identical")
    parser.add_argument("--synthetic_all", required=True, help="predictions_all.csv for synthetic")
    parser.add_argument("--baseline", required=True, help="baseline_predictions.csv")
    parser.add_argument("--output_dir", required=True, help="Directory to save output CSVs")
    args = parser.parse_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source dataset for approximate (needed for alignment enrichment)
    approx_source = pd.read_csv(args.approximate_source)
    for col in ["repeat_locations", "repeat_alignments"]:
        if col in approx_source.columns:
            approx_source[col] = parse_to_list_column(approx_source[col])

    # Load per-position predictions
    approximate_all = pd.read_csv(args.approximate_all)
    identical_all = pd.read_csv(args.identical_all)
    synthetic_all = pd.read_csv(args.synthetic_all)

    # Load and prepare baseline
    baseline = create_baseline_task_df(pd.read_csv(args.baseline))

    # Enrich approximate predictions with alignment columns
    approximate_all = add_approximate_repeat_columns(approx_source, approximate_all)

    # Merge extra source columns needed for task filtering
    extra_cols = approx_source[KEY_COLS + ["repeat_length", "identity_percentage"]].drop_duplicates()
    approximate_all = approximate_all.merge(extra_cols, on=KEY_COLS, how="left")

    _print_excluded_approximate_proteins(approximate_all)

    stats = build_task_comparison(approximate_all, identical_all, synthetic_all, baseline)
    stats.to_csv(output_dir / "tasks_performance.csv", index=False)
    print(f"Saved tasks_performance.csv to {output_dir}")


if __name__ == "__main__":
    main()
