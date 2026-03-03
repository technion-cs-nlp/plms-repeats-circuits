import pandas as pd
import argparse
from pathlib import Path
from utils import (
    parse_to_list_column,
    add_approximate_repeat_columns,
    compute_stats,
    scope_mask,
    KEY_COLS,
)

def sample_df(df, n, seed=42):
    if n <= 0 or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)

def filter_identical(source_df, predictions_all, args):
    """Filter by accuracy threshold, sample, merge with source dataset."""
    per_protein = compute_stats(predictions_all, scope="all")[KEY_COLS + ["accuracy", "avg_prediction_prob"]]
    merged = source_df.merge(per_protein, on=KEY_COLS, how="inner")
    filtered = merged[merged["accuracy"] >= args.accuracy_threshold]
    if len(filtered) < args.sample_size:
        print(f"Warning: only {len(filtered)} samples after filtering (requested {args.sample_size})")
        return filtered
    return sample_df(filtered, args.sample_size, args.random_seed)


def _high_confidence_positions(enriched_all, filter_type, confidence_threshold=0.0):
    """All eligible positions that are correct and meet confidence threshold, grouped per protein."""
    scope = "aligned_identical" if filter_type == "none" else filter_type
    mask = (
        scope_mask(enriched_all, scope)
        & (enriched_all["is_correct"] == True)
        & (enriched_all["true_label_probability"] >= confidence_threshold)
    )
    eligible = enriched_all.loc[mask]
    grouped = eligible.groupby(KEY_COLS)["masked_position"].apply(
        lambda x: sorted(x.unique().tolist())
    ).reset_index()
    grouped = grouped.rename(columns={"masked_position": "high_confidence_positions"})
    return grouped


def filter_approximate(source_df, predictions_all, args):
    """Enrich with alignment columns, filter by accuracy_identical, merge with source."""
    for col in ["repeat_locations", "repeat_alignments"]:
        if col in source_df.columns:
            source_df[col] = parse_to_list_column(source_df[col])

    enriched_all = add_approximate_repeat_columns(source_df, predictions_all)

    # Per-protein accuracy and avg_prob from all positions
    per_protein = compute_stats(predictions_all, scope="all")[KEY_COLS + ["accuracy", "avg_prediction_prob"]]
    result = source_df.merge(per_protein, on=KEY_COLS, how="inner")

    # Scope by protein (indels_count), then compute aligned-identical stats
    if args.filter_type == "near_sub":
        scoped = enriched_all[enriched_all["indels_count"] == 0]
    elif args.filter_type == "near_indel":
        scoped = enriched_all[enriched_all["indels_count"] > 0]
    else:
        scoped = enriched_all

    task_stats = compute_stats(scoped, scope="aligned_identical")
    task_stats = task_stats.rename(columns={
        "accuracy": "accuracy_identical",
        "avg_prediction_prob": "avg_prob_aligned_identical",
        "count": "count_aligned_identical",
        "correct": "correct_aligned_identical",
    })
    result = result.merge(
        task_stats[KEY_COLS + ["accuracy_identical", "avg_prob_aligned_identical",
                               "count_aligned_identical", "correct_aligned_identical"]],
        on=KEY_COLS, how="left",
    )

    # High-confidence positions (all eligible correct positions meeting confidence threshold)
    hc = _high_confidence_positions(
        enriched_all, args.filter_type, args.confidence_threshold
    )
    result = result.merge(hc, on=KEY_COLS, how="left")
    result["high_confidence_positions"] = result["high_confidence_positions"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    has_high_confidence = result["high_confidence_positions"].apply(
        lambda x: len(x) > 0 if isinstance(x, list) else False
    )
    result = result[
        (result["accuracy_identical"] >= args.accuracy_threshold) & has_high_confidence
    ]
    return sample_df(result, args.sample_size, args.random_seed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Filter evaluation results for the corruption experiment"
    )
    parser.add_argument("--dataset_type", required=True,
                        choices=["identical", "approximate", "synthetic"])
    parser.add_argument("--predictions_all", required=True,
                        help="Path to predictions_all.csv from evaluate step")
    parser.add_argument("--source_dataset", required=True,
                        help="Path to original dataset CSV in datasets/")
    parser.add_argument("--output_file", required=True,
                        help="Path for materialized filtered CSV")
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--accuracy_threshold", type=float, default=1.0,
                        help="Min accuracy threshold (identical/synthetic: accuracy; approximate: accuracy_identical)")
    parser.add_argument("--filter_type", choices=["none", "near_sub", "near_indel"],
                        default="none", help="Mutation proximity filter (approximate only)")
    parser.add_argument("--confidence_threshold", type=float, default=0.0,
                        help="Min confidence for high_confidence_positions (approximate only)")
    args = parser.parse_args(args)

    predictions_all = pd.read_csv(args.predictions_all)
    source_df = pd.read_csv(args.source_dataset)

    if args.dataset_type in ("identical", "synthetic"):
        result = filter_identical(source_df, predictions_all, args)
    elif args.dataset_type == "approximate":
        result = filter_approximate(source_df, predictions_all, args)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} rows to {output_path}")


if __name__ == "__main__":
    main()
