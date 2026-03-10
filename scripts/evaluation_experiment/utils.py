"""Shared helpers for the evaluation experiment scripts."""

import pandas as pd
import ast
from plms_repeats_circuits.utils.protein_similiarity_utils import analyze_repeat_positions

KEY_COLS = ["cluster_id", "rep_id", "repeat_key"]


def parse_to_list_column(col):
    """Parse stringified lists (e.g. repeat_locations) in a DataFrame column."""
    return col.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and pd.notnull(x) else [])


def add_approximate_repeat_columns(source_df, all_df):
    """Enrich per-position predictions with alignment-based info from the source dataset.

    Merges seq, repeat_locations, repeat_alignments, indels_count from source_df
    into all_df, then computes per-position alignment properties.
    """
    cols_needed = ["cluster_id", "rep_id", "repeat_key",
                   "seq", "repeat_locations", "repeat_alignments", "indels_count"]
    merged = all_df.merge(source_df[cols_needed], on=KEY_COLS, how="left")
    assert len(merged) == len(all_df), "Merge changed row count unexpectedly"

    new_cols = {
        "is_aligned_matching_identical": [],
        "aligned_matching_absolute_position_in_sequence": [],
        "aligned_matching_identity": [],
        "is_near_sub": [],
        "is_near_indel": [],
    }

    cache = {}
    current_key = None

    for row in merged.itertuples(index=False):
        row_key = (row.cluster_id, row.rep_id, row.repeat_key)

        if row_key != current_key:
            cache = {}
            current_key = row_key

        pos_info = cache.get(row_key)
        if pos_info is None:
            pos_info, _ = analyze_repeat_positions(
                row.seq, row.repeat_locations, row.repeat_alignments
            )
            cache[row_key] = pos_info

        info = pos_info[row.relative_repeat_number - 1][row.masked_position]
        new_cols["is_aligned_matching_identical"].append(info.is_aligned_matching_identical)
        new_cols["aligned_matching_absolute_position_in_sequence"].append(
            info.aligned_matching_absolute_position_in_sequence
        )
        new_cols["aligned_matching_identity"].append(info.aligned_matching_identity)
        new_cols["is_near_sub"].append(info.is_near_sub)
        new_cols["is_near_indel"].append(info.is_near_indel)

    for col_name, col_data in new_cols.items():
        merged[col_name] = col_data

    return merged


def scope_mask(df, scope):
    """Boolean mask for positions in scope.

    scope: "all" | "aligned_identical" | "near_sub" | "near_indel"
    """
    if scope == "all":
        return pd.Series(True, index=df.index)
    aligned = df["is_aligned_matching_identical"] == True
    if scope == "aligned_identical":
        return aligned
    if scope == "near_sub":
        return aligned & (df["is_near_sub"] == True) & (df["indels_count"] == 0)
    if scope == "near_indel":
        return aligned & (df["is_near_indel"] == True) & (df["indels_count"] > 0)
    return aligned


def compute_stats(df, scope="all"):
    """Compute per-protein accuracy and related metrics.

    scope:
      - "all": use all positions (identical/synthetic, or approximate unfiltered)
      - "aligned_identical": aligned-identical positions only (approximate only)
      - "near_sub": aligned-identical + is_near_sub + indels_count==0 (approximate only)
      - "near_indel": aligned-identical + is_near_indel + indels_count>0 (approximate only)
    """
    filtered = df if scope == "all" else df[scope_mask(df, scope)]

    agg = filtered.groupby(KEY_COLS).agg(
        accuracy=("is_correct", "mean"),
        avg_prediction_prob=("true_label_probability", "mean"),
        count=("is_correct", "count"),
        correct=("is_correct", "sum"),
    ).reset_index()
    return agg
