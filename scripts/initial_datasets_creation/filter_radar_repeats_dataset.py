import pandas as pd
import argparse
import ast
import random
import sys
from plms_repeats_circuits.utils.protein_similiarity_utils import compute_alignment_metrics 
from FAIR_algorithm import FAIR_algorithm_sensitive_to_occurences
import os

def overlapping(start1, end1, start2, end2):
    return not (end1 < start2 or start1 > end2)

def contained(start1, end1, start2, end2):
    """Returns True if [start2, end2] is fully contained within [start1, end1]."""
    return start1 <= start2 and end2 <= end1

def filter_repeat_based_on_releated_identical_repeats(
    seq, target_repeat_locations, target_repeat_key, min_len=4, is_identical=True
):
    # This function checks whether a given repeat in a protein sequence is "pure" and non-overlapping
    # according to two scenarios:
    #
    # - Identical case (is_identical=True):
    #     * Confirm the exact repeat (by key) exists in the detected repeats.
    #     * Ensure this repeat has **no related/overlapping repeats** (i.e., is isolated).
    #
    # - Non-identical case (is_identical=False):
    #     * For every possible subrepeat detected by the algorithm:
    #         - For each of the specified repeat occurrences (regions):
    #             + Count how many times this subrepeat appears *fully inside* each region.
    #             + Make sure it appears **at most once per region**.
    #             + Ensure this count is **the same across all regions** (uniformity).
    #         - If the subrepeat is found inside any region, ensure it is **not found outside**
    #           these regions (i.e., all occurrences are accounted for within specified bounds).
    #
    # The function returns True if all these conditions are met, and False otherwise.

    if not target_repeat_locations:
        return True
        
    repeat_dict, related_repeats_dict = FAIR_algorithm_sensitive_to_occurences(
        protein=seq, min=min_len, allow_overlapping=True
    )

    if not repeat_dict:
        return True

    if is_identical:
        repeat_key_in_dict = next((s for s in repeat_dict if target_repeat_key in s), None)
        if repeat_key_in_dict is None:
            return False
        related = related_repeats_dict.get(repeat_key_in_dict)
        return (related is None) or (isinstance(related, list) and len(related) == 0)

    # Non-identical case (unchanged)
    for key, algorithm_locations in repeat_dict.items():
        sorted_repeats_locations_list = sorted(algorithm_locations, key=lambda x: (x[0], x[1]))
        prev_times_k_is_contained_in_occurence = -1
        tot_times_k_is_contained_in_all = 0
        for start_target, end_target in target_repeat_locations:
            curr_times_k_is_contained_in_occurence = 0
            for start_curr, end_curr in sorted_repeats_locations_list:
                if contained(start_target, end_target, start_curr, end_curr):
                    curr_times_k_is_contained_in_occurence += 1
                    tot_times_k_is_contained_in_all += 1
            
            if curr_times_k_is_contained_in_occurence > 1: #avoid cases with have the same identical repeat inside the occurence
                return False
            if (
                prev_times_k_is_contained_in_occurence >= 0
                and prev_times_k_is_contained_in_occurence != curr_times_k_is_contained_in_occurence
            ): #ensure this identical repeat apears the same amount of times as other occurencs
                return False
            prev_times_k_is_contained_in_occurence = curr_times_k_is_contained_in_occurence


        if tot_times_k_is_contained_in_all > 0 and tot_times_k_is_contained_in_all != len(algorithm_locations): #if the key is indeed contained in any repeat occurence, ensure it doesnot apear outside the bounds
            return False

    return True


def validate_repeats_dont_have_related_repeats(df, is_identical):
    valid_indices = []
    invalid_indices = []
    
    total_rows = len(df)
    print(f"Starting validation of {total_rows} repeats...,", flush=True)
    
    # Log progress every 10 rows
    progress_interval = 10
    
    for idx, row in df.iterrows():
        if idx % progress_interval == 0:
            print(f"Validating... {idx}/{total_rows} rows ({(idx/total_rows*100):.1f}%)", flush=True)
        
        seq = row['seq']
        repeat_locations = row['repeat_locations']
        repeat_key = row['repeat_key']
        if not isinstance(repeat_locations, list):
            invalid_indices.append(idx)
            continue
            
        is_valid = filter_repeat_based_on_releated_identical_repeats(
            seq=seq,
            target_repeat_locations=repeat_locations,
            target_repeat_key=repeat_key,
            min_len=4,
            is_identical=is_identical
        )
                
        if is_valid:
            valid_indices.append(idx)
        else:
            invalid_indices.append(idx)
    
    print(f"Validation complete: {len(valid_indices)} valid, {len(invalid_indices)} invalid", flush=True)
    return valid_indices, invalid_indices

def filter_out_related_identical_repeats(df, is_identical):
    valid_indices, invalid_indices = validate_repeats_dont_have_related_repeats(df, is_identical)
    filtered_df = df.loc[valid_indices].copy()
    print(f"Filtered out {len(invalid_indices)} rows due to related identical repeats.", flush=True)
    return filtered_df, len(invalid_indices)
    

def filter_alignments_with_X(df, alignment_col='repeat_alignments'):
    original_len = len(df)

    def is_valid_alignment_list(alns):
        return isinstance(alns, list) and all('X' not in aln.upper() for aln in alns)

    df_filtered = df[df[alignment_col].apply(is_valid_alignment_list)]
    removed_count = original_len - len(df_filtered)

    print(f"Filtered out {removed_count} rows due to 'X' in repeat alignments.", flush=True)
    return df_filtered, removed_count


def parse_repeat_locations(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except Exception as e:
        print(f"Failed to parse repeat_locations: {x} — {e}", flush=True)
        return []

def has_overlap(region1, region2):
    return region1[0] <= region2[1] and region2[0] <= region1[1]

def filter_overlapping_repeats_within_groups(df, group_col="rep_id"):
    if group_col not in df.columns:
        print(f"Column '{group_col}' not found. Skipping group-based filtering.", flush=True)
        return df

    indices_to_discard = set()

    for group_value, group_df in df.groupby(group_col):
        idx_list = list(group_df.index)
        repeat_locs_list = group_df["repeat_locations"].tolist()

        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):  # ensures each pair is checked only once
                locs_i = repeat_locs_list[i]
                locs_j = repeat_locs_list[j]
                for r1 in locs_i:
                    for r2 in locs_j:
                        if has_overlap(r1, r2):
                            indices_to_discard.add(idx_list[i])
                            indices_to_discard.add(idx_list[j])
                            break

    return df.drop(index=list(indices_to_discard))

def clean_alignment(alignment):
    return ''.join([c.upper() for c in alignment if c != '.'])

def validate_segments_match_alignments(df):
    valid_indices = []
    invalid_count = 0
    
    for idx, row in df.iterrows():
        seq = row['seq']
        repeat_locations = row['repeat_locations']
        repeat_alignments = row['repeat_alignments']
        
        if not isinstance(repeat_locations, list) or not isinstance(repeat_alignments, list):
            invalid_count += 1
            continue
            
        if len(repeat_locations) != len(repeat_alignments):
            print(f"Row {idx}: Number of repeat locations ({len(repeat_locations)}) doesn't match number of alignments ({len(repeat_alignments)})", flush=True)
            invalid_count += 1
            continue
            
        is_valid = True
        
        for i, (loc, alignment) in enumerate(zip(repeat_locations, repeat_alignments)):
            start, end = loc[0], loc[1]
            segment = seq[start:end+1]
            
            cleaned_alignment = clean_alignment(alignment)
            
            if segment.upper() != cleaned_alignment.upper():
                print(f"Row {idx}, Repeat {i}: Segment '{segment}' doesn't match cleaned alignment '{cleaned_alignment}'", flush=True)
                is_valid = False
                break
                
        if is_valid:
            valid_indices.append(idx)
        else:
            invalid_count += 1
    
    valid_df = df.loc[valid_indices]
    print(f"Removed {invalid_count} rows that didn't meet the matching condition", flush=True)
    print(f"Kept {len(valid_indices)} valid rows", flush=True)
    
    return valid_df, invalid_count

def fix_location_shift(locations):
    return [[start, end - 1] for start, end in locations]

def trim_percentiles_multi(df, columns, lower=0.05, upper=0.95):
    mask = pd.Series(True, index=df.index)
    for col in columns:
        if col not in df.columns:
            print(f"Warning: column '{col}' not in DataFrame, skipping.", flush=True)
            continue
        low = df[col].quantile(lower)
        high = df[col].quantile(upper)
        mask &= (df[col] >= low) & (df[col] <= high)
        print(f"{col}: {lower*100}% = {low}, {upper*100}% = {high}")
    before, after = len(df), mask.sum()
    print(f"Total removed: {before-after} ({100*(before-after)/before:.2f}%)", flush=True)
    return df[mask]

def main():
    parser = argparse.ArgumentParser(description="Filter rows with overlapping repeats and compute alignment stats.")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV")
    parser.add_argument("--output_csv", required=True, help="Path to output filtered CSV")
    parser.add_argument("--min_repeats_times", type=int, default=2)
    parser.add_argument("--max_repeats_times", type=int, default=2)
    parser.add_argument("--min_seq_len", type=int, default=50)
    parser.add_argument("--min_mutation_percentage", type=float, default=0)
    parser.add_argument("--max_mutation_percentage", type=float, default=50)
    parser.add_argument("--fix_repeat_location_shift_problem", action="store_true", help="Set this flag if input is a RADAR dataset")
    parser.add_argument("--filter_releated_repeats", action="store_true", help="Set this flag if you want to filter based on related identical repeats")
    parser.add_argument("--trim_percentiles_lower", type=float, default=0.03)
    parser.add_argument("--trim_percentiles_upper", type=float, default=0.97)
    args = parser.parse_args()

    print("\n[Args] Script called with the following parameters:", flush=True)
    for k, v in vars(args).items():
        print(f"  {k}: {v}", flush=True)
    print()

    print(f"\n[Step 1] Reading input CSV: {args.input_csv}", flush=True)
    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"Failed to read CSV: {e}", flush=True)
        sys.exit(1)

    print(f"[Info] Input rows loaded: {len(df)}")
    if "repeat_locations" not in df.columns or "repeat_alignments" not in df.columns:
        print("Required columns 'repeat_locations' or 'repeat_alignments' are missing.", flush=True)
        sys.exit(1)

    print("[Step 2] Parsing 'repeat_locations'...", flush=True)
    df['repeat_locations'] = df['repeat_locations'].apply(parse_repeat_locations)

    if args.fix_repeat_location_shift_problem:
        print("[Step 3] Fixing repeat location shift (RADAR adjustment)...", flush=True)
        df['repeat_locations'] = df['repeat_locations'].apply(fix_location_shift)
        
    print("[Step 4] Filtering overlapping repeats within groups...")
    group_col = "rep_id" if "rep_id" in df.columns else "seq"
    df = filter_overlapping_repeats_within_groups(df, group_col)
    print(f"[Info] Rows after overlap filtering: {len(df)}", flush=True)

    print(f"[Step 5] Filtering rows by repeat count ({args.min_repeats_times}-{args.max_repeats_times})...", flush=True)
    df = df[df['repeat_locations'].apply(len).between(args.min_repeats_times, args.max_repeats_times)]
    print(f"[Info] Rows after repeat count filtering: {len(df)}", flush=True)

    if "seq_len" in df.columns:
        print(f"[Step 6] Filtering by minimum sequence length ({args.min_seq_len})...", flush=True)
        df = df[df["seq_len"] >= args.min_seq_len]
        print(f"[Info] Rows after seq_len filtering: {len(df)}", flush=True)

    print("[Step 7] Parsing 'repeat_alignments'...")
    df['repeat_alignments'] = df['repeat_alignments'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    print("[Step 8] Dropping rows with insufficient alignment data...", flush=True)
    df = df[df['repeat_alignments'].apply(lambda x: isinstance(x, list) and len(x) >= 2)]
    print(f"[Info] Rows after alignment count filtering: {len(df)}", flush=True)

    print("[Step 9] Filtering alignments containing 'X'...", flush=True)
    df, removed_x_count = filter_alignments_with_X(df)
    print(f"[Info] Rows after filtering 'X': {len(df)}", flush=True)


    print("[Step 10] Validating alignment/segment matches...", flush=True)
    df, invalid_count = validate_segments_match_alignments(df)
    print(f"[Info] Rows after alignment match validation: {len(df)}", flush=True)
    if df.empty:
        print("[Abort] No data left after validation.", flush=True)
        sys.exit(0)

    print("[Step 11] Annotating with alignment metrics...", flush=True)
    sample_metrics = compute_alignment_metrics(df['repeat_alignments'].iloc[0])
    metrics_columns = list(sample_metrics.keys())
    df[metrics_columns] = df['repeat_alignments'].apply(lambda x: pd.Series(compute_alignment_metrics(x)))

    print(f"[Step 12] Filtering by mutation percentage ({args.min_mutation_percentage}-{args.max_mutation_percentage})...", flush=True)
    df = df[df['mutation_percentage'].between(args.min_mutation_percentage, args.max_mutation_percentage)]
    print(f"[Info] Rows after mutation filtering: {len(df)}", flush=True)
    
    if "cluster_id" not in df.columns:
        print("[Step 13] Adding 'cluster_id' column...", flush=True)
        df["cluster_id"] = df["rep_id"] if "rep_id" in df.columns else pd.Series(["NA"] * len(df), index=df.index)

    if "tax" not in df.columns:
        print("[Step 14] Adding 'tax' column...", flush=True)
        df["tax"] = "UNKNOWN"

    if "repeat_key" not in df.columns:
        print("[Step 15] Adding 'repeat_key' column...", flush=True)
        df["repeat_key"] = df["repeat_alignments"].apply(lambda lst: random.choice(lst) if lst else "")

    print("[Step 16] Trimming percentiles for columns:", ['repeat_length', 'indels_count', 'substitutions_count'], flush=True)
    columns_to_filter = ['repeat_length', 'indels_count', 'substitutions_count']
    df = trim_percentiles_multi(df, columns_to_filter, args.trim_percentiles_lower, args.trim_percentiles_upper)
    print(f"[Info] Rows after percentile trimming: {len(df)}", flush=True)

    print(f"Reset Index: before : {len(df)}")
    
    df = df.reset_index(drop=True)
    print(f"Reset Index: After : {len(df)}")
    if args.filter_releated_repeats:
        print("[Step 17] Filtering out rows with related identical repeats...", flush=True)
        is_identical = args.min_mutation_percentage == args.max_mutation_percentage == 0.0
        print(f"is identical: {is_identical}")
        df, invalid_count = filter_out_related_identical_repeats(df, is_identical)
        print(f"Filtered {invalid_count} rows", flush=True)
        print(f"[Info] Rows after related repeat filtering: {len(df)}", flush=True)

    if df.empty:
        print("[Abort] No data left after final filtering.", flush=True)
        sys.exit(0)
    try:
        print(f"[Final Step] Saving filtered DataFrame to: {args.output_csv}", flush=True)
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print("[Done] Filtered and annotated CSV saved.", flush=True)
    except Exception as e:
        print(f"[Error] Failed to save output CSV: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
