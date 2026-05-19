import argparse
import ast
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import (
    KEY_COLS,
    add_approximate_repeat_columns,
    parse_to_list_column,
    scope_mask,
)


def _parse_repeat_locations(value: Any) -> List[Tuple[int, int]]:
    if isinstance(value, list):
        locs = value
    elif isinstance(value, str) and value.strip():
        # Source CSVs store python-literals like "[(0, 9), (10, 19)]" or "[[79, 92], [95, 108]]"
        locs = ast.literal_eval(value)
    else:
        return []

    out: List[Tuple[int, int]] = []
    for item in locs:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            out.append((int(item[0]), int(item[1])))
    return out


def _add_repeat_relative_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-row relative position inside the matched repeat occurrence.

    Requires columns: masked_position, repeat_locations.
    Produces:
      - repeat_occurrence_idx: 1-based index in sorted repeat_locations
      - repeat_occurrence_start, repeat_occurrence_end
      - relative_position_in_repeat: 1-based offset within occurrence
    """
    required = {"masked_position", "repeat_locations"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for relative position: {sorted(missing)}")

    df = df.copy()
    df["repeat_occurrence_idx"] = np.nan
    df["repeat_occurrence_start"] = np.nan
    df["repeat_occurrence_end"] = np.nan
    df["relative_position_in_repeat"] = np.nan

    # Grouped loop to avoid scanning repeat_locations for every row globally.
    for _, g in df.groupby(KEY_COLS, sort=False):
        idx = g.index
        locs = _parse_repeat_locations(g["repeat_locations"].iloc[0])
        locs = sorted(locs, key=lambda x: x[0])

        positions = g["masked_position"].to_numpy()
        occ_idx = np.full(len(positions), np.nan, dtype=float)
        occ_start = np.full(len(positions), np.nan, dtype=float)
        occ_end = np.full(len(positions), np.nan, dtype=float)
        rel_pos = np.full(len(positions), np.nan, dtype=float)

        for i, pos in enumerate(positions):
            for j, (start, end) in enumerate(locs, start=1):
                if start <= pos <= end:
                    occ_idx[i] = j
                    occ_start[i] = start
                    occ_end[i] = end
                    rel_pos[i] = (pos - start) + 1
                    break

        df.loc[idx, "repeat_occurrence_idx"] = occ_idx
        df.loc[idx, "repeat_occurrence_start"] = occ_start
        df.loc[idx, "repeat_occurrence_end"] = occ_end
        df.loc[idx, "relative_position_in_repeat"] = rel_pos

    df["repeat_occurrence_idx"] = df["repeat_occurrence_idx"].astype("Int64")
    df["relative_position_in_repeat"] = df["relative_position_in_repeat"].astype("Int64")
    df["repeat_occurrence_start"] = df["repeat_occurrence_start"].astype("Int64")
    df["repeat_occurrence_end"] = df["repeat_occurrence_end"].astype("Int64")
    return df


def _add_repeat_boundary_flag(df: pd.DataFrame) -> pd.DataFrame:
    """True if masked token lies within 3 positions of start or end of its repeat occurrence.

    Uses 0-based offsets from the inclusive [start, end] segment:
      - near start: (masked_position - start) in {0, 1, 2}
      - near end: (end - masked_position) in {0, 1, 2}
    Rows without a resolved occurrence (missing start/end) get False.
    """
    df = df.copy()
    if not {"repeat_occurrence_start", "repeat_occurrence_end", "masked_position"}.issubset(df.columns):
        raise ValueError("Need repeat_occurrence_start/end and masked_position for boundary flag")
    start = pd.to_numeric(df["repeat_occurrence_start"], errors="coerce")
    end = pd.to_numeric(df["repeat_occurrence_end"], errors="coerce")
    pos = pd.to_numeric(df["masked_position"], errors="coerce")
    ok = start.notna() & end.notna() & pos.notna()
    off_start = pos - start
    off_end = end - pos
    df["is_repeat_boundary"] = ok & ((off_start <= 2) | (off_end <= 2))
    return df


def failure_boundary_summary_by_task(
    synthetic_all: pd.DataFrame,
    identical_all: pd.DataFrame,
    approximate_all: pd.DataFrame,
) -> pd.DataFrame:
    """Per task: among eligible positions only, count failures at repeat boundary vs not."""
    tasks: List[Tuple[str, pd.DataFrame, pd.Series]] = [
        ("Rand. Identical", synthetic_all, scope_mask(synthetic_all, "all")),
        ("Nat. Identical", identical_all, scope_mask(identical_all, "all")),
        ("Sub-Adjacent", approximate_all, scope_mask(approximate_all, "near_sub")),
        ("Indel-Adjacent", approximate_all, scope_mask(approximate_all, "near_indel")),
    ]
    rows = []
    for task_name, df, elig in tasks:
        sub = df.loc[elig]
        n_eligible = int(len(sub))
        fail = sub.loc[sub["is_correct"] == False]
        n_fail = int(len(fail))
        if n_fail == 0:
            n_bd = 0
        else:
            n_bd = int(fail["is_repeat_boundary"].sum())
        rows.append(
            {
                "Task": task_name,
                "n_eligible_positions": n_eligible,
                "n_failures": n_fail,
                "n_failures_at_repeat_boundary": n_bd,
                "n_failures_not_at_repeat_boundary": n_fail - n_bd,
                "frac_failures_at_repeat_boundary": (n_bd / n_fail) if n_fail else np.nan,
            }
        )
    return pd.DataFrame(rows)


def failure_outcome_by_repeat_summary_by_task(
    synthetic_all: pd.DataFrame,
    identical_all: pd.DataFrame,
    approximate_all: pd.DataFrame,
    approx_per_repeat: Optional[pd.DataFrame] = None,
    all_positions_acc_lt: float = 0.3,
) -> pd.DataFrame:
    """Per task: among repeats (KEY_COLS) with eligible positions, classify repeat outcome.

    Outcome per repeat is based only on eligible positions:
      - all_failed: all eligible positions are incorrect
      - mixed: both correct and incorrect eligible positions exist
      - all_correct: all eligible positions are correct

    For approximate tasks only, merges ``accuracy_all_positions`` from ``approx_per_repeat``
    (same as ``per_repeat_failures_approximate.csv``) and counts repeats / proteins with
    ``accuracy_all_positions < all_positions_acc_lt``.
    """
    tasks: List[Tuple[str, pd.DataFrame, pd.Series, bool]] = [
        ("Rand. Identical", synthetic_all, scope_mask(synthetic_all, "all"), False),
        ("Nat. Identical", identical_all, scope_mask(identical_all, "all"), False),
        ("Sub-Adjacent", approximate_all, scope_mask(approximate_all, "near_sub"), True),
        ("Indel-Adjacent", approximate_all, scope_mask(approximate_all, "near_indel"), True),
    ]

    nan_approx_extra = {
        "n_repeats_accuracy_all_positions_lt_threshold": np.nan,
        "frac_repeats_accuracy_all_positions_lt_threshold": np.nan,
        "n_distinct_proteins_in_task": np.nan,
        "n_proteins_with_any_repeat_accuracy_all_positions_lt_threshold": np.nan,
        "frac_proteins_with_any_repeat_accuracy_all_positions_lt_threshold": np.nan,
        "n_all_failed_repeats_accuracy_all_positions_lt_threshold": np.nan,
        "n_mixed_repeats_accuracy_all_positions_lt_threshold": np.nan,
        "n_all_correct_repeats_accuracy_all_positions_lt_threshold": np.nan,
    }

    rows = []
    for task_name, df, elig, use_approx_acc in tasks:
        eligible = df.loc[elig].copy()
        if eligible.empty:
            row = {
                "Task": task_name,
                "n_repeats_with_eligible_positions": 0,
                "n_repeats_all_failed": 0,
                "n_repeats_mixed": 0,
                "n_repeats_all_correct": 0,
                "frac_repeats_all_failed": np.nan,
                "frac_repeats_mixed": np.nan,
                "frac_repeats_all_correct": np.nan,
                "all_positions_acc_threshold": all_positions_acc_lt,
            }
            row.update(nan_approx_extra)
            rows.append(row)
            continue

        repeat_stats = (
            eligible.groupby(KEY_COLS, sort=False)
            .agg(
                n_eligible_positions=("is_correct", "size"),
                n_correct=("is_correct", "sum"),
            )
            .reset_index()
        )
        repeat_stats["n_failed"] = repeat_stats["n_eligible_positions"] - repeat_stats["n_correct"]

        n_total = int(len(repeat_stats))
        n_all_failed = int((repeat_stats["n_correct"] == 0).sum())
        n_all_correct = int((repeat_stats["n_failed"] == 0).sum())
        n_mixed = int(n_total - n_all_failed - n_all_correct)

        row: Dict[str, Any] = {
            "Task": task_name,
            "n_repeats_with_eligible_positions": n_total,
            "n_repeats_all_failed": n_all_failed,
            "n_repeats_mixed": n_mixed,
            "n_repeats_all_correct": n_all_correct,
            "frac_repeats_all_failed": (n_all_failed / n_total) if n_total else np.nan,
            "frac_repeats_mixed": (n_mixed / n_total) if n_total else np.nan,
            "frac_repeats_all_correct": (n_all_correct / n_total) if n_total else np.nan,
            "all_positions_acc_threshold": all_positions_acc_lt,
        }

        if use_approx_acc and approx_per_repeat is not None and "accuracy_all_positions" in approx_per_repeat.columns:
            acc_df = approx_per_repeat[KEY_COLS + ["accuracy_all_positions"]].drop_duplicates(
                subset=KEY_COLS, keep="first"
            )
            merged = repeat_stats.merge(acc_df, on=KEY_COLS, how="left")
            n_distinct_prot = int(merged[["cluster_id", "rep_id"]].drop_duplicates().shape[0])
            row["n_distinct_proteins_in_task"] = n_distinct_prot
            low = merged["accuracy_all_positions"].notna() & (
                merged["accuracy_all_positions"] < all_positions_acc_lt
            )
            n_low_rep = int(low.sum())
            row["n_repeats_accuracy_all_positions_lt_threshold"] = n_low_rep
            row["frac_repeats_accuracy_all_positions_lt_threshold"] = (
                (n_low_rep / n_total) if n_total else np.nan
            )
            low_keys = merged.loc[low, ["cluster_id", "rep_id"]].drop_duplicates()
            n_prot_low = int(len(low_keys))
            row["n_proteins_with_any_repeat_accuracy_all_positions_lt_threshold"] = n_prot_low
            row["frac_proteins_with_any_repeat_accuracy_all_positions_lt_threshold"] = (
                (n_prot_low / n_distinct_prot) if n_distinct_prot else np.nan
            )

            failed_only = merged["n_correct"] == 0
            mixed_only = (merged["n_correct"] > 0) & (merged["n_failed"] > 0)
            ok_only = merged["n_failed"] == 0
            row["n_all_failed_repeats_accuracy_all_positions_lt_threshold"] = int(
                (failed_only & low).sum()
            )
            row["n_mixed_repeats_accuracy_all_positions_lt_threshold"] = int((mixed_only & low).sum())
            row["n_all_correct_repeats_accuracy_all_positions_lt_threshold"] = int(
                (ok_only & low).sum()
            )
        else:
            row.update(nan_approx_extra)

        rows.append(row)

    return pd.DataFrame(rows)


def mixed_repeat_boundary_comparison_by_task(
    synthetic_all: pd.DataFrame,
    identical_all: pd.DataFrame,
    approximate_all: pd.DataFrame,
) -> pd.DataFrame:
    """For mixed repeats only, compare boundary enrichment in failures vs successes."""
    tasks: List[Tuple[str, pd.DataFrame, pd.Series]] = [
        ("Rand. Identical", synthetic_all, scope_mask(synthetic_all, "all")),
        ("Nat. Identical", identical_all, scope_mask(identical_all, "all")),
        ("Sub-Adjacent", approximate_all, scope_mask(approximate_all, "near_sub")),
        ("Indel-Adjacent", approximate_all, scope_mask(approximate_all, "near_indel")),
    ]

    rows = []
    for task_name, df, elig in tasks:
        eligible = df.loc[elig].copy()
        if eligible.empty:
            rows.append(
                {
                    "Task": task_name,
                    "n_mixed_repeats": 0,
                    "n_failed_positions_mixed": 0,
                    "n_failed_positions_mixed_at_boundary": 0,
                    "frac_failed_positions_mixed_at_boundary": np.nan,
                    "n_success_positions_mixed": 0,
                    "n_success_positions_mixed_at_boundary": 0,
                    "frac_success_positions_mixed_at_boundary": np.nan,
                    "failure_vs_success_boundary_rate_ratio": np.nan,
                }
            )
            continue

        repeat_stats = (
            eligible.groupby(KEY_COLS, sort=False)
            .agg(
                n_eligible_positions=("is_correct", "size"),
                n_correct=("is_correct", "sum"),
            )
            .reset_index()
        )
        repeat_stats["n_failed"] = repeat_stats["n_eligible_positions"] - repeat_stats["n_correct"]
        mixed_keys = repeat_stats[(repeat_stats["n_correct"] > 0) & (repeat_stats["n_failed"] > 0)][KEY_COLS]

        if mixed_keys.empty:
            rows.append(
                {
                    "Task": task_name,
                    "n_mixed_repeats": 0,
                    "n_failed_positions_mixed": 0,
                    "n_failed_positions_mixed_at_boundary": 0,
                    "frac_failed_positions_mixed_at_boundary": np.nan,
                    "n_success_positions_mixed": 0,
                    "n_success_positions_mixed_at_boundary": 0,
                    "frac_success_positions_mixed_at_boundary": np.nan,
                    "failure_vs_success_boundary_rate_ratio": np.nan,
                }
            )
            continue

        mixed = eligible.merge(mixed_keys, on=KEY_COLS, how="inner")
        failed = mixed[mixed["is_correct"] == False]
        succeeded = mixed[mixed["is_correct"] == True]

        n_mixed = int(len(mixed_keys))
        n_fail = int(len(failed))
        n_fail_bd = int(failed["is_repeat_boundary"].sum()) if n_fail else 0
        n_succ = int(len(succeeded))
        n_succ_bd = int(succeeded["is_repeat_boundary"].sum()) if n_succ else 0

        fail_rate = (n_fail_bd / n_fail) if n_fail else np.nan
        succ_rate = (n_succ_bd / n_succ) if n_succ else np.nan
        ratio = (fail_rate / succ_rate) if (pd.notna(fail_rate) and pd.notna(succ_rate) and succ_rate > 0) else np.nan

        rows.append(
            {
                "Task": task_name,
                "n_mixed_repeats": n_mixed,
                "n_failed_positions_mixed": n_fail,
                "n_failed_positions_mixed_at_boundary": n_fail_bd,
                "frac_failed_positions_mixed_at_boundary": fail_rate,
                "n_success_positions_mixed": n_succ,
                "n_success_positions_mixed_at_boundary": n_succ_bd,
                "frac_success_positions_mixed_at_boundary": succ_rate,
                "failure_vs_success_boundary_rate_ratio": ratio,
            }
        )

    return pd.DataFrame(rows)


def within_repeat_failure_boundary_fraction_by_task(
    synthetic_all: pd.DataFrame,
    identical_all: pd.DataFrame,
    approximate_all: pd.DataFrame,
) -> pd.DataFrame:
    """For each task, compute boundary fraction within each repeat, then average.

    Steps (eligible positions only):
      1) Per repeat key: frac_failed_at_boundary_within_repeat =
         (# failed+boundary) / (# failed)
      2) Average this fraction across repeats (unweighted mean).
    Also reports the weighted/global version for reference.
    """
    tasks: List[Tuple[str, pd.DataFrame, pd.Series]] = [
        ("Rand. Identical", synthetic_all, scope_mask(synthetic_all, "all")),
        ("Nat. Identical", identical_all, scope_mask(identical_all, "all")),
        ("Sub-Adjacent", approximate_all, scope_mask(approximate_all, "near_sub")),
        ("Indel-Adjacent", approximate_all, scope_mask(approximate_all, "near_indel")),
    ]

    rows = []
    for task_name, df, elig in tasks:
        eligible = df.loc[elig].copy()
        if eligible.empty:
            rows.append(
                {
                    "Task": task_name,
                    "n_repeats_with_failures": 0,
                    "avg_within_repeat_frac_failures_at_boundary": np.nan,
                    "median_within_repeat_frac_failures_at_boundary": np.nan,
                    "global_frac_failures_at_boundary": np.nan,
                }
            )
            continue

        per_repeat = (
            eligible.groupby(KEY_COLS, sort=False)
            .agg(
                n_failed=("is_correct", lambda s: int((s == False).sum())),
                n_failed_boundary=("is_repeat_boundary", lambda s: 0),  # placeholder
            )
            .reset_index()
        )

        # Compute failed-at-boundary using both columns.
        fb = (
            eligible.assign(_failed_boundary=(eligible["is_correct"] == False) & (eligible["is_repeat_boundary"] == True))
            .groupby(KEY_COLS, sort=False)
            .agg(
                n_failed=("is_correct", lambda s: int((s == False).sum())),
                n_failed_boundary=("_failed_boundary", "sum"),
            )
            .reset_index()
        )
        per_repeat = fb
        per_repeat = per_repeat[per_repeat["n_failed"] > 0].copy()

        if per_repeat.empty:
            rows.append(
                {
                    "Task": task_name,
                    "n_repeats_with_failures": 0,
                    "avg_within_repeat_frac_failures_at_boundary": np.nan,
                    "median_within_repeat_frac_failures_at_boundary": np.nan,
                    "global_frac_failures_at_boundary": np.nan,
                }
            )
            continue

        per_repeat["frac_failures_at_boundary_within_repeat"] = (
            per_repeat["n_failed_boundary"] / per_repeat["n_failed"]
        )

        avg_frac = float(per_repeat["frac_failures_at_boundary_within_repeat"].mean())
        med_frac = float(per_repeat["frac_failures_at_boundary_within_repeat"].median())
        global_frac = float(per_repeat["n_failed_boundary"].sum() / per_repeat["n_failed"].sum())

        rows.append(
            {
                "Task": task_name,
                "n_repeats_with_failures": int(len(per_repeat)),
                "avg_within_repeat_frac_failures_at_boundary": avg_frac,
                "median_within_repeat_frac_failures_at_boundary": med_frac,
                "global_frac_failures_at_boundary": global_frac,
            }
        )

    return pd.DataFrame(rows)


def _positions_list(df: pd.DataFrame, mask: pd.Series) -> Tuple[List[int], List[int]]:
    sub = df.loc[mask, ["masked_position", "is_correct"]]
    if sub.empty:
        return [], []
    success = sorted(sub.loc[sub["is_correct"] == True, "masked_position"].astype(int).unique().tolist())
    failure = sorted(sub.loc[sub["is_correct"] == False, "masked_position"].astype(int).unique().tolist())
    return success, failure


def _sample_one_per_repeat(df: pd.DataFrame, mask: pd.Series, seed: int = 42) -> pd.DataFrame:
    """Match OLD logic: for a task, sample one eligible row per repeat triplet."""
    eligible = df.loc[mask].copy()
    if eligible.empty:
        return eligible
    return (
        eligible.groupby(KEY_COLS, sort=False)
        .sample(n=1, random_state=seed)
        .reset_index(drop=True)
    )


def per_repeat_failure_summary(
    df: pd.DataFrame,
    task_scopes: Dict[str, str],
    extra_group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    extra_group_cols = extra_group_cols or []

    base_cols = KEY_COLS + extra_group_cols
    keep_cols = set(base_cols + ["masked_position", "is_correct"])
    missing = keep_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for summary: {sorted(missing)}")

    df = df.copy()
    grouped = df.groupby(base_cols, sort=False)

    rows: List[Dict[str, Any]] = []
    for key_vals, g in grouped:
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row: Dict[str, Any] = dict(zip(base_cols, key_vals))

        row["n_all_positions"] = int(len(g))
        row["correct_all_positions"] = int((g["is_correct"] == True).sum())
        row["accuracy_all_positions"] = float(g["is_correct"].mean()) if len(g) else np.nan

        all_success, all_failure = _positions_list(g, pd.Series(True, index=g.index))
        row["success_positions_all"] = all_success
        row["failure_positions_all"] = all_failure

        for task_name, scope in task_scopes.items():
            elig_mask = scope_mask(g, scope) if scope != "all" else pd.Series(True, index=g.index)
            elig = g.loc[elig_mask]
            row[f"n_eligible_{task_name}"] = int(len(elig))
            row[f"correct_eligible_{task_name}"] = int((elig["is_correct"] == True).sum()) if len(elig) else 0
            row[f"accuracy_eligible_{task_name}"] = float(elig["is_correct"].mean()) if len(elig) else np.nan

            succ, fail = _positions_list(g, elig_mask)
            row[f"success_positions_{task_name}"] = succ
            row[f"failure_positions_{task_name}"] = fail

        rows.append(row)

    return pd.DataFrame(rows)


def plot_repeat_length_bin_accuracy(
    df: pd.DataFrame,
    tasks: List[Tuple[str, pd.Series]],
    output_path: Path,
    n_per_bin: int = 1000,
):
    import matplotlib.pyplot as plt

    bins = [0, 10, 15, 20, 25, np.inf]
    labels = ["1-10", "11-15", "16-20", "21-25", "26+"]

    df = df.copy()
    df["repeat_bin"] = pd.cut(df["repeat_length"], bins=bins, labels=labels, right=True, include_lowest=True)
    df = df.dropna(subset=["repeat_bin"])

    summaries = []
    for task_name, mask in tasks:
        sub = df.loc[mask].copy()
        sub = sub.dropna(subset=["repeat_bin"])
        if sub.empty:
            continue
        grouped = sub.groupby("repeat_bin", sort=False)
        sampled = grouped.apply(lambda x: x.sample(n=min(len(x), n_per_bin), random_state=42)).reset_index(drop=True)
        summ = sampled.groupby("repeat_bin", sort=False).agg(
            accuracy=("is_correct", "mean"),
            n_samples=("is_correct", "size"),
        ).reset_index()
        summ["task"] = task_name
        summaries.append(summ)

    if not summaries:
        return

    plot_df = pd.concat(summaries, ignore_index=True)
    pivot = plot_df.pivot(index="repeat_bin", columns="task", values="accuracy")

    ax = pivot.plot(kind="bar", figsize=(8, 5), rot=0, edgecolor="black")
    ax.set_xlabel("Repeat Length Bin")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Task accuracy by repeat length bin")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.close()


def plot_accuracy_prob_heatmaps(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    sample_per_bin: int = 500,
    length_bins: Iterable[float] = (0, 10, 15, 20, 25, np.inf),
    length_labels: Iterable[str] = ("1-10", "11-15", "16-20", "21-25", "26+"),
    id_bins: Iterable[float] = (50, 75, 100),
    id_labels: Iterable[str] = ("50-75%", "75-100%"),
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = df.copy()
    df["length_bin"] = pd.cut(
        df["repeat_length"], bins=list(length_bins), labels=list(length_labels), right=True, include_lowest=True
    )
    df["identity_bin"] = pd.cut(
        df["identity_percentage"], bins=list(id_bins), labels=list(id_labels), right=True, include_lowest=True
    )
    df = df.dropna(subset=["length_bin", "identity_bin"])
    if df.empty:
        return

    grouped = df.groupby(["identity_bin", "length_bin"], sort=False)
    sampled = grouped.apply(lambda x: x.sample(min(len(x), sample_per_bin), random_state=42)).reset_index(drop=True)

    acc = sampled.pivot_table(index="identity_bin", columns="length_bin", values="is_correct", aggfunc="mean")
    prob = sampled.pivot_table(
        index="identity_bin", columns="length_bin", values="true_label_probability", aggfunc="mean"
    )
    acc_n = sampled.pivot_table(index="identity_bin", columns="length_bin", values="is_correct", aggfunc="count")
    prob_n = sampled.pivot_table(
        index="identity_bin", columns="length_bin", values="true_label_probability", aggfunc="count"
    )

    acc_annot = acc.round(2).astype(str) + "\n(n=" + acc_n.fillna(0).astype(int).astype(str) + ")"
    prob_annot = prob.round(2).astype(str) + "\n(n=" + prob_n.fillna(0).astype(int).astype(str) + ")"

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sns.heatmap(acc, annot=acc_annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "Accuracy"}, ax=axes[0])
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Repeat Length Bin")
    axes[0].set_ylabel("Identity Percentage Bin")

    sns.heatmap(prob, annot=prob_annot, fmt="", cmap="Purples", vmin=0, vmax=1, cbar_kws={"label": "Mean Probability"}, ax=axes[1])
    axes[1].set_title("Mean probability (correct token)")
    axes[1].set_xlabel("Repeat Length Bin")
    axes[1].set_ylabel("")

    fig.suptitle(title, fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"), dpi=300)
    plt.close(fig)


def main(args=None):
    parser = argparse.ArgumentParser(description="Analyze failures per task and per repeat")
    parser.add_argument("--synthetic_source", required=True)
    parser.add_argument("--synthetic_all", required=True)
    parser.add_argument("--identical_source", required=True)
    parser.add_argument("--identical_all", required=True)
    parser.add_argument("--approximate_source", required=True)
    parser.add_argument("--approximate_all", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_per_bin", type=int, default=500)
    args = parser.parse_args(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets (metadata)
    synthetic_src = pd.read_csv(args.synthetic_source)
    identical_src = pd.read_csv(args.identical_source)
    approximate_src = pd.read_csv(args.approximate_source)

    # Normalize repeat_locations strings
    for src in (synthetic_src, identical_src, approximate_src):
        if "repeat_locations" in src.columns:
            src["repeat_locations"] = src["repeat_locations"].apply(_parse_repeat_locations)

    # Ensure identity_percentage exists for synthetic (it is always identical by construction)
    if "identity_percentage" not in synthetic_src.columns:
        synthetic_src["identity_percentage"] = 100.0

    # Load per-position predictions
    synthetic_all = pd.read_csv(args.synthetic_all)
    identical_all = pd.read_csv(args.identical_all)
    approximate_all = pd.read_csv(args.approximate_all)

    def dedupe_meta(src_df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        missing_cols = [c for c in cols if c not in src_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required metadata columns: {missing_cols}")
        return src_df[cols].groupby(KEY_COLS, as_index=False, sort=False).first()

    # Add repeat-relative info by merging repeat_locations
    synthetic_all = synthetic_all.merge(
        dedupe_meta(synthetic_src, KEY_COLS + ["repeat_locations", "repeat_length", "identity_percentage"]),
        on=KEY_COLS,
        how="left",
    )
    identical_all = identical_all.merge(
        dedupe_meta(identical_src, KEY_COLS + ["repeat_locations", "repeat_length", "identity_percentage"]),
        on=KEY_COLS,
        how="left",
    )

    # Approximate: need alignments to compute eligibility masks (near_sub / near_indel)
    # repeat_locations already normalized to list[tuple[int,int]] above.
    if "repeat_alignments" in approximate_src.columns:
        approximate_src["repeat_alignments"] = parse_to_list_column(approximate_src["repeat_alignments"])
    approximate_all = add_approximate_repeat_columns(approximate_src, approximate_all)
    approximate_all = approximate_all.merge(
        dedupe_meta(
            approximate_src,
            KEY_COLS
            + [
                "repeat_locations",
                "repeat_length",
                "identity_percentage",
                "indels_count",
                "substitutions_count",
            ],
        ),
        on=KEY_COLS,
        how="left",
        suffixes=("", "_meta"),
    )

    # If repeat_locations got duplicated (e.g. repeat_locations + repeat_locations_meta), coalesce.
    if "repeat_locations" not in approximate_all.columns and "repeat_locations_meta" in approximate_all.columns:
        approximate_all = approximate_all.rename(columns={"repeat_locations_meta": "repeat_locations"})
    elif "repeat_locations_meta" in approximate_all.columns:
        approximate_all["repeat_locations"] = approximate_all["repeat_locations_meta"].where(
            approximate_all["repeat_locations_meta"].notna(), approximate_all["repeat_locations"]
        )
        approximate_all = approximate_all.drop(columns=["repeat_locations_meta"])

    # Add relative position columns (useful for debugging; also ensures repeat_locations parsed)
    synthetic_all = _add_repeat_relative_position(synthetic_all)
    identical_all = _add_repeat_relative_position(identical_all)
    approximate_all = _add_repeat_relative_position(approximate_all)
    synthetic_all = _add_repeat_boundary_flag(synthetic_all)
    identical_all = _add_repeat_boundary_flag(identical_all)
    approximate_all = _add_repeat_boundary_flag(approximate_all)

    boundary_summary = failure_boundary_summary_by_task(synthetic_all, identical_all, approximate_all)
    boundary_summary.to_csv(out_dir / "failure_boundary_by_task.csv", index=False)

    # Per-repeat approximate summary (needed for failure_outcome merge + CSV)
    approx_summary = per_repeat_failure_summary(
        approximate_all,
        task_scopes={"sub_adjacent": "near_sub", "indel_adjacent": "near_indel"},
        extra_group_cols=[
            "repeat_length",
            "identity_percentage",
            "indels_count",
            "substitutions_count",
        ],
    )
    approx_summary.to_csv(out_dir / "per_repeat_failures_approximate.csv", index=False)

    repeat_outcome_summary = failure_outcome_by_repeat_summary_by_task(
        synthetic_all,
        identical_all,
        approximate_all,
        approx_per_repeat=approx_summary,
        all_positions_acc_lt=0.3,
    )
    repeat_outcome_summary.to_csv(out_dir / "failure_outcome_by_repeat_by_task.csv", index=False)
    mixed_boundary_summary = mixed_repeat_boundary_comparison_by_task(synthetic_all, identical_all, approximate_all)
    mixed_boundary_summary.to_csv(out_dir / "mixed_repeats_boundary_fail_vs_success_by_task.csv", index=False)
    within_repeat_failure_frac = within_repeat_failure_boundary_fraction_by_task(
        synthetic_all, identical_all, approximate_all
    )
    within_repeat_failure_frac.to_csv(out_dir / "within_repeat_failure_boundary_fraction_by_task.csv", index=False)

    # ------------------------------------------------------------------
    # Task heatmaps + binned accuracy plots (approximate only)
    # ------------------------------------------------------------------
    approx_sub_mask = scope_mask(approximate_all, "near_sub")
    approx_indel_mask = scope_mask(approximate_all, "near_indel")
    # OLD parity: sample one eligible position per repeat before plotting.
    approx_sub_one = _sample_one_per_repeat(approximate_all, approx_sub_mask, seed=42)
    approx_indel_one = _sample_one_per_repeat(approximate_all, approx_indel_mask, seed=42)
    approx_plot_df = pd.concat(
        [
            approx_sub_one.assign(task_name="Sub-Adjacent"),
            approx_indel_one.assign(task_name="Indel-Adjacent"),
        ],
        ignore_index=True,
    )

    plot_repeat_length_bin_accuracy(
        approx_plot_df,
        tasks=[
            ("Sub-Adjacent", approx_plot_df["task_name"] == "Sub-Adjacent"),
            ("Indel-Adjacent", approx_plot_df["task_name"] == "Indel-Adjacent"),
        ],
        output_path=out_dir / "repeat_length_bin_accuracy_approximate",
    )

    # Heatmaps: (1) one eligible position per repeat — matches old script / Table-style view.
    plot_accuracy_prob_heatmaps(
        df=approx_sub_one,
        output_path=out_dir / "accuracy_prob_heatmaps_approximate_sub_adjacent",
        title="Sub-Adjacent task (one eligible position per repeat)",
        sample_per_bin=args.sample_per_bin,
    )
    plot_accuracy_prob_heatmaps(
        df=approx_indel_one,
        output_path=out_dir / "accuracy_prob_heatmaps_approximate_indel_adjacent",
        title="Indel-Adjacent task (one eligible position per repeat)",
        sample_per_bin=args.sample_per_bin,
    )
    # Heatmaps: (2) all eligible positions — no per-repeat downsampling before binning.
    plot_accuracy_prob_heatmaps(
        df=approximate_all.loc[approx_sub_mask],
        output_path=out_dir / "accuracy_prob_heatmaps_approximate_sub_adjacent_all_eligible",
        title="Sub-Adjacent task (all eligible positions)",
        sample_per_bin=args.sample_per_bin,
    )
    plot_accuracy_prob_heatmaps(
        df=approximate_all.loc[approx_indel_mask],
        output_path=out_dir / "accuracy_prob_heatmaps_approximate_indel_adjacent_all_eligible",
        title="Indel-Adjacent task (all eligible positions)",
        sample_per_bin=args.sample_per_bin,
    )

    print(f"Saved failure analysis artifacts to {out_dir}")


if __name__ == "__main__":
    main()

