"""Approximate-repeat failure analysis: Sub-Adjacent and Indel-Adjacent heatmaps only."""

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils import (
    KEY_COLS,
    add_approximate_repeat_columns,
    parse_to_list_column,
    scope_mask,
)


def plot_accuracy_prob_heatmaps(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    sample_per_bin: int = 500,
    length_bins: Iterable[float] = (0, 10, 15, 20, 25, float("inf")),
    length_labels: Iterable[str] = ("1-10", "11-15", "16-20", "21-25", "26+"),
):
    """Bin on mutation % = 100 - identity %.

    [0, 25) mutation   <=>  (75, 100] identity
    [25, 50] mutation  <=>  (50, 75]  identity  (includes 25 and 50)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = df.copy()
    df["length_bin"] = pd.cut(
        df["repeat_length"], bins=list(length_bins), labels=list(length_labels), right=True, include_lowest=True
    )
    m = pd.to_numeric(df["mutation_percentage"], errors="coerce")
    df["mutation_bin"] = np.select(
        [m < 25, (m >= 25) & (m <= 50)],
        ["0-25%", "25-50%"],
        default=pd.NA,
    )
    df["mutation_bin"] = pd.Categorical(df["mutation_bin"], categories=["0-25%", "25-50%"], ordered=True)
    df = df.dropna(subset=["length_bin", "mutation_bin"])
    if df.empty:
        return

    grouped = df.groupby(["mutation_bin", "length_bin"], sort=False)
    sampled = grouped.apply(lambda x: x.sample(min(len(x), sample_per_bin), random_state=42)).reset_index(drop=True)

    acc = sampled.pivot_table(index="mutation_bin", columns="length_bin", values="is_correct", aggfunc="mean")
    prob = sampled.pivot_table(
        index="mutation_bin", columns="length_bin", values="true_label_probability", aggfunc="mean"
    )
    acc_n = sampled.pivot_table(index="mutation_bin", columns="length_bin", values="is_correct", aggfunc="count")
    prob_n = sampled.pivot_table(
        index="mutation_bin", columns="length_bin", values="true_label_probability", aggfunc="count"
    )

    acc_annot = acc.round(2).astype(str) + "\n(n=" + acc_n.fillna(0).astype(int).astype(str) + ")"
    prob_annot = prob.round(2).astype(str) + "\n(n=" + prob_n.fillna(0).astype(int).astype(str) + ")"

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sns.heatmap(acc, annot=acc_annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "Accuracy"}, ax=axes[0])
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Repeat Length Bin")
    axes[0].set_ylabel("Mutation % (100 − identity)")

    sns.heatmap(
        prob, annot=prob_annot, fmt="", cmap="Purples", vmin=0, vmax=1,
        cbar_kws={"label": "Mean Probability"}, ax=axes[1],
    )
    axes[1].set_title("Mean probability (correct token)")
    axes[1].set_xlabel("Repeat Length Bin")
    axes[1].set_ylabel("")

    fig.suptitle(title, fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"), dpi=300)
    plt.close(fig)


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Plot approximate-task accuracy/probability heatmaps (all eligible positions)"
    )
    parser.add_argument(
        "--eval_dataset",
        required=True,
        help="Path to approximate evaluation dataset CSV (e.g. datasets/approximate/evaluation/"
        "approximate_repeats_eval.csv). Provides seq, alignments, repeat_length, identity %%, indels_count.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to approximate per-position predictions from evaluate (e.g. "
        "results/evaluation/<model>/approximate/predictions.csv).",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for heatmap PNG/PDF outputs")
    parser.add_argument(
        "--sample_per_bin",
        type=int,
        default=500,
        help="Max rows sampled per repeat-length x mutation %% cell before averaging",
    )
    args = parser.parse_args(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_csv(args.eval_dataset)
    predictions_df = pd.read_csv(args.predictions)

    for col in ("repeat_locations", "repeat_alignments"):
        if col in eval_df.columns:
            eval_df[col] = parse_to_list_column(eval_df[col])

    predictions_df = add_approximate_repeat_columns(eval_df, predictions_df)
    meta_cols = KEY_COLS + ["repeat_length", "identity_percentage", "indels_count", "substitutions_count"]
    meta = eval_df[meta_cols].groupby(KEY_COLS, as_index=False, sort=False).first()
    for col in meta_cols:
        if col in KEY_COLS:
            continue
        if col in predictions_df.columns:
            predictions_df = predictions_df.drop(columns=[col])
    predictions_df = predictions_df.merge(meta, on=KEY_COLS, how="left")
    predictions_df["mutation_percentage"] = 100.0 - predictions_df["identity_percentage"]

    approx_sub_mask = scope_mask(predictions_df, "near_sub")
    approx_indel_mask = scope_mask(predictions_df, "near_indel")

    plot_accuracy_prob_heatmaps(
        df=predictions_df.loc[approx_sub_mask],
        output_path=out_dir / "accuracy_prob_heatmaps_approximate_sub_adjacent_all_eligible",
        title="Sub-Adjacent task (all eligible positions)",
        sample_per_bin=args.sample_per_bin,
    )
    plot_accuracy_prob_heatmaps(
        df=predictions_df.loc[approx_indel_mask],
        output_path=out_dir / "accuracy_prob_heatmaps_approximate_indel_adjacent_all_eligible",
        title="Indel-Adjacent task (all eligible positions)",
        sample_per_bin=args.sample_per_bin,
    )

    print(f"Saved heatmaps to {out_dir}")


if __name__ == "__main__":
    main()
