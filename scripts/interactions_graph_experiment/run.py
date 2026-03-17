from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.setrecursionlimit(3000)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plms_repeats_circuits.utils.counterfactuals_config import COUNTERFACTUAL_METHODS

from utils import (
    load_edges_and_aggregate,
    cluster_edges_signed,
    plot_connection_heatmap_from_dict,
    CLUSTER_NAMES_TO_LABELS,
    SRC_LABELS,
    DST_LABELS,
)


def _plot_signed_heatmaps(cluster_edges_df: pd.DataFrame, output_dir: Path) -> None:
    to_save = cluster_edges_df[["cluster1", "cluster2", "score_sign", "n_edges", "mean_raw"]].copy()
    to_save["mean_raw_scaled"] = to_save["mean_raw"] * 1000

    pos = to_save[to_save["score_sign"] == "positive"].copy()
    neg = to_save[to_save["score_sign"] == "negative"].copy()

    for df in (pos, neg):
        df["src_label"] = df["cluster1"].map(CLUSTER_NAMES_TO_LABELS)
        df["dst_label"] = df["cluster2"].map(CLUSTER_NAMES_TO_LABELS)
        assert df["src_label"].notna().all(), f"Unmapped cluster in cluster1: {df['cluster1'].unique()}"
        assert df["dst_label"].notna().all(), f"Unmapped cluster in cluster2: {df['cluster2'].unique()}"

    src_to_dst_edge_pos_weights = {
        (row.src_label, row.dst_label): row.mean_raw_scaled for row in pos.itertuples(index=False)
    }
    src_to_dst_edge_neg_weights = {
        (row.src_label, row.dst_label): row.mean_raw_scaled for row in neg.itertuples(index=False)
    }

    # positive
    fig_pos = plot_connection_heatmap_from_dict(
        src_to_dst_edge_pos_weights,
        SRC_LABELS,
        DST_LABELS,
        colorscale="Blues",
        take_abs=False,
    )
    path = output_dir
    fig_pos.write_image(str(path / "interactions_map.pdf"))
    fig_pos.write_image(str(path / "interactions_map.png"))

    # negative
    fig_neg = plot_connection_heatmap_from_dict(
        src_to_dst_edge_neg_weights,
        SRC_LABELS,
        DST_LABELS,
        colorscale="Reds",
        take_abs=True,
    )
    fig_neg.write_image(str(path / "interactions_map_neg.pdf"))
    fig_neg.write_image(str(path / "interactions_map_neg.png"))


def _run_for_model(
    results_root: Path,
    repeat_type: str,
    model_type: str,
    counterfactual_type: str,
    seeds: list[int],
    in_graph_ratio: float,
) -> None:
    """Run interactions graph for a single model type."""
    clustering_path = results_root / "attention_heads_clustering" / model_type / counterfactual_type / "clustering_results.csv"
    if not clustering_path.exists():
        raise FileNotFoundError(f"Clustering results not found: {clustering_path}. Run attention_heads_clustering first.")

    output_dir = results_root / "interactions_graph" / model_type / counterfactual_type
    output_dir.mkdir(parents=True, exist_ok=True)

    edges = load_edges_and_aggregate(
        results_root=results_root,
        repeat_type=repeat_type,
        model_type=model_type,
        counterfactual_type=counterfactual_type,
        clustering_path=clustering_path,
        seeds=seeds,
        in_graph_ratio=in_graph_ratio,
    )

    edges_df = pd.DataFrame([
        {"name": e.name, "node1": e.node1, "node2": e.node2, "cluster1": e.cluster1, "cluster2": e.cluster2, "mean_raw": e.mean_raw, "in_graph": e.in_graph}
        for e in edges
    ])
    before_filter_count = len(edges_df)
    edges_df = edges_df[edges_df["in_graph"] == True]
    if edges_df.empty:
        print(f"[{model_type}] No edges found. Ensure circuit_discovery was run with graph_type=edges.")
        return
    after_filter_count = len(edges_df)
    print(f"[{model_type}] Filtered {before_filter_count - after_filter_count} edges out of {before_filter_count}")
    cluster_edges_df = cluster_edges_signed(edges_df)
    cluster_edges_df.to_csv(output_dir / "cluster_edges_signed.csv", index=False)
    _plot_signed_heatmaps(cluster_edges_df, output_dir)
    print(f"[{model_type}] Output saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactions graph: aggregate circuit edges by cluster, plot heatmaps.")
    parser.add_argument("--repeat_type", type=str, required=True, choices=["identical", "approximate", "synthetic"])
    parser.add_argument(
        "--model_types",
        type=str,
        nargs="+",
        choices=["esm3", "esm-c"],
        default=["esm3", "esm-c"],
        help="Model types to run (default: esm3 esm-c)",
    )
    parser.add_argument("--counterfactual_type", type=str, required=True, choices=sorted(m["name"] for m in COUNTERFACTUAL_METHODS))
    parser.add_argument("--results_root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--in_graph_ratio", type=float, default=0.8, help="Min ratio of seeds where edge must be in_graph (default: 0.8)")
    args = parser.parse_args()

    results_root = args.results_root.resolve()

    for model_type in args.model_types:
        _run_for_model(
            results_root=results_root,
            repeat_type=args.repeat_type,
            model_type=model_type,
            counterfactual_type=args.counterfactual_type,
            seeds=args.seeds,
            in_graph_ratio=args.in_graph_ratio,
        )


if __name__ == "__main__":
    sys.setrecursionlimit(3000)
    main()
