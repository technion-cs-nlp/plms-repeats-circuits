from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

sys.setrecursionlimit(3000)

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plms_repeats_circuits.EAP.graph import Graph, GraphType
from plms_repeats_circuits.EAP.circuit_selection import select_circuit_edges
from plms_repeats_circuits.utils.counterfactuals_config import COUNTERFACTUAL_METHODS


class EdgeInfo:
    """Aggregated edge (node1, node2) with cluster labels and scores across seeds."""

    def __init__(self, name: str, node1: str, node2: str, cluster1: str, cluster2: str):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.cluster1 = cluster1
        self.cluster2 = cluster2
        self.raw_scores: list[float] = []
        self.in_graph_vals: list[int] = []
        self.in_graph: bool | None = None

    @property
    def mean_raw(self) -> float:
        return float(np.mean(self.raw_scores)) if self.raw_scores else np.nan

    def raw_scores_have_same_sign(self):
        if len(self.raw_scores) == 0:
            return True
        signs = np.sign(self.raw_scores)
        return len(np.unique(signs)) == 1

    def decide_in_graph(self, ratio=0.8):
        if len(self.in_graph_vals) == 0:
            return False
        if not self.raw_scores_have_same_sign():
            return False
        return np.mean(self.in_graph_vals) > ratio

    def add_score(self, score: float, raw_score: float) -> None:
        self.raw_scores.append(raw_score)

    def add_in_graph(self, in_graph) -> None:
        val = 1 if in_graph == True else 0
        self.in_graph_vals.append(val)


REPEAT_TYPE_PATHS = {"identical": "identical", "approximate": "approximate", "synthetic": "synthetic"}


def _output_dir(results_root: Path, repeat_type: str, model_type: str, graph_type: str, seed: int) -> Path:
    folder = REPEAT_TYPE_PATHS.get(repeat_type, repeat_type)
    return results_root / "circuit_discovery" / folder / model_type / graph_type / f"seed_{seed}"


def _exp_prefix(repeat_type: str, counterfactual_type: str) -> str:
    return f"{repeat_type}_counterfactual_{counterfactual_type}"


def _find_edge_json(output_dir: Path, exp_prefix: str, seed: int) -> Optional[tuple[str, Path]]:
    """Return (circuit_id, json_path) for the single matching circuit JSON. None if not found. Raises if multiple."""
    matches = []
    for p in output_dir.glob(f"{exp_prefix}_circuit*.json"):
        m = re.search(r"_circuit([a-f0-9-]+)_", p.stem)
        if m:
            matches.append((m.group(1), p))
    if len(matches) > 1:
        raise ValueError(f"Expected single circuit JSON for {exp_prefix} in {output_dir}, found {len(matches)}: {[p.name for _, p in matches]}")
    return matches[0] if matches else None


def _augment_clustering_with_mlp_input_logits(
    clustering_df: pd.DataFrame,
    results_root: Path,
    repeat_type: str,
    model_type: str,
    counterfactual_type: str,
    mlp_ratio_threshold: float = 0.8,
) -> pd.DataFrame:
    df = clustering_df.copy()
    if "cluster" not in df.columns:
        raise ValueError(f"Clustering CSV must have 'cluster', got: {list(df.columns)}")

    def _append_row(nid: str, clabel: str, layer_val: int = -1) -> None:
        nonlocal df
        if nid in df["node_name"].values:
            return
        row = {"node_name": nid, "cluster": clabel}
        if "layer" in df.columns:
            row["layer"] = layer_val
        if "head" in df.columns:
            row["head"] = -1
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    comp_path = results_root / "component_recurrence" / model_type / counterfactual_type / f"nodes_recurrence_{repeat_type}.csv"
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        ratio_col = f"{repeat_type}_ratio_in_graph"
        if ratio_col not in comp_df.columns or "component_type" not in comp_df.columns:
            raise ValueError(f"Component recurrence CSV must have '{ratio_col}' and 'component_type', got: {list(comp_df.columns)}")
        else:
            mlps = comp_df[
                (comp_df["component_type"] == "mlp")
                & (comp_df[ratio_col] >= mlp_ratio_threshold)
            ]
            for _, row in mlps.iterrows():
                _append_row(row["component_id"], "MLP", int(row.get("layer", -1)))

    for node_name, cluster_name in [("input", "Input"), ("logits", "Logits")]:
        _append_row(node_name, cluster_name)


    clusters_to_drop = [
        name for name, g in df.groupby("cluster")
        if name not in ("Input", "Logits") and len(g) == 1
    ]
    if clusters_to_drop:
        print(f"Dropping single-node clusters: {clusters_to_drop}")
        df = df[~df["cluster"].isin(clusters_to_drop)].reset_index(drop=True)

    return df


def load_edges_and_aggregate(
    results_root: Path,
    repeat_type: str,
    model_type: str,
    counterfactual_type: str,
    clustering_path: Path,
    seeds: list[int],
    in_graph_ratio: float = 0.8,
) -> list[EdgeInfo]:

    valid_methods = {m["name"] for m in COUNTERFACTUAL_METHODS}
    if counterfactual_type not in valid_methods:
        raise ValueError(f"counterfactual_type '{counterfactual_type}' not in COUNTERFACTUAL_METHODS. Valid: {sorted(valid_methods)}")
    clustering_df = pd.read_csv(clustering_path)
    clustering_df = _augment_clustering_with_mlp_input_logits(
        clustering_df, results_root, repeat_type, model_type, counterfactual_type
    )
    node_to_cluster = dict(zip(clustering_df["node_name"], clustering_df["cluster"]))
    clustered_nodes = set(node_to_cluster.keys())
    aggregated: dict[str, EdgeInfo] = {}

    for seed in seeds:
        output_dir = _output_dir(results_root, repeat_type, model_type, "edges", seed)
        exp_prefix = _exp_prefix(repeat_type, counterfactual_type)
        match = _find_edge_json(output_dir, exp_prefix, seed)
        if match is None:
            continue
        circuit_id, json_path = match
        info_path = output_dir / "circuit_info_edges.csv"
        if not info_path.exists():
            continue
        info_df = pd.read_csv(info_path)
        info_row = info_df[info_df["circuit_id"] == circuit_id]
        if info_row.empty:
            continue
        row = info_row.iloc[0]
        n_e_val = row.get("n_edges", row.get("real_n_edges", np.nan))
        if pd.isna(n_e_val):
            continue
        n_e = int(float(n_e_val))
        sel_method = str(row.get("circuit_selection_method", "greedy_abs")).strip()
        try:
            graph = Graph.from_json(str(json_path))
        except Exception:
            continue
        if graph.graph_type != GraphType.Edges:
            continue
        select_circuit_edges(graph, selection_method=sel_method, n_edges=n_e, log=None)
        for edge in graph.edges.values():
            pname, cname = edge.parent.name, edge.child.name
            if pname not in clustered_nodes or cname not in clustered_nodes:
                continue
            key = edge.name
            c1 = node_to_cluster[pname]
            c2 = node_to_cluster[cname]
            raw_score = float(edge.score) if edge.score is not None else np.nan
            in_graph_val = bool(edge.in_graph) if edge.in_graph is not None else False
            if key not in aggregated:
                aggregated[key] = EdgeInfo(name=key, node1=pname, node2=cname, cluster1=c1, cluster2=c2)
            aggregated[key].add_score(raw_score, raw_score)
            aggregated[key].add_in_graph(in_graph_val)

    for ei in aggregated.values():
        ei.in_graph = ei.decide_in_graph(ratio=in_graph_ratio)

    return list(aggregated.values())


def cluster_edges_signed(edges_df: pd.DataFrame) -> pd.DataFrame:
    """Add score_sign per edge, then aggregate by cluster1, cluster2, score_sign."""
    edges_df = edges_df.copy()
    edges_df["score_sign"] = np.where(edges_df["mean_raw"] >= 0, "positive", "negative")
    result = (
        edges_df.groupby(["cluster1", "cluster2", "score_sign"], dropna=False)
        .agg(n_edges=("name", "count"), mean_raw=("mean_raw", "mean"))
        .reset_index()
    )
    return result


# Display labels and mapping from cluster names (from clustering_results) to plot labels.
LABELS = [
    "Input<br>Embeddings",
    "AA-biased<br>heads",
    "Rel. Pos.<br>heads",
    "Induction<br>heads",
    "MLPs",
    "Output<br>logits",
]
DST_LABELS = LABELS[1:]
SRC_LABELS = LABELS[:-1]

# Map cluster names (from data) to display labels. Handles variants from clustering_results.
CLUSTER_NAMES_TO_LABELS: dict[str, str] = {
    "Input": LABELS[0],
    "Logits": LABELS[5],
    "MLP": LABELS[4],
    "Relative Position Heads": LABELS[2],
    "AA-biased Heads": LABELS[1],
    "Induction Heads": LABELS[3],
}


def _plot_connection_heatmap_from_dict(
    weight_dict: dict[tuple[str, str], float],
    src_labels: list[str],
    dst_labels: list[str],
    *,
    colorscale: str = "Blues",
    take_abs: bool = False,
) -> go.Figure:
    connection_importance = [
        [
            abs(weight_dict.get((src, dst), 0)) if take_abs else weight_dict.get((src, dst), 0)
            for dst in dst_labels
        ]
        for src in src_labels
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=connection_importance[::-1],
            x=dst_labels,
            y=src_labels[::-1],
            colorscale=colorscale,
            colorbar=dict(title="Importance"),
            showscale=False,
            text=[[f"{v:.2f}" for v in row] for row in connection_importance[::-1]],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        autosize=False,
        title=None,
        xaxis=dict(title=dict(text="Dst components", font=dict(size=16))),
        yaxis=dict(title=dict(text="Src components", font=dict(size=16))),
        width=550,
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_xaxes(tickfont=dict(size=14))
    fig.update_yaxes(tickfont=dict(size=14))
    max_val = max(map(max, connection_importance)) if connection_importance else 0
    for i, row in enumerate(connection_importance[::-1]):
        for j, value in enumerate(row):
            c = "white" if max_val > 0 and value > max_val / 2 else "black"
            fig.add_annotation(
                x=dst_labels[j],
                y=src_labels[::-1][i],
                text=f"{value:.2f}",
                showarrow=False,
                font=dict(size=12, color=c),
            )
    return fig


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
    fig_pos = _plot_connection_heatmap_from_dict(
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
    fig_neg = _plot_connection_heatmap_from_dict(
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
