"""Shared utilities for interactions_graph experiment: edge aggregation, path helpers, and activation-patching dataset."""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plms_repeats_circuits.EAP.graph import Graph, GraphType
from plms_repeats_circuits.EAP.circuit_selection import select_circuit_edges
from plms_repeats_circuits.utils.counterfactuals_config import COUNTERFACTUAL_METHODS
from plms_repeats_circuits.utils.model_utils import mask_protein
from plms_repeats_circuits.utils.esm_utils import replace_short_mask_with_mask_token

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
CLUSTER_NAMES_TO_LABELS: dict[str, str] = {
    "Input": LABELS[0],
    "Logits": LABELS[5],
    "MLP": LABELS[4],
    "Relative Position Heads": LABELS[2],
    "AA-biased Heads": LABELS[1],
    "Induction Heads": LABELS[3],
}


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


def cluster_edges_signed(edges_df: pd.DataFrame, aggregation: str = "mean") -> pd.DataFrame:
    """Add score_sign per edge, then aggregate by cluster1, cluster2, score_sign.
    aggregation: 'mean' or 'sum' for mean_raw values."""
    if aggregation not in ("mean", "sum"):
        raise ValueError(f"aggregation must be 'mean' or 'sum', got {aggregation!r}")
    edges_df = edges_df.copy()
    edges_df["score_sign"] = np.where(edges_df["mean_raw"] >= 0, "positive", "negative")
    result = (
        edges_df.groupby(["cluster1", "cluster2", "score_sign"], dropna=False)
        .agg(n_edges=("name", "count"), mean_raw=("mean_raw", aggregation))
        .reset_index()
    )
    return result


# ---- Plotting (reusable heatmap) ----

def plot_connection_heatmap_from_dict(
    weight_dict: dict[tuple[str, str], float],
    src_labels: list[str],
    dst_labels: list[str],
    *,
    colorscale: str = "Blues",
    take_abs: bool = False,
    decimal_places: int = 2,
) -> go.Figure:
    """Build a Plotly heatmap from (src, dst) -> value. Reusable for interaction maps."""
    connection_importance = [
        [
            abs(weight_dict.get((src, dst), 0)) if take_abs else weight_dict.get((src, dst), 0)
            for dst in dst_labels
        ]
        for src in src_labels
    ]
    fmt = f"{{:.{decimal_places}f}}"
    fig = go.Figure(
        data=go.Heatmap(
            z=connection_importance[::-1],
            x=dst_labels,
            y=src_labels[::-1],
            colorscale=colorscale,
            colorbar=dict(title="Importance"),
            showscale=False,
            text=[[fmt.format(v) for v in row] for row in connection_importance[::-1]],
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
                text=fmt.format(value),
                showarrow=False,
                font=dict(size=12, color=c),
            )
    return fig


# ---- Activation-patching dataset (no circuit_discovery import) ----

def create_evaluation_dataframe(
    df: pd.DataFrame,
    total_n_samples: int,
    random_state: int,
    tokenizer,
    metric: str,
) -> pd.DataFrame:
    if len(df) < total_n_samples:
        raise ValueError(f"Not enough samples: {len(df)} rows available, but {total_n_samples} requested.")
    if total_n_samples < len(df):
        sampled_df = df.sample(n=total_n_samples, random_state=random_state)
    else:
        sampled_df = df

    def process_row(row):
        clean = row["seq"]
        name = f"{row['cluster_id']}_{row['rep_id']}_{row['repeat_key']}"
        corrupted = row["corrupted_sequence"]
        masked_position = int(row["masked_position"])
        clean_masked = mask_protein(clean, masked_position, tokenizer)
        corrupted_masked = mask_protein(corrupted, masked_position, tokenizer)
        corrupted_masked = replace_short_mask_with_mask_token(corrupted_masked, tokenizer)
        labels = [clean[masked_position]]
        if metric == "logit_diff":
            if "corrupted_amino_acid_type" in df.columns:
                labels.append(row["corrupted_amino_acid_type"])
            elif "replacments" in df.columns:
                replacements = ast.literal_eval(row["replacments"])
                if len(replacements) != 1:
                    raise ValueError("got unexpected corrupted amino acids to logit diff metric")
                labels.append(replacements[0])
            else:
                raise ValueError("missing support for corrupted amino acid column")
        tokenized_labels = tokenizer(
            labels,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
        )["input_ids"].squeeze(-1).tolist()
        return pd.Series({
            "clean_masked": clean_masked,
            "corrupted_masked": corrupted_masked,
            "masked_position_after_tokenization": masked_position + 1,
            "tokenized_labels": tokenized_labels,
            "clean_id_names": name,
        })

    return sampled_df.apply(process_row, axis=1)


def collate_activation_patching(xs):
    clean, corrupted, positions, labels, clean_id_names = zip(*xs)
    return (
        list(clean),
        list(corrupted),
        list(positions),
        list(labels),
        list(clean_id_names),
    )


class ActivationPatchingDataset:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        if "clean_id_names" not in self.df.columns:
            raise ValueError("DataFrame must have 'clean_id_names' column.")
        return (
            row["clean_masked"],
            row["corrupted_masked"],
            row["masked_position_after_tokenization"],
            row["tokenized_labels"],
            row["clean_id_names"],
        )

    def to_dataloader(self, batch_size: int):
        from torch.utils.data import DataLoader
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_activation_patching)
