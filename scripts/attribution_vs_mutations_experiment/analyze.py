"""Analysis for attribution vs mutations: induction heads vs substitution rate."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import matplotlib as mpl
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

from utils import nodes_csv_path

try:
    pio.kaleido.scope.mathjax = None
except Exception:
    pass

mpl.rcParams["font.size"] = 10
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["pdf.fonttype"] = 42


def _load_attention_head_clusters(clustering_csv: Path) -> Dict[str, str]:
    df = pd.read_csv(clustering_csv)
    return dict(zip(df["node_name"], df["cluster"]))


def _filter_split_by_sign(
    df: pd.DataFrame, recurrence_csv: Path, repeat_type: str
) -> pd.DataFrame:
    recurrence_df = pd.read_csv(recurrence_csv)
    mean_col = f"{repeat_type}_mean_score"
    if mean_col not in recurrence_df.columns or "component_id" not in recurrence_df.columns:
        raise ValueError(
            f"Recurrence CSV must have component_id and {mean_col}: {recurrence_csv}"
        )

    sign_map: Dict[str, str] = {}
    for _, row in recurrence_df.iterrows():
        comp_id = row["component_id"]
        score = row[mean_col]
        if score > 0:
            sign_map[comp_id] = "positive"
        elif score < 0:
            sign_map[comp_id] = "negative"
        else:
            sign_map[comp_id] = "zero"

    df = df.copy()
    df["sign"] = df["component_id"].map(sign_map)
    return df[df["sign"].notna()].copy()


def _load_nodes_data(nodes_csv: Path, head_to_cluster: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(nodes_csv)
    required_cols = ["n_mutations", "component_id", "component_type", "score"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {nodes_csv}: {missing}")

    def assign_cluster(row):
        if row["component_type"] == "attention":
            return head_to_cluster.get(row["component_id"])
        if row["component_type"] == "mlp":
            return "MLP"
        return None

    df["cluster"] = df.apply(assign_cluster, axis=1)
    return df[df["cluster"].notna()].copy()


def _aggregate_nodes_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
    if "sign" in df.columns:
        aggregated = (
            df.groupby(["cluster", "n_mutations", "sign"])["score"].mean().reset_index()
        )
        aggregated.columns = ["cluster", "n_mutations", "sign", "avg_score"]
    else:
        aggregated = df.groupby(["cluster", "n_mutations"])["score"].mean().reset_index()
        aggregated.columns = ["cluster", "n_mutations", "avg_score"]
    return aggregated


def _plot_induction_heads_stress_test(
    nodes_data_dict: Dict[int, pd.DataFrame],
    output_dir: Path,
    *,
    label_fontsize: int = 12,
    tick_fontsize: int = 12,
    legend_fontsize: int = 12,
    line_width: float = 1.5,
) -> None:
    """Induction Heads only; positive-sign rows when sign column exists."""
    repeat_lengths = sorted(nodes_data_dict.keys())
    if not repeat_lengths:
        raise ValueError("No repeat-length data to plot")

    palette = sns.color_palette("colorblind", len(repeat_lengths))
    length_colors = [mpl.colors.rgb2hex(tuple(c)) for c in palette]
    fig = go.Figure()
    n_traces = 0

    for idx, repeat_len in enumerate(repeat_lengths):
        df = nodes_data_dict[repeat_len]
        induction_df = df[df["cluster"] == "Induction Heads"]
        if "sign" in induction_df.columns:
            induction_df = induction_df[induction_df["sign"] == "positive"].copy()
        if induction_df.empty:
            continue

        plot_df = induction_df.sort_values("n_mutations")
        x_vals = (plot_df["n_mutations"] / float(repeat_len)).astype(float).tolist()
        y_vals = plot_df["avg_score"].astype(float).tolist()

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name=f"Len {repeat_len}",
                line=dict(color=length_colors[idx], width=float(line_width)),
                showlegend=True,
            )
        )
        n_traces += 1

    if n_traces == 0:
        raise ValueError("No Induction Heads traces to plot")

    axis_line = dict(showline=True, linewidth=0.6, linecolor="#A0A0A8", mirror=False)
    grid = dict(showgrid=True, gridcolor="white", gridwidth=1)
    xaxis = dict(
        title=dict(text="Substitution Rate", font=dict(size=label_fontsize), standoff=2),
        tickfont=dict(size=tick_fontsize),
        zeroline=False,
        automargin=True,
        range=[0.0, 1.0],
        tickmode="array",
        tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ticktext=["0", "0.2", "0.4", "0.6", "0.8", "1"],
        **grid,
        **axis_line,
    )
    yaxis = dict(
        title=dict(text="Attribution", font=dict(size=label_fontsize), standoff=4),
        tickfont=dict(size=tick_fontsize),
        zeroline=False,
        automargin=True,
        **grid,
        **axis_line,
    )
    fig.update_layout(
        template=None,
        font=dict(size=12),
        plot_bgcolor="#E2EAF2",
        paper_bgcolor="white",
        width=350,
        height=200,
        margin=dict(l=4, r=4, t=4, b=4, pad=0),
        xaxis=xaxis,
        yaxis=yaxis,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            xref="paper",
            yref="paper",
            font=dict(size=legend_fontsize),
            bgcolor="rgba(255,255,255,0)",
            itemwidth=30,
        ),
        hovermode="x unified",
    )

    os.makedirs(output_dir, exist_ok=True)
    base = output_dir / "induction_heads_stress_test"
    fig.write_image(str(base.with_suffix(".png")), scale=2)
    fig.write_image(str(base.with_suffix(".pdf")))


def run_analyze(
    results_root: Path,
    model_type: str,
    counterfactual_type: str,
    component_recurrence_repeat_type: str,
    output_dir: Path,
    repeat_lengths: list[int],
) -> None:
    """Build induction_heads_stress_test plot from nodes CSVs."""
    results_root = Path(results_root)
    output_dir = Path(output_dir)

    clustering_path = (
        results_root
        / "attention_heads_clustering"
        / model_type
        / counterfactual_type
        / "clustering_results.csv"
    )
    if not clustering_path.exists():
        raise FileNotFoundError(f"Attention clustering not found: {clustering_path}")

    recurrence_path = (
        results_root
        / "component_recurrence"
        / model_type
        / counterfactual_type
        / f"nodes_recurrence_{component_recurrence_repeat_type}.csv"
    )
    if not recurrence_path.exists():
        raise FileNotFoundError(f"Component recurrence not found: {recurrence_path}")

    head_to_cluster = _load_attention_head_clusters(clustering_path)
    nodes_data_dict: Dict[int, pd.DataFrame] = {}

    for repeat_length in repeat_lengths:
        csv_path = nodes_csv_path(output_dir, model_type, repeat_length)
        if not csv_path.exists():
            raise FileNotFoundError(f"Nodes CSV not found: {csv_path}. Run compute step first.")

        df = _load_nodes_data(csv_path, head_to_cluster)
        df = _filter_split_by_sign(df, recurrence_path, component_recurrence_repeat_type)
        nodes_data_dict[repeat_length] = _aggregate_nodes_by_cluster(df)

    _plot_induction_heads_stress_test(nodes_data_dict, output_dir / "plots")
