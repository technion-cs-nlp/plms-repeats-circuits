"""Logit lens analysis: process and plot logit lens across layers and cluster plot."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plms_repeats_circuits.utils.visualization_utils import get_colorblind_palette


def _get_color(cluster_name: str, palette: list[str]) -> str:
    if cluster_name == "Residual":
        return palette[3]
    elif cluster_name == "AA-biased Heads":
        return palette[0]
    elif cluster_name == "Induction Heads":
        return palette[1]
    elif cluster_name == "MLP":
        return palette[4]
    elif cluster_name == "Relative Position Heads":
        return palette[2]
    return palette[7 % len(palette)]


def _get_specific_colors_map(clusters):
    palette = get_colorblind_palette()
    return {c: _get_color(c, palette) for c in sorted(clusters)}


def _get_formal_names_map(cluster_name: str) -> str:
    if cluster_name == "Residual":
        return "Residual"
    elif cluster_name == "AA-biased Heads":
        return "AA-biased heads"
    elif cluster_name == "Induction Heads":
        return "Induction heads"
    elif cluster_name == "MLP":
        return "MLPs"
    elif cluster_name == "Relative Position Heads":
        return "Rel. Pos. heads"
    return cluster_name


def _process_component_type(
    df: pd.DataFrame,
    component_type: str,
    clustering_df: pd.DataFrame | None,
    default_cluster: str | None = None,
) -> pd.DataFrame:
    """Process one component type (resid_post, attention, or mlp)."""
    comp_df = df[df["component_type"] == component_type].copy()
    grouped = comp_df.groupby("component_id")
    metric_cols = ["top1", "correct_prob"]
    avg = grouped[metric_cols].mean()
    row_counts = grouped.size().rename("num_rows")
    layers = grouped["layer"].first()
    summary = avg.join(row_counts).join(layers).reset_index()
    if default_cluster is not None:
        summary["cluster"] = default_cluster
    elif clustering_df is not None:
        summary = summary.merge(
            clustering_df[["node_name", "cluster"]],
            left_on="component_id",
            right_on="node_name",
            how="inner",
        ).drop(columns=["node_name"])
    else:
        summary["cluster"] = "Other"
    grouped2 = summary.groupby(["layer", "cluster"])
    layer_cluster = grouped2[metric_cols].mean().join(
        grouped2.size().rename("num_components")
    ).reset_index()
    return layer_cluster


def _process_full_analysis(
    df: pd.DataFrame,
    clustering_df: pd.DataFrame | None,
) -> pd.DataFrame:
    plot_cols = ["layer", "cluster", "top1", "correct_prob"]
    attention = _process_component_type(df, "attention", clustering_df)
    mlp = _process_component_type(df, "mlp", None, default_cluster="MLP")
    residual = _process_component_type(df, "resid_post", None, default_cluster="Residual")
    return pd.concat([attention[plot_cols], mlp[plot_cols], residual[plot_cols]], ignore_index=True)


def _plot_logit_lens_across_layers_and_cluster(
    plot_df: pd.DataFrame,
    metrics: list[str],
    output_suffix: str,
    output_dir: Path,
    metric_labels: dict | None = None,
    font_size: int = 12,
    width: float = 450 / 1.2,
    height: float = 250 / 1.2,
    legend_title: str | None = None,
) -> None:
    """Plot logit lens across layers and cluster (paper figures 8, 17)."""
    if metric_labels is None:
        metric_labels = {"top1": "Top-1 Acc", "correct_prob": "P(correct)"}
    plot_df = plot_df.sort_values(["cluster", "layer"])
    color_map = _get_specific_colors_map(plot_df["cluster"].unique())
    n_metrics = len(metrics)
    if n_metrics == 1:
        fig = make_subplots(rows=1, cols=1)
    else:
        titles = [metric_labels.get(m, m) for m in metrics]
        fig = make_subplots(rows=1, cols=n_metrics, subplot_titles=titles)
    for cluster_id, g in plot_df.groupby("cluster"):
        color = color_map[cluster_id]
        cluster_name = _get_formal_names_map(cluster_id)
        for i, metric in enumerate(metrics):
            col = 1 if n_metrics == 1 else (i + 1)
            fig.add_trace(
                go.Scatter(
                    x=g["layer"],
                    y=g[metric],
                    mode="lines+markers",
                    name=cluster_name,
                    line=dict(color=color),
                    marker=dict(color=color),
                    showlegend=(i == 0),
                ),
                row=1,
                col=col,
            )
    legend = dict(
        title_text=legend_title or "",
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=font_size),
    )
    fig.update_layout(
        width=width,
        height=height,
        hovermode="x unified",
        font=dict(size=font_size),
        legend=legend,
        margin=dict(l=0.5, r=0.5, t=0.5, b=0.5),
    )
    for i, metric in enumerate(metrics):
        col = 1 if n_metrics == 1 else (i + 1)
        fig.update_xaxes(
            title_text="Layer",
            row=1,
            col=col,
            title_font=dict(size=font_size),
            tickfont=dict(size=font_size),
        )
        fig.update_yaxes(
            title_text=metric_labels.get(metric, metric),
            row=1,
            col=col,
            title_font=dict(size=font_size),
            tickfont=dict(size=font_size),
        )
    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(str(output_dir / f"cluster_metrics{output_suffix}.png"), scale=2)
    fig.write_image(str(output_dir / f"cluster_metrics{output_suffix}.pdf"))


def run_analyze(
    results_root: Path,
    model_type: str,
    counterfactual_type: str,
    repeat_type: str,
    output_dir: Path,
    ratio_threshold: float = 0.8,
) -> None:
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    clustering_path = (
        results_root
        / "attention_heads_clustering"
        / model_type
        / counterfactual_type
        / "clustering_results.csv"
    )
    clustering_df = pd.read_csv(clustering_path) if clustering_path.exists() else None

    csv_path = output_dir / f"logit_lens_{repeat_type}_ratio{ratio_threshold}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Logit lens CSV not found: {csv_path}. Run run_logit_lens first.")

    df = pd.read_csv(csv_path)
    plot_df = _process_full_analysis(df, clustering_df)
    _plot_logit_lens_across_layers_and_cluster(
        plot_df,
        metrics=["correct_prob"],
        output_suffix=f"_{repeat_type}",
        output_dir=output_dir,
        width=450 / 1.2,
        height=250 / 1.2,
        legend_title=None,
    )
