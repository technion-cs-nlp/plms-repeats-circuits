"""Faithfulness analysis: nodes/edges and neurons plots.

Two modes (aligned with compare_modes), supported for both nodes/edges and neurons:
1. across_counterfactual: one plot per repeat type, traces = all counterfactual methods.
2. across_repeats: one plot per counterfactual method, traces = all repeat types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plms_repeats_circuits.utils.counterfactuals_config import find_file_for_method

from analyze_utils import (
    COUNTERFACTUAL_METHOD_COLORS,
    FONT_SIZE,
    GRAPH_TYPE_SETTINGS,
    LOG_PCT_TICKTEXT,
    LOG_PCT_TICKVALS,
    METHOD_DISPLAY_NAMES,
    REPEAT_TYPE_COLORS,
    REPEAT_TYPE_PATHS,
    get_graph_type_config,
    get_repeat_type_labels,
)


def _find_faithfulness_csvs(
    results_root: Path,
    repeat_type: str,
    model_type: str,
    graph_type: str,
    method: str,
    seeds: list[int],
) -> dict[int, Path]:
    """Find faithfulness CSV per seed for (rt, mt, graph_type, method). Works for nodes, edges, and neurons."""
    folder = REPEAT_TYPE_PATHS.get(repeat_type, repeat_type)
    base = results_root / "circuit_discovery" / folder / model_type / graph_type
    out = {}
    for seed in seeds:
        seed_dir = base / f"seed_{seed}"
        if not seed_dir.exists():
            continue
        p = find_file_for_method(method, seed_dir, kind="main", ext="csv")
        if p is not None:
            out[seed] = p
    return out


def _load_and_aggregate_faithfulness(
    csv_files: dict[int, Path],
    graph_type: str,
) -> pd.DataFrame:
    #print(f"  Aggregating {len(csv_files)} seed(s) for {graph_type}:")
    all_dfs = []
    for seed, p in csv_files.items():
        print(f"    seed={seed}: {p}")
        df = pd.read_csv(p)
        df["random_seed"] = seed
        all_dfs.append(df)
    if not all_dfs:
        return pd.DataFrame()
    combined = pd.concat(all_dfs, ignore_index=True)
    cfg = GRAPH_TYPE_SETTINGS.get(graph_type)
    if not cfg:
        return pd.DataFrame()
    size_col = cfg["size_col"]
    if size_col not in combined.columns:
        return pd.DataFrame()
    agg = combined.groupby(size_col).agg(
        {"faithfulness_by_mean": ["mean", "std", "count"]}
    ).reset_index()
    agg.columns = ["circuit_size", "mean_faithfulness", "std_faithfulness", "n_seeds"]
    agg["std_faithfulness"] = agg["std_faithfulness"].fillna(0)
    return agg


def _transform_agg_for_plot(
    agg: pd.DataFrame,
    convert_to_pct: bool,
    max_nodes_mt: int | None,
    max_pct_mt: float | None,
) -> tuple[pd.DataFrame, Any] | None:
    """Transform circuit_size to x_vals, apply max_pct filter. Returns (agg_filtered, x_vals) or None."""
    x_vals = agg["circuit_size"].copy()
    if convert_to_pct and max_nodes_mt:
        x_vals = x_vals / max_nodes_mt
    if convert_to_pct and max_pct_mt:
        mask = x_vals <= max_pct_mt
        agg = agg[mask]
        x_vals = x_vals[mask]
        if agg.empty:
            return None
    return (agg, x_vals)


def _add_faithfulness_scatter_trace(
    fig: go.Figure,
    x_vals: Any,
    agg: pd.DataFrame,
    label: str,
    color: str,
    *,
    marker_size: int = 3,
    error_thickness: float = 1.2,
    error_width: int = 3,
) -> None:
    """Add one trace to faithfulness figure."""
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=agg["mean_faithfulness"],
            error_y=dict(
                type="data",
                array=agg["std_faithfulness"],
                visible=True,
                thickness=error_thickness,
                width=error_width,
                color=color,
            ),
            mode="lines+markers",
            name=label,
            line=dict(width=1.8, color=color),
            marker=dict(size=marker_size),
        )
    )


def _apply_faithfulness_layout(
    fig: go.Figure,
    cfg: dict,
    max_pct_mt: float | None,
    mode: str = "across_repeats",
) -> None:
    """Add grey area, axes, and layout to faithfulness figure."""
    grey_lower = cfg.get("grey_area_lower", 0.84)
    grey_upper = cfg.get("grey_area_upper", 0.85)
    convert_to_pct = cfg.get("convert_to_pct", False)
    max_nodes_mt = cfg.get("max_nodes")
    use_log = cfg.get("use_log_scale", False)
    font_size = cfg.get("font_size", 12)
    legend_font_size = cfg.get("legend_font_size", 10)
    fig_width_key = f"fig_width_{mode}"
    fig_width = cfg.get(fig_width_key, cfg.get("fig_width_across_repeats", 350))
    axis_label = cfg.get("axis_label", "Components")

    fig.add_shape(
        type="rect",
        xref="paper",
        x0=0, x1=1,
        y0=grey_lower, y1=grey_upper,
        fillcolor="gray",
        opacity=0.12,
        line_width=0,
        layer="below",
    )
    fig.add_hline(y=grey_lower, line_width=1.3, line_dash="dash", line_color="black")
    fig.add_hline(y=grey_upper, line_width=1.3, line_dash="dash", line_color="black")

    xaxis_title = f"% {axis_label}" if convert_to_pct else f"Circuit size (% {axis_label})"
    xaxis_cfg: dict = dict(
        title=xaxis_title,
        title_font=dict(size=font_size),
        title_standoff=4,
        tickfont=dict(size=font_size),
        gridcolor="rgba(180,200,255,0.5)",
    )
    if use_log:
        xaxis_cfg["type"] = "log"
        xaxis_cfg["range"] = [-3, 0]
        xaxis_cfg["tickvals"] = LOG_PCT_TICKVALS
        xaxis_cfg["ticktext"] = LOG_PCT_TICKTEXT
    elif convert_to_pct and max_pct_mt is not None:
        max_pct_rounded = int((max_pct_mt * 100) // 10) * 10
        tickvals = [v / 100 for v in range(10, max_pct_rounded + 1, 10)]
        xaxis_cfg["type"] = "linear"
        xaxis_cfg["range"] = [0, max_pct_mt + 0.02]
        xaxis_cfg["tickvals"] = tickvals
        xaxis_cfg["ticktext"] = [f"{v * 100:.0f}%" for v in tickvals]
        xaxis_cfg["zeroline"] = True
        xaxis_cfg["nticks"] = 6
    elif convert_to_pct:
        xaxis_cfg["type"] = "linear"
        xaxis_cfg["range"] = [0, 1.02]
        xaxis_cfg["tickvals"] = [0.2, 0.4, 0.6, 0.8, 1.0]
        xaxis_cfg["ticktext"] = ["20%", "40%", "60%", "80%", "100%"]
        xaxis_cfg["zeroline"] = True
        xaxis_cfg["nticks"] = 6

    fig.update_layout(
        width=fig_width,
        height=fig_width / 1.8,
        margin=dict(l=60, r=10, t=10, b=60),
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="rgba(0, 0, 0, 0.1)",
            borderwidth=1,
            font=dict(size=legend_font_size),
            title=None,
        ),
        xaxis=xaxis_cfg,
        yaxis=dict(
            title="Faithfulness",
            title_standoff=4,
            range=[0, 1.0],
            title_font=dict(size=font_size),
            tickfont=dict(size=font_size),
            gridcolor="rgba(180,200,255,0.5)",
            zeroline=True,
            nticks=6,
        ),
        plot_bgcolor="rgba(230,240,255,1)",
        paper_bgcolor="white",
    )


def _run_faithfulness_across_counterfactual(
    results_root: Path,
    output_dir: Path,
    repeat_types: list[str],
    model_types: list[str],
    methods: list[str],
    seeds: list[int],
    graph_type: str,
    plot_config: dict[str, Any],
) -> None:
    """across_counterfactual: one plot per repeat type, traces = all counterfactual methods."""
    for model_type in model_types:
        cfg = get_graph_type_config(plot_config, model_type, graph_type)
        max_pct_mt = cfg.get("max_pct_across_counterfactual")
        convert_to_pct = cfg.get("convert_to_pct", False)
        max_nodes_mt = cfg.get("max_nodes")

        for repeat_type in repeat_types:
            fig = go.Figure()
            for method in methods:
                csv_files = _find_faithfulness_csvs(
                    results_root, repeat_type, model_type, graph_type, method, seeds
                )
                if not csv_files:
                    continue
                agg = _load_and_aggregate_faithfulness(csv_files, graph_type)
                if agg.empty:
                    continue
                result = _transform_agg_for_plot(agg, convert_to_pct, max_nodes_mt, max_pct_mt)
                if result is None:
                    continue
                agg, x_vals = result
                label = METHOD_DISPLAY_NAMES.get(method, method)
                color = COUNTERFACTUAL_METHOD_COLORS.get(method, "#000000")
                _add_faithfulness_scatter_trace(fig, x_vals, agg, label, color)

            if len(fig.data) == 0:
                continue

            _apply_faithfulness_layout(fig, cfg, max_pct_mt, mode="across_counterfactual")

            out_path = output_dir / model_type / "across_counterfactual" / repeat_type
            out_path.mkdir(parents=True, exist_ok=True)
            fn = f"faithfulness_{repeat_type}_{graph_type}"
            print(f"  → Saving {out_path / fn}.png")
            fig.write_image(out_path / f"{fn}.png", scale=1)
            fig.write_image(out_path / f"{fn}.pdf")


def _run_faithfulness_across_repeats(
    results_root: Path,
    output_dir: Path,
    repeat_types: list[str],
    model_types: list[str],
    methods: list[str],
    seeds: list[int],
    graph_type: str,
    plot_config: dict[str, Any],
) -> None:
    """across_repeats: one plot per counterfactual method, traces = all repeat types."""
    labels = get_repeat_type_labels(plot_config)
    for model_type in model_types:
        cfg = get_graph_type_config(plot_config, model_type, graph_type)
        max_pct_mt = cfg.get("max_pct_across_repeats")
        convert_to_pct = cfg.get("convert_to_pct", False)
        max_nodes_mt = cfg.get("max_nodes")

        for method in methods:
            fig = go.Figure()
            for repeat_type in repeat_types:
                csv_files = _find_faithfulness_csvs(
                    results_root, repeat_type, model_type, graph_type, method, seeds
                )
                print(f"  Found {len(csv_files)} CSV files for {method} in {repeat_type}")
                if not csv_files:
                    continue
                agg = _load_and_aggregate_faithfulness(csv_files, graph_type)
                if agg.empty:
                    continue
                result = _transform_agg_for_plot(agg, convert_to_pct, max_nodes_mt, max_pct_mt)
                if result is None:
                    continue
                agg, x_vals = result
                label = labels.get(repeat_type, repeat_type.capitalize())
                color = REPEAT_TYPE_COLORS.get(repeat_type, "#000000")
                _add_faithfulness_scatter_trace(fig, x_vals, agg, label, color)

            if len(fig.data) == 0:
                continue

            _apply_faithfulness_layout(fig, cfg, max_pct_mt, mode="across_repeats")

            out_path = output_dir / model_type / "across_repeats" / method
            out_path.mkdir(parents=True, exist_ok=True)
            fn = f"faithfulness_across_repeats_{method}_{graph_type}"
            print(f"  → Saving {out_path / fn}.png")
            fig.write_image(out_path / f"{fn}.png", scale=1)
            fig.write_image(out_path / f"{fn}.pdf")


def run_faithfulness_nodes_edges(
    results_root: Path,
    output_dir: Path,
    repeat_types: list[str],
    model_types: list[str],
    methods: list[str],
    seeds: list[int],
    graph_type: str,
    plot_config: dict[str, Any],
) -> None:
    """Plot faithfulness for nodes/edges in both modes: across_counterfactual and across_repeats."""
    _run_faithfulness_across_counterfactual(
        results_root, output_dir, repeat_types, model_types, methods, seeds, graph_type, plot_config
    )
    _run_faithfulness_across_repeats(
        results_root, output_dir, repeat_types, model_types, methods, seeds, graph_type, plot_config
    )


def _extract_neurons_attn_mlp_stats(csv_files: dict[int, Path]) -> tuple[float, float, float, float]:
    """Extract attn/mlp mean±std from neurons CSVs."""
    attn_col = "n_nodes_as_first_step_for_neurons"
    mlp_col = "num_mlp_nodes"
    attn_counts, mlp_counts = [], []
    for p in csv_files.values():
        df = pd.read_csv(p)
        if attn_col in df.columns:
            attn_counts.append(df[attn_col].astype(float).mean())
        if mlp_col in df.columns:
            mlp_counts.append(df[mlp_col].astype(float).mean())
    attn_mean = float(np.mean(attn_counts)) if attn_counts else 0.0
    attn_std = float(np.std(attn_counts)) if attn_counts else 0.0
    mlp_mean = float(np.mean(mlp_counts)) if mlp_counts else 0.0
    mlp_std = float(np.std(mlp_counts)) if mlp_counts else 0.0
    return (attn_mean, attn_std, mlp_mean, mlp_std)


def _load_and_aggregate_neurons(
    results_root: Path,
    repeat_type: str,
    model_type: str,
    method: str,
    seeds: list[int],
) -> tuple[pd.DataFrame, float, float, float, float] | None:
    """Load neurons CSVs across seeds, aggregate. Returns (agg_df, attn_mean, attn_std, mlp_mean, mlp_std) or None."""
    csv_files = _find_faithfulness_csvs(
        results_root, repeat_type, model_type, "neurons", method, seeds
    )
    if not csv_files:
        return None
    agg_df = _load_and_aggregate_faithfulness(csv_files, "neurons")
    if agg_df.empty:
        return None
    attn_mean, attn_std, mlp_mean, mlp_std = _extract_neurons_attn_mlp_stats(csv_files)
    return (agg_df, attn_mean, attn_std, mlp_mean, mlp_std)


def _apply_neurons_layout(
    fig: go.Figure,
    is_per_layer: bool,
    vline_val: float | None,
    use_log_scale: bool | None = None,
) -> None:
    """Apply layout for neurons faithfulness plot. use_log_scale from config takes precedence over auto-detect."""
    pct_title = "Neurons used Per Layer(%)" if is_per_layer else "Neurons used (%)"
    count_title = "Neurons used Per Layer(#)" if is_per_layer else "Neurons used (#)"
    if use_log_scale is not None:
        use_log = use_log_scale
    else:
        all_x = []
        for tr in fig.data:
            if hasattr(tr, "x") and tr.x is not None:
                all_x.extend(tr.x)
        use_log = bool(all_x and all(x <= 1 for x in all_x if x is not None))
    if use_log:
        xaxis_cfg = dict(
            title=pct_title,
            type="log",
            range=[-3, 0],
            tickvals=LOG_PCT_TICKVALS,
            ticktext=LOG_PCT_TICKTEXT,
            tickangle=45,
            title_font=dict(size=FONT_SIZE),
            tickfont=dict(size=FONT_SIZE - 2),
            gridcolor="rgba(180,200,255,0.5)",
            zeroline=False,
        )
    else:
        xaxis_cfg = dict(
            title=count_title,
            type="linear",
            tickangle=45,
            title_font=dict(size=FONT_SIZE),
            tickfont=dict(size=FONT_SIZE),
            gridcolor="rgba(180,200,255,0.5)",
            zeroline=False,
        )
    fig.update_layout(
        width=400,
        height=400 / 1.7,
        margin=dict(l=60, r=40, t=50, b=75),
        showlegend=len(fig.data) > 1,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05,
            font=dict(size=FONT_SIZE),
            title=None,
        ),
        xaxis=xaxis_cfg,
        yaxis=dict(
            title="Faithfulness",
            range=[0, 1.0001],
            title_font=dict(size=FONT_SIZE),
            tickfont=dict(size=FONT_SIZE),
            gridcolor="rgba(180,200,255,0.5)",
            zeroline=False,
            nticks=6,
        ),
        plot_bgcolor="rgba(230,240,255,1)",
        paper_bgcolor="white",
    )
    if vline_val is not None:
        fig.add_vline(x=vline_val, line=dict(color="black", width=1, dash="dash"))
        if use_log:
            fig.add_annotation(
                x=np.log10(vline_val),
                xref="x",
                y=1.02,
                yref="paper",
                text=f"{vline_val * 100:.0f}%",
                showarrow=False,
                font=dict(size=FONT_SIZE - 2),
                bgcolor="rgba(255,255,255,0.7)",
                xanchor="center",
                yanchor="bottom",
            )


def run_faithfulness_neurons(
    results_root: Path,
    output_dir: Path,
    repeat_types: list[str],
    model_types: list[str],
    methods: list[str],
    seeds: list[int],
    plot_config: dict[str, Any],
) -> None:
    """Plot neurons faithfulness: one plot per (model_type, repeat_type, method)."""
    for model_type in model_types:
        cfg = get_graph_type_config(plot_config, model_type, "neurons")
        is_per_layer = cfg.get("is_per_layer", False)
        vline_val = cfg.get("vline")
        use_log_scale = cfg.get("use_log_scale")

        for repeat_type in repeat_types:
            for method in methods:
                res = _load_and_aggregate_neurons(results_root, repeat_type, model_type, method, seeds)
                if res is None:
                    continue
                agg_df, attn_mean, attn_std, mlp_mean, mlp_std = res

                fig = go.Figure()
                label = f"attn={attn_mean:.1f}±{attn_std:.1f}, mlp={mlp_mean:.1f}±{mlp_std:.1f}"
                color = REPEAT_TYPE_COLORS.get(repeat_type, "#000000")
                _add_faithfulness_scatter_trace(
                    fig,
                    agg_df["circuit_size"],
                    agg_df,
                    label,
                    color,
                    marker_size=4,
                    error_thickness=1,
                    error_width=4,
                )

                _apply_neurons_layout(fig, is_per_layer, vline_val, use_log_scale)

                out_path = output_dir / model_type / "neurons" / repeat_type / method
                out_path.mkdir(parents=True, exist_ok=True)
                fn = f"faithfulness_neurons_{repeat_type}_{method}"
                print(f"  → Saving {out_path / fn}.png")
                fig.write_image(out_path / f"{fn}.png", scale=1)
                fig.write_image(out_path / f"{fn}.pdf")
                fig.write_html(out_path / f"{fn}.html")
