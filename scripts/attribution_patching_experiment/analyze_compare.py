"""Compare analysis: cross_task, iou, recall heatmaps and combined figures."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyze_utils import (
    FONT_SIZE,
    REPEAT_TYPE_PATHS,
    load_original_faithfulness,
    extract_repeat_type,
    extract_method,
    plot_heatmap_plotly,
    sort_task_names_for_display,
    task_name_to_display_label,
)


def run_compare_heatmaps(
    results_root: Path,
    output_dir: Path,
    repeat_types: list[str],
    model_types: list[str],
    methods: list[str],
    seeds: list[int],
    graph_type: str,
    compare_modes: list[str],
    compare_metrics: list[str],
    plot_config: dict[str, Any] | None = None,
) -> None:
    """Plot cross_task, iou, recall heatmaps per mode."""
    plot_with_circuit_sizes = (plot_config or {}).get("default", {}).get("plot_with_circuit_sizes", False)
    for mode in compare_modes:
        for mt in model_types:
            if mode == "across_counterfactual":
                _run_compare_across_counterfactual(
                    results_root,
                    output_dir,
                    repeat_types,
                    mt,
                    methods,
                    seeds,
                    graph_type,
                    compare_metrics,
                    plot_with_circuit_sizes,
                    plot_config,
                )
            else:
                _run_compare_across_repeats(
                    results_root,
                    output_dir,
                    repeat_types,
                    mt,
                    methods,
                    seeds,
                    graph_type,
                    compare_metrics,
                    plot_with_circuit_sizes,
                    plot_config,
                )


def _run_compare_across_counterfactual(
    results_root: Path,
    output_dir: Path,
    repeat_types: list[str],
    model_type: str,
    methods: list[str],
    seeds: list[int],
    graph_type: str,
    compare_metrics: list[str],
    plot_with_circuit_sizes: bool,
    plot_config: dict | None,
) -> None:
    for rt in repeat_types:
        cross_path = (
            results_root
            / "circuit_discovery_compare"
            / "cross_task"
            / "across_counterfactual"
            / rt
            / model_type
            / graph_type
            / f"seed_{seeds[0]}"
            / "cross_task_results.csv"
        )
        if "cross_task" in compare_metrics and cross_path.exists():
            _plot_cross_task_heatmap(
                cross_path,
                results_root,
                output_dir / "compare" / "across_counterfactual" / "cross_task" / rt / model_type / graph_type,
                rt,
                model_type,
                graph_type,
                seeds,
                "across_counterfactual",
                plot_with_circuit_sizes,
                is_counterfactual=True,
                plot_config=plot_config,
            )
        for metric in ("iou", "recall"):
            if metric not in compare_metrics:
                continue
            iou_path = (
                results_root
                / "circuit_discovery_compare"
                / "iou_recall"
                / "across_counterfactual"
                / rt
                / model_type
                / graph_type
                / f"seed_{seeds[0]}"
                / f"{metric}_results.csv"
            )
            if iou_path.exists():
                _plot_iou_recall_heatmap(
                    iou_path,
                    output_dir / "compare" / "across_counterfactual" / f"{metric}" / rt / model_type / graph_type,
                    rt,
                    model_type,
                    graph_type,
                    metric,
                    plot_with_circuit_sizes,
                    plot_config=plot_config,
                )


def _run_compare_across_repeats(
    results_root: Path,
    output_dir: Path,
    repeat_types: list[str],
    model_type: str,
    methods: list[str],
    seeds: list[int],
    graph_type: str,
    compare_metrics: list[str],
    plot_with_circuit_sizes: bool,
    plot_config: dict | None,
) -> None:
    for method in methods:
        cross_path = (
            results_root
            / "circuit_discovery_compare"
            / "cross_task"
            / "across_repeats"
            / model_type
            / graph_type
            / f"seed_{seeds[0]}"
            / method
            / "cross_task_results.csv"
        )
        if "cross_task" in compare_metrics and cross_path.exists():
            _plot_cross_task_heatmap(
                cross_path,
                results_root,
                output_dir / "compare" / "across_repeats" / "cross_task" / model_type / graph_type / method,
                None,
                model_type,
                graph_type,
                seeds,
                "across_repeats",
                plot_with_circuit_sizes,
                is_counterfactual=False,
                plot_config=plot_config,
            )
        for metric in ("iou", "recall"):
            if metric not in compare_metrics:
                continue
            iou_path = (
                results_root
                / "circuit_discovery_compare"
                / "iou_recall"
                / "across_repeats"
                / model_type
                / graph_type
                / f"seed_{seeds[0]}"
                / method
                / f"{metric}_results.csv"
            )
            if iou_path.exists():
                _plot_iou_recall_heatmap(
                    iou_path,
                    output_dir / "compare" / "across_repeats" / f"{metric}" / model_type / graph_type / method,
                    None,
                    model_type,
                    graph_type,
                    metric,
                    plot_with_circuit_sizes,
                    plot_config=plot_config,
                )


def _extract_seed(task_name: str) -> int:
    m = re.search(r"_(\d+)$", str(task_name))
    return int(m.group(1)) if m else 0


def _plot_cross_task_heatmap(
    csv_path: Path,
    results_root: Path,
    out_dir: Path,
    repeat_type: str | None,
    model_type: str,
    graph_type: str,
    seeds: list[int],
    mode: str,
    plot_with_circuit_sizes: bool,
    is_counterfactual: bool,
    plot_config: dict | None = None,
) -> None:
    df = pd.read_csv(csv_path)
    df["source_repeat"] = df["source_task_name"].apply(extract_repeat_type)
    df["target_repeat"] = df["target_task_name"].apply(extract_repeat_type)
    df["source_method"] = df["source_task_name"].apply(extract_method)
    df["target_method"] = df["target_task_name"].apply(extract_method)
    df["seed"] = df["source_task_name"].apply(_extract_seed)

    if is_counterfactual:
        def method_to_label(m: str) -> str:
            return task_name_to_display_label(f"approximate_{m}_0", "replacement", plot_config) or str(m)
        df["source_label"] = df["source_method"].apply(method_to_label)
        df["target_label"] = df["target_method"].apply(method_to_label)
    else:
        df["source_label"] = df["source_repeat"].apply(
            lambda x: task_name_to_display_label(f"{x}_blosum_0", "only_repeat_group", plot_config) or str(x)
        )
        df["target_label"] = df["target_repeat"].apply(
            lambda x: task_name_to_display_label(f"{x}_blosum_0", "only_repeat_group", plot_config) or str(x)
        )

    original_faith = {}
    if is_counterfactual and repeat_type:
        original_faith, _ = load_original_faithfulness(
            results_root, repeat_type, model_type, graph_type, seeds, ["blosum"]
        )

    norm_rows = []
    for _, row in df.iterrows():
        norm = row["faithfulness_by_mean"]
        if is_counterfactual and repeat_type:
            key = (row["target_circuit_id"], row["seed"])
            orig = original_faith.get(key)
            if orig and orig > 0:
                norm = row["faithfulness_by_mean"] / orig
        target_size = row.get("real_pct_nodes", row.get("real_n_nodes"))
        norm_rows.append({
            "source_label": row["source_label"],
            "target_label": row["target_label"],
            "faithfulness_by_mean": norm,
            "target_circuit_size": target_size,
        })
    res = pd.DataFrame(norm_rows)
    agg = res.groupby(["source_label", "target_label"]).agg(
        faithfulness_by_mean=("faithfulness_by_mean", "mean"),
        faithfulness_std=("faithfulness_by_mean", "std"),
    ).reset_index()
    agg["faithfulness_std"] = agg["faithfulness_std"].fillna(0).round(2)
    agg["faithfulness_by_mean"] = agg["faithfulness_by_mean"].round(2)

    source_order = sort_task_names_for_display(agg["source_label"].unique().tolist(), plot_config)
    target_order = sort_task_names_for_display(agg["target_label"].unique().tolist(), plot_config)
    agg["source_label"] = pd.Categorical(agg["source_label"], categories=source_order, ordered=True)
    agg["target_label"] = pd.Categorical(agg["target_label"], categories=target_order, ordered=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap_plotly(
        agg,
        x_col="target_label",
        y_col="source_label",
        x_label="Target Task",
        y_label="Source Task",
        metric="faithfulness_by_mean",
        metric_label="Faithfulness",
        std_col="faithfulness_std",
        output_dir=str(out_dir),
        figsize=(500, 350),
        vmin=0,
        vmax=1.1,
        angle=-15,
        filename=f"cross_task_{mode}",
    )


def _plot_iou_recall_heatmap(
    csv_path: Path,
    out_dir: Path,
    repeat_type: str | None,
    model_type: str,
    graph_type: str,
    metric: str,
    plot_with_circuit_sizes: bool,
    plot_config: dict | None = None,
) -> None:
    df = pd.read_csv(csv_path)
    metric_col = f"{metric}_nodes" if graph_type == "nodes" else f"{metric}_edges"
    if metric_col not in df.columns:
        return
    df["source_repeat"] = df["source_task_name"].apply(extract_repeat_type)
    df["target_repeat"] = df["target_task_name"].apply(extract_repeat_type)
    df["source_label"] = df["source_repeat"].apply(
        lambda x: task_name_to_display_label(f"{x}_blosum_0", "only_repeat_group", plot_config) or str(x)
    )
    df["target_label"] = df["target_repeat"].apply(
        lambda x: task_name_to_display_label(f"{x}_blosum_0", "only_repeat_group", plot_config) or str(x)
    )
    agg = df.groupby(["source_label", "target_label"])[metric_col].agg(["mean", "std"]).reset_index()
    agg.columns = ["source_label", "target_label", "metric_mean", "metric_std"]
    agg["metric_std"] = agg["metric_std"].fillna(0).round(2)
    agg["metric_mean"] = agg["metric_mean"].round(2)

    source_order = sort_task_names_for_display(agg["source_label"].unique().tolist(), plot_config)
    target_order = sort_task_names_for_display(agg["target_label"].unique().tolist(), plot_config)
    agg["source_label"] = pd.Categorical(agg["source_label"], categories=source_order, ordered=True)
    agg["target_label"] = pd.Categorical(agg["target_label"], categories=target_order, ordered=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    x_l = "Target Circuit (Ground Truth)" if metric == "recall" else "Circuit"
    y_l = "Source Circuit (Predicted)" if metric == "recall" else "Circuit"
    plot_heatmap_plotly(
        agg,
        x_col="target_label",
        y_col="source_label",
        x_label=x_l,
        y_label=y_l,
        metric="metric_mean",
        metric_label=metric.upper(),
        std_col="metric_std",
        output_dir=str(out_dir),
        figsize=(500, 300),
        vmin=0,
        vmax=1.0,
        angle=-15,
        filename=f"{metric}_{'across_repeats' if repeat_type is None else 'across_counterfactual'}",
    )


def _aggregate_cross_task_for_heatmap(
    cross_path: Path,
    results_root: Path,
    repeat_type: str | None,
    model_type: str,
    graph_type: str,
    seeds: list[int],
    is_counterfactual: bool,
    plot_config: dict | None = None,
) -> pd.DataFrame | None:
    """Load cross_task CSV, aggregate, return DataFrame with source_label, target_label, metric_mean, metric_std."""
    if not cross_path.exists():
        return None
    df = pd.read_csv(cross_path)

    df["source_repeat"] = df["source_task_name"].apply(extract_repeat_type)
    df["target_repeat"] = df["target_task_name"].apply(extract_repeat_type)
    df["source_method"] = df["source_task_name"].apply(extract_method)
    df["target_method"] = df["target_task_name"].apply(extract_method)
    df["seed"] = df["source_task_name"].apply(_extract_seed)

    if is_counterfactual and repeat_type:
        df["source_label"] = df["source_method"].apply(
            lambda m: task_name_to_display_label(f"approximate_{m}_0", "replacement", plot_config) or str(m)
        )
        df["target_label"] = df["target_method"].apply(
            lambda m: task_name_to_display_label(f"approximate_{m}_0", "replacement", plot_config) or str(m)
        )
        original_faith, _ = load_original_faithfulness(
            results_root, repeat_type, model_type, graph_type, seeds, ["blosum"]
        )
        norm_rows = []
        for _, row in df.iterrows():
            norm = row["faithfulness_by_mean"]
            key = (row["target_circuit_id"], row["seed"])
            orig = original_faith.get(key)
            if orig and orig > 0:
                norm = row["faithfulness_by_mean"] / orig
            norm_rows.append({
                "source_label": row["source_label"],
                "target_label": row["target_label"],
                "metric_mean": norm,
            })
        res = pd.DataFrame(norm_rows)
    else:
        df["source_label"] = df["source_repeat"].apply(
            lambda x: task_name_to_display_label(f"{x}_blosum_0", "only_repeat_group", plot_config) or str(x)
        )
        df["target_label"] = df["target_repeat"].apply(
            lambda x: task_name_to_display_label(f"{x}_blosum_0", "only_repeat_group", plot_config) or str(x)
        )
        res = df[["source_label", "target_label"]].copy()
        res["metric_mean"] = df["faithfulness_by_mean"]

    agg = res.groupby(["source_label", "target_label"]).agg(
        metric_mean=("metric_mean", "mean"),
        metric_std=("metric_mean", "std"),
    ).reset_index()
    agg["metric_std"] = agg["metric_std"].fillna(0)
    return agg


def _aggregate_iou_recall_for_heatmap(
    csv_path: Path,
    graph_type: str,
    metric: str,
    plot_config: dict | None = None,
) -> pd.DataFrame | None:
    """Load iou or recall CSV, aggregate, return DataFrame with source_label, target_label, metric_mean, metric_std."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    metric_col = f"{metric}_nodes" if graph_type == "nodes" else f"{metric}_edges"
    if metric_col not in df.columns:
        return None
    df["source_repeat"] = df["source_task_name"].apply(extract_repeat_type)
    df["target_repeat"] = df["target_task_name"].apply(extract_repeat_type)
    df["source_label"] = df["source_repeat"].apply(
        lambda x: task_name_to_display_label(f"{x}_blosum_0", "only_repeat_group", plot_config) or str(x)
    )
    df["target_label"] = df["target_repeat"].apply(
        lambda x: task_name_to_display_label(f"{x}_blosum_0", "only_repeat_group", plot_config) or str(x)
    )
    agg = df.groupby(["source_label", "target_label"])[metric_col].agg(["mean", "std"]).reset_index()
    agg.columns = ["source_label", "target_label", "metric_mean", "metric_std"]
    agg["metric_std"] = agg["metric_std"].fillna(0)
    return agg


def _make_heatmap_trace(
    agg: pd.DataFrame,
    vmin: float,
    vmax: float,
    cmap: str = "Blues",
    show_values: bool = True,
) -> go.Heatmap:
    """Build go.Heatmap trace from aggregated DataFrame."""
    heatmap_data = agg.pivot_table(
        index="source_label",
        columns="target_label",
        values="metric_mean",
        aggfunc="mean",
        observed=False,
    )
    x_labels = heatmap_data.columns.tolist()
    y_labels = heatmap_data.index.tolist()
    z_values = heatmap_data.values

    text = None
    if show_values:
        std_data = agg.pivot_table(
            index="source_label",
            columns="target_label",
            values="metric_std",
            aggfunc="mean",
            observed=False,
        )
        std_values = std_data.values if std_data is not None else np.zeros_like(z_values)
        text = []
        for i in range(z_values.shape[0]):
            row_text = []
            for j in range(z_values.shape[1]):
                if not np.isnan(z_values[i, j]):
                    std_val = std_values[i, j] if not np.isnan(std_values[i, j]) else 0.0
                    row_text.append(f"{z_values[i, j]:.2f}\n±{std_val:.2f}")
                else:
                    row_text.append("")
            text.append(row_text)

    return go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        colorscale=cmap,
        zmin=vmin,
        zmax=vmax,
        text=text,
        texttemplate="%{text}" if text else None,
        textfont={"size": 10},
        showscale=True,
    )


def run_combined_heatmaps(
    results_root: Path,
    output_dir: Path,
    repeat_types: list[str],
    model_types: list[str],
    methods: list[str],
    seeds: list[int],
    graph_type: str,
    compare_modes: list[str],
    plot_config: dict | None = None,
) -> None:
    """Create combined figure with IoU, Recall, Cross-Task Faithfulness subplots."""
    for mode in compare_modes:
        if mode == "across_counterfactual":
            for rt in repeat_types:
                for mt in model_types:
                    _run_combined_one(
                        results_root,
                        output_dir,
                        mode,
                        rt,
                        mt,
                        None,
                        graph_type,
                        seeds,
                        methods,
                        plot_config,
                    )
        else:
            for mt in model_types:
                for method in methods:
                    _run_combined_one(
                        results_root,
                        output_dir,
                        mode,
                        None,
                        mt,
                        method,
                        graph_type,
                        seeds,
                        methods,
                        plot_config,
                    )


def _run_combined_one(
    results_root: Path,
    output_dir: Path,
    mode: str,
    repeat_type: str | None,
    model_type: str,
    method: str | None,
    graph_type: str,
    seeds: list[int],
    methods: list[str],
    plot_config: dict | None = None,
) -> None:
    """Create one combined heatmap figure for a single (mode, rt?, mt, method?)."""
    is_counterfactual = mode == "across_counterfactual"
    if is_counterfactual:
        cross_path = (
            results_root
            / "circuit_discovery_compare"
            / "cross_task"
            / "across_counterfactual"
            / repeat_type
            / model_type
            / graph_type
            / f"seed_{seeds[0]}"
            / "cross_task_results.csv"
        )
        iou_path = (
            results_root
            / "circuit_discovery_compare"
            / "iou_recall"
            / "across_counterfactual"
            / repeat_type
            / model_type
            / graph_type
            / f"seed_{seeds[0]}"
            / "iou_results.csv"
        )
        recall_path = (
            results_root
            / "circuit_discovery_compare"
            / "iou_recall"
            / "across_counterfactual"
            / repeat_type
            / model_type
            / graph_type
            / f"seed_{seeds[0]}"
            / "recall_results.csv"
        )
        out_dir = output_dir / "compare" / "combined" / "across_counterfactual" / repeat_type / model_type / graph_type
    else:
        method = method or (methods[0] if methods else "blosum")
        cross_path = (
            results_root
            / "circuit_discovery_compare"
            / "cross_task"
            / "across_repeats"
            / model_type
            / graph_type
            / f"seed_{seeds[0]}"
            / method
            / "cross_task_results.csv"
        )
        iou_path = (
            results_root
            / "circuit_discovery_compare"
            / "iou_recall"
            / "across_repeats"
            / model_type
            / graph_type
            / f"seed_{seeds[0]}"
            / method
            / "iou_results.csv"
        )
        recall_path = (
            results_root
            / "circuit_discovery_compare"
            / "iou_recall"
            / "across_repeats"
            / model_type
            / graph_type
            / f"seed_{seeds[0]}"
            / method
            / "recall_results.csv"
        )
        out_dir = output_dir / "compare" / "combined" / "across_repeats" / model_type / graph_type / method

    if not cross_path.exists() or not iou_path.exists() or not recall_path.exists():
        return

    cross_agg = _aggregate_cross_task_for_heatmap(
        cross_path, results_root, repeat_type, model_type, graph_type, seeds, is_counterfactual, plot_config
    )
    iou_agg = _aggregate_iou_recall_for_heatmap(iou_path, graph_type, "iou", plot_config)
    recall_agg = _aggregate_iou_recall_for_heatmap(recall_path, graph_type, "recall", plot_config)

    if cross_agg is None or iou_agg is None or recall_agg is None:
        return

    all_labels = set()
    for a in (cross_agg, iou_agg, recall_agg):
        all_labels.update(a["source_label"].unique())
        all_labels.update(a["target_label"].unique())
    order = sort_task_names_for_display(list(all_labels), plot_config)
    for agg in (cross_agg, iou_agg, recall_agg):
        agg["source_label"] = pd.Categorical(agg["source_label"], categories=order, ordered=True)
        agg["target_label"] = pd.Categorical(agg["target_label"], categories=order, ordered=True)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["IoU", "Recall", "Cross-Task Faithfulness"],
        horizontal_spacing=0.08,
        shared_yaxes=True,
    )

    h_iou = _make_heatmap_trace(iou_agg, vmin=0, vmax=1.0)
    h_recall = _make_heatmap_trace(recall_agg, vmin=0, vmax=1.0)
    h_cross = _make_heatmap_trace(cross_agg, vmin=0, vmax=1.1)

    fig.add_trace(h_iou, row=1, col=1)
    fig.add_trace(h_recall, row=1, col=2)
    fig.add_trace(h_cross, row=1, col=3)

    fig.update_xaxes(tickangle=-15, row=1, col=1)
    fig.update_xaxes(tickangle=-15, row=1, col=2)
    fig.update_xaxes(tickangle=-15, row=1, col=3)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=3)

    fig.update_layout(
        width=1200,
        height=400,
        margin=dict(l=100, r=100, t=80, b=80),
        font=dict(size=FONT_SIZE),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.write_image(out_dir / "combined_heatmaps.png", scale=1)
    fig.write_image(out_dir / "combined_heatmaps.pdf")
    print(f"Saved {out_dir / 'combined_heatmaps.png'}")
