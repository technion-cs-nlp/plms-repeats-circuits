"""Compare analysis: cross_task, iou, recall heatmaps and combined figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyze_utils import (
    FONT_SIZE,
    get_heatmap_config,
    load_original_faithfulness,
    extract_repeat_type,
    extract_method,
    extract_seed,
    method_to_display_label,
    repeat_type_to_display_label,
    plot_heatmap_plotly,
    sort_task_names_for_display,
)

# ─── output-path label ────────────────────────────────────────────────────────
_OUTPUT_FOLDER = "heatmaps"


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
    heatmap_cfg = get_heatmap_config(plot_config or {})
    plot_with_circuit_sizes = heatmap_cfg.get("plot_with_circuit_sizes", False)
    print(f"plot_with_circuit_sizes: {plot_with_circuit_sizes}")
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
        cross_paths = [
            results_root
            / "circuit_discovery_compare"
            / "cross_task"
            / "across_counterfactual"
            / rt
            / model_type
            / graph_type
            / f"seed_{seed}"
            / "cross_task_results.csv"
            for seed in seeds
        ]
        cross_exist = [p for p in cross_paths if p.exists()]
        if "cross_task" in compare_metrics and cross_exist:
            _plot_cross_task_heatmap(
                cross_exist,
                results_root,
                output_dir / _OUTPUT_FOLDER / "across_counterfactual" / "cross_task" / rt / model_type / graph_type,
                rt,
                model_type,
                graph_type,
                seeds,
                None,
                "across_counterfactual",
                plot_with_circuit_sizes,
                is_counterfactual=True,
                plot_config=plot_config,
            )
        for metric in ("iou", "recall"):
            if metric not in compare_metrics:
                continue
            iou_paths = [
                results_root
                / "circuit_discovery_compare"
                / "iou_recall"
                / "across_counterfactual"
                / rt
                / model_type
                / graph_type
                / f"seed_{seed}"
                / f"{metric}_results.csv"
                for seed in seeds
            ]
            iou_exist = [p for p in iou_paths if p.exists()]
            if iou_exist:
                _plot_iou_recall_heatmap(
                    iou_exist,
                    output_dir / _OUTPUT_FOLDER / "across_counterfactual" / metric / rt / model_type / graph_type,
                    rt,
                    model_type,
                    graph_type,
                    metric,
                    plot_with_circuit_sizes,
                    is_counterfactual=True,
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
        cross_paths = [
            results_root
            / "circuit_discovery_compare"
            / "cross_task"
            / "across_repeats"
            / model_type
            / graph_type
            / f"seed_{seed}"
            / method
            / "cross_task_results.csv"
            for seed in seeds
        ]
        cross_exist = [p for p in cross_paths if p.exists()]
        if "cross_task" in compare_metrics and cross_exist:
            _plot_cross_task_heatmap(
                cross_exist,
                results_root,
                output_dir / _OUTPUT_FOLDER / "across_repeats" / "cross_task" / model_type / graph_type / method,
                None,
                model_type,
                graph_type,
                seeds,
                repeat_types,
                "across_repeats",
                plot_with_circuit_sizes,
                is_counterfactual=False,
                plot_config=plot_config,
            )
        for metric in ("iou", "recall"):
            if metric not in compare_metrics:
                continue
            iou_paths = [
                results_root
                / "circuit_discovery_compare"
                / "iou_recall"
                / "across_repeats"
                / model_type
                / graph_type
                / f"seed_{seed}"
                / method
                / f"{metric}_results.csv"
                for seed in seeds
            ]
            iou_exist = [p for p in iou_paths if p.exists()]
            if iou_exist:
                _plot_iou_recall_heatmap(
                    iou_exist,
                    output_dir / _OUTPUT_FOLDER / "across_repeats" / metric / model_type / graph_type / method,
                    None,
                    model_type,
                    graph_type,
                    metric,
                    plot_with_circuit_sizes,
                    is_counterfactual=False,
                    plot_config=plot_config,
                )


def _plot_cross_task_heatmap(
    csv_paths: list[Path],
    results_root: Path,
    out_dir: Path,
    repeat_type: str | None,
    model_type: str,
    graph_type: str,
    seeds: list[int],
    repeat_types: list[str] | None,
    mode: str,
    plot_with_circuit_sizes: bool,
    is_counterfactual: bool,
    plot_config: dict | None = None,
) -> None:
    """Plot cross-task heatmap. csv_paths: one CSV per seed (aggregated across seeds)."""
    dfs = []
    for p in csv_paths:
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d["seed"] = d["source_task_name"].apply(lambda x: extract_seed(x) or 0)
        dfs.append(d)
    if not dfs:
        return
    df = pd.concat(dfs, ignore_index=True)

    df["source_repeat"] = df["source_task_name"].apply(extract_repeat_type)
    df["target_repeat"] = df["target_task_name"].apply(extract_repeat_type)
    df["source_method"] = df["source_task_name"].apply(extract_method)
    df["target_method"] = df["target_task_name"].apply(extract_method)

    if is_counterfactual:
        df["source_label"] = df["source_method"].apply(lambda m: method_to_display_label(m, plot_config))
        df["target_label"] = df["target_method"].apply(lambda m: method_to_display_label(m, plot_config))
    else:
        df["source_label"] = df["source_repeat"].apply(lambda x: repeat_type_to_display_label(x, plot_config))
        df["target_label"] = df["target_repeat"].apply(lambda x: repeat_type_to_display_label(x, plot_config))

    # Load original faithfulness: for across_counterfactual one repeat_type; for across_repeats all
    original_faith: dict[str, dict] = {}
    if is_counterfactual and repeat_type:
        of, _ = load_original_faithfulness(
            results_root, repeat_type, model_type, graph_type, seeds, ["blosum"]
        )
        original_faith[repeat_type] = of
    elif not is_counterfactual and repeat_types:
        for rt in repeat_types:
            of, _ = load_original_faithfulness(
                results_root, rt, model_type, graph_type, seeds, ["blosum"]
            )
            original_faith[rt] = of

    norm_rows = []
    for _, row in df.iterrows():
        norm = row["faithfulness_by_mean"]
        target_rt = row["target_repeat"]
        if target_rt in original_faith:
            key = (row["target_circuit_id"], row["seed"])
            orig = original_faith[target_rt].get(key)
            if orig and orig > 0:
                norm = row["faithfulness_by_mean"] / orig
        target_size = row.get("real_pct_nodes", row.get("real_n_nodes"))
        norm_rows.append({
            "source_label": row["source_label"],
            "target_label": row["target_label"],
            "faithfulness_by_mean": norm,
            "target_circuit_size": target_size,
        })

    # Add diagonal entries (self->self = 1.0); cross_task CSVs omit same->same pairs
    if not is_counterfactual and repeat_types:
        circuit_ids_per_rt_seed: dict[tuple[str, int], str] = {}
        for _, row in df.iterrows():
            k = (row["target_repeat"], row["seed"])
            if k not in circuit_ids_per_rt_seed:
                circuit_ids_per_rt_seed[k] = row["target_circuit_id"]
        for (rt, seed), _ in circuit_ids_per_rt_seed.items():
            lbl = repeat_type_to_display_label(rt, plot_config)
            norm_rows.append({
                "source_label": lbl,
                "target_label": lbl,
                "faithfulness_by_mean": 1.0,
                "target_circuit_size": None,
            })
    elif is_counterfactual and repeat_type:
        seen_method_seed: set[tuple[str, int]] = set()
        for _, row in df.iterrows():
            k = (row["target_method"], row["seed"])
            if k not in seen_method_seed:
                seen_method_seed.add(k)
                lbl = method_to_display_label(row["target_method"], plot_config)
                norm_rows.append({
                    "source_label": lbl,
                    "target_label": lbl,
                    "faithfulness_by_mean": 1.0,
                    "target_circuit_size": None,
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

    print(f"  Cross-task {mode}: aggregated {len(csv_paths)} seeds, {len(res)} rows -> {len(agg)} cells")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap_plotly(
        agg,
        x_col="target_label",
        y_col="source_label",
        x_label="Dataset",
        y_label="Circuit",
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
    csv_paths: list[Path],
    out_dir: Path,
    repeat_type: str | None,
    model_type: str,
    graph_type: str,
    metric: str,
    plot_with_circuit_sizes: bool,
    is_counterfactual: bool = False,
    plot_config: dict | None = None,
) -> None:
    """Plot IoU or recall heatmap. csv_paths: one CSV per seed (aggregated across seeds).

    When is_counterfactual: axes are methods (100% BLOSUM, 100% Mask, etc.).
    Otherwise (across_repeats): axes are repeat types (Synthetic, Identical, Approximate).
    """
    dfs = []
    for p in csv_paths:
        if not p.exists():
            continue
        dfs.append(pd.read_csv(p))
    if not dfs:
        return
    df = pd.concat(dfs, ignore_index=True)

    metric_col = f"{metric}_nodes" if graph_type == "nodes" else f"{metric}_edges"
    if metric_col not in df.columns:
        return
    if is_counterfactual:
        df["source_method"] = df["source_task_name"].apply(extract_method)
        df["target_method"] = df["target_task_name"].apply(extract_method)
        df["source_label"] = df["source_method"].apply(lambda m: method_to_display_label(m, plot_config))
        df["target_label"] = df["target_method"].apply(lambda m: method_to_display_label(m, plot_config))
    else:
        df["source_repeat"] = df["source_task_name"].apply(extract_repeat_type)
        df["target_repeat"] = df["target_task_name"].apply(extract_repeat_type)
        df["source_label"] = df["source_repeat"].apply(lambda x: repeat_type_to_display_label(x, plot_config))
        df["target_label"] = df["target_repeat"].apply(lambda x: repeat_type_to_display_label(x, plot_config))
    agg = df.groupby(["source_label", "target_label"])[metric_col].agg(["mean", "std"]).reset_index()
    agg.columns = ["source_label", "target_label", "metric_mean", "metric_std"]
    agg["metric_std"] = agg["metric_std"].fillna(0).round(2)
    agg["metric_mean"] = agg["metric_mean"].round(2)

    source_order = sort_task_names_for_display(agg["source_label"].unique().tolist(), plot_config)
    target_order = sort_task_names_for_display(agg["target_label"].unique().tolist(), plot_config)
    agg["source_label"] = pd.Categorical(agg["source_label"], categories=source_order, ordered=True)
    agg["target_label"] = pd.Categorical(agg["target_label"], categories=target_order, ordered=True)

    print(f"  {metric.upper()} {'across_repeats' if repeat_type is None else 'across_counterfactual'}: aggregated {len(csv_paths)} seeds -> {len(agg)} cells")
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
    cross_paths: list[Path],
    results_root: Path,
    repeat_type: str | None,
    model_type: str,
    graph_type: str,
    seeds: list[int],
    repeat_types: list[str] | None,
    is_counterfactual: bool,
    plot_config: dict | None = None,
) -> pd.DataFrame | None:
    """Load cross_task CSVs (all seeds), aggregate, return DataFrame with source_label, target_label, metric_mean, metric_std."""
    dfs = []
    for p in cross_paths:
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d["seed"] = d["source_task_name"].apply(lambda x: extract_seed(x) or 0)
        dfs.append(d)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)

    df["source_repeat"] = df["source_task_name"].apply(extract_repeat_type)
    df["target_repeat"] = df["target_task_name"].apply(extract_repeat_type)
    df["source_method"] = df["source_task_name"].apply(extract_method)
    df["target_method"] = df["target_task_name"].apply(extract_method)

    original_faith: dict[str, dict] = {}
    if is_counterfactual and repeat_type:
        of, _ = load_original_faithfulness(
            results_root, repeat_type, model_type, graph_type, seeds, ["blosum"]
        )
        original_faith[repeat_type] = of
    elif not is_counterfactual and repeat_types:
        for rt in repeat_types:
            of, _ = load_original_faithfulness(
                results_root, rt, model_type, graph_type, seeds, ["blosum"]
            )
            original_faith[rt] = of

    if is_counterfactual:
        df["source_label"] = df["source_method"].apply(lambda m: method_to_display_label(m, plot_config))
        df["target_label"] = df["target_method"].apply(lambda m: method_to_display_label(m, plot_config))
    else:
        df["source_label"] = df["source_repeat"].apply(lambda x: repeat_type_to_display_label(x, plot_config))
        df["target_label"] = df["target_repeat"].apply(lambda x: repeat_type_to_display_label(x, plot_config))

    norm_rows = []
    for _, row in df.iterrows():
        norm = row["faithfulness_by_mean"]
        target_rt = row["target_repeat"]
        if target_rt in original_faith:
            key = (row["target_circuit_id"], row["seed"])
            orig = original_faith[target_rt].get(key)
            if orig and orig > 0:
                norm = row["faithfulness_by_mean"] / orig
        norm_rows.append({
            "source_label": row["source_label"],
            "target_label": row["target_label"],
            "metric_mean": norm,
        })

    if not is_counterfactual and repeat_types:
        circuit_ids_per_rt_seed: dict[tuple[str, int], str] = {}
        for _, row in df.iterrows():
            k = (row["target_repeat"], row["seed"])
            if k not in circuit_ids_per_rt_seed:
                circuit_ids_per_rt_seed[k] = row["target_circuit_id"]
        for (rt, _) in circuit_ids_per_rt_seed:
            lbl = repeat_type_to_display_label(rt, plot_config)
            norm_rows.append({"source_label": lbl, "target_label": lbl, "metric_mean": 1.0})
    elif is_counterfactual and repeat_type:
        seen_method_seed: set[tuple[str, int]] = set()
        for _, row in df.iterrows():
            k = (row["target_method"], row["seed"])
            if k not in seen_method_seed:
                seen_method_seed.add(k)
                lbl = method_to_display_label(row["target_method"], plot_config)
                norm_rows.append({"source_label": lbl, "target_label": lbl, "metric_mean": 1.0})

    res = pd.DataFrame(norm_rows)
    agg = res.groupby(["source_label", "target_label"]).agg(
        metric_mean=("metric_mean", "mean"),
        metric_std=("metric_mean", "std"),
    ).reset_index()
    agg["metric_std"] = agg["metric_std"].fillna(0)
    return agg


def _aggregate_iou_recall_for_heatmap(
    csv_paths: list[Path],
    graph_type: str,
    metric: str,
    is_counterfactual: bool = False,
    plot_config: dict | None = None,
) -> pd.DataFrame | None:
    """Load iou or recall CSVs (all seeds), aggregate, return DataFrame with source_label, target_label, metric_mean, metric_std."""
    dfs = []
    for p in csv_paths:
        if not p.exists():
            continue
        dfs.append(pd.read_csv(p))
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    metric_col = f"{metric}_nodes" if graph_type == "nodes" else f"{metric}_edges"
    if metric_col not in df.columns:
        return None
    if is_counterfactual:
        df["source_method"] = df["source_task_name"].apply(extract_method)
        df["target_method"] = df["target_task_name"].apply(extract_method)
        df["source_label"] = df["source_method"].apply(lambda m: method_to_display_label(m, plot_config))
        df["target_label"] = df["target_method"].apply(lambda m: method_to_display_label(m, plot_config))
    else:
        df["source_repeat"] = df["source_task_name"].apply(extract_repeat_type)
        df["target_repeat"] = df["target_task_name"].apply(extract_repeat_type)
        df["source_label"] = df["source_repeat"].apply(lambda x: repeat_type_to_display_label(x, plot_config))
        df["target_label"] = df["target_repeat"].apply(lambda x: repeat_type_to_display_label(x, plot_config))
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
    """Build go.Heatmap trace from aggregated DataFrame.

    showscale is always False here; the caller sets it on the rightmost subplot
    after computing the global vmax.
    """
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
        non_nan_std = std_values[~np.isnan(std_values)]
        show_std = len(non_nan_std) > 0 and np.any(non_nan_std > 1e-6)
        text = []
        for i in range(z_values.shape[0]):
            row_text = []
            for j in range(z_values.shape[1]):
                if not np.isnan(z_values[i, j]):
                    if show_std:
                        std_val = std_values[i, j] if not np.isnan(std_values[i, j]) else 0.0
                        row_text.append(f"{z_values[i, j]:.2f}<br>±{std_val:.2f}")
                    else:
                        row_text.append(f"{z_values[i, j]:.2f}")
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
        textfont={"size": FONT_SIZE, "color": "white"},
        showscale=False,
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
                        repeat_types,
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
                        repeat_types,
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
    repeat_types: list[str],
    plot_config: dict | None = None,
) -> None:
    """Create one combined heatmap figure for a single (mode, rt?, mt, method?)."""
    is_counterfactual = mode == "across_counterfactual"
    if is_counterfactual:
        cross_paths = [
            results_root / "circuit_discovery_compare" / "cross_task" / "across_counterfactual"
            / repeat_type / model_type / graph_type / f"seed_{s}" / "cross_task_results.csv"
            for s in seeds
        ]
        iou_paths = [
            results_root / "circuit_discovery_compare" / "iou_recall" / "across_counterfactual"
            / repeat_type / model_type / graph_type / f"seed_{s}" / "iou_results.csv"
            for s in seeds
        ]
        recall_paths = [
            results_root / "circuit_discovery_compare" / "iou_recall" / "across_counterfactual"
            / repeat_type / model_type / graph_type / f"seed_{s}" / "recall_results.csv"
            for s in seeds
        ]
        out_dir = (
            output_dir / _OUTPUT_FOLDER / "combined" / "across_counterfactual" / repeat_type / model_type / graph_type
        )
        fig_width, fig_height = 1200, 450
    else:
        method = method or (methods[0] if methods else "blosum")
        cross_paths = [
            results_root / "circuit_discovery_compare" / "cross_task" / "across_repeats"
            / model_type / graph_type / f"seed_{s}" / method / "cross_task_results.csv"
            for s in seeds
        ]
        iou_paths = [
            results_root / "circuit_discovery_compare" / "iou_recall" / "across_repeats"
            / model_type / graph_type / f"seed_{s}" / method / "iou_results.csv"
            for s in seeds
        ]
        recall_paths = [
            results_root / "circuit_discovery_compare" / "iou_recall" / "across_repeats"
            / model_type / graph_type / f"seed_{s}" / method / "recall_results.csv"
            for s in seeds
        ]
        out_dir = output_dir / _OUTPUT_FOLDER / "combined" / "across_repeats" / model_type / graph_type / method
        fig_width, fig_height = 1000, 350

    cross_exist = [p for p in cross_paths if p.exists()]
    iou_exist = [p for p in iou_paths if p.exists()]
    recall_exist = [p for p in recall_paths if p.exists()]
    if not cross_exist or not iou_exist or not recall_exist:
        return

    cross_agg = _aggregate_cross_task_for_heatmap(
        cross_exist, results_root, repeat_type, model_type, graph_type, seeds,
        repeat_types if not is_counterfactual else None,
        is_counterfactual, plot_config
    )
    iou_agg = _aggregate_iou_recall_for_heatmap(
        iou_exist, graph_type, "iou", is_counterfactual=is_counterfactual, plot_config=plot_config
    )
    recall_agg = _aggregate_iou_recall_for_heatmap(
        recall_exist, graph_type, "recall", is_counterfactual=is_counterfactual, plot_config=plot_config
    )

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

    fig.update_xaxes(tickangle=-15, title_font=dict(size=FONT_SIZE), tickfont=dict(size=FONT_SIZE))
    fig.update_yaxes(autorange="reversed", title_font=dict(size=FONT_SIZE), tickfont=dict(size=FONT_SIZE))
    fig.update_yaxes(title_text=None, row=1, col=2)
    fig.update_yaxes(title_text=None, row=1, col=3)

    # Compute global vmax from all non-NaN z-values across traces
    all_z: list[float] = []
    for trace in fig.data:
        if isinstance(trace, go.Heatmap) and trace.z is not None:
            for row in trace.z:
                for v in row:
                    if v is not None and not np.isnan(float(v)):
                        all_z.append(float(v))

    if all_z:
        vmin_global = 0.0
        vmax_global = min(max(all_z) * 1.05, 1.1) if max(all_z) > 0 else 1.0
        for trace in fig.data:
            if isinstance(trace, go.Heatmap):
                trace.zmin = vmin_global
                trace.zmax = vmax_global
        # Show colorbar only on the rightmost subplot
        fig.update_traces(
            showscale=True,
            colorbar=dict(
                title=dict(text="Value", font=dict(size=FONT_SIZE)),
                len=1.0,
                y=0.5,
                yanchor="middle",
                tickfont=dict(size=FONT_SIZE),
            ),
            row=1,
            col=3,
        )

    fig.update_layout(
        width=fig_width,
        height=fig_height,
        margin=dict(l=80, r=100, t=60, b=80),
        font=dict(size=FONT_SIZE),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Bold subplot titles
    for i, metric_name in enumerate(["IoU", "Recall", "Cross-Task Faithfulness"]):
        fig.layout.annotations[i].update(
            text=f"<b>{metric_name}</b>",
            font=dict(size=15),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.write_image(out_dir / "combined_heatmaps.png", scale=2)
    fig.write_image(out_dir / "combined_heatmaps.pdf")
    print(f"Saved {out_dir / 'combined_heatmaps.png'}")
