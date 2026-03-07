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
    sort_repeat_types_for_display,
    sort_task_names_for_display,
)

# ─── output-path label ────────────────────────────────────────────────────────
_OUTPUT_FOLDER = "heatmaps"


def _update_labels_with_circuit_sizes(
    results_df: pd.DataFrame,
    agg: pd.DataFrame,
    target_order: list,
    graph_type: str,
) -> None:
    """Append circuit size (mean ± std) to target labels. Modifies agg in place."""
    if "target_circuit_size" not in results_df.columns:
        return
    label_size_info: dict[str, str] = {}
    for target_label in target_order:
        target_rows = results_df[results_df["target_label"] == target_label]
        sizes = []
        seen_seeds: set[int | None] = set()
        for _, row in target_rows.iterrows():
            if pd.notna(row.get("target_circuit_size")):
                size = row["target_circuit_size"]
                seed = row.get("seed")
                if seed not in seen_seeds:
                    seen_seeds.add(seed)
                    sizes.append(float(size))
        if len(sizes) > 0:
            if graph_type == "edges":
                mean_sz = np.mean(sizes) * 100
                if len(sizes) > 1:
                    std_sz = np.std(sizes) * 100
                    label_size_info[target_label] = f"{target_label}\n({mean_sz:.2f} ± {std_sz:.2f}%)"
                else:
                    label_size_info[target_label] = f"{target_label}\n({mean_sz:.2f}%)"
            else:
                mean_sz = np.mean(sizes)
                if len(sizes) > 1:
                    std_sz = np.std(sizes)
                    label_size_info[target_label] = f"{target_label}\n({mean_sz:.0f} ± {std_sz:.0f})"
                else:
                    label_size_info[target_label] = f"{target_label}\n({mean_sz:.0f})"
        else:
            label_size_info[target_label] = target_label
    new_target_order = [label_size_info.get(lbl, lbl) for lbl in target_order]
    agg["target_label"] = agg["target_label"].apply(lambda x: label_size_info.get(x, x))
    agg["target_label"] = pd.Categorical(
        agg["target_label"], categories=new_target_order, ordered=True
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
                    results_root,
                    rt,
                    model_type,
                    graph_type,
                    metric,
                    plot_with_circuit_sizes,
                    seeds,
                    [rt],
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
                    results_root,
                    None,
                    model_type,
                    graph_type,
                    metric,
                    plot_with_circuit_sizes,
                    seeds,
                    repeat_types,
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

    # Load original faithfulness and sizes: for across_counterfactual one repeat_type; for across_repeats all
    original_faith: dict[str, dict] = {}
    circuit_sizes: dict[str, dict] = {}
    if is_counterfactual and repeat_type:
        of, sz = load_original_faithfulness(
            results_root, repeat_type, model_type, graph_type, seeds, ["blosum"]
        )
        original_faith[repeat_type] = of
        circuit_sizes[repeat_type] = sz
    elif not is_counterfactual and repeat_types:
        for rt in repeat_types:
            of, sz = load_original_faithfulness(
                results_root, rt, model_type, graph_type, seeds, ["blosum"]
            )
            original_faith[rt] = of
            circuit_sizes[rt] = sz

    def _get_target_size(row: pd.Series, cid: str, seed_val: int, rt: str) -> float | None:
        key = (cid, seed_val)
        if rt in circuit_sizes and key in circuit_sizes[rt]:
            return circuit_sizes[rt][key]
        if graph_type == "nodes":
            return row.get("real_n_nodes", row.get("n_nodes"))
        return row.get("real_pct_edges", row.get("real_pct_nodes", row.get("pct_nodes")))

    norm_rows = []
    for _, row in df.iterrows():
        norm = row["faithfulness_by_mean"]
        target_rt = row["target_repeat"]
        seed_val = int(row["seed"])
        if target_rt in original_faith:
            key = (row["target_circuit_id"], seed_val)
            orig = original_faith[target_rt].get(key)
            if orig and orig > 0:
                norm = row["faithfulness_by_mean"] / orig
        target_size = (
            _get_target_size(row, row["target_circuit_id"], seed_val, target_rt)
            if plot_with_circuit_sizes
            else None
        )
        norm_rows.append({
            "source_label": row["source_label"],
            "target_label": row["target_label"],
            "faithfulness_by_mean": norm,
            "target_circuit_size": target_size,
            "seed": seed_val,
        })

    # Add diagonal entries (self->self = 1.0); cross_task CSVs omit same->same pairs
    if not is_counterfactual and repeat_types:
        circuit_ids_per_rt_seed: dict[tuple[str, int], str] = {}
        for _, row in df.iterrows():
            k = (row["target_repeat"], int(row["seed"]))
            if k not in circuit_ids_per_rt_seed:
                circuit_ids_per_rt_seed[k] = row["target_circuit_id"]
        for (rt, seed), cid in circuit_ids_per_rt_seed.items():
            lbl = repeat_type_to_display_label(rt, plot_config)
            diag_size = (
                circuit_sizes.get(rt, {}).get((cid, seed)) if plot_with_circuit_sizes else None
            )
            norm_rows.append({
                "source_label": lbl,
                "target_label": lbl,
                "faithfulness_by_mean": 1.0,
                "target_circuit_size": diag_size,
                "seed": seed,
            })
    elif is_counterfactual and repeat_type:
        seen_method_seed: set[tuple[str, int]] = set()
        for _, row in df.iterrows():
            k = (row["target_method"], int(row["seed"]))
            if k not in seen_method_seed:
                seen_method_seed.add(k)
                lbl = method_to_display_label(row["target_method"], plot_config)
                cid = row["target_circuit_id"]
                seed_val = int(row["seed"])
                diag_size = (
                    circuit_sizes.get(repeat_type, {}).get((cid, seed_val))
                    if plot_with_circuit_sizes
                    else None
                )
                norm_rows.append({
                    "source_label": lbl,
                    "target_label": lbl,
                    "faithfulness_by_mean": 1.0,
                    "target_circuit_size": diag_size,
                    "seed": seed_val,
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

    if plot_with_circuit_sizes:
        _update_labels_with_circuit_sizes(res, agg, target_order, graph_type)

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
    results_root: Path,
    repeat_type: str | None,
    model_type: str,
    graph_type: str,
    metric: str,
    plot_with_circuit_sizes: bool,
    seeds: list[int],
    repeat_types: list[str],
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
        d = pd.read_csv(p)
        # Extract seed from path (e.g. .../seed_42/...)
        seed_val = 0
        for part in p.parts:
            if part.startswith("seed_"):
                try:
                    seed_val = int(part.split("_")[1])
                    break
                except (IndexError, ValueError):
                    pass
        d["seed"] = seed_val
        dfs.append(d)
    if not dfs:
        return
    df = pd.concat(dfs, ignore_index=True)

    metric_col = f"{metric}_nodes" if graph_type == "nodes" else f"{metric}_edges"
    if metric_col not in df.columns:
        return

    df["target_repeat"] = df["target_task_name"].apply(extract_repeat_type)
    circuit_sizes: dict[str, dict] = {}
    if plot_with_circuit_sizes:
        rts = [repeat_type] if is_counterfactual and repeat_type else repeat_types
        for rt in rts:
            if rt:
                _, sz = load_original_faithfulness(
                    results_root, rt, model_type, graph_type, seeds, ["blosum"]
                )
                circuit_sizes[rt] = sz

    if plot_with_circuit_sizes and circuit_sizes:
        def _size_for(row: pd.Series) -> float | None:
            rt = row["target_repeat"]
            if not rt and is_counterfactual and repeat_type:
                rt = repeat_type
            if rt and rt in circuit_sizes:
                key = (row["target_circuit_id"], row["seed"])
                return circuit_sizes[rt].get(key)
            return None

        df["target_circuit_size"] = df.apply(_size_for, axis=1)
    else:
        df["target_circuit_size"] = None
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

    if plot_with_circuit_sizes and "target_circuit_size" in df.columns and df["target_circuit_size"].notna().any():
        _update_labels_with_circuit_sizes(df, agg, target_order, graph_type)

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
    plot_with_circuit_sizes: bool = False,
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
    circuit_sizes: dict[str, dict] = {}
    if is_counterfactual and repeat_type:
        of, sz = load_original_faithfulness(
            results_root, repeat_type, model_type, graph_type, seeds, ["blosum"]
        )
        original_faith[repeat_type] = of
        circuit_sizes[repeat_type] = sz
    elif not is_counterfactual and repeat_types:
        for rt in repeat_types:
            of, sz = load_original_faithfulness(
                results_root, rt, model_type, graph_type, seeds, ["blosum"]
            )
            original_faith[rt] = of
            circuit_sizes[rt] = sz

    def _get_target_size(row: pd.Series, cid: str, seed_val: int, rt: str) -> float | None:
        key = (cid, seed_val)
        if rt in circuit_sizes and key in circuit_sizes[rt]:
            return circuit_sizes[rt][key]
        if graph_type == "nodes":
            return row.get("real_n_nodes", row.get("n_nodes"))
        return row.get("real_pct_edges", row.get("real_pct_nodes", row.get("pct_nodes")))

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
        seed_val = int(row["seed"])
        if target_rt in original_faith:
            key = (row["target_circuit_id"], seed_val)
            orig = original_faith[target_rt].get(key)
            if orig and orig > 0:
                norm = row["faithfulness_by_mean"] / orig
        target_size = (
            _get_target_size(row, row["target_circuit_id"], seed_val, target_rt)
            if plot_with_circuit_sizes
            else None
        )
        norm_rows.append({
            "source_label": row["source_label"],
            "target_label": row["target_label"],
            "metric_mean": norm,
            "target_circuit_size": target_size,
            "seed": seed_val,
        })

    if not is_counterfactual and repeat_types:
        circuit_ids_per_rt_seed: dict[tuple[str, int], str] = {}
        for _, row in df.iterrows():
            k = (row["target_repeat"], int(row["seed"]))
            if k not in circuit_ids_per_rt_seed:
                circuit_ids_per_rt_seed[k] = row["target_circuit_id"]
        for (rt, seed), cid in circuit_ids_per_rt_seed.items():
            lbl = repeat_type_to_display_label(rt, plot_config)
            diag_size = (
                circuit_sizes.get(rt, {}).get((cid, seed)) if plot_with_circuit_sizes else None
            )
            norm_rows.append({
                "source_label": lbl,
                "target_label": lbl,
                "metric_mean": 1.0,
                "target_circuit_size": diag_size,
                "seed": seed,
            })
    elif is_counterfactual and repeat_type:
        seen_method_seed: set[tuple[str, int]] = set()
        for _, row in df.iterrows():
            k = (row["target_method"], int(row["seed"]))
            if k not in seen_method_seed:
                seen_method_seed.add(k)
                lbl = method_to_display_label(row["target_method"], plot_config)
                cid = row["target_circuit_id"]
                seed_val = int(row["seed"])
                diag_size = (
                    circuit_sizes.get(repeat_type, {}).get((cid, seed_val))
                    if plot_with_circuit_sizes
                    else None
                )
                norm_rows.append({
                    "source_label": lbl,
                    "target_label": lbl,
                    "metric_mean": 1.0,
                    "target_circuit_size": diag_size,
                    "seed": seed_val,
                })

    res = pd.DataFrame(norm_rows)
    agg = res.groupby(["source_label", "target_label"]).agg(
        metric_mean=("metric_mean", "mean"),
        metric_std=("metric_mean", "std"),
    ).reset_index()
    agg["metric_std"] = agg["metric_std"].fillna(0)

    source_order = sort_task_names_for_display(agg["source_label"].unique().tolist(), plot_config)
    target_order = sort_task_names_for_display(agg["target_label"].unique().tolist(), plot_config)
    agg["source_label"] = pd.Categorical(agg["source_label"], categories=source_order, ordered=True)
    agg["target_label"] = pd.Categorical(agg["target_label"], categories=target_order, ordered=True)
    if plot_with_circuit_sizes and "target_circuit_size" in res.columns and res["target_circuit_size"].notna().any():
        _update_labels_with_circuit_sizes(res, agg, target_order, graph_type)

    return agg


def _aggregate_iou_recall_for_heatmap(
    csv_paths: list[Path],
    results_root: Path,
    model_type: str,
    graph_type: str,
    metric: str,
    repeat_type: str | None,
    seeds: list[int],
    repeat_types: list[str],
    plot_with_circuit_sizes: bool,
    is_counterfactual: bool = False,
    plot_config: dict | None = None,
) -> pd.DataFrame | None:
    """Load iou or recall CSVs (all seeds), aggregate, return DataFrame with source_label, target_label, metric_mean, metric_std."""
    dfs = []
    for p in csv_paths:
        if not p.exists():
            continue
        d = pd.read_csv(p)
        seed_val = 0
        for part in p.parts:
            if part.startswith("seed_"):
                try:
                    seed_val = int(part.split("_")[1])
                    break
                except (IndexError, ValueError):
                    pass
        d["seed"] = seed_val
        dfs.append(d)
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    metric_col = f"{metric}_nodes" if graph_type == "nodes" else f"{metric}_edges"
    if metric_col not in df.columns:
        return None

    df["target_repeat"] = df["target_task_name"].apply(extract_repeat_type)
    circuit_sizes: dict[str, dict] = {}
    if plot_with_circuit_sizes:
        rts = [repeat_type] if is_counterfactual and repeat_type else repeat_types
        for rt in rts:
            if rt:
                _, sz = load_original_faithfulness(
                    results_root, rt, model_type, graph_type, seeds, ["blosum"]
                )
                circuit_sizes[rt] = sz

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

    if plot_with_circuit_sizes and circuit_sizes:
        def _size_for(row: pd.Series) -> float | None:
            rt = row["target_repeat"]
            if not rt and is_counterfactual and repeat_type:
                rt = repeat_type
            if rt and rt in circuit_sizes:
                key = (row["target_circuit_id"], row["seed"])
                return circuit_sizes[rt].get(key)
            return None

        df["target_circuit_size"] = df.apply(_size_for, axis=1)
    else:
        df["target_circuit_size"] = None

    agg = df.groupby(["source_label", "target_label"])[metric_col].agg(["mean", "std"]).reset_index()
    agg.columns = ["source_label", "target_label", "metric_mean", "metric_std"]
    agg["metric_std"] = agg["metric_std"].fillna(0)

    source_order = sort_task_names_for_display(agg["source_label"].unique().tolist(), plot_config)
    target_order = sort_task_names_for_display(agg["target_label"].unique().tolist(), plot_config)
    agg["source_label"] = pd.Categorical(agg["source_label"], categories=source_order, ordered=True)
    agg["target_label"] = pd.Categorical(agg["target_label"], categories=target_order, ordered=True)
    if plot_with_circuit_sizes and "target_circuit_size" in df.columns and df["target_circuit_size"].notna().any():
        _update_labels_with_circuit_sizes(df, agg, target_order, graph_type)

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
    heatmap_cfg = get_heatmap_config(plot_config or {})
    plot_with_circuit_sizes = heatmap_cfg.get("plot_with_circuit_sizes", False)
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
                        plot_with_circuit_sizes,
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
                        plot_with_circuit_sizes,
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
    plot_with_circuit_sizes: bool = False,
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
        cross_exist,
        results_root,
        repeat_type,
        model_type,
        graph_type,
        seeds,
        repeat_types if not is_counterfactual else None,
        is_counterfactual,
        plot_with_circuit_sizes=plot_with_circuit_sizes,
        plot_config=plot_config,
    )
    iou_agg = _aggregate_iou_recall_for_heatmap(
        iou_exist,
        results_root,
        model_type,
        graph_type,
        "iou",
        repeat_type,
        seeds,
        repeat_types,
        plot_with_circuit_sizes,
        is_counterfactual=is_counterfactual,
        plot_config=plot_config,
    )
    recall_agg = _aggregate_iou_recall_for_heatmap(
        recall_exist,
        results_root,
        model_type,
        graph_type,
        "recall",
        repeat_type,
        seeds,
        repeat_types,
        plot_with_circuit_sizes,
        is_counterfactual=is_counterfactual,
        plot_config=plot_config,
    )

    if cross_agg is None or iou_agg is None or recall_agg is None:
        return

    source_labels = set()
    target_labels = set()
    for a in (cross_agg, iou_agg, recall_agg):
        source_labels.update(a["source_label"].unique())
        target_labels.update(a["target_label"].unique())
    source_order = sort_task_names_for_display(list(source_labels), plot_config)
    target_order = sort_task_names_for_display(list(target_labels), plot_config)
    for agg in (cross_agg, iou_agg, recall_agg):
        agg["source_label"] = pd.Categorical(agg["source_label"], categories=source_order, ordered=True)
        agg["target_label"] = pd.Categorical(agg["target_label"], categories=target_order, ordered=True)

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

    fig.update_xaxes(
        title_text="Circuit", tickangle=-15, title_font=dict(size=FONT_SIZE), tickfont=dict(size=FONT_SIZE),
        row=1, col=1,
    )
    fig.update_xaxes(
        title_text="Circuit", tickangle=-15, title_font=dict(size=FONT_SIZE), tickfont=dict(size=FONT_SIZE),
        row=1, col=2,
    )
    fig.update_xaxes(
        title_text="Dataset", tickangle=-15, title_font=dict(size=FONT_SIZE), tickfont=dict(size=FONT_SIZE),
        row=1, col=3,
    )
    fig.update_yaxes(
        title_text="Circuit", autorange="reversed", title_font=dict(size=FONT_SIZE), tickfont=dict(size=FONT_SIZE),
        row=1, col=1,
    )
    fig.update_yaxes(title_text=None, autorange="reversed", tickfont=dict(size=FONT_SIZE), row=1, col=2)
    fig.update_yaxes(title_text=None, autorange="reversed", tickfont=dict(size=FONT_SIZE), row=1, col=3)

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


def _build_combined_figure_from_aggs(
    aggs: list[pd.DataFrame],
    subplot_titles: list[str],
    vmin: float,
    vmax: float,
    x_label: str,
    y_label: str,
    metric_label: str,
    fig_width: int,
    fig_height: int,
    square_aspect: bool = False,
    plot_config: dict | None = None,
) -> go.Figure:
    """Build a combined figure from a list of aggs. Shared by all-metrics and per-metric combined."""
    n_cols = len(aggs)
    if n_cols == 0:
        return go.Figure()

    source_labels = set()
    target_labels = set()
    for a in aggs:
        source_labels.update(a["source_label"].unique())
        target_labels.update(a["target_label"].unique())
    source_order = sort_task_names_for_display(list(source_labels), plot_config)
    target_order = sort_task_names_for_display(list(target_labels), plot_config)
    for agg in aggs:
        agg["source_label"] = pd.Categorical(agg["source_label"], categories=source_order, ordered=True)
        agg["target_label"] = pd.Categorical(agg["target_label"], categories=target_order, ordered=True)

    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        shared_yaxes=True,
    )
    all_z: list[float] = []
    for col_idx, agg in enumerate(aggs):
        h = _make_heatmap_trace(agg, vmin=vmin, vmax=vmax)
        fig.add_trace(h, row=1, col=col_idx + 1)
        if h.z is not None:
            for row in h.z:
                for v in row:
                    if v is not None and not np.isnan(float(v)):
                        all_z.append(float(v))
        fig.update_xaxes(
            title_text=x_label,
            tickangle=-15,
            title_font=dict(size=FONT_SIZE),
            tickfont=dict(size=FONT_SIZE),
            row=1,
            col=col_idx + 1,
        )
        y_ax = dict(
            title_text=y_label if col_idx == 0 else None,
            autorange="reversed",
            title_font=dict(size=FONT_SIZE),
            tickfont=dict(size=FONT_SIZE),
            row=1,
            col=col_idx + 1,
        )
        if square_aspect:
            y_ax["scaleanchor"] = "x" if col_idx == 0 else f"x{col_idx + 1}"
            y_ax["scaleratio"] = 1
            y_ax["constrain"] = "domain"
        fig.update_yaxes(**y_ax)

    if all_z:
        vmax_global = min(max(all_z) * 1.05, vmax) if max(all_z) > 0 else vmax
        for trace in fig.data:
            if isinstance(trace, go.Heatmap):
                trace.zmin = vmin
                trace.zmax = vmax_global
        fig.update_traces(
            showscale=True,
            colorbar=dict(
                title=dict(text=metric_label, font=dict(size=FONT_SIZE)),
                len=0.6,
                y=0.5,
                yanchor="middle",
                tickfont=dict(size=FONT_SIZE),
            ),
            row=1,
            col=n_cols,
        )

    fig.update_layout(
        width=fig_width,
        height=fig_height,
        margin=dict(l=80, r=150, t=60, b=130),
        font=dict(size=FONT_SIZE),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    for i, title in enumerate(subplot_titles):
        if i < len(fig.layout.annotations):
            fig.layout.annotations[i].update(text=f"<b>{title}</b>", font=dict(size=15))
    return fig


def run_combined_per_metric_heatmaps(
    results_root: Path,
    output_dir: Path,
    repeat_types: list[str],
    model_types: list[str],
    methods: list[str],
    seeds: list[int],
    graph_type: str,
    compare_modes: list[str],
    compare_metrics: list[str],
    plot_config: dict | None = None,
) -> None:
    # Map user metric names to internal keys
    metric_map = {"cross_task": "cross_task", "iou": "iou", "recall": "recall"}
    metrics_to_run = [metric_map[m] for m in compare_metrics if m in metric_map]
    if not metrics_to_run:
        return

    heatmap_cfg = get_heatmap_config(plot_config or {})
    plot_with_circuit_sizes = heatmap_cfg.get("plot_with_circuit_sizes", False)
    fig_width, fig_height = 1500, 600

    for mode in compare_modes:
        is_counterfactual = mode == "across_counterfactual"
        if is_counterfactual:
            if len(repeat_types) < 3:
                continue
            for mt in model_types:
                for metric_key in metrics_to_run:
                    aggs = []
                    titles = []
                    for rt in sort_repeat_types_for_display(repeat_types, plot_config):
                        if metric_key == "cross_task":
                            paths = [
                                results_root / "circuit_discovery_compare" / "cross_task" / "across_counterfactual"
                                / rt / mt / graph_type / f"seed_{s}" / "cross_task_results.csv"
                                for s in seeds
                            ]
                            agg = _aggregate_cross_task_for_heatmap(
                                [p for p in paths if p.exists()],
                                results_root,
                                rt,
                                mt,
                                graph_type,
                                seeds,
                                None,
                                True,
                                plot_with_circuit_sizes=plot_with_circuit_sizes,
                                plot_config=plot_config,
                            )
                        else:
                            paths = [
                                results_root / "circuit_discovery_compare" / "iou_recall" / "across_counterfactual"
                                / rt / mt / graph_type / f"seed_{s}" / f"{metric_key}_results.csv"
                                for s in seeds
                            ]
                            agg = _aggregate_iou_recall_for_heatmap(
                                [p for p in paths if p.exists()],
                                results_root,
                                mt,
                                graph_type,
                                metric_key,
                                rt,
                                seeds,
                                [rt],
                                plot_with_circuit_sizes,
                                is_counterfactual=True,
                                plot_config=plot_config,
                            )
                        if agg is not None:
                            aggs.append(agg)
                            titles.append(repeat_type_to_display_label(rt, plot_config))
                    if len(aggs) < 2:
                        continue
                    fig = _build_combined_figure_from_aggs(
                        aggs,
                        titles,
                        vmin=0.0,
                        vmax=1.1 if metric_key == "cross_task" else 1.0,
                        x_label="Dataset",
                        y_label="Circuit",
                        metric_label="Faithfulness" if metric_key == "cross_task" else metric_key.upper(),
                        fig_width=fig_width,
                        fig_height=fig_height,
                        square_aspect=True,
                        plot_config=plot_config,
                    )
                    out_dir = output_dir / _OUTPUT_FOLDER / "across_counterfactual"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"combined_{mt}_{graph_type}_{metric_key}"
                    fig.write_image(out_dir / f"{fname}.png", scale=2)
                    fig.write_image(out_dir / f"{fname}.pdf")
                    print(f"Saved {out_dir / (fname + '.png')} (per-metric {metric_key})")
        else:
            if len(methods) < 2:
                continue
            for mt in model_types:
                for metric_key in metrics_to_run:
                    aggs = []
                    titles = []
                    for method in methods:
                        if metric_key == "cross_task":
                            paths = [
                                results_root / "circuit_discovery_compare" / "cross_task" / "across_repeats"
                                / mt / graph_type / f"seed_{s}" / method / "cross_task_results.csv"
                                for s in seeds
                            ]
                            agg = _aggregate_cross_task_for_heatmap(
                                [p for p in paths if p.exists()],
                                results_root,
                                None,
                                mt,
                                graph_type,
                                seeds,
                                repeat_types,
                                False,
                                plot_with_circuit_sizes=plot_with_circuit_sizes,
                                plot_config=plot_config,
                            )
                        else:
                            paths = [
                                results_root / "circuit_discovery_compare" / "iou_recall" / "across_repeats"
                                / mt / graph_type / f"seed_{s}" / method / f"{metric_key}_results.csv"
                                for s in seeds
                            ]
                            agg = _aggregate_iou_recall_for_heatmap(
                                [p for p in paths if p.exists()],
                                results_root,
                                mt,
                                graph_type,
                                metric_key,
                                None,
                                seeds,
                                repeat_types,
                                plot_with_circuit_sizes,
                                is_counterfactual=False,
                                plot_config=plot_config,
                            )
                        if agg is not None:
                            aggs.append(agg)
                            titles.append(method_to_display_label(method, plot_config))
                    if len(aggs) < 2:
                        continue
                    x_label = "Circuit" if metric_key != "cross_task" else "Dataset"
                    fig = _build_combined_figure_from_aggs(
                        aggs,
                        titles,
                        vmin=0.0,
                        vmax=1.1 if metric_key == "cross_task" else 1.0,
                        x_label=x_label,
                        y_label="Circuit",
                        metric_label="Faithfulness" if metric_key == "cross_task" else metric_key.upper(),
                        fig_width=fig_width,
                        fig_height=fig_height,
                        square_aspect=True,
                        plot_config=plot_config,
                    )
                    out_dir = output_dir / _OUTPUT_FOLDER / "across_repeats"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"combined_{mt}_{graph_type}_{metric_key}"
                    fig.write_image(out_dir / f"{fname}.png", scale=2)
                    fig.write_image(out_dir / f"{fname}.pdf")
                    print(f"Saved {out_dir / (fname + '.png')} (per-metric {metric_key})")
