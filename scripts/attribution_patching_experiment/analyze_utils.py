"""Shared utils for attribution patching analysis (plotting, heatmaps, etc.)."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from plms_repeats_circuits.utils.visualization_utils import get_colorblind_palette
from plms_repeats_circuits.utils.counterfactuals_config import COUNTERFACTUAL_METHODS

pio.kaleido.scope.mathjax = None

METHOD_DISPLAY_NAMES = {m["name"]: m["display_name"] for m in COUNTERFACTUAL_METHODS}

FONT_SIZE = 14

# Shared log-scale x-axis for percentage plots
LOG_PCT_TICKVALS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
LOG_PCT_TICKTEXT = ["0.1%", "0.2%", "0.5%", "1%", "2%", "5%", "10%", "20%", "50%", "100%"]

REPEAT_TYPE_PATHS = {"identical": "identical", "approximate": "approximate", "synthetic": "synthetic"}

GRAPH_TYPE_SETTINGS = {
    "edges": {"size_col": "n_edges_input"},
    "nodes": {"size_col": "n_nodes_input"},
    "neurons": {"size_col": "n_neurons_input"},
}

_cb_palette = get_colorblind_palette(10)
REPEAT_TYPE_COLORS = {
    "synthetic": _cb_palette[0],
    "identical": _cb_palette[1],
    "approximate": _cb_palette[2],
}
COUNTERFACTUAL_METHOD_COLORS = {
    "mask": _cb_palette[0],
    "blosum": _cb_palette[1],
    "blosum-opposite50": _cb_palette[2],
    "permutation": _cb_palette[3],
}
# Extend for all methods
for i, m in enumerate(COUNTERFACTUAL_METHODS):
    if m["name"] not in COUNTERFACTUAL_METHOD_COLORS:
        COUNTERFACTUAL_METHOD_COLORS[m["name"]] = _cb_palette[i % 10]


def load_plotting_config(
    plot_config_path: Path | None = None,
    fallback_dir: Path | None = None,
) -> Dict[str, Any]:
    """Load plotting config JSON. If path is None, use fallback_dir/plotting_config.json."""
    if plot_config_path is None and fallback_dir is not None:
        plot_config_path = fallback_dir / "plotting_config.json"
    if plot_config_path is None or not plot_config_path.exists():
        return {}
    with open(plot_config_path) as f:
        return json.load(f)


def get_plot_config_for_model(config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """Merge default and model-specific plotting config (flat keys only)."""
    out = {k: v for k, v in config.get("default", {}).items() if not isinstance(v, dict)}
    out.update({k: v for k, v in config.get(model_type, {}).items() if not isinstance(v, dict)})
    return out


def get_graph_type_config(config: Dict[str, Any], model_type: str, graph_type: str) -> Dict[str, Any]:
    """Merge flat model config with graph-type-specific sub-config (default then model override)."""
    cfg = get_plot_config_for_model(config, model_type)
    cfg.update(config.get("default", {}).get(graph_type, {}))
    cfg.update(config.get(model_type, {}).get(graph_type, {}))
    return cfg


def extract_repeat_type(task_name: str) -> Optional[str]:
    """Extract repeat type from task name, e.g. approximate_blosum_42 -> approximate."""
    m = re.match(r"^(approximate|identical|synthetic)_", str(task_name))
    return m.group(1) if m else None


def extract_method(task_name: str) -> Optional[str]:
    """Extract method from task name, e.g. approximate_blosum_42 -> blosum."""
    m = re.match(r"^[^_]+_([^_]+(?:_[^_]+)?)_\d+$", str(task_name))
    return m.group(1) if m else None


def load_original_faithfulness(
    results_root: Path,
    repeat_type: str,
    model_type: str,
    graph_type: str,
    seeds: List[int],
    methods: List[str],
) -> tuple[Dict[tuple, float], Dict[tuple, float]]:
    """Load (circuit_id, seed) -> faithfulness and circuit size from circuit_info."""
    faith: Dict[tuple, float] = {}
    sizes: Dict[tuple, float] = {}
    folder = REPEAT_TYPE_PATHS.get(repeat_type, repeat_type)
    base = results_root / "circuit_discovery" / folder / model_type / graph_type
    for seed in seeds:
        info_path = base / f"seed_{seed}" / f"circuit_info_{graph_type}.csv"
        if not info_path.exists():
            continue
        df = pd.read_csv(info_path)
        for _, row in df.iterrows():
            cid = row.get("circuit_id")
            if not cid:
                continue
            faith[(cid, seed)] = float(row.get("faithfulness_by_mean", 0))
            if graph_type == "edges":
                sizes[(cid, seed)] = float(row.get("pct_n_edges", row.get("real_pct_n_edges", 0)))
            else:
                sizes[(cid, seed)] = int(float(row.get("n_nodes", row.get("real_n_nodes", 0))))
    return faith, sizes


DEFAULT_REPEAT_TYPE_LABELS = {"synthetic": "Synthetic", "identical": "Identical", "approximate": "Approximate"}


def get_repeat_type_labels(config: Dict[str, Any] | None) -> Dict[str, str]:
    """Repeat type display labels: from config JSON if present, else default."""
    if config:
        labels = config.get("display_labels", {}).get("repeat_type", {})
        if labels:
            return dict(labels)
    return dict(DEFAULT_REPEAT_TYPE_LABELS)


def _get_repeat_type_labels(config: Dict[str, Any] | None) -> Dict[str, str]:
    """Alias for get_repeat_type_labels (used by task_name_to_display_label)."""
    return get_repeat_type_labels(config)


def sort_task_names_for_display(
    tasks_names: List[str],
    config: Dict[str, Any] | None = None,
) -> List[str]:
    """Return labels sorted for heatmap axis ordering. Uses display_order from config.
    display_order.method can use method names (e.g. mask, blosum) which are mapped to display names."""
    order = (config or {}).get("display_order", {})
    repeat_order = order.get("repeat_type", [])
    raw_method_order = order.get("method") or [m["name"] for m in COUNTERFACTUAL_METHODS]
    method_order = [METHOD_DISPLAY_NAMES.get(x, x) for x in raw_method_order]
    full_order = list(repeat_order) + [x for x in method_order if x not in repeat_order]

    def key(name: str) -> int:
        try:
            return full_order.index(name)
        except ValueError:
            return len(full_order)

    unique = list(set(tasks_names))
    return sorted(unique, key=key)


def task_name_to_display_label(
    task_name: str,
    mode: str,
    config: Dict[str, Any] | None = None,
) -> Optional[str]:
    """
    Map task name to display label. mode: 'only_repeat_group' (repeat type) or 'replacement' (method).
    Method labels come from COUNTERFACTUAL_METHODS; repeat_type from config. Task format: {repeat_type}_{method}_{seed}.
    """
    if mode not in ("only_repeat_group", "replacement"):
        raise ValueError(f"Unknown mode: {mode}. Expected 'only_repeat_group' or 'replacement'.")

    parts = (task_name or "").split("_")
    rt = parts[0] if parts else ""
    method = parts[1] if len(parts) >= 2 else None

    if mode == "only_repeat_group":
        labels = _get_repeat_type_labels(config)
        return labels.get(rt, rt) or rt
    if mode == "replacement" and method:
        return METHOD_DISPLAY_NAMES.get(method, method)
    return None


def plot_heatmap_plotly(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    metric: str,
    metric_label: str,
    title: Optional[str] = None,
    output_dir: Optional[str] = None,
    figsize: tuple = (600, 400),
    cmap: str = "Blues",
    show_values: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    angle: int = 20,
    std_col: Optional[str] = None,
    filename: Optional[str] = None,
) -> go.Figure:
    """Create Plotly heatmap with same styling as legacy plot_utils."""
    heatmap_data = df.pivot_table(
        index=y_col,
        columns=x_col,
        values=metric,
        aggfunc="mean",
        observed=False,
    )
    x_labels = heatmap_data.columns.tolist()
    y_labels = heatmap_data.index.tolist()
    z_values = heatmap_data.values

    if show_values:
        if std_col is not None:
            std_data = df.pivot_table(
                index=y_col,
                columns=x_col,
                values=std_col,
                aggfunc="mean",
                observed=False,
            )
            std_values = std_data.values
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
        else:
            text = np.round(z_values, 3).astype(str)
    else:
        text = None

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale=cmap,
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(
                title=metric_label,
                len=1.0,
                y=0.5,
                yanchor="middle",
            ),
            text=text if show_values else None,
            texttemplate="%{text}" if show_values else None,
            textfont={"size": 12},
        )
    )

    fig.update_layout(
        title=dict(
            text=title or "",
            x=0.5,
            y=0.85,
            xanchor="center",
            yanchor="top",
            font=dict(size=14),
        ),
        xaxis=dict(title=x_label, tickangle=angle),
        yaxis=dict(title=y_label, autorange="reversed"),
        width=figsize[0],
        height=figsize[1],
        margin=dict(l=100, r=50, t=60, b=100),
        xaxis_domain=[0, 0.85],
        yaxis_domain=[0, 1],
    )

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = (title or f"{metric}_by_{x_col}_vs_{y_col}").replace(" ", "_").replace("/", "_").replace("\n", "_")
        fig.write_image(output_path / f"{filename}.png", scale=1)
        fig.write_image(output_path / f"{filename}.pdf")
        print(f"Saved Plotly figure to {output_path}/{filename}.png and .pdf")

    return fig
