"""
Analyze neurons analysis experiment outputs.

Produces:
1. Concepts Categories - bar chart by concept tag/category
2. plot_auroc_scatter_columns_per_bin - scatter of AUROC by layer bin
3. plot_biochemical_concepts_comparison_between_models - ESM-C vs ESM-3 biochemical concept counts (when both models)
4. plot_layer_cluster_heatmap - heatmap of attribution scores (layer x cluster) from component_recurrence + neurons
"""
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from plms_repeats_circuits.utils.visualization_utils import get_colorblind_palette


def _save_fig(fig, output_dir: Path, stem: str):
    """Save figure to output_dir as PNG and PDF."""
    for ext in (".png", ".pdf"):
        try:
            fig.write_image(str(output_dir / f"{stem}{ext}"), scale=2)
        except Exception:
            pass


def plot_concepts_categories(concepts_df, output_dir: Path, title="Concepts Categories"):
    """Bar chart of concept counts by final_tag."""
    from utils import assign_final_tag

    concepts_df = concepts_df.copy()
    if "final_tag" not in concepts_df.columns:
        concepts_df["final_tag"] = concepts_df.apply(
            lambda r: assign_final_tag(r["concept_category"], r.get("concept_name", "")),
            axis=1,
        )
    dist = concepts_df["final_tag"].value_counts(dropna=False).reset_index()
    dist.columns = ["final_tag", "count"]
    dist = dist.sort_values("count")

    fig = px.bar(
        dist,
        x="count",
        y="final_tag",
        color="final_tag",
        text="count",
        orientation="h",
        title=title,
    )
    fig.update_xaxes(range=[0, 130])
    fig.update_traces(textposition="outside")
    fig.update_layout(
        width=600,
        height=450,
        bargap=0.15,
        showlegend=False,
        xaxis_title="Count",
        yaxis_title="",
        margin=dict(l=140, r=60, t=50, b=30),
    )
    _save_fig(fig, output_dir, "concepts_categories")
    return fig


def plot_auroc_scatter_columns_per_bin(
    df,
    output_dir: Path,
    baseline_df=None,
    layer_col="layer",
    score_col="final_auroc",
    tag_col="final_tag",
    bin_size=10,
    jitter=0.045,
    width=650,
    height=450,
    title="",
    tags_to_exclude=["Special_Token"],
):
    """Scatter of AUROC by layer bin, colored by concept tag group."""
    df = df.copy()
    if tags_to_exclude:
        df = df[~df[tag_col].isin(tags_to_exclude)]

    df["layer_bin"] = (df[layer_col] // bin_size).astype(int)

    tag_map = {
        "Amino_Acid": "Amino Acid",
        "Repeat": "Repeat",
        "Special_Token": "Special Token",
        "Blosum62": "Blosum62",
    }
    df["group"] = df[tag_col].map(tag_map).fillna("Biological (Other)")

    groups = ["Amino Acid", "Repeat", "Blosum62", "Biological (Other)", "Special Token"]

    safe_colors = get_colorblind_palette()
    colors = [safe_colors[0], safe_colors[1], safe_colors[2], safe_colors[4], safe_colors[5]]
    palette = dict(zip(groups, colors))
    palette["Baseline"] = safe_colors[7]

    bins = sorted(df["layer_bin"].unique())
    x_base = {b: i for i, b in enumerate(bins)}

    offsets = {
        "Amino Acid": -0.18,
        "Repeat": -0.06,
        "Blosum62": 0.06,
        "Biological (Other)": 0.18,
        "Special Token": 0.30,
        "Baseline": 0.0,
    }

    rng = np.random.default_rng(42)
    df["x"] = (
        df["layer_bin"].map(x_base)
        + df["group"].map(offsets)
        + rng.uniform(-jitter, jitter, len(df))
    )

    fig = go.Figure()

    if baseline_df is not None:
        baseline_df = baseline_df.copy()
        baseline_df["layer_bin"] = (baseline_df[layer_col] // bin_size).astype(int)
        baseline_df["x"] = (
            baseline_df["layer_bin"].map(x_base)
            + offsets["Baseline"]
            + rng.uniform(-jitter, jitter, len(baseline_df))
        )
        fig.add_trace(
            go.Scattergl(
                x=baseline_df["x"],
                y=baseline_df[score_col],
                mode="markers",
                name="Baseline",
                marker=dict(size=3.5, opacity=0.35, color=palette["Baseline"]),
                legendgroup="Baseline",
            )
        )

    for g in groups:
        s = df[df["group"] == g]
        if s.empty:
            continue
        marker_size = 4.5 if g == "Biological (Other)" else 3
        fig.add_trace(
            go.Scattergl(
                x=s["x"],
                y=s[score_col],
                mode="markers",
                name=g,
                marker=dict(size=marker_size, opacity=0.6, color=palette[g]),
                legendgroup=g,
            )
        )

    y_ticks = np.arange(0.5, 1.01, 0.1)
    shapes = [
        dict(
            type="line",
            x0=-0.5,
            x1=len(bins) - 0.5,
            y0=y,
            y1=y,
            line=dict(color="gray", width=1, dash="dot"),
            layer="below",
        )
        for y in y_ticks
    ]

    fig.update_layout(
        width=width,
        height=height,
        title=title,
        plot_bgcolor="white",
        paper_bgcolor="white",
        shapes=shapes,
        font=dict(family="Helvetica", size=14),
        title_font=dict(size=14),
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.2,
            xanchor="center",
            itemsizing="constant",
            font=dict(size=14),
        ),
        xaxis=dict(
            title="Layer bin",
            title_font=dict(size=14),
            tickfont=dict(size=14),
            tickvals=list(x_base.values()),
            ticktext=[f"{b*bin_size}–{(b+1)*bin_size-1}" for b in bins],
            range=[-0.5, len(bins) - 0.5],
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="outside",
        ),
        yaxis=dict(
            title="AUROC",
            title_font=dict(size=14),
            tickfont=dict(size=14),
            range=[0.5, 1.0],
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            ticks="outside",
            tickvals=y_ticks,
            showgrid=True,
            gridcolor="#B0B0B0",
            gridwidth=1,
            griddash="dot",
        ),
    )
    _save_fig(fig, output_dir, "auroc_scatter_by_bin")
    return fig


def _formalize_concept_name(concept: str, include_aas: bool = False) -> str:
    """Map raw concept names to display names (from ESM-C_ESM-3 notebook)."""
    mapping = {
        "aliphatic_amino_acid_IMGT_11": ("Aliphatic", "A, I, L, V"),
        "aromatic_ring_amino_acid": ("Aromatic Ring", "F, W, Y, H"),
        "helix_breakers_amino_acid_propensity": ("Helix Breakers", "G, P"),
        "hydrogen_donor_amino_acid_IMGT": ("H-Bond Donors", "R, K, W"),
        "hydrophilic_amino_acid_IMGT": ("Hydrophilic", "R, N, D, Q, E, K"),
        "hydrophobic_amino_acid_IMGT": ("Hydrophobic", "A, C, I, L, M, F, W, V"),
        "polar_amino_acid_IMGT": ("Polar", "R, N, D, Q, E, H, K, S, T, Y"),
        "small_amino_acid_IMGT": ("Small", "N, D, C, P, T"),
        "sulfur_amino_acid_IMGT": ("Sulfur-Containing", "C, M"),
    }
    if concept not in mapping:
        return str(concept)
    name, aas = mapping[concept]
    return f"{name} ({aas})" if include_aas else name


def plot_biochemical_concepts_comparison_between_models(
    esmc_df: pd.DataFrame,
    esm3_df: pd.DataFrame,
    output_dir: Path,
    min_auroc: float = 0.75,
    concepts_to_filter: list | None = None,
) -> go.Figure:
    """Side-by-side bar chart of biological concept counts (ESM-C vs ESM-3)."""
    if concepts_to_filter is None:
        concepts_to_filter = ["Amino_Acid", "Special_Token", "Repeat", "Blosum62"]

    esmc_df = esmc_df.copy()
    esm3_df = esm3_df.copy()
    esmc_df = esmc_df[esmc_df["final_auroc"] > min_auroc]
    esm3_df = esm3_df[esm3_df["final_auroc"] > min_auroc]
    esmc_df = esmc_df[~esmc_df["final_tag"].isin(concepts_to_filter)]
    esm3_df = esm3_df[~esm3_df["final_tag"].isin(concepts_to_filter)]

    if "other_concept_names" in esmc_df.columns:
        esmc_df["other_concept_names"] = esmc_df["other_concept_names"].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )
    if "other_concept_names" in esm3_df.columns:
        esm3_df["other_concept_names"] = esm3_df["other_concept_names"].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )

    esmc_df["concept_name"] = esmc_df["concept_name"].astype(str).apply(_formalize_concept_name)
    esm3_df["concept_name"] = esm3_df["concept_name"].astype(str).apply(_formalize_concept_name)

    esmc_counts = esmc_df["concept_name"].value_counts().sort_index()
    esm3_counts = esm3_df["concept_name"].value_counts().sort_index()
    all_concepts = sorted(set(esmc_counts.index) | set(esm3_counts.index))
    esmc_values = np.array([esmc_counts.get(c, 0) for c in all_concepts])
    esm3_values = np.array([esm3_counts.get(c, 0) for c in all_concepts])

    order = np.argsort(np.maximum(esmc_values, esm3_values))[::-1]
    all_concepts = np.array(all_concepts)[order]
    esmc_values = esmc_values[order]
    esm3_values = esm3_values[order]

    palette = get_colorblind_palette()
    font_size = 14

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="ESM-C",
            x=all_concepts,
            y=esmc_values,
            text=[f"{int(v)}" for v in esmc_values],
            textposition="outside",
            marker_color=palette[0],
        )
    )
    fig.add_trace(
        go.Bar(
            name="ESM-3",
            x=all_concepts,
            y=esm3_values,
            text=[f"{int(v)}" for v in esm3_values],
            textposition="outside",
            marker_color=palette[1],
        )
    )
    y_max = float(max(esmc_values.max(), esm3_values.max()))
    fig.update_layout(
        barmode="group",
        width=500,
        height=300,
        font=dict(size=font_size),
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(
            tickangle=-30,
            tickfont=dict(size=font_size),
            title_font=dict(size=font_size),
            categoryorder="array",
            categoryarray=all_concepts.tolist(),
        ),
        yaxis=dict(
            range=[0, y_max + 2],
            tickfont=dict(size=font_size),
            title_font=dict(size=font_size),
        ),
        legend=dict(
            orientation="v",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=font_size),
        ),
        margin=dict(l=40, r=40, t=60, b=120),
    )
    _save_fig(fig, output_dir, "biochemical_concepts_comparison_between_models")
    return fig


def plot_layer_cluster_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    metric: str = "mean_score",
    layer_col: str = "layer",
    cluster_col: str = "cluster",
    title: str = "",
    colorscale: str = "RdBu",
    color_percentile: float = 95,
    tick_step: int = 3,
    cluster_order: tuple | None = None,
    width: int = 600,
    output_suffix: str = "",
) -> go.Figure:
    """Create heatmap of attribution scores across layers and clusters."""
    font = dict(size=14)

    tmp = df[[layer_col, cluster_col, metric]].copy()
    tmp[layer_col] = pd.to_numeric(tmp[layer_col], errors="coerce")
    tmp = tmp.dropna(subset=[layer_col])

    layers = sorted(tmp[layer_col].unique())
    layer_labels = [str(int(l)) for l in layers]
    tmp["_layer_str"] = tmp[layer_col].astype(int).astype(str)

    def _nice(c):
        s = str(c)
        s = s.replace("Relative Position Heads", "Relative Pos")
        s = s.replace("Induction Heads", "Induction")
        s = s.replace("AA-biased Heads", "AA-biased")
        return s

    tmp["_cluster_nice"] = tmp[cluster_col].map(_nice)
    present = list(dict.fromkeys(tmp["_cluster_nice"]))

    if cluster_order is None:
        cluster_order = (
            "Relative Pos", "AA-biased", "Induction", "MLP", "Residual",
            "Amino Acid", "Biochemical Sim.", "Repeat", "Special Token", "Biological (Other)",
        )
    ordered = [c for c in cluster_order if c in present] + [c for c in present if c not in cluster_order]

    pivot = (
        tmp.pivot(index="_cluster_nice", columns="_layer_str", values=metric)
        .reindex(index=ordered, columns=layer_labels)
    )
    z = pivot.to_numpy(dtype=float)
    finite = z[np.isfinite(z)]

    if finite.size:
        M = np.percentile(np.abs(finite), color_percentile)
        M = 1e-12 if M == 0 else M
        lo, hi = -M, M
    else:
        lo, hi = -1.0, 1.0

    z_draw = np.where(np.isfinite(z), z, None)
    customdata = np.where(np.isfinite(z), z, None)
    tickvals = layer_labels[::tick_step]

    fig = go.Figure(go.Heatmap(
        x=pivot.columns,
        y=pivot.index.astype(str),
        z=z_draw,
        customdata=customdata,
        colorscale=colorscale,
        zmin=lo,
        zmax=hi,
        hoverongaps=False,
        colorbar=dict(
            thickness=12,
            len=1.0,
            y=0.5,
            tickfont=font,
            title_font=font,
            outlinewidth=0,
        ),
        hovertemplate=f"{metric}: %{{customdata:.6f}}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=font),
        width=width,
        height=80 + 20 * len(ordered),
        margin=dict(l=110, r=40, t=40, b=40),
        font=font,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
    )
    fig.update_xaxes(
        title_text="Layer",
        tickmode="array",
        tickvals=tickvals,
        showgrid=False,
        zeroline=False,
        ticks="outside",
        ticklen=3,
        linecolor="black",
        mirror=True,
        showspikes=False,
        tickfont=font,
        title_font=font,
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        ticks="",
        linecolor="black",
        mirror=True,
        autorange="reversed",
        showspikes=False,
        tickfont=font,
        title_font=font,
    )
    stem = f"heatmap_layer_cluster_{metric}"
    if output_suffix:
        stem = f"{stem}{output_suffix}"
    _save_fig(fig, output_dir, stem)
    return fig


def _build_layer_cluster_summary_from_recurrence(
    component_recurrence_csv: Path,
    neurons_to_best_concepts_csv: Path,
    repeat_type: str,
    model_type: str,
    auroc_threshold: float = 0.75,
    concepts_to_drop: list | None = None,
) -> pd.DataFrame | None:
    """Build layer x cluster summary from component_recurrence (mean scores) + neurons_to_best_concepts (cluster tags)."""
    if concepts_to_drop is None:
        concepts_to_drop = ["Special_Token"]

    rec_df = pd.read_csv(component_recurrence_csv)
    neu_df = pd.read_csv(neurons_to_best_concepts_csv)

    # Match notebook load_neuron_concepts: filter by AUROC and drop concepts
    neu_df = neu_df[~neu_df["final_tag"].isin(concepts_to_drop)].copy()
    neu_df = neu_df[neu_df["final_auroc"] >= auroc_threshold].copy()

    mean_col = f"{repeat_type}_mean_score"
    if mean_col not in rec_df.columns:
        return None

    # Match notebook assign_neuron_cluster: Amino_Acid, Repeat, Special_Token get names; all else -> Biochemical Sim.
    def _assign_neuron_cluster(final_tag):
        if final_tag == "Amino_Acid":
            return "Amino Acid"
        elif final_tag == "Repeat":
            return "Repeat"
        elif final_tag == "Special_Token":
            return "Special Token"
        else:
            return "Biochemical Sim."

    neu_df["cluster"] = neu_df["final_tag"].apply(_assign_neuron_cluster)

    merge_cols = ["component_id"] if "component_id" in rec_df.columns and "component_id" in neu_df.columns else ["layer", "neuron_idx"]
    merged = rec_df.merge(neu_df[merge_cols + ["cluster"]], on=merge_cols, how="inner")
    if merged.empty:
        return None

    summary = (
        merged.groupby(["layer", "cluster"], as_index=False)[mean_col]
        .mean()
        .rename(columns={mean_col: "mean_score"})
    )
    return summary


def plot_layer_cluster_heatmap_neurons(
    exp_results_dir: Path,
    results_root: Path,
    model_type: str,
    counterfactual_type: str,
    repeat_type: str,
) -> go.Figure | None:
    """Plot layer x cluster heatmap when component_recurrence and neurons results exist. Returns the figure or None if inputs are missing."""
    exp_results_dir = Path(exp_results_dir)
    results_root = Path(results_root)

    comp_path = results_root / "component_recurrence" / model_type / counterfactual_type / f"neurons_recurrence_{repeat_type}.csv"
    best_path = exp_results_dir / f"{model_type}_neurons_to_best_concepts.csv"
    if not comp_path.exists() or not best_path.exists():
        return None

    summary = _build_layer_cluster_summary_from_recurrence(
        comp_path, best_path, repeat_type, model_type
    )
    if summary is None or summary.empty:
        return None

    cluster_order = ("Amino Acid", "Biochemical Sim.", "Repeat")
    fig = plot_layer_cluster_heatmap(
        summary,
        exp_results_dir,
        metric="mean_score",
        cluster_order=cluster_order,
        output_suffix="_neurons",
    )
    return fig


def analyze_model(
    exp_results_dir: Path,
    model_type: str = "esm3",
    exp_baseline_results_dir: Path | None = None,
) -> None:
    """Analyze single model: concepts categories bar chart and AUROC scatter by layer bin."""
    exp_results_dir = Path(exp_results_dir)
    if not exp_results_dir.exists():
        raise FileNotFoundError(f"Experiment results dir not found: {exp_results_dir}")

    concepts_path = exp_results_dir / "concepts.csv"
    if concepts_path.exists():
        concepts_df = pd.read_csv(concepts_path)
        plot_concepts_categories(concepts_df, exp_results_dir)

    best_path = exp_results_dir / f"{model_type}_neurons_to_best_concepts.csv"
    if not best_path.exists():
        print(f"  Skipping scatter: {best_path} not found")
        return

    best_df = pd.read_csv(best_path)
    if "other_concept_names" in best_df.columns:
        best_df["other_concept_names"] = best_df["other_concept_names"].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )

    baseline_df = None
    if exp_baseline_results_dir:
        baseline_path = Path(exp_baseline_results_dir) / f"{model_type}_neurons_to_best_concepts.csv"
        if baseline_path.exists():
            baseline_df = pd.read_csv(baseline_path)

    plot_auroc_scatter_columns_per_bin(best_df, exp_results_dir, baseline_df=baseline_df)


def compare_models(
    esm3_results_dir: Path,
    esmc_results_dir: Path,
    comparison_output_dir: Path,
) -> Path:
    """Compare ESM-C vs ESM-3: biochemical concepts comparison figure."""
    esm3_results_dir = Path(esm3_results_dir)
    esmc_results_dir = Path(esmc_results_dir)
    comparison_output_dir = Path(comparison_output_dir)

    esm3_path = esm3_results_dir / "esm3_neurons_to_best_concepts.csv"
    esmc_path = esmc_results_dir / "esm-c_neurons_to_best_concepts.csv"
    if not esm3_path.exists():
        print(f"  Skipping compare: {esm3_path} not found")
        return comparison_output_dir
    if not esmc_path.exists():
        print(f"  Skipping compare: {esmc_path} not found")
        return comparison_output_dir

    comparison_output_dir.mkdir(parents=True, exist_ok=True)
    esm3_df = pd.read_csv(esm3_path)
    esmc_df = pd.read_csv(esmc_path)
    plot_biochemical_concepts_comparison_between_models(
        esmc_df, esm3_df, comparison_output_dir
    )
    return comparison_output_dir

