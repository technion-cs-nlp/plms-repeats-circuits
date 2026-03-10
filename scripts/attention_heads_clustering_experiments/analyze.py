from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from statsmodels.stats.oneway import anova_oneway

STANDARD_AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def filter_dataset_single_example_per_protein(
    dataset_path: str,
) -> tuple[pd.DataFrame, set[str]]:
    """Load circuit_discovery CSV, drop rows where (cluster_id, rep_id) appears more than once.

    Returns:
        (df, filtered_names): filtered DataFrame and set of names that were excluded.
    """
    df = pd.read_csv(dataset_path)
    df["name"] = (
        df["cluster_id"].astype(str)
        + "_"
        + df["rep_id"].astype(str)
        + "_"
        + df["repeat_key"].astype(str)
    )

    original_count = len(df)

    pair_counts = df.groupby(["cluster_id", "rep_id"]).size().rename("count")
    pairs_appear_more_than_once = set(
        (c, r) for (c, r), cnt in pair_counts.items() if cnt > 1
    )

    filtered_names = set()
    for _, row in df.iterrows():
        if (row["cluster_id"], row["rep_id"]) in pairs_appear_more_than_once:
            filtered_names.add(row["name"])

    df = df[~df["name"].isin(filtered_names)].reset_index(drop=True)
    filtered_count = len(df)
    return df, filtered_names


def compute_amino_acid_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """From dataset seq column, Counter over canonical AAs, return freq DataFrame (aa index, frequency column)."""
    sequences = df["seq"].str.upper().tolist()
    all_seq = "".join(sequences)
    counts = Counter(all_seq)
    filtered = {aa: counts.get(aa, 0) for aa in STANDARD_AA_LIST}
    total = sum(filtered.values())
    if total == 0:
        raise ValueError("No standard amino acids found in sequences")
    freqs = {aa: c / total for aa, c in filtered.items()}
    return pd.Series(freqs, name="frequency").rename_axis("aa").to_frame()


def _normalized_js_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    assert np.all(p >= -eps) and np.all(q >= -eps)
    assert np.isclose(p.sum(), 1.0, atol=1e-6)
    assert np.isclose(q.sum(), 1.0, atol=1e-6)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
    kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    js = 0.5 * (kl_pm + kl_qm)
    return float(np.sqrt(js / np.log(2.0)))


def add_bio_jsd_score(
    df: pd.DataFrame,
    aa_frequencies: pd.DataFrame,
    eps: float = 1e-10,
) -> pd.DataFrame:
    """From aa_bias_{aa} columns vs background, add bio_jsd using normalized JSD. Same logic as original add_bio_score."""
    cols_aa = [f"aa_bias_{aa}" for aa in STANDARD_AA_LIST]
    bg = aa_frequencies.loc[STANDARD_AA_LIST, "frequency"].values.astype(float)
    assert np.isclose(bg.sum(), 1.0, atol=1e-6)

    jsd_scores = []
    for _, row in df.iterrows():
        head = row[cols_aa].values.astype(float)
        head = head / (head.sum() + eps)
        jsd_scores.append(_normalized_js_distance(head, bg, eps=eps))

    df = df.copy()
    df["bio_jsd"] = jsd_scores
    return df


def build_per_head_feature_matrix(
    features_path: str | Path,
    dataset_df: pd.DataFrame,
    excluded_names: set[str],
    aa_frequencies: pd.DataFrame,
) -> pd.DataFrame:
    """Load features CSV, filter by excluded_names, groupby layer/head/node_name (mean), add bio scores."""
    df = pd.read_csv(features_path)
    df = df[~df["seq_name"].isin(excluded_names)]

    id_cols = ["layer", "head", "node_name"]
    feature_cols = [c for c in df.columns if c not in id_cols and c != "seq_name"]
    agg_dict = {c: "mean" for c in feature_cols}
    agg_dict["seq_name"] = "count"

    grouped = (
        df.groupby(id_cols)
        .agg(agg_dict)
        .rename(columns={"seq_name": "n_examples"})
        .reset_index()
    )
    grouped = grouped.sort_values(by=["layer", "head"]).reset_index(drop=True)
    return add_bio_jsd_score(grouped, aa_frequencies)


def add_derived_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add induction_score, mean_attn_to_bos_eos, pos_score."""
    df = df.copy()
    df["induction_score"] = (df["copy_score"] + df["pattern_matching_score"]) / 2
    df["mean_attn_to_bos_eos"] = (df["attn_to_bos"] + df["attn_to_eos"]) / 2
    df["pos_score"] = 1 - df["position_score"]
    return df


def fit_kmeans_and_assign_clusters(
    df: pd.DataFrame,
    clustering_cols: list[str],
    k: int = 3,
    n_clusters_range: tuple[int, int] = (2, 10),
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """StandardScaler, KMeans k=2..10, silhouette + inertia; chosen k=3; return labels, metrics, X_scaled."""
    X = df[clustering_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_min, k_max = n_clusters_range
    inertias = []
    silhouettes = []
    for ki in range(k_min, k_max + 1):
        km = KMeans(
            n_clusters=ki,
            init="k-means++",
            n_init=20,
            max_iter=500,
            random_state=42,
            algorithm="lloyd",
        )
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    metrics_df = pd.DataFrame(
        {"k": range(k_min, k_max + 1), "inertia": inertias, "silhouette": silhouettes}
    )

    km_final = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=20,
        max_iter=500,
        random_state=42,
        algorithm="lloyd",
    )
    labels = km_final.fit_predict(X_scaled)
    return labels, metrics_df, X_scaled


def assign_cluster_labels_from_centroids(
    df: pd.DataFrame,
    labels: np.ndarray,
    clustering_cols: list[str],
) -> pd.DataFrame:
    """Map cluster ids to Induction Heads, Relative Position Heads, AA-biased Heads via argmax of centroid means. Same logic as OLD notebook."""
    df = df.copy()
    df["cluster"] = labels

    summary = df.groupby("cluster")[clustering_cols].mean()
    cluster_names = {}

    # Find global extremes once (same as OLD notebook)
    max_induction = summary["induction_score"].idxmax() 
    max_pos = summary["pos_score"].idxmax() 
    max_jsd = summary["bio_jsd"].idxmax()

    for cluster_id in summary.index:
        if cluster_id == max_induction:
            cluster_names[cluster_id] = "Induction Heads"
        elif cluster_id == max_pos:
            cluster_names[cluster_id] = "Relative Position Heads"
        elif cluster_id == max_jsd:
            cluster_names[cluster_id] = "AA-biased Heads"
        else:
            print(f"Cluster {cluster_id} is not a global extreme")
            cluster_names[cluster_id] = f"Cluster {cluster_id}"

    df["cluster_label"] = df["cluster"].map(cluster_names)
    return df


def compute_cluster_summaries(
    feat_df: pd.DataFrame,
    labels: np.ndarray,
    clustering_cols: list[str],
    X_scaled: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute norm_summary (scaled) and orig_summary (original). Same as OLD notebook."""
    cluster_col = "cluster"
    df_with_cluster = feat_df.copy()
    df_with_cluster[cluster_col] = labels

    # normalized_summary: from scaled data
    scaled_df = pd.DataFrame(X_scaled, columns=clustering_cols)
    scaled_df[cluster_col] = labels
    norm_summary = scaled_df.groupby(cluster_col).agg(
        size=(cluster_col, "count"),
        **{col: (col, "mean") for col in clustering_cols},
    )

    # orig_summary: from original data
    orig_summary = df_with_cluster.groupby(cluster_col).agg(
        size=(cluster_col, "count"),
        **{col: (col, "mean") for col in clustering_cols},
    )

    return norm_summary, orig_summary


def run_welch_anova_per_feature(
    df: pd.DataFrame,
    cluster_col: str,
    features: list[str],
) -> pd.DataFrame:
    """statsmodels anova_oneway (Welch) per clustering feature."""
    results = []
    for feat in features:
        vals = df[feat].values
        groups = df[cluster_col].astype(str).values
        res = anova_oneway(vals, groups=groups, use_var="unequal", welch_correction=True)
        results.append({"feature": feat, "Welch F-stat": res.statistic, "p-value": res.pvalue})
    return pd.DataFrame(results)


def identify_repeat_focus_outliers(
    df: pd.DataFrame,
    repeat_focus_col: str = "repeat_focus",
    iqr_factor: float = 1.5,
) -> pd.DataFrame:
    """IQR on repeat_focus (q3 + 1.5*iqr)."""
    q1 = df[repeat_focus_col].quantile(0.25)
    q3 = df[repeat_focus_col].quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + iqr_factor * iqr
    return df[df[repeat_focus_col] > threshold].copy()


def plot_inertia_silhouette_curves(
    metrics_df: pd.DataFrame,
    out_path: str | Path,
    display_inline: bool = False,
) -> None:
    """Plot inertia and silhouette vs k (2 subplots). Save png/pdf."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    pio.kaleido.scope.mathjax = None

    k_range = metrics_df["k"].tolist()
    inertias = metrics_df["inertia"].tolist()
    silhouettes = metrics_df["silhouette"].tolist()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Elbow Method (Inertia)", "Silhouette Score"),
    )

    fig.add_trace(
        go.Scatter(
            x=k_range, y=inertias,
            mode="lines+markers",
            marker=dict(size=8, color="darkblue"),
            line=dict(width=2, color="darkblue"),
            name="Inertia",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=k_range, y=silhouettes,
            mode="lines+markers",
            marker=dict(size=8, color="darkred"),
            line=dict(width=2, color="darkred"),
            name="Silhouette Score",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title_text="",
        title_x=0.5,
        width=950, height=400,
        font=dict(size=14),
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=60),
    )

    fig.update_xaxes(title_text="Number of clusters (k)", row=1, col=1)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_xaxes(title_text="Number of clusters (k)", row=1, col=2)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)

    base = str(out_path).rsplit(".", 1)[0] if "." in str(out_path) else str(out_path)
    fig.write_image(f"{base}.pdf")
    fig.write_image(f"{base}.png")

    if display_inline:
        fig.show()


def plot_umap_cluster_visualization(
    df: pd.DataFrame,
    clustering_cols: list[str],
    cluster_col: str,
    out_path: str | Path,
    random_state: int = 42,
    display_inline: bool = False,
    cluster_to_label: dict | None = None,
    n_neighbors: int = 30,
) -> None:
    """UMAP 2D, scatter by cluster, convex hulls (Plotly). Matches OLD notebook style."""
    try:
        import umap
    except ImportError as e:
        raise ImportError("umap-learn is required for plot_umap_cluster_visualization") from e

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from plms_repeats_circuits.utils.experiment_utils import set_random_seed
    from plms_repeats_circuits.utils.visualization_utils import get_colorblind_palette

    set_random_seed(random_state)

    data = df[clustering_cols + [cluster_col]].copy()
    if "layer" in data.columns and "head" in data.columns:
        data = data.sort_values(by=["layer", "head"]).reset_index(drop=True)
    X = data[clustering_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    reducer = umap.UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=n_neighbors,
        n_jobs=1,
    )
    embeddings = reducer.fit_transform(X_scaled)

    palette = get_colorblind_palette(10)
    KNOWN_CLUSTER_COLORS = {
        "Induction Heads": palette[1],
        "Relative Position Heads": palette[2],
        "AA-biased Heads": palette[0],
    }
    my_palette = [palette[2], palette[1], palette[0]]
    cluster_ids = sorted(data[cluster_col].unique())
    cluster_to_color = {
        cid: KNOWN_CLUSTER_COLORS.get(cid, my_palette[i % len(my_palette)])
        for i, cid in enumerate(cluster_ids)
    }

    cluster_to_label = cluster_to_label or {}
    fig = make_subplots(rows=1, cols=1)

    for cluster_id in cluster_ids:
        idx = data[data[cluster_col] == cluster_id].index
        pts = embeddings[np.asarray(idx)]
        color = cluster_to_color[cluster_id]
        label = cluster_to_label.get(cluster_id, str(cluster_id))

        if len(pts) > 2:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            fig.add_trace(
                go.Scatter(
                    x=hull_pts[:, 0],
                    y=hull_pts[:, 1],
                    mode="lines",
                    line=dict(color=color, width=1),
                    fill="toself",
                    fillcolor=color,
                    opacity=0.15,
                    legendgroup=str(cluster_id),
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=pts[:, 0],
                y=pts[:, 1],
                mode="markers",
                marker=dict(size=6, color=color, opacity=0.8),
                name=label,
                legendgroup=str(cluster_id),
                showlegend=True,
            )
        )

    fig.update_layout(
        title_x=0.5,
        width=320,
        height=340,
        font=dict(size=14),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.4,
            xanchor="center",
            x=0.5,
            bordercolor="black",
            borderwidth=0.5,
        ),
    )
    fig.update_xaxes(title="UMAP-1", title_standoff=2, title_font=dict(size=14))
    fig.update_yaxes(title="UMAP-2", title_standoff=2, title_font=dict(size=14))
    fig.update_layout(showlegend=False)

    if display_inline:
        fig.show()
    else:
        import plotly.io as pio

        pio.kaleido.scope.mathjax = None
        base = str(out_path).rsplit(".", 1)[0] if "." in str(out_path) else str(out_path)
        fig.write_image(f"{base}.pdf")
        fig.write_image(f"{base}.png")


def run_analyze(
    features_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    clustering_cols: list[str] = [
        "induction_score",
        "mean_attn_to_bos_eos",
        "pos_score",
        "attn_entropy",
        "contribution_to_residual_stream_mean_tokens",
        "vocab_entropy",
        "bio_jsd",
        "diff_corrupted_clean",
        "repeat_focus",
    ],
    n_clusters: int = 3,
) -> None:
    """Run full analyze pipeline: filter, build matrix, cluster, ANOVA, outliers, UMAP."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated_features_path = output_dir / "aggregated_features.csv"
    if aggregated_features_path.exists():
        feat_df = pd.read_csv(aggregated_features_path)
    else:
        dataset_df, excluded = filter_dataset_single_example_per_protein(str(dataset_path))

        aa_freq = compute_amino_acid_frequencies(dataset_df)

        feat_df = build_per_head_feature_matrix(
            features_path, dataset_df, excluded, aa_freq
        )
        feat_df = add_derived_clustering_features(feat_df)
        feat_df.to_csv(aggregated_features_path, index=False)

    labels, k_metrics, X_scaled = fit_kmeans_and_assign_clusters(
        feat_df, clustering_cols, k=n_clusters
    )
    k_metrics.to_csv(output_dir / "kmeans_metrics.csv", index=False)

    plot_inertia_silhouette_curves(
        k_metrics,
        output_dir / "inertia_silhouette_curves",
    )

    feat_df = assign_cluster_labels_from_centroids(feat_df, labels, clustering_cols)

    norm_summary, orig_summary = compute_cluster_summaries(
        feat_df, labels, clustering_cols, X_scaled
    )
    norm_summary.to_csv(output_dir / "cluster_summary_normalized.csv")
    orig_summary.to_csv(output_dir / "cluster_summary_original.csv")

    anova_df = run_welch_anova_per_feature(
        feat_df, "cluster_label", clustering_cols
    )
    anova_df.to_csv(output_dir / "anova_results.csv", index=False)

    outliers = identify_repeat_focus_outliers(feat_df)
    outliers_out = outliers[["layer", "head", "node_name", "repeat_focus", "cluster_label"]].sort_values(
        "repeat_focus", ascending=False
    )
    outliers_out.to_csv(output_dir / "outlier_heads_repeat_focus.csv", index=False)

    plot_umap_cluster_visualization(
        feat_df, clustering_cols, "cluster_label",
        output_dir / "cluster_visualization",
    )

    clustering_results = feat_df[["node_name", "layer", "head"]].copy()
    clustering_results["cluster"] = feat_df["cluster_label"]
    clustering_results.to_csv(output_dir / "clustering_results.csv", index=False)
