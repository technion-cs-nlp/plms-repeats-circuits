"""Activation-patching cluster-cluster ablation: ablate top p% of positive edges per (c1,c2), evaluate, write CSV per p."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import (
    load_edges_and_aggregate,
    create_evaluation_dataframe,
    ActivationPatchingDataset,
    plot_connection_heatmap_from_dict,
    CLUSTER_NAMES_TO_LABELS,
    SRC_LABELS,
    DST_LABELS,
)
from plms_repeats_circuits.EAP.graph import Graph, GraphType
from plms_repeats_circuits.EAP.circuit_selection import set_edges_in_graph_except_names
from plms_repeats_circuits.EAP.evaluate import evaluate_graph, evaluate_baseline
from plms_repeats_circuits.utils.patching_metrics import create_loss_and_metric
from plms_repeats_circuits.utils.model_utils import get_device
from plms_repeats_circuits.utils.esm_utils import load_model, load_tokenizer_by_model_type
from plms_repeats_circuits.utils.counterfactuals_config import find_file_for_method


def _get_ablated_exclusions(
    positive_df: pd.DataFrame,
    cluster1: str,
    cluster2: str,
    p: float,
) -> tuple[set, int, float, float, int, float]:
    subset = positive_df[
        (positive_df["cluster1"] == cluster1)
        & (positive_df["cluster2"] == cluster2)
    ]
    n = len(subset)
    if n == 0:
        raise ValueError(
            f"No positive edges for cluster pair ({cluster1!r}, {cluster2!r}); "
            "pair came from positive_df so at least one was expected."
        )
    subset = subset.sort_values("mean_raw", ascending=False)
    mean_score = subset["mean_raw"].mean()
    k = math.ceil(p * n)
    p_actual = k / n
    if k == 0:
        return set(), n, mean_score, float("nan"), 0, p_actual
    top_k = subset.head(k)
    p_ablated_mean_score = top_k["mean_raw"].mean()
    return set(top_k["name"].tolist()), n, mean_score, p_ablated_mean_score, k, p_actual


def _ablation_csv_path(
    output_dir: Path,
    exp_name: str,
    repeat_type: str,
    counterfactual_type: str,
    random_state: int,
    p: float,
) -> Path:
    """Path to the ablation CSV for a given p (same pattern as written in activation step)."""
    counterfactual_safe = str(counterfactual_type).replace("/", "_")
    return output_dir / f"{exp_name}_{repeat_type}_{counterfactual_safe}_rs{random_state}_p_{p}.csv"


def _run_analyze_step(
    output_dir: Path,
    percentages: list[float],
    repeat_type: str,
    counterfactual_type: str,
    random_state: int,
    metric: str,
) -> None:
    """Load ablation CSVs per p, build heatmap of diff_metric (impact of ablating cluster pair), save figures."""
    exp_name = "activation_patching_ablation"
    diff_col = f"diff_{metric}"
    for p in percentages:
        csv_path = _ablation_csv_path(
            output_dir, exp_name, repeat_type, counterfactual_type, random_state, p
        )
        if not csv_path.exists():
            print(f"[analyze] Skip p={p}: {csv_path} not found.")
            continue
        df = pd.read_csv(csv_path)
        if diff_col not in df.columns:
            print(f"[analyze] Skip p={p}: column {diff_col!r} not in {csv_path.name}.")
            continue
        df["src_label"] = df["cluster1"].map(CLUSTER_NAMES_TO_LABELS)
        df["dst_label"] = df["cluster2"].map(CLUSTER_NAMES_TO_LABELS)
        missing = df[df["src_label"].isna() | df["dst_label"].isna()]
        if not missing.empty:
            print(f"[analyze] Skip p={p}: unmapped cluster names in {csv_path.name}: {missing[['cluster1', 'cluster2']].drop_duplicates().to_dict('records')}")
            continue
        weight_dict = {
            (row.src_label, row.dst_label): float(getattr(row, diff_col)) for row in df.itertuples(index=False)
        }
        decimal_places = 4 if metric == "probability" else 2
        fig = plot_connection_heatmap_from_dict(
            weight_dict,
            SRC_LABELS,
            DST_LABELS,
            colorscale="Blues",
            take_abs=False,
            decimal_places=decimal_places,
        )
        stem = csv_path.stem
        fig.write_image(str(output_dir / f"{stem}_heatmap.pdf"))
        fig.write_image(str(output_dir / f"{stem}_heatmap.png"))
        print(f"[analyze] Wrote {stem}_heatmap.pdf / .png")


def _run_activation_step(
    args: argparse.Namespace,
    output_dir: Path,
    results_root: Path,
    datasets_root: Path,
) -> None:
    """Run ablation for each p, write one CSV per p."""
    log_prefix = "[run_activation_patching_clusters]"
    print(f"{log_prefix} Starting.")
    print(f"{log_prefix} Params: repeat_type={args.repeat_type}, model_type={args.model_type}, counterfactual_type={args.counterfactual_type}")
    print(f"{log_prefix}   results_root={results_root}, datasets_root={datasets_root}")
    print(f"{log_prefix}   seeds={args.seeds}, in_graph_ratio={args.in_graph_ratio}, metric={args.metric}")
    print(f"{log_prefix}   total_n_samples={args.total_n_samples}, percentages={args.percentages}")
    print(f"{log_prefix}   output_dir={output_dir}, batch_size={args.batch_size}, random_state={args.random_state}, quiet={args.quiet}")
    if args.total_n_samples is None:
        raise ValueError("--total_n_samples is required for the run_activation_patching_clusters step.")
    circuit_dir = datasets_root / args.repeat_type / args.model_type / "circuit_discovery"
    csv_path = find_file_for_method(args.counterfactual_type, circuit_dir, kind="main", ext="csv")
    if csv_path is None:
        raise FileNotFoundError(f"No circuit_discovery CSV for '{args.counterfactual_type}' in {circuit_dir}")
    print(f"{log_prefix} Circuit discovery CSV: {csv_path}")

    device = get_device()
    model = load_model(
        model_type=args.model_type,
        device=device,
        use_transformer_lens_model=True,
        cache_attention_activations=True,
        cache_mlp_activations=True,
        output_type="sequence",
        cache_attn_pattern=False,
        split_qkv_input=True,
    )
    tokenizer = load_tokenizer_by_model_type(args.model_type)
    print(f"{log_prefix} Model and tokenizer loaded.")

    df = pd.read_csv(csv_path)
    eval_df = create_evaluation_dataframe(
        df,
        total_n_samples=args.total_n_samples,
        random_state=args.random_state,
        tokenizer=tokenizer,
        metric=args.metric,
    )
    dataset = ActivationPatchingDataset(eval_df)
    dataloader = dataset.to_dataloader(args.batch_size)
    print(f"{log_prefix} Evaluation dataset ready (n_samples={len(eval_df)}).")

    _, metric_eval = create_loss_and_metric(args.metric)

    clustering_path = results_root / "attention_heads_clustering" / args.model_type / args.counterfactual_type / "clustering_results.csv"
    if not clustering_path.exists():
        raise FileNotFoundError(f"Clustering not found: {clustering_path}")

    edges = load_edges_and_aggregate(
        results_root=results_root,
        repeat_type=args.repeat_type,
        model_type=args.model_type,
        counterfactual_type=args.counterfactual_type,
        clustering_path=clustering_path,
        seeds=args.seeds,
        in_graph_ratio=args.in_graph_ratio,
    )
    edges_df = pd.DataFrame([
        {
            "name": e.name,
            "node1": e.node1,
            "node2": e.node2,
            "cluster1": e.cluster1,
            "cluster2": e.cluster2,
            "mean_raw": e.mean_raw,
            "in_graph": e.in_graph,
        }
        for e in edges
    ]    )
    edges_df = edges_df[edges_df["in_graph"] == True]
    print(f"{log_prefix} Edges loaded and filtered (n_edges={len(edges_df)}).")

    graph = Graph.from_model(model, graph_type=GraphType.Edges)
    graph.set_all_edges_in_graph(in_graph=True)
    real_total_edges = graph.count_included_edges()
    print(f"{log_prefix} Graph built (edges in graph: {real_total_edges}).")

    original_scores = evaluate_baseline(
        model=model,
        dataloader=dataloader,
        metrics=metric_eval,
        device=device,
        run_corrupted=False,
        quiet=args.quiet,
        calc_clean_logits_for_metric=False,
    )
    original_metric = original_scores.mean().item()
    print(f"{log_prefix} Baseline evaluation done ({args.metric}={original_metric:.4f}).")

    positive_df = edges_df[edges_df["mean_raw"] > 0]
    cluster_pairs = positive_df[["cluster1", "cluster2"]].drop_duplicates()
    n_pairs = len(cluster_pairs)
    print(f"{log_prefix} Cluster pairs (positive edges): {n_pairs}.")

    n_percentages = len(args.percentages)
    for idx, p in enumerate(args.percentages):
        print(f"{log_prefix} p={p} ({idx + 1}/{n_percentages} percentages), {n_pairs} pairs...")
        rows = []
        for _, pair_row in cluster_pairs.iterrows():
            c1, c2 = pair_row["cluster1"], pair_row["cluster2"]
            print(f"{log_prefix} Analyzing pair ({c1}, {c2}).")
            ablated_exclusions, total_edges, mean_score, p_ablated_mean_score, k_ablated, p_actual = _get_ablated_exclusions(
                positive_df, c1, c2, p
            )
            if len(ablated_exclusions) == 0:
                ablated_metric = original_metric
                diff_metric = 0.0
            else:
                graph.set_all_edges_in_graph(in_graph=True)
                set_edges_in_graph_except_names(graph, ablated_exclusions, log=None)
                circuit_eval = evaluate_graph(
                    model=model,
                    graph=graph,
                    dataloader=dataloader,
                    metrics=metric_eval,
                    device=device,
                    prune=False,
                    quiet=args.quiet,
                    calc_input_for_nodes_not_in_graph=False,
                    debug_corrupted_construction=False,
                    calc_clean_logits=False,
                )
                ablated_metric = circuit_eval.mean().item()
                diff_metric = (original_scores - circuit_eval).mean().item()

            rows.append({
                "cluster1": c1,
                "cluster2": c2,
                "mean_score": mean_score,
                "p_ablated_mean_score": p_ablated_mean_score,
                "total_edges": total_edges,
                "p_ablated": p,
                "k_ablated": k_ablated,
                "p_actual": p_actual,
                f"ablated_{args.metric}": ablated_metric,
                f"original_{args.metric}": original_metric,
                f"diff_{args.metric}": diff_metric,
            })

        out_csv = _ablation_csv_path(
            output_dir, "activation_patching_ablation", args.repeat_type, args.counterfactual_type, args.random_state, p
        )
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"{log_prefix} Wrote {out_csv}")
    print(f"{log_prefix} Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Activation-patching cluster-cluster ablation: one CSV per p."
    )
    parser.add_argument("--repeat_type", type=str, required=True, choices=["identical", "approximate", "synthetic"])
    parser.add_argument("--model_type", type=str, required=True, choices=["esm3", "esm-c"])
    parser.add_argument("--counterfactual_type", type=str, required=True)
    parser.add_argument("--results_root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--datasets_root", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--in_graph_ratio", type=float, default=0.8)
    parser.add_argument(
        "--metric",
        type=str,
        default="logit_diff",
        choices=["logit_diff", "log_prob", "probability"],
    )
    parser.add_argument("--total_n_samples", type=int, default=None, help="Required for activation step.")
    parser.add_argument(
        "--percentages",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=42, help="Random state for samples")
    parser.add_argument("--quiet", action="store_true", default=False, help="Suppress verbose output from evaluation.")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["run_activation_patching_clusters", "analyze"],
        default=["run_activation_patching_clusters", "analyze"],
        help="Steps: 'run_activation_patching_clusters' (run ablation, write CSVs), 'analyze' (load CSVs, plot heatmaps per p).",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    datasets_root = args.datasets_root.resolve()
    output_dir = results_root / "interactions_graph" / args.model_type / args.counterfactual_type / "activation_patching_ablation" / args.metric
    output_dir.mkdir(parents=True, exist_ok=True)

    if "run_activation_patching_clusters" in args.steps:
        _run_activation_step(args, output_dir, results_root, datasets_root)
    if "analyze" in args.steps:
        _run_analyze_step(
            output_dir=output_dir,
            percentages=args.percentages,
            repeat_type=args.repeat_type,
            counterfactual_type=args.counterfactual_type,
            random_state=args.random_state,
            metric=args.metric,
        )
    print(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    sys.setrecursionlimit(3000)
    main()
