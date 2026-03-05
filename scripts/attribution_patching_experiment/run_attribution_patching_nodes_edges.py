import argparse
import csv
import logging
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

import pandas as pd
import torch

from utils import (
    SearchMinCircuitAboveThreshold,
    build_experiment_name,
    circuit_info_csv_path,
    create_induction_dataset_pandas,
    get_clean_corrupted_baselines,
    get_edges_or_nodes_number,
    search_min_circuit_binary_search,
)
from EAP_dataset import EAPDataset
from plms_repeats_circuits.EAP.graph import Graph, GraphType
from plms_repeats_circuits.EAP.attribute import attribute
from plms_repeats_circuits.EAP.circuit_selection import select_circuit_edges, select_circuit_nodes
from plms_repeats_circuits.EAP.evaluate import (
    evaluate_graph,
    evaluate_graph_node,
    compute_faithfulness,
)
from plms_repeats_circuits.utils.patching_metrics import create_loss_and_metric
from plms_repeats_circuits.utils.experiment_utils import split_train_test
from plms_repeats_circuits.utils.model_utils import get_device
from plms_repeats_circuits.utils.esm_utils import load_model, load_tokenizer_by_model_type

DEFAULT_N_COMPONENTS_IN_CIRCUIT_LIST = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def _load_graph_and_perform_attribution_if_needed(
    experiment_name,
    output_dir,
    train_df,
    model,
    device,
    train_loss,
    attribution_patching_method,
    abs_score,
    aggregation_method,
    circuit_json_path,
    EAP_IG_steps,
    batch_size,
    graph_type: str,
    all_examples_scores_npy_path: str | None = None,
):
    """Load graph from JSON or create new, run attribution if needed, save to JSON."""
    if circuit_json_path is None:
        file_path = Path(output_dir) / f"{experiment_name}.json"
    else:
        file_path = Path(circuit_json_path)

    perform_attribution = not os.path.exists(file_path) or file_path.suffix.lower() != ".json"

    if not perform_attribution:
        logging.info(f"Loading existing graph from JSON: {file_path}")
        g = Graph.from_json(file_path)
        if graph_type == "edges":
            edge = next(iter(g.edges.values()), None)
            perform_attribution = edge is None or edge.score is None
        else:
            node = next(iter(g.nodes.values()), None)
            perform_attribution = node is None or node.score is None
    else:
        logging.info("Creating new graph from model configuration.")
        graph_type_enum = GraphType.Edges if graph_type == "edges" else GraphType.Nodes
        g = Graph.from_model(model, graph_type=graph_type_enum)

    if perform_attribution:
        logging.info("Performing attribution patching.")
        train_ds = EAPDataset(train_df)
        train_dataloader = train_ds.to_dataloader(batch_size)
        attribute(
            model=model,
            graph=g,
            dataloader=train_dataloader,
            metric=train_loss,
            device=device,
            aggregation=aggregation_method,
            method=attribution_patching_method,
            quiet=False,
            abs_per_pos=abs_score,
            are_clean_logits_needed=False,
            eap_ig_steps=EAP_IG_steps,
            all_examples_scores_npy_path=all_examples_scores_npy_path,
        )
        logging.info("Attribution patching completed.")
        g.to_json(file_path)
        logging.info(f"Graph saved to {file_path}.")

    return g


def main(
    csv_path: str,
    output_dir: str,
    total_n_samples: int,
    model_type: str,
    graph_type: str,
    attribution_patching_method: str = "EAP",
    circuit_selection_method: str = "greedy_abs",
    n_components_in_circuit_list: list[float] | None = None,
    metric: str = "logit_diff",
    abs_score: bool = False,
    aggregation_method: str = "sum",
    circuit_json_path: str | None = None,
    EAP_IG_steps: int = 5,
    batch_size: int = 1,
    random_state: int = 42,
    train_ratio: float = 0.5,
    exp_prefix: str = "",
    enable_min_circuit_search: bool = False,
    min_performance_threshold: float = 0.8,
    max_performance_threshold: float = 0.85,
    min_circuit_size: float = -1,
    max_circuit_size: float = -1,
    max_search_steps: int = 20,
    save_results_csv: bool = True,
    save_scores_per_example_npy: bool = False,
    **kwargs,
):
    """
    Run attribution patching for edges or nodes.

    Pipeline:
    1. Load CSV, create induction dataset, split train/test.
    2. Build graph and run attribution (EAP or EAP-IG) on train, or load from circuit_json_path.
    3. For each circuit size in n_components_in_circuit_list: select circuit, evaluate faithfulness on test.
    4. Save graph JSON. Optionally run binary search for minimal circuit (--enable_min_circuit_search).

    Returns:
        None if save_results_csv=True (writes {experiment_name}.csv).
        pd.DataFrame of results if save_results_csv=False (for programmatic callers, e.g. run_cross_task).
    """
    # Validation: needed for programmatic callers (run_cross_task etc.) who don't use argparse
    if graph_type not in ("edges", "nodes"):
        raise ValueError(f"graph_type must be 'edges' or 'nodes', got {graph_type}")
    if metric not in ("logit_diff", "log_prob"):
        raise ValueError(f"metric must be 'logit_diff' or 'log_prob', got {metric}")
    # circuit_selection_method depends on graph_type (edges: greedy/top_n/none; nodes: top_n only)
    supported_edges = ["greedy", "greedy_abs", "top_n", "top_n_abs", "none"]
    supported_nodes = ["top_n", "top_n_abs"]
    supported = supported_edges if graph_type == "edges" else supported_nodes
    if circuit_selection_method not in supported:
        raise ValueError(f"circuit_selection_method must be in {supported} for graph_type={graph_type}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if n_components_in_circuit_list is None:
        n_components_in_circuit_list = DEFAULT_N_COMPONENTS_IN_CIRCUIT_LIST

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    circuit_id = str(uuid.uuid4())

    experiment_name = build_experiment_name(
        circuit_id=circuit_id,
        total_n_samples=total_n_samples,
        random_state=random_state,
        attribution_patching_method=attribution_patching_method,
        circuit_selection_method=circuit_selection_method,
        aggregation_method=aggregation_method,
        abs_score=abs_score,
        metric=metric,
        graph_type=graph_type,
        exp_prefix=exp_prefix or None
    )

    all_examples_scores_npy_path = (
        str(output_dir / f"{experiment_name}_scores_per_example.npy")
        if save_scores_per_example_npy
        else None
    )

    log_file = output_dir / f"{experiment_name}.log"
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    for k, v in (
        ("csv_path", csv_path),
        ("output_dir", str(output_dir)),
        ("total_n_samples", total_n_samples),
        ("model_type", model_type),
        ("graph_type", graph_type),
        ("attribution_patching_method", attribution_patching_method),
        ("circuit_selection_method", circuit_selection_method),
        ("n_components_in_circuit_list", n_components_in_circuit_list),
        ("metric", metric),
        ("abs_score", abs_score),
        ("aggregation_method", aggregation_method),
        ("circuit_json_path", circuit_json_path),
        ("EAP_IG_steps", EAP_IG_steps),
        ("batch_size", batch_size),
        ("random_state", random_state),
        ("train_ratio", train_ratio),
        ("exp_prefix", exp_prefix),
        ("enable_min_circuit_search", enable_min_circuit_search),
        ("min_performance_threshold", min_performance_threshold),
        ("max_performance_threshold", max_performance_threshold),
        ("min_circuit_size", min_circuit_size),
        ("max_circuit_size", max_circuit_size),
        ("max_search_steps", max_search_steps),
        ("save_results_csv", save_results_csv),
        ("save_scores_per_example_npy", save_scores_per_example_npy),
    ):
        logging.info("%s=%s", k, v)
    logging.info(f"Starting run_attribution_patching_nodes_edges, experiment_name={experiment_name}")

    device = get_device()
    model = load_model(
        model_type=model_type,
        device=device,
        use_transformer_lens_model=True,
        cache_attention_activations=True,
        cache_mlp_activations=True,
        output_type="sequence",
        cache_attn_pattern=False,
        split_qkv_input=True,
    )
    tokenizer = load_tokenizer_by_model_type(model_type)

    df = pd.read_csv(csv_path)
    df = create_induction_dataset_pandas(df, total_n_samples, random_state, tokenizer=tokenizer, metric=metric)
    train_df, test_df = split_train_test(df, train_ratio)
    logging.info(f"Train size: {len(train_df)}, test size: {len(test_df)}")

    train_loss, metric_eval = create_loss_and_metric(metric)

    g = _load_graph_and_perform_attribution_if_needed(
        experiment_name=experiment_name,
        output_dir=str(output_dir),
        train_df=train_df,
        model=model,
        device=device,
        train_loss=train_loss,
        attribution_patching_method=attribution_patching_method,
        abs_score=abs_score,
        aggregation_method=aggregation_method,
        circuit_json_path=circuit_json_path,
        EAP_IG_steps=EAP_IG_steps,
        batch_size=batch_size,
        graph_type=graph_type,
        all_examples_scores_npy_path=all_examples_scores_npy_path,
    )

    total = g.count_total_edges() if graph_type == "edges" else g.count_total_nodes()
    test_ds = EAPDataset(test_df)
    test_dataloader = test_ds.to_dataloader(batch_size)

    clean_baseline, corrupted_baseline = get_clean_corrupted_baselines(
        model, test_dataloader, metric_eval, device
    )
    clean_mean = clean_baseline.mean().item()
    corrupted_mean = corrupted_baseline.mean().item()
    logging.info(f"clean_mean: {clean_mean}, corrupted_mean: {corrupted_mean}")

    results = []
    rows = []
    csv_path_out = output_dir / f"{experiment_name}.csv"

    csv_file = open(csv_path_out, "a", newline="") if save_results_csv else None
    writer = csv.writer(csv_file) if csv_file else None
    try:
        if save_results_csv and csv_file and csv_file.tell() == 0:
            if graph_type == "edges":
                writer.writerow([
                    "n_edges_input", "n_edges", "real_n_edges", "real_pct_n_edges", "circuit_selection_method",
                    "clean_mean", "corrupted_mean", "circuit_mean", "faithfulness_by_mean", "faithfulness_per_element",
                ])
            else:
                writer.writerow([
                    "n_nodes_input", "n_nodes", "real_n_nodes", "real_pct_n_nodes", "circuit_selection_method",
                    "clean_mean", "corrupted_mean", "circuit_mean", "faithfulness_by_mean", "faithfulness_per_element",
                ])

        for n_input in sorted(n_components_in_circuit_list):
            n = get_edges_or_nodes_number(n_input, total)
            if graph_type == "edges":
                select_circuit_edges(
                    graph=g,
                    selection_method=circuit_selection_method,
                    n_edges=n,
                    log=logging.getLogger(),
                )
            else:
                select_circuit_nodes(
                    graph=g,
                    selection_method=circuit_selection_method,
                    n_nodes=n,
                    log=logging.getLogger(),
                )

            empty_circuit = not g.nodes["logits"].in_graph
            if empty_circuit and n != 0:
                logging.info(f"Skipping n={n} (empty circuit)")
                continue

            real_n = g.count_included_edges() if graph_type == "edges" else g.count_included_nodes()
            real_pct = real_n / total

            if graph_type == "edges":
                circuit_evaluation = evaluate_graph(
                    model=model,
                    graph=g,
                    dataloader=test_dataloader,
                    metrics=metric_eval,
                    device=device,
                    prune=False,
                    quiet=False,
                    calc_input_for_nodes_not_in_graph=False,
                    debug_corrupted_construction=False,
                    calc_clean_logits=False,
                )
            else:
                circuit_evaluation = evaluate_graph_node(
                    model=model,
                    graph=g,
                    dataloader=test_dataloader,
                    metrics=metric_eval,
                    device=device,
                    quiet=False,
                    debug=False,
                    calc_clean_logits=False,
                )

            faithfulness_by_mean = compute_faithfulness(
                clean_baseline=clean_baseline,
                corrupted_baseline=corrupted_baseline,
                target_baseline=circuit_evaluation,
                per_element_faithfulness=False,
            ).item()
            faithfulness_per_elem = compute_faithfulness(
                clean_baseline=clean_baseline,
                corrupted_baseline=corrupted_baseline,
                target_baseline=circuit_evaluation,
                per_element_faithfulness=True,
            ).item()
            circuit_mean = circuit_evaluation.mean().item()

            row = [
                n_input, n, real_n, real_pct, circuit_selection_method,
                clean_mean, corrupted_mean, circuit_mean,
                faithfulness_by_mean, faithfulness_per_elem,
            ]
            rows.append(row)
            if save_results_csv and writer:
                writer.writerow(row)
            results.append((n, real_n, faithfulness_by_mean))
            logging.info(f"n_input={n_input} -> n={n}: faithfulness_by_mean={faithfulness_by_mean}")
    finally:
        if csv_file:
            csv_file.close()


    search_min_circuit = None
    if enable_min_circuit_search:
        circuit_info_path = circuit_info_csv_path(output_dir, graph_type)
        search_min_circuit = SearchMinCircuitAboveThreshold(
            global_csv_circuit_information=str(circuit_info_path),
            min_performance_threshold=min_performance_threshold,
            max_performance_threshold=max_performance_threshold,
            min_circuit_size=int(min_circuit_size) if min_circuit_size >= 0 else -1,
            max_circuit_size=int(max_circuit_size) if max_circuit_size >= 0 else -1,
            max_steps=max_search_steps,
        )
        try:
            search_min_circuit_binary_search(
                circuit_id=circuit_id,
                model=model,
                metric_evaluation=metric_eval,
                device=device,
                g=g,
                circuit_selection_method=circuit_selection_method,
                search_min_circuit=search_min_circuit,
                results=results,
                clean_baseline=clean_baseline,
                corrupted_baseline=corrupted_baseline,
                test_dataloader=test_dataloader,
                graph_type=graph_type,
                use_file_lock=True,
            )
        except Exception as e:
            logging.error(f"search_min_circuit_binary_search failed: {e}")

    logging.info("Finished run_attribution_patching_nodes_edges")

    if save_results_csv:
        return None

    # Return results as DataFrame
    cols = (
        ["n_edges_input", "n_edges", "real_n_edges", "real_pct_n_edges", "circuit_selection_method"]
        if graph_type == "edges"
        else ["n_nodes_input", "n_nodes", "real_n_nodes", "real_pct_n_nodes", "circuit_selection_method"]
    )
    cols += ["clean_mean", "corrupted_mean", "circuit_mean", "faithfulness_by_mean", "faithfulness_per_element"]
    return pd.DataFrame(rows, columns=cols)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Attribution patching for edges or nodes. "
        "Loads data, runs EAP/EAP-IG attribution, selects circuits by size, evaluates faithfulness."
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV (circuit_discovery format).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for logs, CSVs, and circuit JSON.")
    parser.add_argument("--total_n_samples", type=int, required=True, help="Number of samples to use from CSV.")
    parser.add_argument("--model_type", type=str, required=True, choices=["esm3", "esm-c"], help="Model to load.")
    parser.add_argument(
        "--graph_type",
        type=str,
        default="edges",
        choices=["edges", "nodes"],
        help="Circuit granularity: edges (per-edge scores) or nodes (per-node scores).",
    )
    parser.add_argument(
        "--attribution_patching_method",
        type=str,
        default="EAP",
        choices=["EAP", "EAP-IG"],
        help="Attribution method: EAP or EAP-IG.",
    )
    parser.add_argument(
        "--circuit_selection_method",
        type=str,
        default="greedy_abs",
        help="Circuit selection: greedy, greedy_abs, top_n, top_n_abs, or none.",
    )
    parser.add_argument(
        "--n_components_in_circuit_list",
        type=float,
        nargs="+",
        default=DEFAULT_N_COMPONENTS_IN_CIRCUIT_LIST,
        help="Circuit sizes to evaluate. Values <1 = fraction of total; >=1 = absolute count.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="logit_diff",
        choices=["logit_diff", "log_prob"],
        help="Evaluation metric.",
    )
    parser.add_argument("--abs_score", action="store_true", help="Use absolute attribution scores per position before aggregation of a score, see EAP documentation.")
    parser.add_argument(
        "--aggregation_method",
        type=str,
        default="sum",
        choices=["sum", "pos_mean"],
        help="How to aggregate attribution scores we get per position in a full sequence, see EAP documentation.",
    )
    parser.add_argument(
        "--circuit_json_path",
        type=str,
        default=None,
        help="Load graph from JSON (skip attribution if file exists).",
    )
    parser.add_argument("--EAP_IG_steps", type=int, default=5, help="IG steps for EAP-IG.")
    parser.add_argument("--batch_size", type=int, default=1, help="Dataloader batch size.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for train/test split and sampling.")
    parser.add_argument("--train_ratio", type=float, default=0.5, help="Fraction of data for train (rest for test).")
    parser.add_argument("--exp_prefix", type=str, default="", help="Prefix for experiment name in filenames.")
    parser.add_argument(
        "--enable_min_circuit_search",
        action="store_true",
        help="Run binary search for minimal circuit within performance threshold.",
    )
    parser.add_argument(
        "--min_performance_threshold",
        type=float,
        default=0.8,
        help="Min faithfulness for minimal circuit search.",
    )
    parser.add_argument(
        "--max_performance_threshold",
        type=float,
        default=0.85,
        help="Max faithfulness for minimal circuit search.",
    )
    parser.add_argument(
        "--min_circuit_size",
        type=float,
        default=-1,
        help="Min circuit size for search. >=1=count, <1=fraction, -1=infer from results.",
    )
    parser.add_argument(
        "--max_circuit_size",
        type=float,
        default=-1,
        help="Max circuit size for search. Same rules as min_circuit_size.",
    )
    parser.add_argument("--max_search_steps", type=int, default=20, help="Max binary-search iterations.")
    parser.add_argument(
        "--save_scores_per_example_npy",
        action="store_true",
        help="Save per-example attribution scores to NPY, see EAP documentation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.setrecursionlimit(3000)
    args = _parse_args()
    kwargs = vars(args).copy()
    # CLI always writes results to CSV. save_results_csv is not in argparse; programmatic
    # callers (e.g. run_cross_task) pass save_results_csv=False to get a DataFrame.
    kwargs["save_results_csv"] = True

    try:
        main(**kwargs)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
