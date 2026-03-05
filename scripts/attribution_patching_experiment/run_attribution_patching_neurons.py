import argparse
import csv
import logging
import os
import sys
import traceback
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    SearchMinCircuitAboveThreshold,
    build_experiment_name,
    circuit_info_csv_path,
    create_induction_dataset_pandas,
    get_clean_corrupted_baselines,
    get_edges_or_nodes_number,
    search_min_circuit_binary_search_neurons,
)
from EAP_dataset import EAPDataset
from plms_repeats_circuits.EAP.graph import Graph, GraphType, NeuronGraph
from plms_repeats_circuits.EAP.attribute import attribute
from plms_repeats_circuits.EAP.circuit_selection import select_circuit_nodes, select_circuit_neurons
from plms_repeats_circuits.EAP.evaluate import (
    evaluate_graph_neurons,
    compute_faithfulness,
)
from plms_repeats_circuits.utils.patching_metrics import create_loss_and_metric
from plms_repeats_circuits.utils.experiment_utils import split_train_test
from plms_repeats_circuits.utils.model_utils import get_device
from plms_repeats_circuits.utils.esm_utils import load_model, load_tokenizer_by_model_type


def _build_neurons_graph_with_attribution(
    nodes_graph_json: str,
    model,
    train_df: pd.DataFrame,
    batch_size: int,
    output_dir: Path,
    experiment_name: str,
    n_nodes: int,
    circuit_selection_method: str,
    train_loss,
    attribution_patching_method: str,
    aggregation_method: str,
    abs_score: bool,
    EAP_IG_steps: int,
    device,
) -> tuple[NeuronGraph, int, int]:
    """Load nodes graph, create NeuronGraph, run attribution, verify scores, select top n_nodes, save graph."""
    nodes_graph = Graph.from_json(Path(nodes_graph_json))
    neurons_graph = NeuronGraph.from_model(model, graph_type=GraphType.Nodes)
    train_ds = EAPDataset(train_df)
    train_dl = train_ds.to_dataloader(batch_size)
    attribute(
        model=model,
        graph=neurons_graph,
        dataloader=train_dl,
        metric=train_loss,
        device=device,
        aggregation=aggregation_method,
        method=attribution_patching_method,
        quiet=False,
        abs_per_pos=abs_score,
        are_clean_logits_needed=False,
        eap_ig_steps=EAP_IG_steps,
        all_examples_scores_npy_path=None,
    )
    for name, node in nodes_graph.nodes.items():
        ng_node = neurons_graph.nodes.get(name)
        if ng_node is None:
            continue
        if node.score is not None and ng_node.score is not None:
            if not np.allclose(float(node.score), float(ng_node.score), rtol=1e-5, atol=1e-8):
                raise ValueError(f"Node score mismatch for {name}: nodes_graph={node.score}, neurons_graph={ng_node.score}")
    select_circuit_nodes(
        graph=neurons_graph,
        selection_method=circuit_selection_method,
        n_nodes=n_nodes,
        log=logging.getLogger(),
    )
    n_attn = neurons_graph.count_attention_nodes(filter_by_in_graph=True)
    n_mlp = neurons_graph.count_mlp_nodes(filter_by_in_graph=True)
    neurons_graph_path = output_dir / f"{experiment_name}.json"
    neurons_graph.to_json(neurons_graph_path)
    logging.info(f"Neurons graph saved to {neurons_graph_path}")
    return neurons_graph, n_attn, n_mlp


def main(
    csv_path: str,
    output_dir: str,
    total_n_samples: int,
    model_type: str,
    nodes_graph_json: str,
    n_nodes: int,
    attribution_patching_method: str = "EAP",
    circuit_selection_method: str = "top_n_abs",
    n_neurons_in_circuit_list: list[float] | None = None,
    min_performance_threshold_neurons: float = 0.89,
    max_performance_threshold_neurons: float = 0.9,
    is_per_layer_neurons: bool = False,
    metric: str = "logit_diff",
    abs_score: bool = False,
    aggregation_method: str = "sum",
    EAP_IG_steps: int = 5,
    batch_size: int = 1,
    random_state: int = 42,
    train_ratio: float = 0.5,
    exp_prefix: str = "",
    enable_min_circuit_search: bool = False,
    min_circuit_size: float = -1,
    max_circuit_size: float = -1,
    max_search_steps: int = 20,
    **kwargs,
):
    """
    Run attribution patching for neurons. Requires a ready nodes graph JSON and n_nodes.
    Loads nodes graph, creates NeuronGraph, runs attribution, selects top n_nodes, then finds neurons.
    """
    # Validation (for programmatic callers)
    if not nodes_graph_json or n_nodes is None:
        raise ValueError("nodes_graph_json and n_nodes are required")
    if not os.path.exists(nodes_graph_json):
        raise FileNotFoundError(f"Nodes graph JSON not found: {nodes_graph_json}")
    if circuit_selection_method not in ("top_n", "top_n_abs"):
        raise ValueError("circuit_selection_method must be top_n or top_n_abs for neurons")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if n_neurons_in_circuit_list is None:
        n_neurons_in_circuit_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    circuit_id = str(uuid.uuid4())
    graph_type = "neurons"

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
        exp_prefix=exp_prefix or None,
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
        ("nodes_graph_json", nodes_graph_json),
        ("n_nodes", n_nodes),
        ("attribution_patching_method", attribution_patching_method),
        ("circuit_selection_method", circuit_selection_method),
        ("n_neurons_in_circuit_list", n_neurons_in_circuit_list),
        ("min_performance_threshold_neurons", min_performance_threshold_neurons),
        ("max_performance_threshold_neurons", max_performance_threshold_neurons),
        ("is_per_layer_neurons", is_per_layer_neurons),
        ("metric", metric),
        ("abs_score", abs_score),
        ("aggregation_method", aggregation_method),
        ("EAP_IG_steps", EAP_IG_steps),
        ("batch_size", batch_size),
        ("random_state", random_state),
        ("train_ratio", train_ratio),
        ("exp_prefix", exp_prefix),
        ("enable_min_circuit_search", enable_min_circuit_search),
        ("min_circuit_size", min_circuit_size),
        ("max_circuit_size", max_circuit_size),
        ("max_search_steps", max_search_steps),
    ):
        logging.info("%s=%s", k, v)
    logging.info("Starting run_attribution_patching_neurons (ready nodes)")

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
    train_loss, metric_eval = create_loss_and_metric(metric)

    neurons_graph, n_attn, n_mlp = _build_neurons_graph_with_attribution(
        nodes_graph_json=nodes_graph_json,
        model=model,
        train_df=train_df,
        batch_size=batch_size,
        output_dir=output_dir,
        experiment_name=experiment_name,
        n_nodes=n_nodes,
        circuit_selection_method=circuit_selection_method,
        train_loss=train_loss,
        attribution_patching_method=attribution_patching_method,
        aggregation_method=aggregation_method,
        abs_score=abs_score,
        EAP_IG_steps=EAP_IG_steps,
        device=device,
    )

    test_ds = EAPDataset(test_df)
    test_dataloader = test_ds.to_dataloader(batch_size)
    clean_baseline, corrupted_baseline = get_clean_corrupted_baselines(model, test_dataloader, metric_eval, device)

    search_min_circuit = None
    if enable_min_circuit_search:
        circuit_info_path = circuit_info_csv_path(output_dir, "neurons")
        search_min_circuit = SearchMinCircuitAboveThreshold(
            global_csv_circuit_information=str(circuit_info_path),
            min_performance_threshold=min_performance_threshold_neurons,
            max_performance_threshold=max_performance_threshold_neurons,
            min_circuit_size=int(min_circuit_size) if min_circuit_size >= 0 else -1,
            max_circuit_size=int(max_circuit_size) if max_circuit_size >= 0 else -1,
            max_steps=max_search_steps,
        )

    csv_path_out = output_dir / f"{experiment_name}_neurons.csv"
    clean_mean = clean_baseline.mean().item()
    corrupted_mean = corrupted_baseline.mean().item()
    results = []
    csv_file = open(csv_path_out, "a", newline="")
    writer = csv.writer(csv_file)
    try:
        if csv_file.tell() == 0:
            writer.writerow([
                "n_neurons_input", "n_neurons", "is_per_layer", "pct_per_layer",
                "n_nodes_as_first_step_for_neurons", "n_attention_nodes_as_first_step", "mlp_nodes_as_first_step",
                "tot_n_neurons", "tot_pct_n_neurons", "circuit_size", "circuit_size_pct", "num_mlp_nodes",
                "circuit_selection_method",
                "clean_mean", "corrupted_mean", "circuit_mean", "faithfulness_by_mean", "faithfulness_per_element",
            ])
        total = neurons_graph.n_neurons_per_mlp if is_per_layer_neurons else neurons_graph.count_neurons(filter_by_in_graph=False)
        for n_input in sorted(n_neurons_in_circuit_list):
            n = get_edges_or_nodes_number(n_input, total)
            pct_per_layer = n / neurons_graph.n_neurons_per_mlp if is_per_layer_neurons else None
            select_circuit_neurons(
                graph=neurons_graph,
                selection_method=circuit_selection_method,
                n_neurons=n,
                n_nodes_as_first_step_for_neurons=n_nodes,
                is_per_layer=is_per_layer_neurons,
                log=logging.getLogger(),
            )
            empty_circuit = not neurons_graph.nodes["logits"].in_graph
            if empty_circuit and n != 0:
                continue
            circuit_evaluation = evaluate_graph_neurons(
                model=model, graph=neurons_graph, dataloader=test_dataloader, metrics=metric_eval,
                device=device, quiet=False, debug=False, calc_clean_logits=False,
            )
            faithfulness_by_mean = compute_faithfulness(
                clean_baseline=clean_baseline, corrupted_baseline=corrupted_baseline,
                target_baseline=circuit_evaluation, per_element_faithfulness=False,
            ).item()
            faithfulness_per_elem = compute_faithfulness(
                clean_baseline=clean_baseline, corrupted_baseline=corrupted_baseline,
                target_baseline=circuit_evaluation, per_element_faithfulness=True,
            ).item()
            circuit_mean = circuit_evaluation.mean().item()
            num_attention_nodes = neurons_graph.count_attention_nodes(filter_by_in_graph=True)
            num_mlp_nodes = neurons_graph.count_mlp_nodes(filter_by_in_graph=True)
            tot_n_neurons = neurons_graph.count_neurons(filter_by_in_graph=True)
            tot_pct_n_neurons = tot_n_neurons / neurons_graph.count_neurons(filter_by_in_graph=False)
            circuit_size = neurons_graph.count_included_nodes()
            circuit_size_pct = circuit_size / neurons_graph.count_total_nodes()
            writer.writerow([
                n_input, n, is_per_layer_neurons, pct_per_layer,
                n_nodes, n_attn, n_mlp,
                tot_n_neurons, tot_pct_n_neurons, circuit_size, circuit_size_pct, num_mlp_nodes,
                circuit_selection_method,
                clean_mean, corrupted_mean, circuit_mean, faithfulness_by_mean, faithfulness_per_elem,
            ])
            results.append((n, faithfulness_by_mean))
    finally:
        csv_file.close()

    if search_min_circuit is not None:
        search_min_circuit_binary_search_neurons(
            circuit_id=circuit_id,
            model=model,
            metric_evaluation=metric_eval,
            device=device,
            g=neurons_graph,
            circuit_selection_method=circuit_selection_method,
            search_min_circuit=search_min_circuit,
            results=results,
            clean_baseline=clean_baseline,
            corrupted_baseline=corrupted_baseline,
            test_dataloader=test_dataloader,
            is_per_layer=is_per_layer_neurons,
            n_nodes_as_first_step_for_neurons=n_nodes,
            use_file_lock=True,
        )

    logging.info("Finished run_attribution_patching_neurons")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Attribution patching for neurons. Requires a ready nodes graph JSON from run_attribution_patching_nodes_edges.",
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV (circuit_discovery format).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for logs, CSVs, and circuit JSONs.")
    parser.add_argument("--total_n_samples", type=int, required=True, help="Number of samples to use from CSV.")
    parser.add_argument("--model_type", type=str, required=True, choices=["esm3", "esm-c"], help="Model to load.")
    parser.add_argument(
        "--attribution_patching_method",
        type=str,
        default="EAP",
        choices=["EAP", "EAP-IG"],
        help="Attribution method.",
    )
    parser.add_argument(
        "--circuit_selection_method",
        type=str,
        default="top_n_abs",
        choices=["top_n", "top_n_abs"],
        help="Circuit selection for nodes and neurons (top_n or top_n_abs).",
    )
    parser.add_argument(
        "--n_neurons_in_circuit_list",
        type=float,
        nargs="+",
        default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        help="Neuron circuit sizes to evaluate. Values <1 = fraction; >=1 = absolute count.",
    )
    parser.add_argument("--min_performance_threshold_neurons", type=float, default=0.89,
                        help="Min faithfulness for neuron-level binary search.")
    parser.add_argument("--max_performance_threshold_neurons", type=float, default=0.9,
                        help="Max faithfulness for neuron-level binary search.")
    parser.add_argument(
        "--is_per_layer_neurons",
        action="store_true",
        help="Select neurons per MLP layer.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="logit_diff",
        choices=["logit_diff", "log_prob"],
        help="Evaluation metric.",
    )
    parser.add_argument("--abs_score", action="store_true", help="Use absolute attribution scores before aggregation.")
    parser.add_argument(
        "--aggregation_method",
        type=str,
        default="sum",
        choices=["sum", "pos_mean"],
        help="How to aggregate attribution scores.",
    )
    parser.add_argument(
        "--nodes_graph_json",
        type=str,
        required=True,
        help="Path to nodes graph JSON (from run_attribution_patching_nodes_edges).",
    )
    parser.add_argument(
        "--n_nodes",
        type=int,
        required=True,
        help="Number of nodes to use (must match the circuit in nodes_graph_json).",
    )
    parser.add_argument("--EAP_IG_steps", type=int, default=5, help="IG steps for EAP-IG.")
    parser.add_argument("--batch_size", type=int, default=1, help="Dataloader batch size.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument("--train_ratio", type=float, default=0.5, help="Fraction of data for train.")
    parser.add_argument("--exp_prefix", type=str, default="", help="Prefix for experiment name in filenames.")
    parser.add_argument(
        "--enable_min_circuit_search",
        action="store_true",
        help="Run binary search for minimal neuron circuit within threshold.",
    )
    parser.add_argument(
        "--min_circuit_size",
        type=float,
        default=-1,
        help="Min circuit size for neuron search. >=1=count, <1=fraction, -1=infer.",
    )
    parser.add_argument(
        "--max_circuit_size",
        type=float,
        default=-1,
        help="Max circuit size for neuron search. Same rules as min_circuit_size.",
    )
    parser.add_argument("--max_search_steps", type=int, default=20, help="Max binary-search iterations.")
    return parser.parse_args()


if __name__ == "__main__":
    sys.setrecursionlimit(3000)
    args = _parse_args()
    try:
        main(**vars(args))
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
