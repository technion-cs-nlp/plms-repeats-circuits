import argparse
import pandas as pd
import os
from utils import create_induction_dataset_pandas
from EAP_dataset import EAPDataset
from transformer_lens import HookedESM3, HookedESMC
from plms_repeats_circuits.EAP.graph import Graph, Edge, GraphType, NeuronGraph
from plms_repeats_circuits.EAP.attribute import attribute
from plms_repeats_circuits.EAP.circuit_selection import (
    select_circuit_edges,
    select_circuit_nodes,
    select_circuit_neurons,
)
from plms_repeats_circuits.EAP.evaluate import evaluate_baseline, evaluate_graph, compute_faithfulness, evaluate_graph_node, evaluate_graph_neurons
from plms_repeats_circuits.utils.patching_metrics import create_loss_and_metric
from pathlib import Path
import logging
import time
from functools import partial
from plms_repeats_circuits.utils.experiment_utils import split_train_test
from plms_repeats_circuits.utils.model_utils import get_device
import csv
import traceback
import sys
import math
import torch
from dataclasses import dataclass
import uuid
from plms_repeats_circuits.utils.esm_utils import load_model, load_tokenizer_by_model_type


@dataclass
class SearchMinCircuitAboveThreshold:
    global_csv_circuit_information: str  # Path to a CSV file
    min_performance_threshold: float
    max_performance_threshold: float   
    min_circuit_size: int = -1           
    max_circuit_size: int =-1
    max_steps: int =20                 

@dataclass
class CrossTaskFaithfulnessArgs:
    source_task_name: str
    source_circuit_id: str
    target_task_name: str
    target_circuit_id: str
    save_results_csv: str 

@dataclass
class FindNeuronsCircuitsTwoStepsArgs:
    is_per_layer: bool
    min_performance_threshold_neurons: float
    max_performance_threshold_neurons: float
    two_steps_neuron_circuit_node_circuit_list_sizes: list[float]

def load_graph_and_perform_attribution_patching_if_needed(
    experiment_name, output_dir, train_df, model, device, train_loss, attribution_patching_method, abs_score, aggregation_method, circuit_json_path, EAP_IG_steps, batch_size, graph_type="edges",
    neurons_graph: bool=False,
    all_examples_scores_npy_path: str=None
):
    logging.info("Entering load_graph_and_perform_attribution_patching_if_needed")
    
    if circuit_json_path is None:
        file_path = Path(output_dir) / f'{experiment_name}.json'
    else:
        file_path = Path(circuit_json_path)

    perform_attribution = not os.path.exists(file_path) or file_path.suffix.lower() != ".json"
    
    if not perform_attribution:
        logging.info(f"Loading existing graph from JSON.- {file_path}")
        g = NeuronGraph.from_json(file_path) if (graph_type == "nodes" and neurons_graph) else Graph.from_json(file_path)

        logging.info(f"Graph type: {type(g)}")
        if neurons_graph and not isinstance(g, NeuronGraph):
            raise ValueError("Graph provided is not a neurons graph.")

        if graph_type == "edges" and isinstance(g, NeuronGraph):
            raise ValueError("Graph provided is a neurons graph, but you were asked for edges graph.")

        if graph_type == "edges":
            edge = next(iter(g.edges.values()), None)
            perform_attribution = edge is None or edge.score is None
        else:
            node = next(iter(g.nodes.values()), None)
            perform_attribution = node is None or node.score is None
    else:
        logging.info("Creating new graph from model configuration.")
        graph_type_enum = GraphType.Edges if graph_type=="edges" else GraphType.Nodes
        g = Graph.from_model(model, graph_type=graph_type_enum) if neurons_graph==False else NeuronGraph.from_model(model, graph_type=graph_type_enum)
    
    if perform_attribution:
        logging.info("Performing attribution patching.")
        train_ds = EAPDataset(train_df)
        train_dataloader = train_ds.to_dataloader(batch_size)
        attribute(
            model=model, graph=g, dataloader=train_dataloader, metric=train_loss, 
            device=device, aggregation=aggregation_method, method=attribution_patching_method, 
            quiet=False, abs_per_pos=abs_score, are_clean_logits_needed=False, eap_ig_steps=EAP_IG_steps, 
            all_examples_scores_npy_path=all_examples_scores_npy_path
        )

        logging.info("Attribution patching completed.")
        g.to_json(file_path)
        logging.info(f"Graph saved to {file_path}.")
    
    logging.info("Exiting load_graph_and_perform_attribution_patching_if_needed")
    return g

def get_clean_corrupted_baselines(model, test_dataloader, metric, device):
    logging.info("Entering get_clean_corrupted_baselines")
    prev_attn_res = model.cfg.use_attn_result
    prev_use_split_qkv_input =model.cfg.use_split_qkv_input
    prev_use_hook_mlp_in = model.cfg.use_hook_mlp_in
    model.cfg.use_attn_result= False
    model.cfg.use_split_qkv_input= False
    model.cfg.use_hook_mlp_in=False
    clean = evaluate_baseline(model=model, dataloader=test_dataloader, metrics=metric, device=device, run_corrupted=False, quiet=False)
    corrupted = evaluate_baseline(model=model, dataloader=test_dataloader, metrics=metric, device=device, run_corrupted=True, quiet=False)
    model.cfg.use_attn_result= prev_attn_res
    model.cfg.use_split_qkv_input=prev_use_split_qkv_input
    model.cfg.use_hook_mlp_in=prev_use_hook_mlp_in
    logging.info(f"use_attn_result:{model.cfg.use_attn_result}, use_split_qkv_input:{model.cfg.use_split_qkv_input},use_hook_mlp_in:{model.cfg.use_hook_mlp_in }")
    logging.info(f"Finished get_clean_corrupted_baselines")
    return clean, corrupted

def get_edges_or_nodes_number(n, total):
    if n > 1:
        if n != int(n):
            raise ValueError(f"n_edges must be an integer when > 1. Got: {n}")
        return int(n)
    if n < 0:
        raise ValueError(f"n_edges must be non-negative. Got: {n}")
    
    return math.ceil(n * total)

def save_cross_task_results(cross_task_faithfulness_args: CrossTaskFaithfulnessArgs, results, total, graph_type="edges"):
    
    logging.info("Entering save_cross_task_results")
    if len(results) == 0:
        logging.warning("No results to save.")
        return
    csv_dir = os.path.dirname(cross_task_faithfulness_args.save_results_csv)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
    
    with open(cross_task_faithfulness_args.save_results_csv, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if csv_file.tell() == 0:  # Check if the file pointer is at the start (file is empty)
            if graph_type == "edges":
                writer.writerow(["source_task_name", "source_circuit_id", "target_task_name", "target_circuit_id", "n_edges", "real_n_edges", "pct_edges", "real_pct_edges", "faithfulness_by_mean"])
            else: 
                writer.writerow(["source_task_name", "source_circuit_id", "target_task_name", "target_circuit_id", "n_nodes", "real_n_nodes", "pct_nodes", "real_pct_nodes", "faithfulness_by_mean"])
        
        #writer.writerow([cross_task_faithfulness_args.source_task_name, cross_task_faithfulness_args.source_circuit_id, cross_task_faithfulness_args.target_task_name, cross_task_faithfulness_args.target_circuit_id, results])
        #results.append((n_edges, real_n_edges ,faithfulness_by_mean)

        for n, real_n, faithfulness_by_mean in results:
            pct = n / total
            real_pct = real_n / total
            writer.writerow([cross_task_faithfulness_args.source_task_name, cross_task_faithfulness_args.source_circuit_id, cross_task_faithfulness_args.target_task_name,
                             cross_task_faithfulness_args.target_circuit_id, n, real_n, pct, real_pct, faithfulness_by_mean])
            
    logging.info(f"Results saved to {cross_task_faithfulness_args.save_results_csv}.")


def validate_cross_task_faithfulness_input_args_and_build_data_class(args) -> CrossTaskFaithfulnessArgs:
    logging.info("Entering validate_cross_task_faithfulness_input_args_and_build_data_class")
    if args.source_task_name is None or args.source_circuit_id is None:
        raise ValueError("source_task_name and source_circuit_id must be provided for cross-task faithfulness evaluation.")
    if args.target_task_name is None or args.target_circuit_id is None:
        raise ValueError("target_task_name and target_circuit_id must be provided for cross-task faithfulness evaluation.")
    if args.save_results_csv is None:
        raise ValueError("save_results_csv must be provided for cross-task faithfulness evaluation.")
    
    cross_task_faithfulness_args = CrossTaskFaithfulnessArgs(
        source_task_name=args.source_task_name,
        source_circuit_id=args.source_circuit_id,
        target_task_name=args.target_task_name,
        target_circuit_id=args.target_circuit_id,
        save_results_csv=args.save_results_csv
    )
    
    logging.info(f"Cross-task faithfulness arguments: {cross_task_faithfulness_args}")
    logging.info("Exiting validate_cross_task_faithfulness_input_args_and_build_data_class")
    return cross_task_faithfulness_args

def search_min_circuit_binary_search(circuit_id, task_name, model, metric_evaluation, device, g:Graph, 
                                     post_attribution_patching_processing, 
                                     search_min_circuit:SearchMinCircuitAboveThreshold, 
                                     results, clean_baseline, corrupted_baseline, test_dataloader, graph_type="edges"):
    
    best_circuits_csv = search_min_circuit.global_csv_circuit_information
    # Add at the beginning of the function
    total = len(g.edges) if graph_type == "edges" else g.count_total_nodes()
    csv_dir = os.path.dirname(best_circuits_csv)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
    def _get_init_search_range():
        if search_min_circuit.min_circuit_size >= 0 and search_min_circuit.max_circuit_size > search_min_circuit.min_circuit_size:
            return get_edges_or_nodes_number(search_min_circuit.min_circuit_size, total), get_edges_or_nodes_number(search_min_circuit.max_circuit_size,total), None, None
        if results is None or len(results) == 0:
            raise ValueError("there isnt a way to compute valid range for finding best circuit")
        sorted_results = sorted(results, key=lambda x: x[0])
        start = 0
        end = len(g.edges.values()) if graph_type == "edges" else g.count_total_nodes()
        start_performance = 0.0
        real_n = None
        for i, (n, curr_real_n, performance) in enumerate(sorted_results):
            if performance > search_min_circuit.min_performance_threshold:
                end = n
                if i != 0:
                    start = sorted_results[i-1][0]
                start_performance = performance
                real_n = curr_real_n
                break
        return start, end, start_performance, real_n

    start, end, curr_performance, curr_real_n = _get_init_search_range()
    res_n, res_performance, res_real = None, None, None
    logging.info(f"Search best circuit: initial: start: {start}, end: {end}, curr_performance {curr_performance}, curr_real_n {curr_real_n}")
    # Track the closest performance to the desired window in case we never land inside it
    closest_n, closest_real, closest_perf, closest_dist = None, None, None, None
    def _distance_to_window(value: float, low: float, high: float) -> float:
        if value < low:
            return low - value
        if value > high:
            return value - high
        return 0.0
    # early stopping
    if (curr_performance is not None and curr_real_n is not None and start is not None and end is not None and
        search_min_circuit.min_performance_threshold <= curr_performance <= search_min_circuit.max_performance_threshold):
        res_performance = curr_performance
        res_n = end
        res_real = curr_real_n
    else:
        steps = 0
        while start <= end and steps < search_min_circuit.max_steps:
            steps += 1
            n =(start + end) // 2
            logging.info(f"Search best circuit: Evaluating circuit for input edge midpoint: {n}, start: {start}, end: {end}")
            
            if graph_type == "edges":
                select_circuit_edges(graph=g, selection_method=post_attribution_patching_processing, n_edges=n, log=logging.getLogger())
            else:
                select_circuit_nodes(graph=g, selection_method=post_attribution_patching_processing, n_nodes=n, log=logging.getLogger())
            empty_circuit = not g.nodes['logits'].in_graph
            
            if not empty_circuit:
                if graph_type == "edges":
                    circuit_evaluation = evaluate_graph(model=model, graph=g, dataloader=test_dataloader, 
                                                    metrics=metric_evaluation, device=device, prune=False, 
                                                    quiet=False, calc_input_for_nodes_not_in_graph=False, 
                                                    debug_corrupted_construction=False, calc_clean_logits=False)
                else:
                    if isinstance(g, NeuronGraph):
                        circuit_evaluation = evaluate_graph_neurons(model=model, graph=g, dataloader=test_dataloader, metrics=metric_evaluation,
                                                            device=device, quiet=False, debug=False, calc_clean_logits=False)
                    else:
                        circuit_evaluation = evaluate_graph_node(model=model, graph=g, dataloader=test_dataloader, 
                                                        metrics=metric_evaluation, device=device, quiet=False, 
                                                        debug=False, calc_clean_logits=False)
                    
                faithfulness_by_mean = compute_faithfulness(clean_baseline=clean_baseline, 
                                                            corrupted_baseline=corrupted_baseline, 
                                                            target_baseline=circuit_evaluation, 
                                                            per_element_faithfulness=False).item()
                logging.info(f"Search best circuit: {n} : faithfulness_by_mean: {faithfulness_by_mean}")
                curr_performance = faithfulness_by_mean 
            else:
                curr_performance = 0.0
            
            #real_n = len([e for e in g.edges.values() if e.in_graph]) if graph_type == "edges" else len([n for n in g.nodes.values() if n.in_graph])
            real_n = g.count_included_edges() if graph_type == "edges" else g.count_included_nodes()
            if isinstance(g, NeuronGraph):
                logging.info(f"num attention nodes in graph: {g.count_attention_nodes(filter_by_in_graph=True)}, num neurons in graph: {g.count_neurons(filter_by_in_graph=True)}")
            else:
                logging.info(f"num attention nodes in graph: {g.count_attention_nodes(filter_by_in_graph=True)}, num mlp nodes in graph: {g.count_mlp_nodes(filter_by_in_graph=True)}")

            # Update closest candidate if needed
            curr_dist = _distance_to_window(curr_performance, search_min_circuit.min_performance_threshold, search_min_circuit.max_performance_threshold)
            if closest_dist is None or curr_dist < closest_dist or (curr_dist == closest_dist and closest_n is not None and n < closest_n):
                closest_dist, closest_n, closest_real, closest_perf = curr_dist, n, real_n, curr_performance

            if search_min_circuit.min_performance_threshold <= curr_performance <= search_min_circuit.max_performance_threshold:
                res_performance = curr_performance
                res_n = n
                res_real = real_n
                break
            
            elif curr_performance > search_min_circuit.max_performance_threshold:
                end = n-1 # we can reduce 1 cause anyway result is too hight
            else:  # curr_performance < search_min_circuit.min_performance_threshold
                start = n+1 # we can add 1 cause anyway result is too hight
                
    if res_n is None or res_performance is None:
        # Fallback to the closest performance observed during search
        if closest_n is not None and closest_perf is not None:
            logging.warning(f"Search best circuit: no performance in range; selecting closest performance {closest_perf} at n={closest_n} (distance {closest_dist})")
            res_n, res_real, res_performance = closest_n, closest_real, closest_perf
        else:
            logging.warning(f"Search best circuit: could not find any circuit that has performance inside this range in {search_min_circuit.max_steps} steps and no closest candidate available")
            return  # Early return if no valid result

    # Only write to CSV if we have valid results
    with open(best_circuits_csv, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if csv_file.tell() == 0:  # Check if the file pointer is at the start (file is empty)
            if graph_type == "edges":
                writer.writerow(["circuit_id", "task_name", "post_attribution_patching_processing",
                            "n_edges", "pct_n_edges", "real_n_edges", "real_pct_n_edges", "faithfulness_by_mean"])
            else:
                writer.writerow(["circuit_id", "task_name", "post_attribution_patching_processing",
                            "n_nodes", "pct_n_nodes", "real_n_nodes", "real_pct_n_nodes", "faithfulness_by_mean"])
        
        pct_n = res_n / total if res_n is not None else None
        real_pct_n_edges = res_real / total if res_real is not None else None
        
        writer.writerow([circuit_id, task_name, post_attribution_patching_processing, 
                       res_n, pct_n, res_real, real_pct_n_edges, res_performance])
    
    return res_n, res_real, res_performance


def search_min_circuit_binary_search_neurons(circuit_id, task_name, model, metric_evaluation, device, g:NeuronGraph, 
                                     post_attribution_patching_processing, 
                                     search_min_circuit:SearchMinCircuitAboveThreshold, 
                                     results, clean_baseline, corrupted_baseline, test_dataloader,
                                     is_per_layer, n_nodes_as_first_step_for_neurons,
                                     n_attention_nodes_as_first_step_for_neurons, n_mlp_nodes_as_first_step_for_neurons):
    
    if not isinstance(g, NeuronGraph):
        raise ValueError("search_min_circuit_binary_search_neurons: g must be a NeuronGraph")
    
    best_circuits_csv = search_min_circuit.global_csv_circuit_information
    # Calculate total based on is_per_layer
    total = g.n_neurons_per_mlp if is_per_layer else g.count_neurons(filter_by_in_graph=False)


    csv_dir = os.path.dirname(best_circuits_csv)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
    
    def _get_init_search_range():
        if search_min_circuit.min_circuit_size >= 0 and search_min_circuit.max_circuit_size > search_min_circuit.min_circuit_size:
            return get_edges_or_nodes_number(search_min_circuit.min_circuit_size, total), get_edges_or_nodes_number(search_min_circuit.max_circuit_size, total), None
        if results is None or len(results) == 0:
            raise ValueError("there isnt a way to compute valid range for finding best circuit")
        # Results format is (n, faithfulness_by_mean)
        sorted_results = sorted(results, key=lambda x: x[0])
        start = 0
        end = total
        start_performance = 0.0
        for i, (n, performance) in enumerate(sorted_results):
            if performance > search_min_circuit.min_performance_threshold:
                end = n
                if i != 0:
                    start = sorted_results[i-1][0]
                start_performance = performance
                break
        return start, end, start_performance

    start, end, curr_performance = _get_init_search_range()
    res_n, res_performance = None, None
    logging.info(f"Search best circuit (neurons): initial: start: {start}, end: {end}, curr_performance {curr_performance}, is_per_layer: {is_per_layer}, n_nodes_as_first_step: {n_nodes_as_first_step_for_neurons}")
    
    # Track the closest performance to the desired window in case we never land inside it
    closest_n, closest_perf, closest_dist = None, None, None
    
    def _distance_to_window(value: float, low: float, high: float) -> float:
        if value < low:
            return low - value
        if value > high:
            return value - high
        return 0.0
    
    # early stopping
    if (curr_performance is not None and start is not None and end is not None and
        search_min_circuit.min_performance_threshold <= curr_performance <= search_min_circuit.max_performance_threshold):
        res_performance = curr_performance
        res_n = end
    else:
        steps = 0
        while start <= end and steps < search_min_circuit.max_steps:
            steps += 1
            n = (start + end) // 2
            logging.info(f"Search best circuit (neurons): Evaluating circuit for input neurons midpoint: {n}, start: {start}, end: {end}, is_per_layer: {is_per_layer}")
            
            select_circuit_neurons(graph=g, selection_method=post_attribution_patching_processing, 
                                          n_neurons=n, n_nodes_as_first_step_for_neurons=n_nodes_as_first_step_for_neurons,
                                          is_per_layer=is_per_layer, log=logging.getLogger())

            empty_circuit = not g.nodes['logits'].in_graph
            
            if not empty_circuit:
                circuit_evaluation = evaluate_graph_neurons(model=model, graph=g, dataloader=test_dataloader, metrics=metric_evaluation,
                                                            device=device, quiet=False, debug=False, calc_clean_logits=False)
                
                faithfulness_by_mean = compute_faithfulness(clean_baseline=clean_baseline, 
                                                            corrupted_baseline=corrupted_baseline, 
                                                            target_baseline=circuit_evaluation, 
                                                            per_element_faithfulness=False).item()
                logging.info(f"Search best circuit (neurons): {n} : faithfulness_by_mean: {faithfulness_by_mean}")
                curr_performance = faithfulness_by_mean 
            else:
                curr_performance = 0.0
            
            num_attention_nodes = g.count_attention_nodes(filter_by_in_graph=True)
            num_mlp_nodes = g.count_mlp_nodes(filter_by_in_graph=True)
            num_neurons = g.count_neurons(filter_by_in_graph=True)
            logging.info(f"Search best circuit (neurons) for {n} neurons, is_per_layer: {is_per_layer}: num attention nodes in graph: {num_attention_nodes}, num mlp nodes in graph: {num_mlp_nodes}, num neurons in graph: {num_neurons}")

            # Update closest candidate if needed
            curr_dist = _distance_to_window(curr_performance, search_min_circuit.min_performance_threshold, search_min_circuit.max_performance_threshold)
            if closest_dist is None or curr_dist < closest_dist or (curr_dist == closest_dist and closest_n is not None and n < closest_n):
                closest_dist, closest_n, closest_perf = curr_dist, n, curr_performance

            if search_min_circuit.min_performance_threshold <= curr_performance <= search_min_circuit.max_performance_threshold:
                res_performance = curr_performance
                res_n = n
                break
            
            elif curr_performance > search_min_circuit.max_performance_threshold:
                end = n - 1 # we can reduce 1 cause anyway result is too high
            else:  # curr_performance < search_min_circuit.min_performance_threshold
                start = n + 1 # we can add 1 cause anyway result is too low
                
    if res_n is None or res_performance is None:
        # Fallback to the closest performance observed during search
        if closest_n is not None and closest_perf is not None:
            logging.warning(f"Search best circuit (neurons): no performance in range; selecting closest performance {closest_perf} at n={closest_n}, is_per_layer: {is_per_layer}, n_nodes_as_first_step_for_neurons: {n_nodes_as_first_step_for_neurons}, (distance {closest_dist})")
            res_n, res_performance = closest_n, closest_perf
        else:
            logging.warning(f"Search best circuit (neurons): could not find any circuit that has performance inside this range in {search_min_circuit.max_steps} steps and no closest candidate available")
            return  # Early return if no valid result

    # Re-evaluate the final circuit to get all metrics
    
    select_circuit_neurons(graph=g, selection_method=post_attribution_patching_processing, 
                                  n_neurons=res_n, n_nodes_as_first_step_for_neurons=n_nodes_as_first_step_for_neurons,
                                  is_per_layer=is_per_layer, log=logging.getLogger())
    
    # Compute neuron-related metrics

    pct_n_neurons_per_layer = res_n / g.n_neurons_per_mlp if is_per_layer else None
    tot_n_neurons = g.count_neurons(filter_by_in_graph=True)
    tot_pct_n_neurons = tot_n_neurons / g.count_neurons(filter_by_in_graph=False)
    
    circuit_size_as_neurons = g.count_included_nodes()
    circuit_size_as_neurons_pct = circuit_size_as_neurons / g.count_total_nodes()
    
    num_attention_nodes = g.count_attention_nodes(filter_by_in_graph=True)
    if num_attention_nodes != n_attention_nodes_as_first_step_for_neurons:
        raise ValueError(f"num_attention_nodes: {num_attention_nodes} != n_attention_nodes_as_first_step_for_neurons: {n_attention_nodes_as_first_step_for_neurons}")

    num_mlp_nodes = g.count_mlp_nodes(filter_by_in_graph=True)
    if is_per_layer and num_mlp_nodes != n_mlp_nodes_as_first_step_for_neurons:
        raise ValueError(f"num_mlp_nodes: {num_mlp_nodes} != n_mlp_nodes_as_first_step_for_neurons: {n_mlp_nodes_as_first_step_for_neurons}")

    with open(best_circuits_csv, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if csv_file.tell() == 0:  # Check if the file pointer is at the start (file is empty)
            writer.writerow(["circuit_id", "task_name", "post_attribution_patching_processing",
                        "n_nodes_as_first_step_for_neurons", 
                        "n_neurons", "is_per_layer", "pct_n_neurons_per_layer",
                        "tot_n_neurons", "tot_pct_n_neurons",
                        "circuit_size", "circuit_size_pct", "num_mlp_nodes", "faithfulness_by_mean"])
        
        writer.writerow([circuit_id, task_name, post_attribution_patching_processing,
                       n_nodes_as_first_step_for_neurons, res_n, is_per_layer, pct_n_neurons_per_layer,
                       tot_n_neurons, tot_pct_n_neurons,
                       circuit_size_as_neurons, circuit_size_as_neurons_pct, num_mlp_nodes, res_performance])
    
    return res_n, res_performance


def _find_circuit_nodes_as_first_step_for_neurons(neurons_graph,output_dir, model, device, experiment_name, n_edges_or_nodes_in_circuit_list,
    post_attribution_patching_processing, test_df, batch_size, metric_evaluation, clean_baseline, corrupted_baseline, min_performance_threshold_neurons, 
    max_performance_threshold_neurons, circuit_id, task_name):

    if not isinstance(neurons_graph, NeuronGraph):
        raise ValueError("_find_circuit_nodes_as_first_step_for_neurons: neurons_graph must be a NeuronGraph")

    g = Graph.from_model(model, graph_type=GraphType.Nodes)
    for node in g.nodes.values():
        node.score = neurons_graph.nodes[node.name].score
    
    # save graph to json
    g.to_json(Path(output_dir) / f'{experiment_name}_nodes_as_first_step_for_neurons.json')
    total =  g.count_total_nodes()
    csv_path = Path(output_dir) / f'{experiment_name}_nodes_as_first_step_for_neurons.csv'

    test_ds = EAPDataset(test_df)
    test_dataloader = test_ds.to_dataloader(batch_size)


    clean_mean = clean_baseline.mean().item()
    corruped_mean = corrupted_baseline.mean().item()

    results= []
    with open(csv_path , "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if csv_file.tell() == 0:  # Check if the file pointer is at the start (file is empty)
                writer.writerow(["n_nodes_input","n_nodes", "real_n_nodes", "real_pct_n_nodes", "clean_mean", "corrupted_mean", "circuit_mean", "faithfulness_by_mean", "faithfulness_per_element"])

        sorted_num = sorted(n_edges_or_nodes_in_circuit_list)
        for n_input in sorted_num:
            logging.info(f"Evaluating circuit for input edge/node value: {n_input}")
            n = get_edges_or_nodes_number(n_input, total)
            logging.info(f"Computed number of edges/nodes to use: {n}")
            select_circuit_nodes(graph=g, selection_method=post_attribution_patching_processing, n_nodes=n, log=logging.getLogger())

            empty_circuit = not g.nodes['logits'].in_graph
            if empty_circuit and n!=0:
                logging.info(f"skipping {n} due to empty circuit")
                continue

            real_n =  g.count_included_nodes()
            real_pct = real_n / total
            circuit_evaluation = evaluate_graph_node(model=model, graph=g, dataloader=test_dataloader, metrics=metric_evaluation, device=device, quiet=False, 
                                                            debug=False, calc_clean_logits=False)
                    
            faithfulness_by_mean = compute_faithfulness(clean_baseline=clean_baseline, corrupted_baseline=corrupted_baseline, target_baseline=circuit_evaluation, per_element_faithfulness=False).item()
            faithfulness_by_elem = compute_faithfulness(clean_baseline=clean_baseline, corrupted_baseline=corrupted_baseline, target_baseline=circuit_evaluation, per_element_faithfulness=True).item()
            circuit_mean = circuit_evaluation.mean().item()
            logging.info(f"{n_input} -> {n} edge/nodes: circuit_mean: {circuit_mean} faithfulness_by_mean: {faithfulness_by_mean} faithfulness_by_elem:{faithfulness_by_elem}")
      
            logging.info(f"num attention nodes in first step for neurons graph: {g.count_attention_nodes(filter_by_in_graph=True)}, num mlp nodes in first step for neurons graph: {g.count_mlp_nodes(filter_by_in_graph=True)}")
            writer.writerow([n_input, n, real_n, real_pct, clean_mean, corruped_mean ,circuit_mean, faithfulness_by_mean, faithfulness_by_elem])
            results.append((n, real_n ,faithfulness_by_mean))
            logging.info(f"results: {results}")

        global_csv_circuit_information = Path(output_dir) / f'circuit_info_nodes_as_first_step_for_neurons.csv'
        search_min_circuit = SearchMinCircuitAboveThreshold(
            global_csv_circuit_information=global_csv_circuit_information,
            min_performance_threshold=min_performance_threshold_neurons,
            max_performance_threshold=max_performance_threshold_neurons,
            max_steps=20
        )
        try:
            logging.info(f"Searching best circuit for nodes as first step for neurons with min_performance_threshold: {min_performance_threshold_neurons}, max_performance_threshold: {max_performance_threshold_neurons}")
            res_n, res_real, res_performance = search_min_circuit_binary_search(circuit_id=circuit_id, task_name=task_name, model=model, metric_evaluation=metric_evaluation, device=device, g=g, 
                                            post_attribution_patching_processing=post_attribution_patching_processing, 
                                            search_min_circuit=search_min_circuit, 
                                            results=results, clean_baseline=clean_baseline, corrupted_baseline=corrupted_baseline, test_dataloader=test_dataloader, graph_type="nodes")
            logging.info(f"Search best circuit for nodes as first step for neurons: res_n: {res_n}, res_real: {res_real}, res_performance: {res_performance}")
                        
            
            g.set_all_edges_in_graph(in_graph=False)
            select_circuit_nodes(graph=g, selection_method=post_attribution_patching_processing, n_nodes=res_n, log=logging.getLogger())
            logging.info(f"best circuit for nodes as first step for neurons: Attention nodes in first step for neurons graph: {g.count_attention_nodes(filter_by_in_graph=True)}, mlp nodes in first step for neurons graph: {g.count_mlp_nodes(filter_by_in_graph=True)}")
            return res_n, res_real, res_performance, g.count_attention_nodes(filter_by_in_graph=True), g.count_mlp_nodes(filter_by_in_graph=True)

        except Exception as e:
            logging.error(f"An error occured while trying to perform search_min_circuit_binary_search :{e}")
            raise ValueError(f"An error occured while trying to perform search_min_circuit_binary_search in _find_circuit_nodes_as_first_step_for_neurons:{e}")

def _find_circuit_neurons_as_second_step_for_neurons(neurons_graph,output_dir, model, device, experiment_name,
 n_neurons_in_circuit_list, is_per_layer, n_nodes_as_first_step_for_neurons, n_attention_nodes_as_first_step_for_neurons, n_mlp_nodes_as_first_step_for_neurons, post_attribution_patching_processing, test_df, batch_size, 
 metric_evaluation, clean_baseline, corrupted_baseline, circuit_id, task_name, search_min_circuit):

    #here we are in the second step for neurons, where we already chose attention heads and mlp nodes in the first step
    if not isinstance(neurons_graph, NeuronGraph):
        raise ValueError("__find_circuit_neurons_as_second_step_for_neurons: neurons_graph must be a NeuronGraph")

    results= []
    csv_path = Path(output_dir) / f'{experiment_name}_neurons.csv'

    test_ds = EAPDataset(test_df)
    test_dataloader = test_ds.to_dataloader(batch_size)

    clean_mean = clean_baseline.mean().item()
    corruped_mean = corrupted_baseline.mean().item()

    with open(csv_path , "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if csv_file.tell() == 0: 
                writer.writerow([
                "n_neurons_input", "n_neurons", "is_per_layer", "pct_per_layer", "n_nodes_as_first_step_for_neurons", "n_attention_nodes_as_first_step_for_neurons", "mlp_nodes_as_first_step_for_neurons",
                "tot_n_neurons", "tot_pct_n_neurons",
                "circuit_size", "circuit_size_pct", "num_mlp_nodes",
                "clean_mean", "corrupted_mean", "circuit_mean", "faithfulness_by_mean", "faithfulness_per_element"])

        total = neurons_graph.n_neurons_per_mlp if is_per_layer else neurons_graph.count_neurons(filter_by_in_graph=False)
        sorted_num = sorted(n_neurons_in_circuit_list)
        for n_input in sorted_num:
            logging.info(f"Evaluating circuit for input neurons value: {n_input}")
            n = get_edges_or_nodes_number(n_input, total)
            logging.info(f"Computed number of neurons to use: {n}")

            pct_per_layer = n / neurons_graph.n_neurons_per_mlp if is_per_layer else None

            select_circuit_neurons(graph=neurons_graph, selection_method=post_attribution_patching_processing, n_neurons=n, n_nodes_as_first_step_for_neurons=n_nodes_as_first_step_for_neurons,
             is_per_layer=is_per_layer, log=logging.getLogger())

            empty_circuit = not neurons_graph.nodes['logits'].in_graph
            if empty_circuit and n!=0:
                logging.info(f"skipping {n} due to empty circuit")
                continue
            
            circuit_evaluation = evaluate_graph_neurons(model=model, graph=neurons_graph, dataloader=test_dataloader, metrics=metric_evaluation,
                                                            device=device, quiet=False, debug=False, calc_clean_logits=False)
                    
            faithfulness_by_mean = compute_faithfulness(clean_baseline=clean_baseline, corrupted_baseline=corrupted_baseline, target_baseline=circuit_evaluation, per_element_faithfulness=False).item()
            faithfulness_by_elem = compute_faithfulness(clean_baseline=clean_baseline, corrupted_baseline=corrupted_baseline, target_baseline=circuit_evaluation, per_element_faithfulness=True).item()
            circuit_mean = circuit_evaluation.mean().item()
            logging.info(f"{n_input} -> {n} neurons, is per layer: {is_per_layer} circuit_mean: {circuit_mean} faithfulness_by_mean: {faithfulness_by_mean} faithfulness_by_elem:{faithfulness_by_elem}")

            num_attention_nodes = neurons_graph.count_attention_nodes(filter_by_in_graph=True)
            if num_attention_nodes != n_attention_nodes_as_first_step_for_neurons:
                raise ValueError(f"num_attention_nodes: {num_attention_nodes} != n_attention_nodes_as_first_step_for_neurons: {n_attention_nodes_as_first_step_for_neurons}")

            num_mlp_nodes = neurons_graph.count_mlp_nodes(filter_by_in_graph=True)
            if is_per_layer and num_mlp_nodes != n_mlp_nodes_as_first_step_for_neurons:
                raise ValueError(f"num_mlp_nodes: {num_mlp_nodes} != n_mlp_nodes_as_first_step_for_neurons: {n_mlp_nodes_as_first_step_for_neurons}")

            tot_n_neurons = neurons_graph.count_neurons(filter_by_in_graph=True)
            tot_pct_n_neurons = tot_n_neurons / neurons_graph.count_neurons(filter_by_in_graph=False)


            circuit_size_as_neurons = neurons_graph.count_included_nodes()
            circuit_size_as_neurons_pct = circuit_size_as_neurons / neurons_graph.count_total_nodes()

            logging.info(f"input {n_input} -> {n} neurons, is per layer: {is_per_layer}, num attention nodes in graph: {num_attention_nodes}, num mlp nodes in graph: {num_mlp_nodes}, num neurons in graph: {tot_n_neurons}, pct neurons in graph: {tot_pct_n_neurons}")
                
            writer.writerow([n_input, n, is_per_layer, pct_per_layer, n_nodes_as_first_step_for_neurons, n_attention_nodes_as_first_step_for_neurons, n_mlp_nodes_as_first_step_for_neurons,
            tot_n_neurons, tot_pct_n_neurons, circuit_size_as_neurons, circuit_size_as_neurons_pct, num_mlp_nodes,
            clean_mean, corruped_mean ,circuit_mean, faithfulness_by_mean, faithfulness_by_elem])
            results.append((n ,faithfulness_by_mean))
            logging.info(f"results: {results}")
    
    if search_min_circuit is not None:
        try:

           _, _ = search_min_circuit_binary_search_neurons(circuit_id=circuit_id, task_name=task_name, model=model, metric_evaluation=metric_evaluation, device=device, g=neurons_graph, 
                                            post_attribution_patching_processing=post_attribution_patching_processing, 
                                            search_min_circuit=search_min_circuit, 
                                            results=results, clean_baseline=clean_baseline, corrupted_baseline=corrupted_baseline, test_dataloader=test_dataloader,
                                            is_per_layer=is_per_layer, n_nodes_as_first_step_for_neurons=n_nodes_as_first_step_for_neurons,
                                            n_attention_nodes_as_first_step_for_neurons=n_attention_nodes_as_first_step_for_neurons, n_mlp_nodes_as_first_step_for_neurons=n_mlp_nodes_as_first_step_for_neurons)
        except Exception as e:
            logging.error(f"An error occured while trying to perform search_min_circuit_binary_search :{e}")


    logging.info(f"Finished run_experiment")


def find_neurons_circuits_two_steps(
    neurons_graph,
    output_dir,
    model,
    device,
    experiment_name,
    n_edges_or_nodes_in_circuit_list,
    is_per_layer,
    post_attribution_patching_processing,
    test_df,
    batch_size,
    metric_evaluation,
    clean_baseline,
    corrupted_baseline,
    circuit_id,
    task_name,
    search_min_circuit,
    min_performance_threshold_neurons,
    max_performance_threshold_neurons,
    two_steps_neuron_circuit_node_circuit_list_sizes,
):
    logging.info("=== Starting two-step neuron circuit search ===")
    logging.info(f"Processing type={post_attribution_patching_processing}, "
                 f"per-layer={is_per_layer}, circuit_id={circuit_id}")

    # --- STEP 1: node-level selection ---
    (
        res_n,
        res_real,
        res_performance,
        n_attention_nodes_as_first_step_for_neurons,
        n_mlp_nodes_as_first_step_for_neurons,
    ) = _find_circuit_nodes_as_first_step_for_neurons(
        neurons_graph=neurons_graph,
        output_dir=output_dir,
        model=model,
        device=device,
        experiment_name=experiment_name,
        n_edges_or_nodes_in_circuit_list=two_steps_neuron_circuit_node_circuit_list_sizes,
        post_attribution_patching_processing=post_attribution_patching_processing,
        test_df=test_df,
        batch_size=batch_size,
        metric_evaluation=metric_evaluation,
        clean_baseline=clean_baseline,
        corrupted_baseline=corrupted_baseline,
        circuit_id=circuit_id,
        task_name=task_name,
        min_performance_threshold_neurons=min_performance_threshold_neurons,
        max_performance_threshold_neurons=max_performance_threshold_neurons,
    )

    # --- Validation sanity check (fixed precedence bug) ---
    if res_n != res_real and (
        post_attribution_patching_processing in {"top_n", "top_n_abs"}
    ):
        raise RuntimeError(
            f"Unexpected mismatch: res_n ({res_n}) != res_real ({res_real}) "
            f"under mode {post_attribution_patching_processing}"
        )

    if res_real is None:
        logging.warning("No valid circuit found in node-level search. Skipping neuron-level stage.")
        return

    # --- STEP 2: neuron-level selection ---
    _find_circuit_neurons_as_second_step_for_neurons(
        neurons_graph=neurons_graph,
        output_dir=output_dir,
        model=model,
        device=device,
        experiment_name=experiment_name,
        n_neurons_in_circuit_list=n_edges_or_nodes_in_circuit_list,
        is_per_layer=is_per_layer,
        n_nodes_as_first_step_for_neurons=res_real,
        n_attention_nodes_as_first_step_for_neurons=n_attention_nodes_as_first_step_for_neurons,
        n_mlp_nodes_as_first_step_for_neurons=n_mlp_nodes_as_first_step_for_neurons,
        post_attribution_patching_processing=post_attribution_patching_processing,
        test_df=test_df,
        batch_size=batch_size,
        metric_evaluation=metric_evaluation,
        clean_baseline=clean_baseline,
        corrupted_baseline=corrupted_baseline,
        circuit_id=circuit_id,
        task_name=task_name,
        search_min_circuit=search_min_circuit,
    )

    logging.info(f"=== Completed neuron circuit search for task {task_name} ===")



def run_experiment(
    experiment_name,
    output_dir,
    train_df, test_df, 
    train_loss,
    metric_evaluation,
    n_edges_or_nodes_in_circuit_list,
    attribution_patching_method,
    post_attribution_patching_processing,
    device,
    abs_score,
    aggregation_method,
    circuit_json_path,
    EAP_IG_steps,
    batch_size,
    task_name: str,
    circuit_id: str,
    search_min_circuit:SearchMinCircuitAboveThreshold=None,
    save_cross_task_faithfulness_args: CrossTaskFaithfulnessArgs=None,
    graph_type="nodes",
    neurons_graph: bool=False,
    find_neurons_circuits_two_steps_args: FindNeuronsCircuitsTwoStepsArgs=None,
    all_examples_scores_npy_path: str=None,
    model=None):

    logging.info(f"Entered run_experiment")
    if model is None:
        raise ValueError("model is None")
   
    g = load_graph_and_perform_attribution_patching_if_needed(experiment_name=experiment_name, output_dir=output_dir, train_df=train_df, model=model, device=device, 
                                                              train_loss=train_loss, attribution_patching_method=attribution_patching_method, abs_score=abs_score,
                                                              aggregation_method=aggregation_method, circuit_json_path=circuit_json_path, EAP_IG_steps=EAP_IG_steps, batch_size=batch_size, graph_type=graph_type, 
                                                              neurons_graph=neurons_graph,
                                                              all_examples_scores_npy_path=all_examples_scores_npy_path)

    total = len(g.edges) if graph_type == "edges" else g.count_total_nodes()
    test_ds = EAPDataset(test_df)
    test_dataloader = test_ds.to_dataloader(batch_size)

    clean_baseline, corrupted_baseline = get_clean_corrupted_baselines(model, test_dataloader, metric_evaluation, device)
    clean_mean = clean_baseline.mean().item()
    corruped_mean = corrupted_baseline.mean().item()
    logging.info(f"clean_mean: {clean_mean}, corrupted_mean: {corruped_mean}")


    if neurons_graph and find_neurons_circuits_two_steps_args is not None:
        find_neurons_circuits_two_steps(
            neurons_graph=g,
            output_dir=output_dir,
            model=model,
            device=device,
            experiment_name=experiment_name,
            n_edges_or_nodes_in_circuit_list=n_edges_or_nodes_in_circuit_list,
            is_per_layer= find_neurons_circuits_two_steps_args.is_per_layer,
            post_attribution_patching_processing=post_attribution_patching_processing,
            test_df=test_df,
            batch_size=batch_size,
            metric_evaluation=metric_evaluation,
            clean_baseline=clean_baseline,
            corrupted_baseline=corrupted_baseline,
            circuit_id=circuit_id,
            task_name=task_name,
            search_min_circuit=search_min_circuit,
            min_performance_threshold_neurons=find_neurons_circuits_two_steps_args.min_performance_threshold_neurons,
            max_performance_threshold_neurons=find_neurons_circuits_two_steps_args.max_performance_threshold_neurons,
            two_steps_neuron_circuit_node_circuit_list_sizes=find_neurons_circuits_two_steps_args.two_steps_neuron_circuit_node_circuit_list_sizes,
        )
        logging.info(f"Finished find_neurons_circuits_two_steps")
        return  # No further action in run_experiment after two-step neuron search

    results= []
    csv_path = Path(output_dir) / f'{experiment_name}.csv'
    with open(csv_path , "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if csv_file.tell() == 0:  # Check if the file pointer is at the start (file is empty)
            if graph_type == "edges":
                writer.writerow(["n_edges_input","n_edges", "real_n_edges", "real_pct_n_edges", "clean_mean", "corrupted_mean", "circuit_mean", "faithfulness_by_mean", "faithfulness_per_element"])
            else:
                writer.writerow(["n_nodes_input","n_nodes", "real_n_nodes", "real_pct_n_nodes", "clean_mean", "corrupted_mean", "circuit_mean", "faithfulness_by_mean", "faithfulness_per_element"])

        sorted_num = sorted(n_edges_or_nodes_in_circuit_list)
        for n_input in sorted_num:
            logging.info(f"Evaluating circuit for input edge/node value: {n_input}")
            n = get_edges_or_nodes_number(n_input, total)
            logging.info(f"Computed number of edges/nodes to use: {n}")
            if graph_type == "edges":
                select_circuit_edges(graph=g, selection_method=post_attribution_patching_processing, n_edges=n, log=logging.getLogger())
            else:
                select_circuit_nodes(graph=g, selection_method=post_attribution_patching_processing, n_nodes=n, log=logging.getLogger())

            empty_circuit = not g.nodes['logits'].in_graph
            if empty_circuit and n!=0:
                logging.info(f"skipping {n} due to empty circuit")
                continue

            #real_n = len([e for e in g.edges.values() if e.in_graph]) if graph_type == "edges" else len([n for n in g.nodes.values() if n.in_graph])
            real_n = g.count_included_edges() if graph_type == "edges" else g.count_included_nodes()
            real_pct = real_n / total
            if graph_type == "edges":

                circuit_evaluation = evaluate_graph(model=model, graph=g, dataloader=test_dataloader, metrics=metric_evaluation, device=device, prune=False, quiet=False, calc_input_for_nodes_not_in_graph=False,
                                                     debug_corrupted_construction=False, calc_clean_logits=False)
            elif graph_type == "nodes":
                if isinstance(g, NeuronGraph):
                    circuit_evaluation = evaluate_graph_neurons(model=model, graph=g, dataloader=test_dataloader, metrics=metric_evaluation,
                                                            device=device, quiet=False, debug=False, calc_clean_logits=False)
                else:
                    circuit_evaluation = evaluate_graph_node(model=model, graph=g, dataloader=test_dataloader, metrics=metric_evaluation, device=device, quiet=False, 
                                                            debug=False, calc_clean_logits=False)
                    
            faithfulness_by_mean = compute_faithfulness(clean_baseline=clean_baseline, corrupted_baseline=corrupted_baseline, target_baseline=circuit_evaluation, per_element_faithfulness=False).item()
            faithfulness_by_elem = compute_faithfulness(clean_baseline=clean_baseline, corrupted_baseline=corrupted_baseline, target_baseline=circuit_evaluation, per_element_faithfulness=True).item()
            circuit_mean = circuit_evaluation.mean().item()
            logging.info(f"{n_input} -> {n} edge/nodes: circuit_mean: {circuit_mean} faithfulness_by_mean: {faithfulness_by_mean} faithfulness_by_elem:{faithfulness_by_elem}")
            if isinstance(g, NeuronGraph):
                logging.info(f"num attention nodes in graph: {g.count_attention_nodes(filter_by_in_graph=True)}, num neurons in graph: {g.count_neurons(filter_by_in_graph=True)}")
            else:
                logging.info(f"num attention nodes in graph: {g.count_attention_nodes(filter_by_in_graph=True)}, num mlp nodes in graph: {g.count_mlp_nodes(filter_by_in_graph=True)}")
            writer.writerow([n_input, n, real_n, real_pct, clean_mean, corruped_mean ,circuit_mean, faithfulness_by_mean, faithfulness_by_elem])
            results.append((n, real_n ,faithfulness_by_mean))
            logging.info(f"results: {results}")
    
    if search_min_circuit is not None:
        try:
            _ ,_ ,_ = search_min_circuit_binary_search(circuit_id=circuit_id, task_name=task_name, model=model, metric_evaluation=metric_evaluation, device=device, g=g, 
                                            post_attribution_patching_processing=post_attribution_patching_processing, 
                                            search_min_circuit=search_min_circuit, 
                                            results=results, clean_baseline=clean_baseline, corrupted_baseline=corrupted_baseline, test_dataloader=test_dataloader, graph_type=graph_type)
        except Exception as e:
            logging.error(f"An error occured while trying to perform search_min_circuit_binary_search :{e}")

    if save_cross_task_faithfulness_args is not None:
        try:
            save_cross_task_results(cross_task_faithfulness_args=save_cross_task_faithfulness_args, results=results, total=total, graph_type=graph_type)
        except Exception as e:
            logging.error(f"An error occured while trying to perform save_cross_task_results :{e}")

    logging.info(f"Finished run_experiment")

def main():
    try:
        print("starting attribution patching!")
        print("trying to print gpu uuid...")
        print([str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else 'No GPU available')
        parser = argparse.ArgumentParser(description="Run attribution patching on a proteins dataset.")
        parser.add_argument("--csv_path", type=str, required=True, 
                            help="Path to the input CSV file containing the dataset.")
        parser.add_argument("--output_dir", type=str, required=True, 
                            help="Path to save the processed dataset or experiment results.")
        parser.add_argument("--total_n_samples", type=int, required=True, 
                            help="Total number of samples to include in the experiment.")
        parser.add_argument("--random_state", type=int, default=42, 
                            help="Random seed for reproducibility when sampling the dataset (default: 42).")
        parser.add_argument("--train_ratio", type=float, default=0.5, 
                            help="Proportion of the dataset to use for training. The remainder is used for testing (default: 0.5).")

        parser.add_argument(
            "--attribution_patching_method", 
            type=str, 
            default="EAP", 
            choices=["EAP", "EAP-IG"],
            help="Attribution method: 'EAP' or 'EAP-IG'.")

        parser.add_argument("--post_attribution_patching_processing", type=str, default="greedy_abs", help="choose greedy_abs/greedy/top_n/ top_n_abs")

        parser.add_argument(
            "--n_edges_in_circuit_list", 
            type=float, 
            nargs='+', 
            default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
            help="List of edges to keep. Values >= 1 are treated as absolute number of edges. Values < 1 are treated as percentage (fraction) of total edges. For example: 5 10 0.1 1.0"
)


        parser.add_argument("--metric", type=str, default="logit_diff", help="choose logit_diff/log prob")

        parser.add_argument(
            "--abs_score",
            action="store_true",
            help="If set, use the absolute value of the score.")

        parser.add_argument(
            "--aggregation_method",
            type=str,
            default="sum",
            choices=["sum", "pos_mean"],
            help="Aggregation method to use. Choose from: 'sum', 'pos_mean'"
        )
        parser.add_argument("--circuit_json_path", type=str, required=False, default=None, 
                        help="Path to the input json file containing the circuit.")

        parser.add_argument(
            "--EAP_IG_steps",
            type=int,
            default=5,
            help="EAP_IG_steps. Default: 5"
        )

        parser.add_argument("--batch_size", type=int, default=1, 
                    help="Batch size for processing (default: 1).")

        parser.add_argument("--task_name", type=str, required=True, 
                            help="Name to give the experiment when saving results.")
        
        parser.add_argument(
            "--enable_min_circuit_search",
            action="store_true",
            help="Enable binary search to find minimal circuit meeting performance thresholds."
        )
        
        parser.add_argument(
            "--min_circuit_csv",
            type=str,
            default=None,
            help="Path to CSV file where minimal circuit information will be saved."
        )
        
        parser.add_argument(
            "--min_performance_threshold",
            type=float,
            default=0.8,
            help="Minimum performance threshold for minimal circuit search (default: 0.7)."
        )
        
        parser.add_argument(
            "--max_performance_threshold",
            type=float,
            default=0.85,
            help="Maximum performance threshold for minimal circuit search (default: 0.9)."
        )
        
        parser.add_argument(
            "--min_circuit_size",
            type=float,
            default=-1,
            help="Minimum circuit size to consider. Values >= 1 are treated as absolute number of edges. Values < 1 are treated as percentage. -1 means use results from standard circuit evaluation (default: -1)."
        )
        
        parser.add_argument(
            "--max_circuit_size",
            type=float,
            default=-1,
            help="Maximum circuit size to consider. Values >= 1 are treated as absolute number of edges. Values < 1 are treated as percentage. -1 means use results from standard circuit evaluation (default: -1)."
        )
        
        parser.add_argument(
            "--max_search_steps",
            type=int,
            default=20,
            help="Maximum number of steps for binary search (default: 20)."
        )

        parser.add_argument(
            "--is_cross_task_faithfulness",
            action="store_true",
            help="is cross-task faithfulness evaluation."
        )
        parser.add_argument(
            "--source_task_name",
            type=str,
            default=None,
            help="Source task name for cross-task faithfulness evaluation."
        )
        parser.add_argument(
            "--source_circuit_id",
            type=str,
            default=None,
            help="Source circuit ID for cross-task faithfulness evaluation."
        )
        parser.add_argument(
            "--target_task_name",
            type=str,
            default=None,
            help="Target task name for cross-task faithfulness evaluation."
        )
        parser.add_argument(
            "--target_circuit_id",
            type=str,
            default=None,
            help="Target circuit ID for cross-task faithfulness evaluation."
        )
        parser.add_argument(
            "--save_results_csv",
            type=str,
            default=None,
            help="Path to save results for cross-task faithfulness evaluation."
        )
        parser.add_argument(
            "--graph_type",
            type=str,
            default="edges",
            choices=["nodes", "edges"],
            help= "Type of graph to use for the experiment. Choose from: 'nodes', 'edges' (default: 'nodes')."
        )
        parser.add_argument(
            "--neurons_graph",
            action="store_true",
            help="Use neurons graph.",
            default=False
        )

        parser.add_argument(
            "--find_neurons_circuits_two_steps",
            action="store_true",
            help="Find neurons circuits in two steps.",
            default=False
        )
        
        parser.add_argument(
            "--min_performance_threshold_neurons",
            type=float,
            default=0.89,
            help="Minimum performance threshold for neurons circuits in two steps.",
        )
        parser.add_argument(
            "--max_performance_threshold_neurons",
            type=float,
            default=0.9,
            help="Maximum performance threshold for neurons circuits in two steps.",
        )
        parser.add_argument(
            "--is_per_layer_neurons",
            action="store_true",
            help="Is per layer for neurons circuits in two steps.",
            default = False
        )
        parser.add_argument(
            "--two_steps_neuron_circuit_node_circuit_list_sizes",
            type=float,
            nargs='+',
            default=[0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
            help="List of nodes as first step for neurons.",
        )
        parser.add_argument(
            "--save_scores_per_example_npy",
            action="store_true",
            help="Save scores per example in npy file.",
            default=False
        )

        parser.add_argument(
            "--model_type",
            type=str,
            required=True,
            choices=["esm3", "esm-c"],
            help="Type of model to use for the experiment. Choose from: 'esm3', 'esm-c'."
        )

        args = parser.parse_args()
        if args.neurons_graph and args.graph_type == "edges":
            raise ValueError("Neurons graph is not supported for edges graph.")

        if args.neurons_graph and args.find_neurons_circuits_two_steps and args.is_cross_task_faithfulness:
            raise ValueError("Find neurons circuits in two steps is not supported for cross-task faithfulness evaluation.")

        if args.neurons_graph and args.find_neurons_circuits_two_steps and \
            args.post_attribution_patching_processing not in {"top_n", "top_n_abs"}:
                raise ValueError(
                    "Two-step neurons mode requires post_attribution_patching_processing ∈ {top_n, top_n_abs}"
                )

        
        output_dir = args.output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        circuit_id = str(uuid.uuid4())

        experiment_name = f"circuit{circuit_id}_{args.task_name}_n{args.total_n_samples}_random_state{args.random_state}_train_ratio{args.train_ratio}_{args.attribution_patching_method}_{args.post_attribution_patching_processing}_agg{args.aggregation_method}_abs{args.abs_score}_metric{args.metric}"
 
        log_file = Path(output_dir) / f'{experiment_name}.log'
        logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s")

        if args.save_scores_per_example_npy:
            save_scores_per_example_npy_path = Path(output_dir) / f'{experiment_name}_scores_per_example.npy'
        else:
            save_scores_per_example_npy_path = None

        try:
            with open(log_file, 'w') as f:
                f.write("Logging test\n")
            print(f"Successfully wrote to log file: {log_file}")
        except Exception as e:
            print(f"Error writing to log file: {e}")

        logging.info([str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else 'No GPU available')

        if args.enable_min_circuit_search and args.min_circuit_csv is None:
            args.min_circuit_csv = str(Path(output_dir) / f'min_circuits_{experiment_name}.csv')
            
        search_min_circuit = None
        if args.enable_min_circuit_search:
            search_min_circuit = SearchMinCircuitAboveThreshold(
                global_csv_circuit_information=args.min_circuit_csv,
                min_performance_threshold=args.min_performance_threshold,
                max_performance_threshold=args.max_performance_threshold,
                min_circuit_size=args.min_circuit_size,
                max_circuit_size=args.max_circuit_size,
                max_steps=args.max_search_steps
            )
            
            # Create directory for min circuit CSV if it doesn't exist
            min_circuit_dir = os.path.dirname(args.min_circuit_csv)
            if min_circuit_dir and not os.path.exists(min_circuit_dir):
                os.makedirs(min_circuit_dir, exist_ok=True)
                
            logging.info(f"Binary search for minimal circuit enabled with parameters:")
            logging.info(f"  Min performance threshold: {args.min_performance_threshold}")
            logging.info(f"  Max performance threshold: {args.max_performance_threshold}")
            logging.info(f"  Min circuit size: {args.min_circuit_size}")
            logging.info(f"  Max circuit size: {args.max_circuit_size}")
            logging.info(f"  Max search steps: {args.max_search_steps}")
            logging.info(f"  Results will be saved to: {args.min_circuit_csv}")


        if args.is_cross_task_faithfulness:
            cross_task_faithfulness_args = validate_cross_task_faithfulness_input_args_and_build_data_class(args)
            logging.info(f"Cross-task faithfulness arguments: {cross_task_faithfulness_args}")
        else:
            cross_task_faithfulness_args = None

        start_time = time.time()
        logging.info("Starting the script...")

        logging.info("Arguments passed to the script:")
        for arg, value in vars(args).items():
            logging.info(f"{arg}: {value}")

        supported_post_processing_method = ["greedy_abs", "greedy", "top_n", "top_n_abs", "none"]
        if not args.post_attribution_patching_processing in supported_post_processing_method:
            raise ValueError(f"Unsupported post_attribution_patching_processing: {args.post_attribution_patching_processing}.")

        supported_metrics = ["logit_diff", "log_prob"]
        if not args.metric in supported_metrics:
            raise ValueError(f"Unsupported metric: {args.metric}.")

        if not os.path.exists(args.csv_path):
            raise FileNotFoundError(f"CSV file not found at {args.csv_path}")    

        df = pd.read_csv(args.csv_path)
        device = get_device()
        if args.model_type is None:
            raise ValueError("model_type is required")
            
        model = load_model(
            model_type=args.model_type,
            device=device,
            use_transformer_lens_model=True,
            cache_attention_activations=True,
            cache_mlp_activations=True,
            output_type="sequence",
            cache_attn_pattern=False,
            split_qkv_input=True)

        tokenizer = load_tokenizer_by_model_type(args.model_type)

        df = create_induction_dataset_pandas(df, args.total_n_samples, args.random_state, tokenizer=tokenizer, metric=args.metric)

        train, test = split_train_test(df, args.train_ratio)
        logging.info(f"loaded train and test dataset. train size:{len(train)}. test:{len(test)}")

        loss, metric = create_loss_and_metric(args.metric)

        find_neurons_circuits_two_steps_args = None
        if args.neurons_graph and args.find_neurons_circuits_two_steps:
            find_neurons_circuits_two_steps_args = FindNeuronsCircuitsTwoStepsArgs(
                min_performance_threshold_neurons=args.min_performance_threshold_neurons,
                max_performance_threshold_neurons=args.max_performance_threshold_neurons,
                is_per_layer=args.is_per_layer_neurons,
                two_steps_neuron_circuit_node_circuit_list_sizes=args.two_steps_neuron_circuit_node_circuit_list_sizes
            )

        run_experiment(
            experiment_name=experiment_name, 
            output_dir=output_dir, 
            train_df=train, 
            test_df=test, 
            train_loss=loss,
            metric_evaluation=metric, 
            n_edges_or_nodes_in_circuit_list=args.n_edges_in_circuit_list, 
            attribution_patching_method=args.attribution_patching_method, 
            post_attribution_patching_processing=args.post_attribution_patching_processing,
            device=device, 
            abs_score=args.abs_score, 
            aggregation_method=args.aggregation_method,
            circuit_json_path=args.circuit_json_path, 
            EAP_IG_steps=args.EAP_IG_steps, 
            batch_size=args.batch_size,
            task_name=args.task_name,
            circuit_id=circuit_id,
            search_min_circuit=search_min_circuit,  # Pass the search_min_circuit parameter
            save_cross_task_faithfulness_args=cross_task_faithfulness_args,
            graph_type=args.graph_type,
            neurons_graph=args.neurons_graph,
            find_neurons_circuits_two_steps_args=find_neurons_circuits_two_steps_args,
            all_examples_scores_npy_path=save_scores_per_example_npy_path,
            model=model
        )
    
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Script completed in {elapsed_time:.2f} seconds.")
    except Exception as e:
        # Print the error message
        print(f"Error while running script: {e}")
        
        # Print the detailed traceback
        print("Detailed traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    sys.setrecursionlimit(3000)
    print(f"Recursion limit: {sys.getrecursionlimit()}")
    main()