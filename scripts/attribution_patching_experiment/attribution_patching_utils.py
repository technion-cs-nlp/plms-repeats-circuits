"""Shared utilities for attribution patching experiments."""
import math
import os
import csv
import ast
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager

import pandas as pd

from plms_repeats_circuits.utils.model_utils import mask_protein
from plms_repeats_circuits.utils.esm_utils import replace_short_mask_with_mask_token


# ---- Induction dataset ----
def create_induction_dataset_pandas(df, total_n_samples, random_state, tokenizer, metric):
    """Process induction dataset. Expects df to be pre-filtered (e.g. max seq length 400)."""
    if len(df) < total_n_samples:
        raise ValueError(f"Not enough samples: {len(df)} rows available, but {total_n_samples} requested.")
    if total_n_samples < len(df):
        sampled_df = df.sample(n=total_n_samples, random_state=random_state)
    else:
        print(f"total_n_samples {total_n_samples} >= dataframe size {len(df)}. Using all {len(df)} samples.")
        sampled_df = df

    def process_row(row):
        clean = row['seq']
        name = f"{row['cluster_id']}_{row['rep_id']}_{row['repeat_key']}"
        corrupted = row['corrupted_sequence']
        masked_position = int(row['masked_position'])
        clean_masked = mask_protein(clean, masked_position, tokenizer)
        corrupted_masked = mask_protein(corrupted, masked_position, tokenizer)
        corrupted_masked = replace_short_mask_with_mask_token(corrupted_masked, tokenizer)
        labels = [clean[masked_position]]
        if metric == "logit_diff":
            if 'corrupted_amino_acid_type' in df.columns:
                labels.append(row['corrupted_amino_acid_type'])
            elif 'replacments' in df.columns:
                replacements = ast.literal_eval(row['replacments'])
                if len(replacements) != 1:
                    raise ValueError("got unexpected corrupted amino acids to logit diff metric")
                labels.append(replacements[0])
            else:
                raise ValueError("missing support for corrupted amino acid column")
        tokenized_labels = tokenizer(
            labels,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False
        )['input_ids'].squeeze(-1).tolist()
        return pd.Series({
            'clean_masked': clean_masked,
            'corrupted_masked': corrupted_masked,
            'masked_position_after_tokenization': masked_position + 1,
            'tokenized_labels': tokenized_labels,
            'clean_id_names': name
        })

    return sampled_df.apply(process_row, axis=1)


# ---- Dataclasses ----
@dataclass
class SearchMinCircuitAboveThreshold:
    global_csv_circuit_information: str  # Path to circuit_info CSV
    min_performance_threshold: float
    max_performance_threshold: float
    min_circuit_size: int = -1
    max_circuit_size: int = -1
    max_steps: int = 20


# ---- Helpers ----
def get_edges_or_nodes_number(n, total):
    if n > 1:
        if n != int(n):
            raise ValueError(f"n must be integer when > 1. Got: {n}")
        return int(n)
    if n < 0:
        raise ValueError(f"n must be non-negative. Got: {n}")
    return math.ceil(n * total)


def _distance_to_window(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


# ---- File locking for shared CSVs ----
@contextmanager
def _lock_csv(path: str | Path):
    """Acquire lock when appending to shared CSV. Uses fcntl on Unix."""
    path = Path(path)
    lock_path = path.with_suffix(path.suffix + '.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = None
    try:
        lock_fd = open(lock_path, 'w')
        try:
            import fcntl
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass  # Windows: no fcntl, proceed without lock
        yield
    finally:
        if lock_fd is not None:
            try:
                import fcntl
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            except ImportError:
                pass
            lock_fd.close()
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass


def get_clean_corrupted_baselines(model, test_dataloader, metric, device):
    from plms_repeats_circuits.EAP.evaluate import evaluate_baseline
    import logging

    logging.info("Entering get_clean_corrupted_baselines")
    prev_attn_res = model.cfg.use_attn_result
    prev_use_split_qkv_input = model.cfg.use_split_qkv_input
    prev_use_hook_mlp_in = model.cfg.use_hook_mlp_in
    model.cfg.use_attn_result = False
    model.cfg.use_split_qkv_input = False
    model.cfg.use_hook_mlp_in = False
    clean = evaluate_baseline(model=model, dataloader=test_dataloader, metrics=metric, device=device, run_corrupted=False, quiet=False)
    corrupted = evaluate_baseline(model=model, dataloader=test_dataloader, metrics=metric, device=device, run_corrupted=True, quiet=False)
    model.cfg.use_attn_result = prev_attn_res
    model.cfg.use_split_qkv_input = prev_use_split_qkv_input
    model.cfg.use_hook_mlp_in = prev_use_hook_mlp_in
    logging.info("Finished get_clean_corrupted_baselines")
    return clean, corrupted


# ---- Experiment name ----
def build_experiment_name(
    circuit_id: str,
    total_n_samples: int,
    random_state: int,
    attribution_patching_method: str,
    circuit_selection_method: str,
    aggregation_method: str,
    abs_score: bool,
    metric: str,
    graph_type: str,
    exp_prefix: str | None = None,
) -> str:
    """Build experiment name: include graph_type, optional exp_prefix; no task_name (not supported)."""
    parts = [
        f"circuit{circuit_id}",
        f"n{total_n_samples}",
        f"seed{random_state}",
        attribution_patching_method,
        circuit_selection_method,
        f"agg_{aggregation_method}",
        f"abs_{abs_score}",
        f"metric_{metric}",
        f"graph_type_{graph_type}",
    ]
    base = "_".join(parts)
    if exp_prefix:
        return f"{exp_prefix}_{base}"
    return base

# ---- Circuit info path ----
def circuit_info_csv_path(output_dir: Path | str, graph_type: str) -> Path:
    """Path for circuit_info CSV: circuit_info_edges.csv, circuit_info_nodes.csv, circuit_info_neurons.csv."""
    return Path(output_dir) / f"circuit_info_{graph_type}.csv"


# ---- Binary search: edges/nodes ----
def search_min_circuit_binary_search(
    circuit_id,
    model,
    metric_evaluation,
    device,
    g,
    circuit_selection_method,
    search_min_circuit: SearchMinCircuitAboveThreshold,
    results,
    clean_baseline,
    corrupted_baseline,
    test_dataloader,
    graph_type="edges",
    use_file_lock: bool = True,
):
    from plms_repeats_circuits.EAP.graph import Graph, NeuronGraph
    from plms_repeats_circuits.EAP.circuit_selection import select_circuit_edges, select_circuit_nodes
    from plms_repeats_circuits.EAP.evaluate import (
        evaluate_graph,
        evaluate_graph_node,
        evaluate_graph_neurons,
        compute_faithfulness,
    )
    import logging

    best_circuits_csv = search_min_circuit.global_csv_circuit_information
    total = g.count_total_edges() if graph_type == "edges" else g.count_total_nodes()
    csv_dir = os.path.dirname(best_circuits_csv)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    def _get_init_search_range():
        if search_min_circuit.min_circuit_size >= 0 and search_min_circuit.max_circuit_size > search_min_circuit.min_circuit_size:
            return (
                get_edges_or_nodes_number(search_min_circuit.min_circuit_size, total),
                get_edges_or_nodes_number(search_min_circuit.max_circuit_size, total),
                None,
                None,
            )
        if results is None or len(results) == 0:
            raise ValueError("No way to compute valid range for finding best circuit")
        sorted_results = sorted(results, key=lambda x: x[0])
        start, end = 0, g.count_total_edges() if graph_type == "edges" else g.count_total_nodes()
        start_performance, real_n = 0.0, None
        for i, (n, curr_real_n, performance) in enumerate(sorted_results):
            if performance > search_min_circuit.min_performance_threshold:
                end = n
                if i != 0:
                    start = sorted_results[i - 1][0]
                start_performance = performance
                real_n = curr_real_n
                break
        return start, end, start_performance, real_n

    start, end, curr_performance, curr_real_n = _get_init_search_range()
    res_n, res_performance, res_real = None, None, None
    logging.info(f"Search best circuit: initial start={start}, end={end}, curr_performance={curr_performance}")

    closest_n, closest_real, closest_perf, closest_dist = None, None, None, None

    if (
        curr_performance is not None
        and curr_real_n is not None
        and start is not None
        and end is not None
        and search_min_circuit.min_performance_threshold <= curr_performance <= search_min_circuit.max_performance_threshold
    ):
        res_performance = curr_performance
        res_n = end
        res_real = curr_real_n
    else:
        steps = 0
        while start <= end and steps < search_min_circuit.max_steps:
            steps += 1
            n = (start + end) // 2
            logging.info(f"Search best circuit: Evaluating midpoint n={n}")

            if graph_type == "edges":
                select_circuit_edges(graph=g, selection_method=circuit_selection_method, n_edges=n, log=logging.getLogger())
            else:
                select_circuit_nodes(graph=g, selection_method=circuit_selection_method, n_nodes=n, log=logging.getLogger())

            empty_circuit = not g.nodes['logits'].in_graph
            if not empty_circuit:
                if graph_type == "edges":
                    circuit_evaluation = evaluate_graph(
                        model=model, graph=g, dataloader=test_dataloader, metrics=metric_evaluation,
                        device=device, prune=False, quiet=False, calc_input_for_nodes_not_in_graph=False,
                        debug_corrupted_construction=False, calc_clean_logits=False,
                    )
                else:
                    if isinstance(g, NeuronGraph):
                        circuit_evaluation = evaluate_graph_neurons(
                            model=model, graph=g, dataloader=test_dataloader, metrics=metric_evaluation,
                            device=device, quiet=False, debug=False, calc_clean_logits=False,
                        )
                    else:
                        circuit_evaluation = evaluate_graph_node(
                            model=model, graph=g, dataloader=test_dataloader, metrics=metric_evaluation,
                            device=device, quiet=False, debug=False, calc_clean_logits=False,
                        )
                faithfulness_by_mean = compute_faithfulness(
                    clean_baseline=clean_baseline,
                    corrupted_baseline=corrupted_baseline,
                    target_baseline=circuit_evaluation,
                    per_element_faithfulness=False,
                ).item()
                curr_performance = faithfulness_by_mean
            else:
                curr_performance = 0.0

            real_n = g.count_included_edges() if graph_type == "edges" else g.count_included_nodes()
            curr_dist = _distance_to_window(
                curr_performance,
                search_min_circuit.min_performance_threshold,
                search_min_circuit.max_performance_threshold,
            )
            if closest_dist is None or curr_dist < closest_dist or (curr_dist == closest_dist and closest_n is not None and n < closest_n):
                closest_dist, closest_n, closest_real, closest_perf = curr_dist, n, real_n, curr_performance

            if search_min_circuit.min_performance_threshold <= curr_performance <= search_min_circuit.max_performance_threshold:
                res_performance, res_n, res_real = curr_performance, n, real_n
                break
            elif curr_performance > search_min_circuit.max_performance_threshold:
                end = n - 1
            else:
                start = n + 1

    if res_n is None or res_performance is None:
        if closest_n is not None and closest_perf is not None:
            logging.warning(f"Search best circuit: selecting closest performance {closest_perf} at n={closest_n}")
            res_n, res_real, res_performance = closest_n, closest_real, closest_perf
        else:
            logging.warning("Search best circuit: no valid circuit found")
            return

    def _write_csv():
        with open(best_circuits_csv, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if csv_file.tell() == 0:
                if graph_type == "edges":
                    writer.writerow(["circuit_id", "circuit_selection_method", "n_edges", "pct_n_edges", "real_n_edges", "real_pct_n_edges", "faithfulness_by_mean"])
                else:
                    writer.writerow(["circuit_id", "circuit_selection_method", "n_nodes", "pct_n_nodes", "real_n_nodes", "real_pct_n_nodes", "faithfulness_by_mean"])
            pct_n = res_n / total if res_n is not None else None
            real_pct_n = res_real / total if res_real is not None else None
            writer.writerow([circuit_id, circuit_selection_method, res_n, pct_n, res_real, real_pct_n, res_performance])

    if use_file_lock:
        with _lock_csv(best_circuits_csv):
            _write_csv()
    else:
        _write_csv()

    return res_n, res_real, res_performance


# ---- Binary search: neurons ----
def search_min_circuit_binary_search_neurons(
    circuit_id,
    model,
    metric_evaluation,
    device,
    g,
    circuit_selection_method,
    search_min_circuit: SearchMinCircuitAboveThreshold,
    results,
    clean_baseline,
    corrupted_baseline,
    test_dataloader,
    is_per_layer,
    n_nodes_as_first_step_for_neurons,
    use_file_lock: bool = True,
):
    from plms_repeats_circuits.EAP.graph import NeuronGraph
    from plms_repeats_circuits.EAP.circuit_selection import select_circuit_neurons
    from plms_repeats_circuits.EAP.evaluate import evaluate_graph_neurons, compute_faithfulness
    import logging

    if not isinstance(g, NeuronGraph):
        raise ValueError("search_min_circuit_binary_search_neurons: g must be a NeuronGraph")

    best_circuits_csv = search_min_circuit.global_csv_circuit_information
    total = g.n_neurons_per_mlp if is_per_layer else g.count_neurons(filter_by_in_graph=False)
    csv_dir = os.path.dirname(best_circuits_csv)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    def _get_init_search_range():
        if search_min_circuit.min_circuit_size >= 0 and search_min_circuit.max_circuit_size > search_min_circuit.min_circuit_size:
            return (
                get_edges_or_nodes_number(search_min_circuit.min_circuit_size, total),
                get_edges_or_nodes_number(search_min_circuit.max_circuit_size, total),
                None,
            )
        if results is None or len(results) == 0:
            raise ValueError("No way to compute valid range for neurons circuit")
        sorted_results = sorted(results, key=lambda x: x[0])
        start, end = 0, total
        start_performance = 0.0
        for i, (n, performance) in enumerate(sorted_results):
            if performance > search_min_circuit.min_performance_threshold:
                end = n
                if i != 0:
                    start = sorted_results[i - 1][0]
                start_performance = performance
                break
        return start, end, start_performance

    start, end, curr_performance = _get_init_search_range()
    res_n, res_performance = None, None
    closest_n, closest_perf, closest_dist = None, None, None

    if (
        curr_performance is not None
        and start is not None
        and end is not None
        and search_min_circuit.min_performance_threshold <= curr_performance <= search_min_circuit.max_performance_threshold
    ):
        res_performance = curr_performance
        res_n = end
    else:
        steps = 0
        while start <= end and steps < search_min_circuit.max_steps:
            steps += 1
            n = (start + end) // 2
            select_circuit_neurons(
                graph=g,
                selection_method=circuit_selection_method,
                n_neurons=n,
                n_nodes_as_first_step_for_neurons=n_nodes_as_first_step_for_neurons,
                is_per_layer=is_per_layer,
                log=logging.getLogger(),
            )
            empty_circuit = not g.nodes['logits'].in_graph
            if not empty_circuit:
                circuit_evaluation = evaluate_graph_neurons(
                    model=model, graph=g, dataloader=test_dataloader, metrics=metric_evaluation,
                    device=device, quiet=False, debug=False, calc_clean_logits=False,
                )
                curr_performance = compute_faithfulness(
                    clean_baseline=clean_baseline,
                    corrupted_baseline=corrupted_baseline,
                    target_baseline=circuit_evaluation,
                    per_element_faithfulness=False,
                ).item()
            else:
                curr_performance = 0.0

            curr_dist = _distance_to_window(
                curr_performance,
                search_min_circuit.min_performance_threshold,
                search_min_circuit.max_performance_threshold,
            )
            if closest_dist is None or curr_dist < closest_dist or (curr_dist == closest_dist and closest_n is not None and n < closest_n):
                closest_dist, closest_n, closest_perf = curr_dist, n, curr_performance

            if search_min_circuit.min_performance_threshold <= curr_performance <= search_min_circuit.max_performance_threshold:
                res_performance, res_n = curr_performance, n
                break
            elif curr_performance > search_min_circuit.max_performance_threshold:
                end = n - 1
            else:
                start = n + 1

    if res_n is None or res_performance is None:
        if closest_n is not None and closest_perf is not None:
            res_n, res_performance = closest_n, closest_perf
        else:
            logging.warning("Search best circuit (neurons): no valid circuit found")
            return

    select_circuit_neurons(
        graph=g,
        selection_method=circuit_selection_method,
        n_neurons=res_n,
        n_nodes_as_first_step_for_neurons=n_nodes_as_first_step_for_neurons,
        is_per_layer=is_per_layer,
        log=logging.getLogger(),
    )

    pct_n_neurons_per_layer = res_n / g.n_neurons_per_mlp if is_per_layer else None
    tot_n_neurons = g.count_neurons(filter_by_in_graph=True)
    tot_pct_n_neurons = tot_n_neurons / g.count_neurons(filter_by_in_graph=False)
    circuit_size_as_neurons = g.count_included_nodes()
    circuit_size_as_neurons_pct = circuit_size_as_neurons / g.count_total_nodes()
    num_mlp_nodes = g.count_mlp_nodes(filter_by_in_graph=True)

    def _write_csv():
        with open(best_circuits_csv, "a", newline="") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow([
                    "circuit_id", "circuit_selection_method",
                    "n_nodes_as_first_step_for_neurons", "n_neurons", "is_per_layer", "pct_n_neurons_per_layer",
                    "tot_n_neurons", "tot_pct_n_neurons", "circuit_size", "circuit_size_pct", "num_mlp_nodes", "faithfulness_by_mean",
                ])
            w.writerow([
                circuit_id, circuit_selection_method,
                n_nodes_as_first_step_for_neurons, res_n, is_per_layer, pct_n_neurons_per_layer,
                tot_n_neurons, tot_pct_n_neurons, circuit_size_as_neurons, circuit_size_as_neurons_pct, num_mlp_nodes, res_performance,
            ])

    if use_file_lock:
        with _lock_csv(best_circuits_csv):
            _write_csv()
    else:
        _write_csv()

    return res_n, res_performance
