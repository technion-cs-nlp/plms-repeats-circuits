"""Circuit selection: choose which edges, nodes, or neurons form the circuit from attribution scores."""

from typing import Optional
import logging

from .graph import Graph, NeuronGraph


def _log_info(log: Optional[logging.Logger], msg: str) -> None:
    if log is not None:
        log.info(msg)
    else:
        print(msg)


def _log_error(log: Optional[logging.Logger], msg: str) -> None:
    if log is not None:
        log.error(msg)
    else:
        print(msg)


def select_circuit_edges(
    graph: Graph,
    selection_method: str,
    n_edges: int,
    *,
    log: Optional[logging.Logger] = None,
) -> None:
    """Select edges for the circuit from an edge-level graph. Uses greedy, top_n, or top_n_abs."""
    _log_info(log, "Entering select_circuit_edges")
    _log_info(log, f"Selection method: {selection_method}, n_edges: {n_edges}")

    if n_edges >= len(graph.edges):
        graph.set_all_edges_in_graph(in_graph=True)
        _log_info(log, "All edges assigned to the graph.")
    elif selection_method == "greedy":
        graph.apply_greedy(n_edges=n_edges, reset=True, absolute=False)
    elif selection_method == "greedy_abs":
        graph.apply_greedy(n_edges=n_edges, reset=True, absolute=True)
    elif selection_method == "top_n":
        graph.apply_topn(n=n_edges, absolute=False)
    elif selection_method == "top_n_abs":
        graph.apply_topn(n=n_edges, absolute=True)
    elif selection_method == "none":
        pass
    else:
        _log_error(log, f"Unsupported selection method: {selection_method}")
        raise RuntimeError(f"Unsupported selection method: {selection_method}")

    _log_info(log, "Pruning dead nodes after circuit selection.")
    graph.prune_dead_nodes()
    _log_info(log, "Dead nodes pruned successfully.")
    n_nodes = len([n for n in graph.nodes.values() if n.in_graph])
    _log_info(log, f"Number of nodes in the graph after selection: {n_nodes}")
    n_edges_after = len([e for e in graph.edges.values() if e.in_graph])
    _log_info(log, f"Number of edges in the graph after selection: {n_edges_after}")
    _log_info(log, "Exiting select_circuit_edges")


def select_circuit_nodes(
    graph: Graph,
    selection_method: str,
    n_nodes: int,
    *,
    log: Optional[logging.Logger] = None,
) -> None:
    """Select nodes for the circuit from a node-level graph. Uses top_n or top_n_abs."""
    _log_info(log, "Entering select_circuit_nodes")
    _log_info(log, f"Selection method: {selection_method}, n_nodes: {n_nodes}")

    n_total = len(graph.nodes) if isinstance(graph, NeuronGraph) else graph.count_total_nodes()
    if n_nodes >= n_total:
        _log_info(log, "All nodes assigned to the graph.")
        graph.set_all_nodes_in_graph(in_graph=True)
        _log_info(log, "Exiting select_circuit_nodes")
        return

    if selection_method == "top_n":
        absolute = False
    elif selection_method == "top_n_abs":
        absolute = True
    else:
        _log_error(log, f"Unsupported selection method: {selection_method}")
        raise RuntimeError(f"Unsupported selection method: {selection_method}")

    if isinstance(graph, NeuronGraph):
        graph.apply_topn_on_nodes(n=n_nodes, absolute=absolute)
    else:
        graph.apply_topn(n=n_nodes, absolute=absolute)

    _log_info(log, "Exiting select_circuit_nodes")


def select_circuit_neurons(
    graph: NeuronGraph,
    selection_method: str,
    n_neurons: int,
    n_nodes_as_first_step_for_neurons: int,
    is_per_layer: bool,
    *,
    log: Optional[logging.Logger] = None,
) -> NeuronGraph:
    """Select neurons for the circuit from a NeuronGraph. Uses top_n or top_n_abs, with optional per-layer selection."""
    _log_info(log, "Entering select_circuit_neurons")
    _log_info(
        log,
        f"Selection method: {selection_method}, "
        f"n_neurons: {n_neurons}, n_nodes_as_first_step_for_neurons: {n_nodes_as_first_step_for_neurons}, "
        f"is_per_layer: {is_per_layer}",
    )

    if not isinstance(graph, NeuronGraph):
        raise ValueError("select_circuit_neurons: graph must be a NeuronGraph")

    if selection_method == "top_n":
        if is_per_layer:
            graph.apply_topn_neurons_per_layer(
                n_neurons=n_neurons,
                n_nodes=n_nodes_as_first_step_for_neurons,
                absolute=False,
            )
        else:
            graph.apply_topn_only_neurons(
                n_neurons=n_neurons,
                n_nodes=n_nodes_as_first_step_for_neurons,
                absolute=False,
            )
    elif selection_method == "top_n_abs":
        if is_per_layer:
            graph.apply_topn_neurons_per_layer(
                n_neurons=n_neurons,
                n_nodes=n_nodes_as_first_step_for_neurons,
                absolute=True,
            )
        else:
            graph.apply_topn_only_neurons(
                n_neurons=n_neurons,
                n_nodes=n_nodes_as_first_step_for_neurons,
                absolute=True,
            )
    else:
        _log_error(log, f"Unsupported selection method: {selection_method}")
        raise RuntimeError(f"Unsupported selection method: {selection_method}")

    _log_info(log, "Exiting select_circuit_neurons")
    return graph
