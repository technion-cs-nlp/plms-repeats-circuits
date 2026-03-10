import argparse
import csv
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plms_repeats_circuits.EAP.graph import Graph
from plms_repeats_circuits.EAP import circuit_selection
from attribution_patching_utils import get_edges_or_nodes_number


def compute_iou(g1: Graph, g2: Graph, mode: str) -> float:
    """Compute Intersection over Union between two graphs. mode is 'edges' or 'nodes'."""
    if mode == "edges":
        set1 = g1.get_edges_in_graph()
        set2 = g2.get_edges_in_graph()
    elif mode == "nodes":
        set1 = g1.get_nodes_in_graph()
        set2 = g2.get_nodes_in_graph()
    else:
        raise ValueError(f"mode must be 'edges' or 'nodes', got {mode}")

    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0:
        return 0.0
    return len(intersection) / len(union)


def compute_recall(g1: Graph, g2: Graph, mode: str) -> float:
    """Compute Recall: |g1 ∩ g2| / |g2| (how much of g2 is covered by g1)."""
    if mode == "edges":
        set1 = g1.get_edges_in_graph()
        set2 = g2.get_edges_in_graph()
    elif mode == "nodes":
        set1 = g1.get_nodes_in_graph()
        set2 = g2.get_nodes_in_graph()
    else:
        raise ValueError(f"mode must be 'edges' or 'nodes', got {mode}")

    intersection = set1.intersection(set2)
    if len(set2) == 0:
        return 0.0
    return len(intersection) / len(set2)


def _load_and_apply_circuit_size(
    json_path: str,
    graph_type: str,
    circuit_size: float | None,
    circuit_selection_method: str,
) -> Graph:
    """Load graph from JSON and optionally apply circuit size selection."""
    g = Graph.from_json(json_path)

    if circuit_size is not None:
        g.reset_graph_state()
        if graph_type == "edges":
            total = g.count_total_edges()
            actual_size = get_edges_or_nodes_number(circuit_size, total)
            circuit_selection.select_circuit_edges(g, circuit_selection_method, actual_size)
        else:  # nodes
            total = g.count_total_nodes()
            actual_size = get_edges_or_nodes_number(circuit_size, total)
            circuit_selection.select_circuit_nodes(g, circuit_selection_method, actual_size)
    # else: use graph as loaded from JSON (in_graph state preserved)

    return g


def main(
    circuit_json_1: str,
    circuit_json_2: str,
    graph_type: str,
    save_results_csv: str,
    metric: str = "iou",
    circuit_size_1: float | None = None,
    circuit_size_2: float | None = None,
    circuit_selection_method: str | None = None,
    circuit_selection_method_1: str | None = None,
    circuit_selection_method_2: str | None = None,
    source_task_name: str = "",
    source_circuit_id: str = "",
    target_task_name: str = "",
    target_circuit_id: str = "",
    **kwargs,
) -> dict:
    """
    Compute IOU or recall between two circuits and append to CSV.

    Returns dict with the result row.
    """
    if graph_type not in ("edges", "nodes"):
        raise ValueError(f"graph_type must be 'edges' or 'nodes', got {graph_type}")
    if metric not in ("iou", "recall"):
        raise ValueError(f"metric must be 'iou' or 'recall', got {metric}")
    if not os.path.exists(circuit_json_1):
        raise FileNotFoundError(f"Circuit JSON 1 not found: {circuit_json_1}")
    if not os.path.exists(circuit_json_2):
        raise FileNotFoundError(f"Circuit JSON 2 not found: {circuit_json_2}")

    EDGES_METHODS = {"greedy", "greedy_abs", "top_n", "top_n_abs", "none"}
    NODES_METHODS = {"top_n", "top_n_abs"}
    default_sel = "greedy_abs" if graph_type == "edges" else "top_n_abs"
    sel1 = circuit_selection_method_1 or circuit_selection_method or default_sel
    sel2 = circuit_selection_method_2 or circuit_selection_method or default_sel
    for sel, label in [(sel1, "circuit 1"), (sel2, "circuit 2")]:
        if graph_type == "edges" and sel not in EDGES_METHODS:
            raise ValueError(
                f"circuit_selection_method for {label} (edges) must be one of {sorted(EDGES_METHODS)}, got {sel}"
            )
        if graph_type == "nodes" and sel not in NODES_METHODS:
            raise ValueError(
                f"circuit_selection_method for {label} (nodes) must be one of {sorted(NODES_METHODS)}, got {sel}"
            )

    g1 = _load_and_apply_circuit_size(circuit_json_1, graph_type, circuit_size_1, sel1)
    g2 = _load_and_apply_circuit_size(circuit_json_2, graph_type, circuit_size_2, sel2)

    if metric == "iou":
        value = compute_iou(g1, g2, graph_type)
    else:
        value = compute_recall(g1, g2, graph_type)

    if graph_type == "edges":
        n1 = g1.count_included_edges()
        total1 = g1.count_total_edges()
        n2 = g2.count_included_edges()
        total2 = g2.count_total_edges()
        metric_col = f"{metric}_edges"
    else:
        n1 = g1.count_included_nodes()
        total1 = g1.count_total_nodes()
        n2 = g2.count_included_nodes()
        total2 = g2.count_total_nodes()
        metric_col = f"{metric}_nodes"

    pct1 = n1 / total1 if total1 > 0 else 0.0
    pct2 = n2 / total2 if total2 > 0 else 0.0

    row = {
        "source_task_name": source_task_name or Path(circuit_json_1).stem,
        "source_circuit_id": source_circuit_id or "",
        "target_task_name": target_task_name or Path(circuit_json_2).stem,
        "target_circuit_id": target_circuit_id or "",
        f"n_{graph_type}_source": n1,
        f"real_pct_{graph_type}_source": pct1,
        f"n_{graph_type}_target": n2,
        f"real_pct_{graph_type}_target": pct2,
        metric_col: value,
    }

    csv_path = Path(save_results_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    columns = [
        "source_task_name", "source_circuit_id", "target_task_name", "target_circuit_id",
        f"n_{graph_type}_source", f"real_pct_{graph_type}_source",
        f"n_{graph_type}_target", f"real_pct_{graph_type}_target",
        metric_col,
    ]
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)

    return row


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compute IOU or recall between a pair of circuits (edges or nodes only)."
    )
    parser.add_argument("--circuit_json_1", type=str, required=True, help="Path to first circuit JSON")
    parser.add_argument("--circuit_json_2", type=str, required=True, help="Path to second circuit JSON")
    parser.add_argument("--graph_type", type=str, required=True, choices=["edges", "nodes"])
    parser.add_argument("--save_results_csv", type=str, required=True, help="Path to append results")
    parser.add_argument("--metric", type=str, default="iou", choices=["iou", "recall"])
    parser.add_argument("--circuit_size_1", type=float, default=None, help="Circuit 1 size (absolute or fraction). If omitted, use graph as loaded.")
    parser.add_argument("--circuit_size_2", type=float, default=None, help="Circuit 2 size (absolute or fraction). If omitted, use graph as loaded.")
    parser.add_argument(
        "--circuit_selection_method",
        type=str,
        default=None,
        help="Fallback for both circuits. Edges: greedy, greedy_abs, top_n, top_n_abs, none. Nodes: top_n, top_n_abs.",
    )
    parser.add_argument(
        "--circuit_selection_method_1",
        type=str,
        default=None,
        help="Selection method for circuit 1. Overrides --circuit_selection_method.",
    )
    parser.add_argument(
        "--circuit_selection_method_2",
        type=str,
        default=None,
        help="Selection method for circuit 2. Overrides --circuit_selection_method.",
    )
    parser.add_argument("--source_task_name", type=str, default="", help="For CSV columns")
    parser.add_argument("--source_circuit_id", type=str, default="", help="For CSV columns")
    parser.add_argument("--target_task_name", type=str, default="", help="For CSV columns")
    parser.add_argument("--target_circuit_id", type=str, default="", help="For CSV columns")
    return parser.parse_args()


if __name__ == "__main__":
    sys.setrecursionlimit(3000)
    args = _parse_args()
    try:
        main(**vars(args))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
