"""
Cross-task faithfulness: evaluate a source circuit on target data.
Calls run_attribution_patching_nodes_edges.main() and aggregates results into cross_task.csv.
Exposes main(**kwargs) for programmatic use; validation inside main().
"""
import argparse
import csv
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

from run_attribution_patching_nodes_edges import main as run_nodes_edges_main


@dataclass
class CrossTaskFaithfulnessArgs:
    source_task_name: str
    source_circuit_id: str
    target_task_name: str
    target_circuit_id: str
    save_results_csv: str


def main(
    source_task_name: str,
    source_circuit_id: str,
    source_circuit_json: str,
    target_task_name: str,
    target_circuit_id: str,
    target_csv_path: str,
    save_results_csv: str,
    output_dir: str,
    total_n_samples: int,
    model_type: str,
    graph_type: str = "nodes",
    attribution_patching_method: str = "EAP",
    circuit_selection_method: str = "greedy_abs",
    n_components_in_circuit_list: list[float] | None = None,
    metric: str = "logit_diff",
    abs_score: bool = False,
    aggregation_method: str = "sum",
    EAP_IG_steps: int = 5,
    batch_size: int = 1,
    random_state: int = 42,
    train_ratio: float = 0.5,
    exp_prefix: str = "",
    delete_per_experiment_csv: bool = False,
    **kwargs,
):
    """
    Run cross-task faithfulness: load source circuit, evaluate on target data,
    append results to cross_task.csv.
    """
    # Validation
    if not source_task_name or not source_circuit_id:
        raise ValueError("source_task_name and source_circuit_id required")
    if not target_task_name or not target_circuit_id:
        raise ValueError("target_task_name and target_circuit_id required")
    if not save_results_csv:
        raise ValueError("save_results_csv required")
    if not os.path.exists(source_circuit_json):
        raise FileNotFoundError(f"Source circuit JSON not found: {source_circuit_json}")
    if not os.path.exists(target_csv_path):
        raise FileNotFoundError(f"Target CSV not found: {target_csv_path}")

    if n_components_in_circuit_list is None:
        n_components_in_circuit_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "cross_task.log"
    log = logging.getLogger("run_cross_task")
    if not log.handlers:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)

    log.info("Running cross-task faithfulness: source->target")
    df = run_nodes_edges_main(
        csv_path=target_csv_path,
        output_dir=str(output_dir),
        total_n_samples=total_n_samples,
        model_type=model_type,
        graph_type=graph_type,
        attribution_patching_method=attribution_patching_method,
        circuit_selection_method=circuit_selection_method,
        n_components_in_circuit_list=n_components_in_circuit_list,
        metric=metric,
        abs_score=abs_score,
        aggregation_method=aggregation_method,
        circuit_json_path=source_circuit_json,
        EAP_IG_steps=EAP_IG_steps,
        batch_size=batch_size,
        random_state=random_state,
        train_ratio=train_ratio,
        exp_prefix=exp_prefix or None,
        enable_min_circuit_search=False,
        save_results_csv=False,
        save_scores_per_example_npy=False,
    )

    if df is None or len(df) == 0:
        log.warning("No results from nodes_edges run.")
        return

    # Map DataFrame columns to (n, real_n, pct, real_pct, faithfulness_by_mean)
    if graph_type == "edges":
        n_col, real_col, pct_col, real_pct_col = "n_edges", "real_n_edges", "pct_n_edges", "real_pct_n_edges"
    else:
        n_col, real_col, pct_col, real_pct_col = "n_nodes", "real_n_nodes", "pct_n_nodes", "real_pct_n_nodes"

    csv_dir = os.path.dirname(save_results_csv)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    write_header = not os.path.exists(save_results_csv) or os.path.getsize(save_results_csv) == 0
    with open(save_results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            if graph_type == "edges":
                w.writerow(["source_task_name", "source_circuit_id", "target_task_name", "target_circuit_id", "n_edges", "real_n_edges", "pct_edges", "real_pct_edges", "faithfulness_by_mean"])
            else:
                w.writerow(["source_task_name", "source_circuit_id", "target_task_name", "target_circuit_id", "n_nodes", "real_n_nodes", "pct_nodes", "real_pct_nodes", "faithfulness_by_mean"])
        for _, row in df.iterrows():
            n = row[n_col]
            real_n = row[real_col]
            pct = row[pct_col]
            real_pct = row[real_pct_col]
            faithfulness = row["faithfulness_by_mean"]
            w.writerow([source_task_name, source_circuit_id, target_task_name, target_circuit_id, n, real_n, pct, real_pct, faithfulness])

    log.info(f"Cross-task results appended to {save_results_csv}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Cross-task faithfulness evaluation.")
    parser.add_argument("--source_task_name", type=str, required=True, help="Task name of the source circuit")
    parser.add_argument("--source_circuit_id", type=str, required=True, help="UUID of the source circuit")
    parser.add_argument("--source_circuit_json", type=str, required=True, help="Path to source circuit JSON (edges or nodes)")
    parser.add_argument("--target_task_name", type=str, required=True, help="Task name of the target evaluation")
    parser.add_argument("--target_circuit_id", type=str, required=True, help="UUID for the target run (for logging)")
    parser.add_argument("--target_csv_path", type=str, required=True, help="CSV path to target dataset")
    parser.add_argument("--save_results_csv", type=str, required=True, help="Path to append cross-task results")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for per-experiment files")
    parser.add_argument("--total_n_samples", type=int, required=True, help="Number of samples from target dataset")
    parser.add_argument("--model_type", type=str, required=True, choices=["esm3", "esm-c"], help="Model type")
    parser.add_argument("--graph_type", type=str, default="nodes", choices=["edges", "nodes"], help="Graph type (edges or nodes)")
    parser.add_argument("--attribution_patching_method", type=str, default="EAP", choices=["EAP", "EAP-IG"], help="Attribution method")
    parser.add_argument("--circuit_selection_method", type=str, default="greedy_abs", help="Circuit selection (e.g. greedy_abs, top_n_abs)")
    parser.add_argument("--n_components_in_circuit_list", type=float, nargs="+", default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0], help="Fractions/numbers of components to evaluate")
    parser.add_argument("--metric", type=str, default="logit_diff", choices=["logit_diff", "log_prob"], help="Evaluation metric")
    parser.add_argument("--abs_score", action="store_true", help="Use absolute scores in attribution")
    parser.add_argument("--aggregation_method", type=str, default="sum", choices=["sum", "pos_mean"], help="How to aggregate attribution scores")
    parser.add_argument("--EAP_IG_steps", type=int, default=5, help="Steps for EAP-IG integration")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--train_ratio", type=float, default=0.5, help="Train/test split ratio (unused; kept for API compatibility)")
    parser.add_argument("--exp_prefix", type=str, default="", help="Prefix for experiment names")
    parser.add_argument("--delete_per_experiment_csv", action="store_true", help="Delete per-experiment CSVs after appending to cross-task results")
    return parser.parse_args()


if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(3000)
    args = _parse_args()
    try:
        main(**vars(args))
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)
