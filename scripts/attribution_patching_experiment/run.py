import argparse
import csv
import sys
from pathlib import Path

from plms_repeats_circuits.utils.counterfactuals_config import (
    COUNTERFACTUAL_METHODS,
    find_file_for_method,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_attribution_patching_nodes_edges import main as run_nodes_edges_main
from run_attribution_patching_neurons import main as run_neurons_main

# Discover step params (aligned with smart_attribution_launcher.py)
DISCOVER_ATTRIBUTION_METHOD = "EAP-IG"
DISCOVER_AGGREGATION = "sum"
DISCOVER_ABS_SCORE = False
DISCOVER_EAP_IG_STEPS = 5
DISCOVER_TRAIN_RATIO = 0.5
DISCOVER_MIN_EDGES_NODES, DISCOVER_MAX_EDGES_NODES = 0.84, 0.85
DISCOVER_MIN_NEURONS, DISCOVER_MAX_NEURONS = 0.8, 0.81
DISCOVER_MIN_CIRCUIT_SIZE = -1
DISCOVER_MAX_CIRCUIT_SIZE = -1
DISCOVER_MAX_SEARCH_STEPS = 50
# nodes (no neurons): same as old launcher n_edges_in_circuit_list
DISCOVER_N_COMPONENTS_NODES_ESM3 = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
DISCOVER_N_COMPONENTS_NODES_ESMC = [0, 50, 100, 200, 300, 400, 500, 600, 700]
# edges + nodes (with fractions): same as old neurons n_edges_in_circuit_list
DISCOVER_N_COMPONENTS_FRACTIONS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

REPEAT_TYPE_PATHS = {
    "identical": "identical",
    "approximate": "approximate",
    "synthetic": "synthetic",
}


def _circuit_discovery_dir(datasets_root: Path, repeat_type: str, model_type: str) -> Path:
    folder = REPEAT_TYPE_PATHS.get(repeat_type, repeat_type)
    return datasets_root / folder / model_type / "circuit_discovery"


def _output_dir(
    results_root: Path,
    repeat_type: str,
    model_type: str,
    graph_type: str,
    seed: int,
) -> Path:
    folder = REPEAT_TYPE_PATHS.get(repeat_type, repeat_type)
    return results_root / "circuit_discovery" / folder / model_type / graph_type / f"seed_{seed}"


def _validate_methods(
    datasets_root: Path,
    repeat_types: list[str],
    model_types: list[str],
    requested_methods: list[str],
) -> None:
    """Validate that all requested methods have circuit_discovery CSVs. Exit if any missing."""
    valid_names = {m["name"] for m in COUNTERFACTUAL_METHODS}
    unknown = [m for m in requested_methods if m not in valid_names]
    if unknown:
        print("Unknown methods (must be in COUNTERFACTUAL_METHODS):", unknown, flush=True)
        sys.exit(1)

    missing = []
    for rt in repeat_types:
        for mt in model_types:
            circuit_dir = _circuit_discovery_dir(datasets_root, rt, mt)
            for method in requested_methods:
                path = find_file_for_method(method, circuit_dir, kind="main", ext="csv")
                if path is None:
                    missing.append(str(circuit_dir / f"*counterfactual_{method}*.csv"))
    if missing:
        print("Missing circuit_discovery CSVs (run counterfactual filter first):", flush=True)
        for p in missing:
            print(f"  - {p}", flush=True)
        sys.exit(1)


def _enumerate_jobs(
    datasets_root: Path,
    repeat_types: list[str],
    model_types: list[str],
    seeds: list[int],
    requested_methods: list[str],
) -> list[tuple[str, str, int, str, Path, Path]]:
    """Return list of (repeat_type, model_type, seed, method, csv_path, output_dir)."""
    jobs = []
    for rt in repeat_types:
        for mt in model_types:
            circuit_dir = _circuit_discovery_dir(datasets_root, rt, mt)
            for method in requested_methods:
                path = find_file_for_method(method, circuit_dir, kind="main", ext="csv")
                if path is not None:
                    for seed in seeds:
                        jobs.append((rt, mt, seed, method, path))
    return jobs


def _find_matching_nodes_circuit(
    nodes_output_dir: Path,
    exp_prefix: str,
    seed: int,
    graph_type: str = "nodes",
) -> tuple[str, int, Path]:
    """Find the single matching (circuit_id, real_n_nodes, json_path) for neurons.

    Requires exactly one match. Verifies seed appears in JSON filename.
    Fails with clear message if 0 or 2+ matches.
    """
    circuit_info_path = nodes_output_dir / f"circuit_info_{graph_type}.csv"
    if not circuit_info_path.exists():
        print(
            f"Neurons: circuit_info_nodes.csv not found at {circuit_info_path}. "
            "Run discover with graph_type=nodes first.",
            flush=True,
        )
        sys.exit(1)

    matches = []
    with open(circuit_info_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print(
            "Neurons: circuit_info_nodes.csv is empty. Run discover with graph_type=nodes first.",
            flush=True,
        )
        sys.exit(1)

    # We know exp_prefix and the output dir is seed-specific (nodes/seed_{seed}/).
    # Match by circuit_id from CSV + glob for {exp_prefix}_circuit{circuit_id}_*.json.
    for row in rows:
        circuit_id = row.get("circuit_id")
        real_n_nodes = row.get("real_n_nodes")
        if circuit_id is None or real_n_nodes is None:
            continue
        try:
            n_nodes_val = int(real_n_nodes)
        except (ValueError, TypeError):
            continue
        pattern = f"{exp_prefix}_circuit{circuit_id}_*.json"
        jsons = list(nodes_output_dir.glob(pattern))
        if not jsons:
            continue
        if len(jsons) > 1:
            print(
                f"Neurons: multiple JSONs for circuit_id {circuit_id}: {jsons}",
                flush=True,
            )
            sys.exit(1)
        json_path = jsons[0]
        if f"seed{seed}" not in json_path.stem:
            print(
                f"Neurons: JSON {json_path.name} does not contain seed{seed} in filename. Skipping.",
                flush=True,
            )
            continue
        matches.append((circuit_id, n_nodes_val, json_path))

    if len(matches) == 0:
        print(
            f"Neurons: no matching nodes circuit for exp_prefix={exp_prefix} in {nodes_output_dir}. "
            "Run discover with graph_type=nodes first.",
            flush=True,
        )
        for row in rows:
            print(f"  circuit_id={row.get('circuit_id')} real_n_nodes={row.get('real_n_nodes')}", flush=True)
        sys.exit(1)
    if len(matches) > 1:
        print(
            f"Neurons: multiple matching nodes circuits ({len(matches)}). Expected exactly one. "
            "This indicates ambiguous or duplicate runs.",
            flush=True,
        )
        for cid, nn, jp in matches:
            print(f"  circuit_id={cid} n_nodes={nn} json={jp.name}", flush=True)
        sys.exit(1)

    return matches[0]


def run_discover(
    datasets_root: Path,
    results_root: Path,
    repeat_types: list[str],
    model_types: list[str],
    seeds: list[int],
    graph_type: str,
    methods: list[str],
    n_examples: int = 1000,
) -> None:
    """Run the discover step: attribution patching for edges, nodes, or neurons."""
    _validate_methods(datasets_root, repeat_types, model_types, methods)
    jobs = _enumerate_jobs(datasets_root, repeat_types, model_types, seeds, methods)

    for repeat_type, model_type, seed, method, csv_path in jobs:
        exp_prefix = csv_path.stem
        output_dir = _output_dir(results_root, repeat_type, model_type, graph_type, seed)
        output_dir.mkdir(parents=True, exist_ok=True)

        if graph_type == "edges":
            run_nodes_edges_main(
                csv_path=str(csv_path),
                output_dir=str(output_dir),
                total_n_samples=n_examples,
                model_type=model_type,
                graph_type="edges",
                attribution_patching_method=DISCOVER_ATTRIBUTION_METHOD,
                circuit_selection_method="greedy_abs",
                n_components_in_circuit_list=DISCOVER_N_COMPONENTS_FRACTIONS,
                metric="log_prob",
                abs_score=DISCOVER_ABS_SCORE,
                aggregation_method=DISCOVER_AGGREGATION,
                exp_prefix=exp_prefix,
                train_ratio=DISCOVER_TRAIN_RATIO,
                enable_min_circuit_search=True,
                min_performance_threshold=DISCOVER_MIN_EDGES_NODES,
                max_performance_threshold=DISCOVER_MAX_EDGES_NODES,
                min_circuit_size=DISCOVER_MIN_CIRCUIT_SIZE,
                max_circuit_size=DISCOVER_MAX_CIRCUIT_SIZE,
                max_search_steps=DISCOVER_MAX_SEARCH_STEPS,
                random_state=seed,
                EAP_IG_steps=DISCOVER_EAP_IG_STEPS,
            )
        elif graph_type == "nodes":
            n_components_nodes = (
                DISCOVER_N_COMPONENTS_NODES_ESMC if model_type == "esm-c" else DISCOVER_N_COMPONENTS_NODES_ESM3
            )
            run_nodes_edges_main(
                csv_path=str(csv_path),
                output_dir=str(output_dir),
                total_n_samples=n_examples,
                model_type=model_type,
                graph_type="nodes",
                attribution_patching_method=DISCOVER_ATTRIBUTION_METHOD,
                circuit_selection_method="top_n_abs",
                n_components_in_circuit_list=n_components_nodes,
                metric="log_prob",
                abs_score=DISCOVER_ABS_SCORE,
                aggregation_method=DISCOVER_AGGREGATION,
                exp_prefix=exp_prefix,
                train_ratio=DISCOVER_TRAIN_RATIO,
                enable_min_circuit_search=True,
                min_performance_threshold=DISCOVER_MIN_EDGES_NODES,
                max_performance_threshold=DISCOVER_MAX_EDGES_NODES,
                min_circuit_size=DISCOVER_MIN_CIRCUIT_SIZE,
                max_circuit_size=DISCOVER_MAX_CIRCUIT_SIZE,
                max_search_steps=DISCOVER_MAX_SEARCH_STEPS,
                random_state=seed,
                EAP_IG_steps=DISCOVER_EAP_IG_STEPS,
            )
        elif graph_type == "neurons":
            nodes_output_dir = _output_dir(
                results_root, repeat_type, model_type, "nodes", seed
            )
            circuit_id, n_nodes, nodes_json_path = _find_matching_nodes_circuit(
                nodes_output_dir,
                exp_prefix=exp_prefix,
                seed=seed,
            )
            run_neurons_main(
                csv_path=str(csv_path),
                output_dir=str(output_dir),
                total_n_samples=n_examples,
                model_type=model_type,
                nodes_graph_json=str(nodes_json_path),
                n_nodes=n_nodes,
                attribution_patching_method=DISCOVER_ATTRIBUTION_METHOD,
                circuit_selection_method="top_n_abs",
                metric="log_prob",
                abs_score=DISCOVER_ABS_SCORE,
                aggregation_method=DISCOVER_AGGREGATION,
                exp_prefix=exp_prefix,
                train_ratio=DISCOVER_TRAIN_RATIO,
                enable_min_circuit_search=True,
                min_performance_threshold_neurons=DISCOVER_MIN_NEURONS,
                max_performance_threshold_neurons=DISCOVER_MAX_NEURONS,
                min_circuit_size=DISCOVER_MIN_CIRCUIT_SIZE,
                max_circuit_size=DISCOVER_MAX_CIRCUIT_SIZE,
                max_search_steps=DISCOVER_MAX_SEARCH_STEPS,
                random_state=seed,
                EAP_IG_steps=DISCOVER_EAP_IG_STEPS,
            )
        else:
            raise ValueError(f"graph_type must be edges, nodes, or neurons, got {graph_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Run attribution patching circuit discovery pipeline."
    )
    parser.add_argument(
        "--datasets_root",
        type=Path,
        default=REPO_ROOT / "datasets",
        help="Root for circuit_discovery CSVs",
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=REPO_ROOT / "results",
        help="Root for circuit_discovery outputs",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["discover"],
        choices=["discover"],
        help="Steps to run",
    )
    parser.add_argument(
        "--repeat_types",
        nargs="+",
        default=["identical", "approximate", "synthetic"],
        help="Repeat types to process",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds (e.g. 42 43 44 45 46)",
    )
    parser.add_argument(
        "--model_types",
        nargs="+",
        default=["esm3"],
        choices=["esm3", "esm-c"],
        help="Models to run",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        required=True,
        choices=["edges", "nodes", "neurons"],
        help="Graph type for circuit discovery",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="Counterfactual methods (validated against datasets)",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=1000,
        help="Samples per run",
    )
    args = parser.parse_args()

    datasets_root = args.datasets_root.resolve()
    results_root = args.results_root.resolve()

    print("=" * 60, flush=True)
    print("Attribution Patching Circuit Discovery", flush=True)
    for k, v in vars(args).items():
        print(f"  {k}: {v}", flush=True)
    print("=" * 60, flush=True)

    if "discover" in args.steps:
        print("\n=== Step: discover ===", flush=True)
        run_discover(
            datasets_root=datasets_root,
            results_root=results_root,
            repeat_types=args.repeat_types,
            model_types=args.model_types,
            seeds=args.seeds,
            graph_type=args.graph_type,
            methods=args.methods,
            n_examples=args.n_examples,
        )
        print("=== Step: discover done ===\n", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
