import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CircuitInfo:
    """Circuit metadata for compare steps."""

    repeat_type: str
    model_type: str
    seed: int
    method: str
    circuit_id: str
    json_path: Path
    circuit_size: int | float
    task_name: str
    circuit_selection_method: str

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
from run_iou_recall import main as run_iou_recall_main
from run_cross_task import main as run_cross_task_main

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


def _find_circuit_json(output_dir: Path, circuit_id: str, exp_prefix: str) -> Path | None:
    """Find circuit JSON by circuit_id and exp_prefix. Returns None if not found."""
    pattern = f"{exp_prefix}_circuit{circuit_id}_*.json"
    candidates = list(output_dir.glob(pattern))
    return candidates[0] if candidates else None


def _enumerate_circuits_for_compare(
    results_root: Path,
    datasets_root: Path,
    repeat_types: list[str],
    model_types: list[str],
    seeds: list[int],
    methods: list[str],
    graph_type: str,
) -> list[CircuitInfo]:
    """
    Enumerate circuits for compare steps. graph_type must be edges or nodes.
    """
    circuits = []
    for rt in repeat_types:
        for mt in model_types:
            circuit_dir = _circuit_discovery_dir(datasets_root, rt, mt)
            for seed in seeds:
                output_dir = _output_dir(results_root, rt, mt, graph_type, seed)
                info_path = output_dir / f"circuit_info_{graph_type}.csv"
                if not info_path.exists():
                    print(f"Skipping (no circuit_info): {info_path}", flush=True)
                    continue
                with open(info_path) as f:
                    rows = list(csv.DictReader(f))
                for method in methods:
                    path = find_file_for_method(method, circuit_dir, kind="main", ext="csv")
                    if path is None:
                        print(f"Skipping (no dataset for method): {method} in {circuit_dir}", flush=True)
                        continue
                    exp_prefix = path.stem
                    for row in rows:
                        cid = row.get("circuit_id")
                        if not cid:
                            print(f"Skipping (no circuit_id in row): {info_path} row={dict(row)}", flush=True)
                            continue
                        json_path = _find_circuit_json(output_dir, cid, exp_prefix)
                        if json_path is None:
                            print(f"Skipping (no JSON for circuit): cid={cid} exp_prefix={exp_prefix} in {output_dir}", flush=True)
                            continue
                        if graph_type == "edges":
                            sz = row.get("real_n_edges")
                        else:
                            sz = row.get("real_n_nodes")
                        try:
                            circuit_size = int(float(sz)) if sz is not None else 1
                        except (ValueError, TypeError):
                            circuit_size = 1
                        task_name = f"{rt}_{method}_{seed}"
                        sel_method = row.get("circuit_selection_method") or (
                            "greedy_abs" if graph_type == "edges" else "top_n_abs"
                        )
                        circuits.append(
                            CircuitInfo(
                                repeat_type=rt,
                                model_type=mt,
                                seed=seed,
                                method=method,
                                circuit_id=cid,
                                json_path=json_path,
                                circuit_size=circuit_size,
                                task_name=task_name,
                                circuit_selection_method=sel_method,
                            )
                        )
    return circuits


def run_compare_iou_recall(
    datasets_root: Path,
    results_root: Path,
    repeat_types: list[str],
    model_types: list[str],
    seeds: list[int],
    graph_type: str,
    methods: list[str],
    metric: str = "iou",
    compare_modes: list[str] | None = None,
    all_circuits: list[CircuitInfo] | None = None,
) -> None:
    """Run IOU/recall comparison for across_counterfactual and across_repeats modes. Requires graph_type in (edges, nodes). compare_modes must be a list (not None)."""
    if all_circuits is None:
        all_circuits = _enumerate_circuits_for_compare(
            results_root, datasets_root, repeat_types, model_types, seeds, methods, graph_type
        )
    if not all_circuits:
        print("No circuits found for compare_iou_recall. Run discover first.", flush=True)
        return

    def _run_pair(c1: CircuitInfo, c2: CircuitInfo, out_csv: Path) -> None:
        assert c1.seed == c2.seed, "Compare only within same seed"
        run_iou_recall_main(
            circuit_json_1=str(c1.json_path),
            circuit_json_2=str(c2.json_path),
            graph_type=graph_type,
            save_results_csv=str(out_csv),
            metric=metric,
            circuit_size_1=c1.circuit_size,
            circuit_size_2=c2.circuit_size,
            circuit_selection_method_1=c1.circuit_selection_method,
            circuit_selection_method_2=c2.circuit_selection_method,
            source_task_name=c1.task_name,
            source_circuit_id=c1.circuit_id,
            target_task_name=c2.task_name,
            target_circuit_id=c2.circuit_id,
        )

    if "across_counterfactual" in compare_modes:
        for rt in repeat_types:
            for mt in model_types:
                for seed in seeds:
                    group = [c for c in all_circuits if c.repeat_type == rt and c.model_type == mt and c.seed == seed]
                    if len(group) < 2:
                        continue
                    out_dir = results_root / "circuit_discovery_compare" / "iou_recall" / "across_counterfactual" / rt / mt / graph_type / f"seed_{seed}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_csv = out_dir / f"{metric}_results.csv"
                    for i, c1 in enumerate(group):
                        j_start = 0  # Compute all pairs for both IOU and recall
                        for j in range(j_start, len(group)):
                            c2 = group[j]
                            _run_pair(c1, c2, out_csv)

    if "across_repeats" in compare_modes:
        for mt in model_types:
            for seed in seeds:
                for method in methods:
                    group = [c for c in all_circuits if c.model_type == mt and c.seed == seed and c.method == method]
                    if len(group) < 2:
                        continue
                    out_dir = results_root / "circuit_discovery_compare" / "iou_recall" / "across_repeats" / mt / graph_type / f"seed_{seed}" / method
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_csv = out_dir / f"{metric}_results.csv"
                    for i, c1 in enumerate(group):
                        j_start = 0  # Compute all pairs for both IOU and recall
                        for j in range(j_start, len(group)):
                            c2 = group[j]
                            _run_pair(c1, c2, out_csv)


def run_compare_cross_task(
    datasets_root: Path,
    results_root: Path,
    repeat_types: list[str],
    model_types: list[str],
    seeds: list[int],
    graph_type: str,
    methods: list[str],
    n_examples: int = 1000,
    compare_modes: list[str] | None = None,
    all_circuits: list[CircuitInfo] | None = None,
) -> None:
    """Run cross-task faithfulness comparison for across_counterfactual and across_repeats modes. Requires graph_type in (edges, nodes). compare_modes must be a list (not None)."""
    if all_circuits is None:
        all_circuits = _enumerate_circuits_for_compare(
            results_root, datasets_root, repeat_types, model_types, seeds, methods, graph_type
        )
    if not all_circuits:
        print("No circuits found for compare_cross_task. Run discover first.", flush=True)
        return

    circuit_dir_fn = _circuit_discovery_dir

    def _run_pair(c1: CircuitInfo, c2: CircuitInfo, out_csv: Path) -> None:
        assert c1.seed == c2.seed, "Compare only within same seed"
        csv_dir = circuit_dir_fn(datasets_root, c2.repeat_type, c2.model_type)
        target_path = find_file_for_method(c2.method, csv_dir, kind="main", ext="csv")
        if target_path is None:
            return
        output_dir = results_root / "circuit_discovery_compare" / "cross_task" / "temp" / f"{c1.circuit_id}_to_{c2.circuit_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        run_cross_task_main(
            source_task_name=c1.task_name,
            source_circuit_id=c1.circuit_id,
            source_circuit_json=str(c1.json_path),
            target_task_name=c2.task_name,
            target_circuit_id=c2.circuit_id,
            target_csv_path=str(target_path),
            save_results_csv=str(out_csv),
            output_dir=str(output_dir),
            total_n_samples=n_examples,
            model_type=c2.model_type,
            graph_type=graph_type,
            random_state=c2.seed,
            n_components_in_circuit_list=[1.0],
            metric="log_prob",
            attribution_patching_method=DISCOVER_ATTRIBUTION_METHOD,
            circuit_selection_method=c1.circuit_selection_method,
        )

    if "across_counterfactual" in compare_modes:
        for rt in repeat_types:
            for mt in model_types:
                for seed in seeds:
                    group = [c for c in all_circuits if c.repeat_type == rt and c.model_type == mt and c.seed == seed]
                    if len(group) < 2:
                        continue
                    out_dir = results_root / "circuit_discovery_compare" / "cross_task" / "across_counterfactual" / rt / mt / graph_type / f"seed_{seed}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_csv = out_dir / "cross_task_results.csv"
                    for i, c1 in enumerate(group):
                        for c2 in group:
                            if c1.circuit_id == c2.circuit_id:
                                continue
                            _run_pair(c1, c2, out_csv)

    if "across_repeats" in compare_modes:
        for mt in model_types:
            for seed in seeds:
                for method in methods:
                    group = [c for c in all_circuits if c.model_type == mt and c.seed == seed and c.method == method]
                    if len(group) < 2:
                        continue
                    out_dir = results_root / "circuit_discovery_compare" / "cross_task" / "across_repeats" / mt / graph_type / f"seed_{seed}" / method
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_csv = out_dir / "cross_task_results.csv"
                    for i, c1 in enumerate(group):
                        for c2 in group:
                            if c1.circuit_id == c2.circuit_id:
                                continue
                            _run_pair(c1, c2, out_csv)


def run_compare(
    datasets_root: Path,
    results_root: Path,
    repeat_types: list[str],
    model_types: list[str],
    seeds: list[int],
    graph_type: str,
    methods: list[str],
    compare_metrics: list[str],
    compare_modes: list[str] | None = None,
    n_examples: int = 1000,
) -> None:
    """Run compare step: IOU/recall and/or cross_task based on compare_metrics. compare_modes must be a non-empty list (validated in main)."""
    all_circuits = _enumerate_circuits_for_compare(
        results_root, datasets_root, repeat_types, model_types, seeds, methods, graph_type
    )
    if not all_circuits:
        print("No circuits found for compare. Run discover first.", flush=True)
        return

    iou_recall_metrics = [m for m in compare_metrics if m in ("iou", "recall")]
    for metric in iou_recall_metrics:
        print(f"  compare metric: {metric}", flush=True)
        run_compare_iou_recall(
            datasets_root=datasets_root,
            results_root=results_root,
            repeat_types=repeat_types,
            model_types=model_types,
            seeds=seeds,
            graph_type=graph_type,
            methods=methods,
            metric=metric,
            compare_modes=compare_modes,
            all_circuits=all_circuits,
        )

    if "cross_task" in compare_metrics:
        print("  compare metric: cross_task", flush=True)
        run_compare_cross_task(
            datasets_root=datasets_root,
            results_root=results_root,
            repeat_types=repeat_types,
            model_types=model_types,
            seeds=seeds,
            graph_type=graph_type,
            methods=methods,
            n_examples=n_examples,
            compare_modes=compare_modes,
            all_circuits=all_circuits,
        )


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
        choices=["discover", "compare"],
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
    parser.add_argument(
        "--compare_modes",
        nargs="+",
        default=["across_counterfactual", "across_repeats"],
        choices=["across_counterfactual", "across_repeats"],
        help="Modes for compare steps (default: across_counterfactual across_repeats)",
    )
    parser.add_argument(
        "--compare_metrics",
        nargs="+",
        default=["iou", "cross_task"],
        choices=["iou", "recall", "cross_task"],
        help="Compare metrics to run: iou, recall, cross_task (default: iou cross_task)",
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

    if "compare" in args.steps:
        if args.graph_type not in ("edges", "nodes"):
            raise ValueError(
                "compare requires graph_type edges or nodes; neurons graph type is not supported"
            )
        compare_modes = args.compare_modes or ["across_counterfactual", "across_repeats"]
        print("\n=== Step: compare ===", flush=True)
        run_compare(
            datasets_root=datasets_root,
            results_root=results_root,
            repeat_types=args.repeat_types,
            model_types=args.model_types,
            seeds=args.seeds,
            graph_type=args.graph_type,
            methods=args.methods,
            compare_metrics=args.compare_metrics,
            compare_modes=compare_modes,
            n_examples=args.n_examples,
        )
        print("=== Step: compare done ===\n", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    sys.setrecursionlimit(3000)
    main()
