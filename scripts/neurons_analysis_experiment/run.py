"""
Neurons analysis experiment entry point.

Resolves paths from datasets_root/results_root and invokes run_concept_experiment.
Supports --steps: 'run' (full experiment), 'analyze' (plots via analyze.py).
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plms_repeats_circuits.utils.counterfactuals_config import find_file_for_method

from run_concept_experiment import main as run_concept_experiment_main
from analyze import analyze_model, compare_models, plot_layer_cluster_heatmap_neurons


def _circuit_discovery_dataset_dir(datasets_root: Path, repeat_type: str, model_type: str) -> Path:
    """Directory containing circuit discovery datasets (input CSVs)."""
    return datasets_root / repeat_type / model_type / "circuit_discovery"


def _radar_dataset_dir(datasets_root: Path, repeat_type: str, model_type: str) -> Path:
    """Directory containing RADAR CSVs for suspected repeats."""
    return datasets_root / repeat_type / model_type / "radar"


def _output_parent_dir(
    results_root: Path, model_type: str, counterfactual_type: str, repeat_type: str
) -> Path:
    """Parent dir for neurons analysis outputs."""
    return results_root / "neurons_analysis" / model_type / counterfactual_type / repeat_type


def _regular_output_dir(parent: Path) -> Path:
    """Output directory for regular (non-baseline) experiment."""
    return parent / "concept_exp_results"


def _baseline_output_dir(parent: Path) -> Path:
    """Output directory for baseline (shuffled concept positions) experiment."""
    return parent / "concept_exp_results_Baseline"


def _invoke_run_concept_experiment(
    *,
    csv_path: Path | None,
    radar_csv_path: Path | None,
    neurons_path: Path,
    output_dir: Path,
    model_type: str,
    repeat_type: str,
    ratio_threshold: float,
    n_samples: int,
    seed: int,
    min_samples: int,
    save_activations: bool,
    save_neuron_stats: bool,
    save_all_results: bool,
    load_activations: str | None,
    baseline_mode: str,
) -> None:
    """Build sys.argv and invoke run_concept_experiment."""
    argv = [
        "run_concept_experiment",
        "--neurons_csv", str(neurons_path),
        "--output_dir", str(output_dir),
        "--model_type", model_type,
        "--repeat_type", repeat_type,
        "--ratio_threshold", str(ratio_threshold),
        "--n_samples", str(n_samples),
        "--seed", str(seed),
        "--min_samples", str(min_samples),
        "--baseline_mode", baseline_mode,
    ]
    if csv_path is not None:
        argv.extend(["--csv_path", str(csv_path)])
    if radar_csv_path is not None:
        argv.extend(["--radar_csv", str(radar_csv_path)])
    if load_activations is not None:
        argv.extend(["--load_activations", load_activations])
    if save_activations:
        argv.append("--save_activations")
    if save_neuron_stats:
        argv.append("--save_neuron_stats")
    if save_all_results:
        argv.append("--save_all_results")
    sys.argv = argv
    run_concept_experiment_main()


def run_experiment_step(
    datasets_root: Path,
    results_root: Path,
    model_type: str,
    counterfactual_type: str,
    repeat_type: str,
    n_samples: int,
    seed: int,
    ratio_threshold: float,
    min_samples: int,
    save_activations: bool,
    save_neuron_stats: bool,
    save_all_results: bool,
    run_baseline: bool,
) -> Path:
    """Run full experiment (regular, then optionally baseline)."""
    circuit_discovery_dir = _circuit_discovery_dataset_dir(
        datasets_root, repeat_type, model_type
    )
    circuit_discovery_csv = find_file_for_method(
        counterfactual_type, circuit_discovery_dir, kind="main", ext="csv"
    )
    if circuit_discovery_csv is None:
        raise FileNotFoundError(
            f"No circuit discovery dataset for '{counterfactual_type}' in {circuit_discovery_dir}"
        )

    radar_dir = _radar_dataset_dir(datasets_root, repeat_type, model_type)
    radar_csv_path = find_file_for_method(
        counterfactual_type, radar_dir, kind="main", ext="csv"
    ) if radar_dir.exists() else None

    neurons_path = (
        results_root
        / "component_recurrence"
        / model_type
        / counterfactual_type
        / f"neurons_recurrence_{repeat_type}.csv"
    )
    if not neurons_path.exists():
        raise FileNotFoundError(f"Neurons CSV not found: {neurons_path}")

    parent = _output_parent_dir(results_root, model_type, counterfactual_type, repeat_type)
    regular_dir = _regular_output_dir(parent)
    baseline_dir = _baseline_output_dir(parent)
    regular_dir.mkdir(parents=True, exist_ok=True)

    # 1. Regular mode
    _invoke_run_concept_experiment(
        csv_path=circuit_discovery_csv,
        radar_csv_path=radar_csv_path,
        neurons_path=neurons_path,
        output_dir=regular_dir,
        model_type=model_type,
        repeat_type=repeat_type,
        ratio_threshold=ratio_threshold,
        n_samples=n_samples,
        seed=seed,
        min_samples=min_samples,
        save_activations=save_activations,
        save_neuron_stats=save_neuron_stats,
        save_all_results=save_all_results,
        load_activations=None,
        baseline_mode="none",
    )

    # 2. Baseline mode (after regular)
    if run_baseline:
        baseline_dir.mkdir(parents=True, exist_ok=True)
        load_from = str(regular_dir / "activations_with_concepts.pt") if save_activations else None
        load_from_exists = load_from is not None and Path(load_from).exists()
        _invoke_run_concept_experiment(
            csv_path=None if load_from_exists else circuit_discovery_csv,
            radar_csv_path=radar_csv_path,
            neurons_path=neurons_path,
            output_dir=baseline_dir,
            model_type=model_type,
            repeat_type=repeat_type,
            ratio_threshold=ratio_threshold,
            n_samples=n_samples,
            seed=seed,
            min_samples=min_samples,
            save_activations=False,
            save_neuron_stats=False,
            save_all_results=False,
            load_activations=load_from if load_from_exists else None,
            baseline_mode="shuffled_concept_positions",
        )

    return regular_dir


def run_analyze_step(
    output_dir: Path,
    baseline_dir: Path,
    model_type: str,
    results_root: Path,
    counterfactual_type: str,
    repeat_type: str,
) -> Path:
    """Run analyze_model: concepts bar chart, AUROC scatter, and layer-cluster heatmap. Always saves figures."""
    if not output_dir.exists():
        raise FileNotFoundError(
            f"Output dir not found: {output_dir}. Run 'run' step first."
        )
    analyze_model(
        exp_results_dir=output_dir,
        model_type=model_type,
        exp_baseline_results_dir=baseline_dir if baseline_dir.exists() else None,
    )
    plot_layer_cluster_heatmap_neurons(
        exp_results_dir=output_dir,
        results_root=results_root,
        model_type=model_type,
        counterfactual_type=counterfactual_type,
        repeat_type=repeat_type,
    )
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Neurons analysis: run experiment and/or analyze from saved activations."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["run", "analyze", "compare"],
        default=["run"],
        help="Steps: 'run' (full neuron concept matching), 'analyze' (plots from results), 'compare' (ESM-C vs ESM-3 biochemical concepts). Default: run",
    )
    parser.add_argument(
        "--datasets_root",
        type=str,
        default=str(REPO_ROOT / "datasets"),
        help="Root for input datasets (circuit_discovery, RADAR).",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=str(REPO_ROOT / "results"),
        help="Root for all output results.",
    )
    parser.add_argument(
        "--repeat_type",
        type=str,
        default="approximate",
        choices=["identical", "approximate", "synthetic"],
        help="Repeat type for dataset and neurons selection. Default: approximate.",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        nargs="+",
        choices=["esm3", "esm-c"],
        default=["esm3", "esm-c"],
        help="Model types to run (default: esm3 esmc).",
    )
    parser.add_argument(
        "--counterfactual_type",
        type=str,
        default="blosum",
        help="Counterfactual method (e.g. blosum, mask, permutation).",
    )
    parser.add_argument(
        "--ratio_threshold",
        type=float,
        default=0.8,
        help="Min {repeat_type}_ratio_in_graph for neuron inclusion.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5000,
        help="Number of protein sequences to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=3000,
        help="Min concept size for AUROC (concepts with fewer samples are skipped).",
    )
    parser.add_argument(
        "--run_baseline",
        action="store_true",
        help="Run baseline (shuffled concept positions) after regular; loads activations if --save_activations, else computes.",
    )
    parser.add_argument(
        "--save_activations",
        action="store_true",
        help="Save activations to activations_with_concepts.pt.",
    )
    parser.add_argument(
        "--save_neuron_stats",
        action="store_true",
        help="Save neuron statistics to CSV.",
    )
    parser.add_argument(
        "--save_all_results",
        action="store_true",
        help="Save full neuron_analysis_results_all_concepts.csv.",
    )
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root)
    results_root = Path(args.results_root)

    for model_type in args.model_types:
        parent = _output_parent_dir(
            results_root, model_type, args.counterfactual_type, args.repeat_type
        )
        regular_dir = _regular_output_dir(parent)
        baseline_dir = _baseline_output_dir(parent)

        if "run" in args.steps:
            run_experiment_step(
                datasets_root=datasets_root,
                results_root=results_root,
                model_type=model_type,
                counterfactual_type=args.counterfactual_type,
                repeat_type=args.repeat_type,
                n_samples=args.n_samples,
                seed=args.seed,
                ratio_threshold=args.ratio_threshold,
                min_samples=args.min_samples,
                save_activations=args.save_activations,
                save_neuron_stats=args.save_neuron_stats,
                save_all_results=args.save_all_results,
                run_baseline=args.run_baseline,
            )
            print(f"[{model_type}] Results saved to {regular_dir}")

        if "analyze" in args.steps:
            run_analyze_step(
                output_dir=regular_dir,
                baseline_dir=baseline_dir,
                model_type=model_type,
                results_root=results_root,
                counterfactual_type=args.counterfactual_type,
                repeat_type=args.repeat_type,
            )
            print(f"[{model_type}] Plots saved to {regular_dir}")

    # ESM-C vs ESM-3 comparison (requires both model types)
    if "compare" in args.steps and set(args.model_types) >= {"esm3", "esm-c"}:
        comparison_dir = (
            results_root
            / "neurons_analysis"
            / "comparison"
            / args.counterfactual_type
            / args.repeat_type
        )
        esm3_parent = _output_parent_dir(
            results_root, "esm3", args.counterfactual_type, args.repeat_type
        )
        esmc_parent = _output_parent_dir(
            results_root, "esm-c", args.counterfactual_type, args.repeat_type
        )
        compare_models(
            esm3_results_dir=_regular_output_dir(esm3_parent),
            esmc_results_dir=_regular_output_dir(esmc_parent),
            comparison_output_dir=comparison_dir,
        )
        print(f"[compare] biochemical_concepts_comparison_between_models saved to {comparison_dir}")


if __name__ == "__main__":
    main()
