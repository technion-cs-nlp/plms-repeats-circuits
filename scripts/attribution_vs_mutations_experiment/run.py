"""Main entry for attribution vs mutations experiment. Orchestrates compute and analyze."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plms_repeats_circuits.utils.counterfactuals_config import find_file_for_method

from analyze import run_analyze
from run_attribution_vs_mutations import run_attribution_vs_mutations
from utils import nodes_csv_path


def _circuit_discovery_dataset_dir(
    datasets_root: Path, repeat_type: str, model_type: str
) -> Path:
    return datasets_root / repeat_type / model_type / "circuit_discovery"


def _results_dir(results_root: Path, model_type: str, counterfactual_type: str) -> Path:
    return results_root / "attribution_vs_mutations" / model_type / counterfactual_type


def run_compute_step(
    datasets_root: Path,
    results_root: Path,
    model_type: str,
    dataset_repeat_type: str,
    component_recurrence_repeat_type: str,
    counterfactual_type: str,
    repeat_lengths: list[int],
    n_samples: int,
    random_state: int,
    component_recurrence_threshold: float,
    eap_ig_steps: int,
    batch_size: int,
    aggregation: str,
    metric: str,
) -> list[Path]:
    """Run attribution vs mutations for each repeat length. Returns output CSV paths."""
    circuit_dir = _circuit_discovery_dataset_dir(
        datasets_root, dataset_repeat_type, model_type
    )
    dataset_path = find_file_for_method(
        counterfactual_type, circuit_dir, kind="main", ext="csv"
    )
    if dataset_path is None:
        raise FileNotFoundError(
            f"No circuit_discovery dataset CSV found for method '{counterfactual_type}' "
            f"in {circuit_dir}"
        )

    comp_recurrence_path = (
        results_root
        / "component_recurrence"
        / model_type
        / counterfactual_type
        / f"nodes_recurrence_{component_recurrence_repeat_type}.csv"
    )
    if not comp_recurrence_path.exists():
        raise FileNotFoundError(f"Component recurrence not found: {comp_recurrence_path}")

    out_dir = _results_dir(results_root, model_type, counterfactual_type)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for repeat_length in repeat_lengths:
        out_csv = nodes_csv_path(out_dir, model_type, repeat_length)
        run_attribution_vs_mutations(
            dataset_csv=str(dataset_path),
            component_recurrence_csv=str(comp_recurrence_path),
            out_csv_path=str(out_csv),
            model_type=model_type,
            component_recurrence_repeat_type=component_recurrence_repeat_type,
            repeat_length=repeat_length,
            n_samples=n_samples,
            random_state=random_state,
            component_recurrence_threshold=component_recurrence_threshold,
            eap_ig_steps=eap_ig_steps,
            batch_size=batch_size,
            aggregation=aggregation,
            metric=metric,
        )
        output_paths.append(out_csv)
    return output_paths


def run_analyze_step(
    results_root: Path,
    model_type: str,
    counterfactual_type: str,
    component_recurrence_repeat_type: str,
    repeat_lengths: list[int],
) -> None:
    """Plot induction heads vs substitution rate."""
    out_dir = _results_dir(results_root, model_type, counterfactual_type)
    run_analyze(
        results_root=results_root,
        model_type=model_type,
        counterfactual_type=counterfactual_type,
        component_recurrence_repeat_type=component_recurrence_repeat_type,
        output_dir=out_dir,
        repeat_lengths=repeat_lengths,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attribution vs mutations: run_attribution_vs_mutations and/or analyze."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["run_attribution_vs_mutations", "analyze"],
        default=["run_attribution_vs_mutations", "analyze"],
        help="Steps to run (default: both)",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        nargs="+",
        choices=["esm3", "esm-c"],
        default=["esm-c"],
        help="Model types (default: esm-c)",
    )
    parser.add_argument(
        "--dataset_repeat_type",
        type=str,
        default="identical",
        choices=["identical", "approximate", "synthetic"],
        help="Repeat type for circuit_discovery dataset (default: identical)",
    )
    parser.add_argument(
        "--component_recurrence_repeat_type",
        type=str,
        default="approximate",
        choices=["identical", "approximate", "synthetic"],
        help="Repeat type for nodes_recurrence CSV and sign filter (default: approximate)",
    )
    parser.add_argument(
        "--counterfactual_type",
        type=str,
        default="blosum",
        help="Counterfactual method (default: blosum)",
    )
    parser.add_argument(
        "--repeat_lengths",
        type=int,
        nargs="+",
        default=[10, 20, 30],
        help="Repeat lengths to process (default: 10 20 30)",
    )
    parser.add_argument(
        "--datasets_root",
        type=Path,
        default=REPO_ROOT / "datasets",
        help="Root for circuit discovery datasets",
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=REPO_ROOT / "results",
        help="Root for all results",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=40,
        help="Number of sequences per repeat length (default: 40)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--component_recurrence_threshold",
        type=float,
        default=0.8,
        help="Min {component_recurrence_repeat_type}_ratio_in_graph (default: 0.8)",
    )
    parser.add_argument(
        "--eap_ig_steps",
        type=int,
        default=5,
        help="EAP-IG steps (default: 5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="sum",
        choices=["sum", "pos_mean"],
        help="Attribution aggregation (default: sum)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="log_prob",
        choices=["logit_diff", "log_prob"],
        help="Patching metric (default: log_prob)",
    )
    args = parser.parse_args()

    datasets_root = args.datasets_root.resolve()
    results_root = args.results_root.resolve()

    for model_type in args.model_types:
        if "run_attribution_vs_mutations" in args.steps:
            paths = run_compute_step(
                datasets_root=datasets_root,
                results_root=results_root,
                model_type=model_type,
                dataset_repeat_type=args.dataset_repeat_type,
                component_recurrence_repeat_type=args.component_recurrence_repeat_type,
                counterfactual_type=args.counterfactual_type,
                repeat_lengths=args.repeat_lengths,
                n_samples=args.n_samples,
                random_state=args.random_state,
                component_recurrence_threshold=args.component_recurrence_threshold,
                eap_ig_steps=args.eap_ig_steps,
                batch_size=args.batch_size,
                aggregation=args.aggregation,
                metric=args.metric,
            )
            for p in paths:
                print(f"[{model_type}] Attribution vs mutations saved to {p}")

        if "analyze" in args.steps:
            run_analyze_step(
                results_root=results_root,
                model_type=model_type,
                counterfactual_type=args.counterfactual_type,
                component_recurrence_repeat_type=args.component_recurrence_repeat_type,
                repeat_lengths=args.repeat_lengths,
            )
            out_dir = _results_dir(results_root, model_type, args.counterfactual_type)
            plot_dir = out_dir / "plots"
            print(f"[{model_type}] Analysis saved to {plot_dir}")


if __name__ == "__main__":
    main()
