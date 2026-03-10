"""Main entry for attention heads clustering. Orchestrates collect_features and analyze."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add script dir for local imports
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plms_repeats_circuits.utils.counterfactuals_config import find_file_for_method

from run_collect_features import collect_features
from analyze import run_analyze


def _circuit_discovery_dataset_dir(
    datasets_root: Path, repeat_type: str, model_type: str
) -> Path:
    """Directory containing datasets for circuit discovery (input CSVs). Not results/circuit_discovery."""
    return datasets_root / repeat_type / model_type / "circuit_discovery"


def _results_dir(results_root: Path, model_type: str, counterfactual_type: str) -> Path:
    """Output directory for attention heads clustering."""
    return results_root / "attention_heads_clustering" / model_type / counterfactual_type


def run_collect_step(
    datasets_root: Path,
    results_root: Path,
    model_type: str,
    counterfactual_type: str,
    repeat_type: str,
    batch_size: int,
    component_recurrence_threshold: float,
) -> Path:
    """Run feature collection. Returns path to features.csv."""
    circuit_discovery_dataset_dir = _circuit_discovery_dataset_dir(
        datasets_root, repeat_type, model_type
    )
    dataset_path = find_file_for_method(
        counterfactual_type, circuit_discovery_dataset_dir, kind="main", ext="csv"
    )
    if dataset_path is None:
        raise FileNotFoundError(
            f"No circuit_discovery dataset CSV found for method '{counterfactual_type}' "
            f"in {circuit_discovery_dataset_dir}"
        )

    comp_recurrence_path = (
        results_root
        / "component_recurrence"
        / model_type
        / counterfactual_type
        / f"nodes_recurrence_{repeat_type}.csv"
    )
    if not comp_recurrence_path.exists():
        raise FileNotFoundError(f"Component recurrence not found: {comp_recurrence_path}")

    out_dir = _results_dir(results_root, model_type, counterfactual_type)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "collected_features.csv"

    collect_features(
        dataset_csv=str(dataset_path),
        out_csv_path=str(features_path),
        model_type=model_type,
        component_recurrence_csv=str(comp_recurrence_path),
        batch_size=batch_size,
        repeat_type=repeat_type,
        component_recurrence_threshold=component_recurrence_threshold,
    )
    return features_path


def run_analyze_step(
    results_root: Path,
    model_type: str,
    counterfactual_type: str,
    features_path: Path,
    dataset_path: Path,
    n_clusters: int,
) -> None:
    """Run clustering analysis."""
    out_dir = _results_dir(results_root, model_type, counterfactual_type)
    run_analyze(
        features_path=features_path,
        dataset_path=dataset_path,
        output_dir=out_dir,
        n_clusters=n_clusters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attention heads clustering: collect features and/or analyze."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["collect_features", "analyze"],
        default=["collect_features", "analyze"],
        help="Steps to run (default: both)",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        nargs="+",
        choices=["esm3", "esm-c"],
        default=["esm3", "esm-c"],
        help="Model types to run (default: esm3 esm-c)",
    )
    parser.add_argument(
        "--counterfactual_type",
        type=str,
        default="blosum",
        help="Counterfactual method (e.g. blosum, mask)",
    )
    parser.add_argument(
        "--repeat_type",
        type=str,
        default="approximate",
        choices=["identical", "approximate", "synthetic"],
        help="Repeat type",
    )
    parser.add_argument(
        "--datasets_root",
        type=Path,
        default=REPO_ROOT / "datasets",
        help="Root for datasets used in circuit discovery (datasets/<repeat_type>/<model>/circuit_discovery)",
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=REPO_ROOT / "results",
        help="Root for all results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for feature collection",
    )
    parser.add_argument(
        "--component_recurrence_threshold",
        type=float,
        default=0.8,
        help="Min ratio_in_graph for component inclusion",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=3,
        help="Number of KMeans clusters (default: 3)",
    )
    args = parser.parse_args()

    datasets_root = args.datasets_root.resolve()
    results_root = args.results_root.resolve()

    for model_type in args.model_types:
        features_path = None
        dataset_path = None

        if "collect_features" in args.steps:
            circuit_discovery_dataset_dir = _circuit_discovery_dataset_dir(
                datasets_root, args.repeat_type, model_type
            )
            dp = find_file_for_method(
                args.counterfactual_type, circuit_discovery_dataset_dir, kind="main", ext="csv"
            )
            if dp is None:
                raise FileNotFoundError(
                    f"No dataset for {args.counterfactual_type} in {circuit_discovery_dataset_dir}"
                )
            dataset_path = dp

            features_path = run_collect_step(
                datasets_root=datasets_root,
                results_root=results_root,
                model_type=model_type,
                counterfactual_type=args.counterfactual_type,
                repeat_type=args.repeat_type,
                batch_size=args.batch_size,
                component_recurrence_threshold=args.component_recurrence_threshold,
            )
            print(f"[{model_type}] Features saved to {features_path}")

        if "analyze" in args.steps:
            out_dir = _results_dir(results_root, model_type, args.counterfactual_type)
            if features_path is None:
                features_path = out_dir / "collected_features.csv"
            if not features_path.exists():
                raise FileNotFoundError(
                    f"Features not found: {features_path}. Run collect_features first."
                )
            if dataset_path is None:
                circuit_discovery_dataset_dir = _circuit_discovery_dataset_dir(
                    datasets_root, args.repeat_type, model_type
                )
                dp = find_file_for_method(
                    args.counterfactual_type, circuit_discovery_dataset_dir, kind="main", ext="csv"
                )
                if dp is None:
                    raise FileNotFoundError(
                        f"No dataset for {args.counterfactual_type} in {circuit_discovery_dataset_dir}"
                    )
                dataset_path = dp

            run_analyze_step(
                results_root=results_root,
                model_type=model_type,
                counterfactual_type=args.counterfactual_type,
                features_path=features_path,
                dataset_path=dataset_path,
                n_clusters=args.n_clusters,
            )
            print(f"[{model_type}] Analysis saved to {out_dir}")


if __name__ == "__main__":
    main()
