"""Main entry for logit lens experiment. Orchestrates run_logit_lens and analyze."""

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

from run_logit_lens import run_logit_lens
from analyze import run_analyze


def _circuit_discovery_dataset_dir(
    datasets_root: Path, repeat_type: str, model_type: str
) -> Path:
    return datasets_root / repeat_type / model_type / "circuit_discovery"


def _logit_lens_results_dir(
    results_root: Path, model_type: str, counterfactual_type: str
) -> Path:
    return results_root / "logit_lens" / model_type / counterfactual_type


def _run_logit_lens_step(
    datasets_root: Path,
    results_root: Path,
    model_type: str,
    repeat_type: str,
    counterfactual_type: str,
    n_samples: int,
    batch_size: int,
    random_state: int,
    ratio_threshold: float,
) -> Path:
    """Run logit lens for one (model_type, repeat_type). Returns path to output CSV."""
    circuit_dir = _circuit_discovery_dataset_dir(datasets_root, repeat_type, model_type)
    dataset_path = find_file_for_method(
        counterfactual_type, circuit_dir, kind="main", ext="csv"
    )
    if dataset_path is None:
        raise FileNotFoundError(
            f"No circuit_discovery dataset for '{counterfactual_type}' in {circuit_dir}"
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

    out_dir = _logit_lens_results_dir(results_root, model_type, counterfactual_type)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"logit_lens_{repeat_type}_ratio{ratio_threshold}.csv"

    run_logit_lens(
        dataset_csv=str(dataset_path),
        component_recurrence_csv=str(comp_recurrence_path),
        out_csv_path=str(out_csv),
        model_type=model_type,
        repeat_type=repeat_type,
        n_samples=n_samples,
        random_state=random_state,
        batch_size=batch_size,
        ratio_threshold=ratio_threshold,
    )
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logit lens experiment: run_logit_lens and/or analyze."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["run_logit_lens", "analyze"],
        default=["run_logit_lens", "analyze"],
        help="Steps to run (default: both)",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        nargs="+",
        choices=["esm3", "esm-c"],
        default=["esm3", "esm-c"],
        help="Model types (default: esm3 esm-c)",
    )
    parser.add_argument(
        "--repeat_type",
        type=str,
        required=True,
        choices=["identical", "approximate", "synthetic"],
        help="Repeat type (one per run)",
    )
    parser.add_argument(
        "--counterfactual_type",
        type=str,
        default="blosum",
        help="Counterfactual method (default: blosum)",
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
        default=5000,
        help="Number of samples for logit lens (default: 5000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for logit lens (default: 1)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--ratio_threshold",
        type=float,
        default=0.8,
        help="Component recurrence ratio threshold (default: 0.8)",
    )
    args = parser.parse_args()

    datasets_root = args.datasets_root.resolve()
    results_root = args.results_root.resolve()

    for model_type in args.model_types:
        if "run_logit_lens" in args.steps:
            out_csv = _run_logit_lens_step(
                datasets_root=datasets_root,
                results_root=results_root,
                model_type=model_type,
                repeat_type=args.repeat_type,
                counterfactual_type=args.counterfactual_type,
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                random_state=args.random_state,
                ratio_threshold=args.ratio_threshold,
            )
            print(f"[{model_type}] Logit lens saved to {out_csv}")

        if "analyze" in args.steps:
            out_dir = _logit_lens_results_dir(
                results_root, model_type, args.counterfactual_type
            )
            run_analyze(
                results_root=results_root,
                model_type=model_type,
                counterfactual_type=args.counterfactual_type,
                repeat_type=args.repeat_type,
                output_dir=out_dir,
                ratio_threshold=args.ratio_threshold,
            )
            print(f"[{model_type}] Analysis saved to {out_dir}")


if __name__ == "__main__":
    main()
