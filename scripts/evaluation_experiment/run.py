import argparse
import shutil
import sys
from pathlib import Path

import evaluate as evaluate_module
import filter as filter_module
import analyze as analyze_module

REPO_ROOT = Path(__file__).resolve().parents[2]

# Expected dataset files relative to datasets_root
DATASET_FILES = {
    "synthetic": "synthetic/evaluation/synthetic_repeats_eval.csv",
    "identical": "identical/evaluation/identical_repeats_eval.csv",
    "approximate": "approximate/evaluation/approximate_repeats_eval.csv",
    "baseline": "baseline/evaluation/baseline.csv",
}

REPEAT_DATASETS = ["synthetic", "identical", "approximate"]

FILTER_CONFIGS = {
    "synthetic": {"dataset_type": "synthetic", "accuracy_threshold": 1.0, "sample_size": 5000},
    "identical": {"dataset_type": "identical", "accuracy_threshold": 1.0, "sample_size": 5000},
    "approximate": {
        "dataset_type": "approximate",
        "accuracy_threshold": 0.8,
        "sample_size": 5000,
        "filter_types": ["near_sub"],
    },
}


def verify_datasets_structure(datasets_root, datasets):
    """Verify required dataset files exist. Exit if any missing."""
    missing = []
    for name in datasets:
        path = datasets_root / DATASET_FILES[name]
        if not path.exists():
            missing.append(str(path))
    if missing:
        print("Datasets root missing expected files:")
        for p in missing:
            print(f"  - {p}")
        sys.exit(1)


def verify_prediction_results(results_root, model_type, datasets):
    """Verify prediction files exist for filter step. Exit if any missing."""
    res = results_dir(results_root, model_type)
    missing = []
    for name in datasets:
        if name == "baseline":
            continue
        pred_path = res / name / "predictions.csv"
        if not pred_path.exists():
            missing.append(str(pred_path))
    if missing:
        print("Missing predictions (run evaluate first):")
        for p in missing:
            print(f"  - {p}")
        sys.exit(1)


def verify_analyze_inputs(results_root, model_type):
    """Verify all prediction files exist for analyze. Exit if any missing."""
    res = results_dir(results_root, model_type)
    required = [
        res / "approximate" / "predictions.csv",
        res / "identical" / "predictions.csv",
        res / "synthetic" / "predictions.csv",
        res / "baseline" / "baseline_predictions.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("Missing predictions for analyze (run evaluate first):")
        for p in missing:
            print(f"  - {p}")
        sys.exit(1)


def dataset_path(datasets_root, name):
    return datasets_root / DATASET_FILES[name]


def results_dir(results_root, model_type, name=None):
    """Path: <results_root>/evaluation/<model_type>/[name]. name=None returns base dir."""
    base = results_root / "evaluation" / model_type
    return base if name is None else base / name


def run_evaluate(datasets_root, results_root, model_type, datasets, batch_size):
    verify_datasets_structure(datasets_root, datasets)

    # Delete previous results for datasets we're recomputing
    for name in datasets:
        out = results_dir(results_root, model_type, name)
        if out.exists():
            shutil.rmtree(out)
            print(f"[evaluate] Removed previous results for {name}")

    for name in datasets:
        src = dataset_path(datasets_root, name)
        out = results_dir(results_root, model_type, name)
        out.mkdir(parents=True, exist_ok=True)
        mode = "baseline" if name == "baseline" else "repeat"
        print(f"[evaluate] {name} ({mode}) -> {out}")

        evaluate_module.main([
            "--input_file", str(src),
            "--output_dir", str(out),
            "--model_type", model_type,
            "--batch_size", str(batch_size),
            "--mode", mode,
        ])


def filtered_output_dir(datasets_root, repeat_name, model_type):
    """Path: datasets_root/repeat_name/model_type/counterfactuals/"""
    return datasets_root / repeat_name / model_type / "counterfactuals"


def run_filter(datasets_root, results_root, model_type, datasets, configs):
    filter_datasets = [n for n in datasets if n != "baseline"]
    verify_prediction_results(results_root, model_type, filter_datasets)

    for name in filter_datasets:
        res = results_dir(results_root, model_type, name)
        pred_path = res / "predictions.csv"
        src = dataset_path(datasets_root, name)
        cfg = configs.get(name, {})
        out_dir = filtered_output_dir(datasets_root, name, model_type)
        out_dir.mkdir(parents=True, exist_ok=True)

        filter_types = cfg.get("filter_types", None)
        if filter_types is not None:
            runs = [(f"{name}_eval_filtered_{ft}.csv", ft) for ft in filter_types]
        else:
            runs = [(f"{name}_eval_filtered.csv", cfg.get("filter_type", "none"))]

        # Delete previous filter results for this dataset
        for out_name, _ in runs:
            out_path = out_dir / out_name
            if out_path.exists():
                out_path.unlink()
                print(f"[filter] Removed previous {out_name} for {name}")

        for out_name, filter_type in runs:
            out_path = out_dir / out_name
            print(f"[filter] {name} filter_type={filter_type} -> {out_path}")

            filter_module.main([
                "--dataset_type", cfg.get("dataset_type", name),
                "--predictions_all", str(pred_path),
                "--source_dataset", str(src),
                "--output_file", str(out_path),
                "--sample_size", str(cfg.get("sample_size", 5000)),
                "--accuracy_threshold", str(cfg.get("accuracy_threshold", 1.0)),
                "--filter_type", filter_type,
            ])


def run_analyze(datasets_root, results_root, model_type):
    verify_analyze_inputs(results_root, model_type)

    res = results_dir(results_root, model_type)
    out = res / "analysis"

    # Delete previous analysis results
    if out.exists():
        shutil.rmtree(out)
        print("[analyze] Removed previous analysis results")

    out.mkdir(parents=True, exist_ok=True)

    required = {
        "approximate_all": res / "approximate" / "predictions.csv",
        "identical_all": res / "identical" / "predictions.csv",
        "synthetic_all": res / "synthetic" / "predictions.csv",
        "baseline": res / "baseline" / "baseline_predictions.csv",
    }

    print(f"[analyze] -> {out}")
    analyze_module.main([
        "--approximate_source", str(dataset_path(datasets_root, "approximate")),
        "--approximate_all", str(required["approximate_all"]),
        "--identical_all", str(required["identical_all"]),
        "--synthetic_all", str(required["synthetic_all"]),
        "--baseline", str(required["baseline"]),
        "--output_dir", str(out),
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Run the evaluation experiment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--datasets_root", type=Path, default=REPO_ROOT / "datasets",
                        help="Root folder for inputs: <root>/synthetic/evaluation/..., etc.")
    parser.add_argument("--results_root", type=Path, default=REPO_ROOT / "results",
                        help="Root folder for outputs: <root>/evaluation/<model_type>/<dataset>/..., etc.")
    parser.add_argument("--model_type", choices=["esm3", "esm-c"], default="esm3")
    parser.add_argument("--steps", nargs="+", default=["evaluate", "filter", "analyze"],
                        choices=["evaluate", "filter", "analyze"],
                        help="Which pipeline steps to run")
    parser.add_argument("--datasets", nargs="+",
                        default=["synthetic", "identical", "approximate", "baseline"],
                        choices=["synthetic", "identical", "approximate", "baseline"],
                        help="Which datasets to process")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    datasets_root = args.datasets_root.resolve()
    results_root = args.results_root.resolve()

    if "evaluate" in args.steps:
        run_evaluate(datasets_root, results_root, args.model_type, args.datasets, args.batch_size)

    if "filter" in args.steps:
        run_filter(datasets_root, results_root, args.model_type, args.datasets, FILTER_CONFIGS)

    if "analyze" in args.steps:
        required_for_analyze = {"synthetic", "identical", "approximate", "baseline"}
        missing = required_for_analyze - set(args.datasets)
        if missing:
            print("Analyze step requires all datasets. Missing:", sorted(missing))
            sys.exit(1)
        run_analyze(datasets_root, results_root, args.model_type)

    print("Done.")


if __name__ == "__main__":
    main()
