import argparse
import shutil
import sys
from pathlib import Path

import evaluate as evaluate_module
import filter as filter_module
import analyze as analyze_module

from plms_repeats_circuits.utils.counterfactuals_config import COUNTERFACTUAL_METHODS

REPO_ROOT = Path(__file__).resolve().parents[2]

# Path structure: {datasets_root}/{folder}/{model_type}/counterfactuals/<input>.csv
REPEAT_TYPE_PATHS = {
    "identical": "identical",
    "approximate": "approximate",
    "similar": "approximate",
    "synthetic": "synthetic",
}


def _input_dir(datasets_root: Path, folder: str, model_type: str) -> Path:
    return datasets_root / folder / model_type / "counterfactuals"


def _results_dir(results_root: Path, folder: str, model_type: str) -> Path:
    return results_root / "counterfactual" / folder / model_type


def verify_input_structure(datasets_root: Path, repeat_types: list[str], model_type: str) -> None:
    """Verify input counterfactual CSVs exist. Exit if any missing."""
    missing = []
    for rt in repeat_types:
        folder = REPEAT_TYPE_PATHS.get(rt, rt)
        d = _input_dir(datasets_root, folder, model_type)
        if not d.exists() or not list(d.glob("*.csv")):
            missing.append(str(d) + "/*.csv")
    if missing:
        print("Missing input counterfactual files (run evaluation filter first):", flush=True)
        for p in missing:
            print(f"  - {p}", flush=True)
        sys.exit(1)


def verify_evaluate_outputs(
    results_root: Path,
    repeat_types: list[str],
    model_type: str,
    filter_methods: list[str],
) -> None:
    """Verify evaluate output CSVs exist for filter_methods. Exit if any missing."""
    from plms_repeats_circuits.utils.counterfactuals_config import METHOD_PATTERNS
    missing = []
    for rt in repeat_types:
        folder = REPEAT_TYPE_PATHS.get(rt, rt)
        out_dir = _results_dir(results_root, folder, model_type)
        for method in filter_methods:
            pattern = METHOD_PATTERNS.get(method)
            if pattern is None:
                continue
            matches = list(out_dir.rglob(f"*{pattern}*.csv")) if out_dir.exists() else []
            if not matches:
                missing.append(f"{out_dir} (method={method})")
    if missing:
        print("Missing evaluate outputs for filter step (run evaluate first):", flush=True)
        for p in missing:
            print(f"  - {p}", flush=True)
        sys.exit(1)


def verify_analyze_inputs(results_root: Path, repeat_types: list[str], model_type: str) -> None:
    """Verify evaluate output directories are non-empty for analyze. Exit if any missing."""
    missing = []
    for rt in repeat_types:
        folder = REPEAT_TYPE_PATHS.get(rt, rt)
        out_dir = _results_dir(results_root, folder, model_type)
        if not out_dir.exists() or not list(out_dir.glob("*.csv")):
            missing.append(str(out_dir) + "/*.csv")
    if missing:
        print("Missing evaluate outputs for analyze step (run evaluate first):", flush=True)
        for p in missing:
            print(f"  - {p}", flush=True)
        sys.exit(1)


def _find_input_file(input_dir: Path) -> Path:
    csvs = list(input_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV in {input_dir}")
    if len(csvs) > 1:
        raise ValueError(f"Multiple CSVs in {input_dir}, specify explicitly")
    return csvs[0]


def run_evaluate(
    datasets_root: Path,
    results_root: Path,
    model_type: str,
    repeat_types: list[str],
    main_only: bool = False,
    random_seed: int = 42,
    dtype: str = "float32",
    eval_methods: list[str] = [],
):
    """Run evaluate for each (repeat_type, model) and each method."""
    verify_input_structure(datasets_root, repeat_types, model_type)
    for rt in repeat_types:
        folder = REPEAT_TYPE_PATHS.get(rt, rt)
        input_dir = _input_dir(datasets_root, folder, model_type)
        input_file = _find_input_file(input_dir)
        output_dir = _results_dir(results_root, folder, model_type)
        output_dir.mkdir(parents=True, exist_ok=True)

        experiments = []
        for m in COUNTERFACTUAL_METHODS:
            if m["name"] not in eval_methods:
                continue
            if main_only:
                if m["main_eval"]:
                    experiments.append(("main", m["main_eval"], m["name"]))
            else:
                if m["main_eval"]:
                    experiments.append(("main", m["main_eval"], m["name"]))
                if m.get("baseline_eval"):
                    experiments.append(("baseline", m["baseline_eval"], f"{m['name']}_baseline"))

        for i, (kind, spec, name) in enumerate(experiments):
            mode = "random_only_identical" if (rt in ("approximate", "similar")) else "random"
            if kind == "baseline":
                mode = "baseline_experiment_similiar" if rt in ("approximate", "similar") else "baseline_experiment_identical"
            base_name = m["name"]
            output_filename = f"{folder}_counterfactual_{base_name}" if kind == "main" else f"{folder}_counterfactual_{base_name}_baseline"
            args = [
                "--input_file", str(input_file),
                "--output_dir", str(output_dir),
                "--output_filename", output_filename,
                "--model_type", model_type,
                "--dtype", dtype,
                "--mode", mode,
                "--method", spec["method"],
                "--random_seed", str(random_seed),
            ]
            t1 = spec.get("type1")
            t2 = spec.get("type2")
            if t1 is not None:
                args.extend(["--corrupted_amino_acid_type1", str(t1)])
            if t2 is not None:
                args.extend(["--corrupted_amino_acid_type2", str(t2)])
            if spec.get("pct") is not None:
                args.extend(["--pct_corruption", str(spec["pct"])])
            print(f"[evaluate] {rt} {name}")
            evaluate_module.main(args)


def run_filter_step(
    datasets_root: Path,
    results_root: Path,
    model_type: str,
    repeat_types: list[str],
    methods: list[str],
):
    """Run filter: merge evaluate outputs with source, output to circuit_discovery."""
    verify_evaluate_outputs(results_root, repeat_types, model_type, methods)
    for rt in repeat_types:
        folder = REPEAT_TYPE_PATHS.get(rt, rt)
        eval_dir = _results_dir(results_root, folder, model_type)
        source_dir = _input_dir(datasets_root, folder, model_type)
        source_file = _find_input_file(source_dir)
        output_dir = datasets_root / folder / model_type / "circuit_discovery"
        output_dir.mkdir(parents=True, exist_ok=True)
        args = [
            "--source_dataset", str(source_file),
            "--input_dir", str(eval_dir),
            "--methods", *methods,
            "--output_dir", str(output_dir),
        ]
        print(f"[filter] {rt} -> {output_dir}")
        filter_module.main(args)


def run_analyze_step(
    results_root: Path,
    model_type: str,
    repeat_types: list[str],
):
    """Run analyze on evaluate output dirs."""
    verify_analyze_inputs(results_root, repeat_types, model_type)
    for rt in repeat_types:
        folder = REPEAT_TYPE_PATHS.get(rt, rt)
        input_dir = _results_dir(results_root, folder, model_type)
        output_dir = input_dir / "analysis"
        print(f"[analyze] {rt} -> {output_dir}")
        analyze_module.run_analysis(input_dir=input_dir, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description="Run counterfactual experiment pipeline")
    parser.add_argument("--datasets_root", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--results_root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--model_type", choices=["esm3", "esm-c"], default="esm3")
    parser.add_argument("--steps", nargs="+", default=["evaluate", "filter", "analyze"],
                        choices=["evaluate", "filter", "analyze"])
    parser.add_argument("--repeat_types", nargs="+", default=["identical", "approximate", "synthetic"],
                        help="Repeat types to process")
    parser.add_argument("--eval_methods", nargs="+", required=False, default=["all"],
                        help="Methods to run in evaluate step. Use 'all' for all methods, or list specific names e.g. mask blosum50")
    parser.add_argument("--filter_methods", nargs="+", default=["mask", "blosum100", "blosum-opposite50", "permutation"],
                        help="Methods for filter intersection")
    parser.add_argument("--main_only", action="store_true", help="Run only main experiments (no baseline)")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32", help="Model dtype for evaluate step")
    args = parser.parse_args()

    datasets_root = args.datasets_root.resolve()
    results_root = args.results_root.resolve()

    all_method_names = [m["name"] for m in COUNTERFACTUAL_METHODS]
    if "evaluate" in args.steps:
        if not args.eval_methods:
            print("Error: --eval_methods must be 'all' or a non-empty list of method names.", flush=True)
            sys.exit(1)
        eval_methods = all_method_names if "all" in args.eval_methods else args.eval_methods
        unknown = [m for m in eval_methods if m not in all_method_names]
        if unknown:
            print(f"Error: Unknown eval_methods: {unknown}", flush=True)
            print(f"  Available: {all_method_names}", flush=True)
            sys.exit(1)

    if "evaluate" in args.steps:
        print("=== Step: evaluate ===", flush=True)
        run_evaluate(
            datasets_root, results_root, args.model_type, args.repeat_types,
            main_only=args.main_only, random_seed=args.random_seed, dtype=args.dtype,
            eval_methods=eval_methods,
        )
    if "filter" in args.steps:
        print("=== Step: filter ===", flush=True)
        run_filter_step(datasets_root, results_root, args.model_type, args.repeat_types, args.filter_methods)
    if "analyze" in args.steps:
        print("=== Step: analyze ===", flush=True)
        run_analyze_step(results_root, args.model_type, args.repeat_types)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
