# Evaluation Experiment

This experiment evaluates language models (ESM3, ESM-C) on masked-token prediction over repeat and baseline datasets. The pipeline produces model accuracy on repeat prediction tasks (Table 1 in the paper) and filtered datasets for the counterfactuals methods experiment.

**3 phases (use run.py):**

1) **evaluate.py** — This script gets an input dataset (proteins with repeats or without repeats) and evaluates if the model correctly predicts (top-1) the masked token. It operates in two modes: repeat and baseline. For repeat mode it tests the model on all repeat positions masked per protein. For baseline mode it randomly chooses one position per protein and tests the model on that. The script produces a file called `predictions.csv` (or `baseline_predictions.csv` for baseline) where it saves per-position true label, predicted label, probabilities, and is_correct. For the purpose of our experiments we run the script per model on all 4 datasets (synthetic, identical, approximate, baseline).

2) **analyze.py** — This script takes all four prediction files from evaluate (synthetic, identical, approximate, baseline). It computes per-protein accuracy for each task. For approximate repeats it splits into two tasks: Sub-Adjacent (positions near substitutions only) and Indel-Adjacent (positions near indels only). It aggregates into a summary table with one row per task: Rand. Identical (synthetic), Nat. Identical (identical), Sub-Adjacent, Indel-Adjacent, and Baseline. Each row has Task name, N (number of proteins), Accuracy, and Accuracy_std. The script writes this to `tasks_performance.csv`, which is Table 1 in the paper.

3) **filter.py** — This script creates filtered datasets based on successful instances per model that are used in counterfactuals—meaning we take only proteins where each model had high accuracy on the repeat prediction task (e.g. accuracy ≥ 1.0 for synthetic/identical, or accuracy_identical ≥ 0.8 for approximate). For approximate repeats we also save which positions are identical in both repeat instances, predicted correctly by the model, and near a substitution, so we can later choose one position at random. The output goes to `datasets/{repeat_type}/{model}/counterfactuals/`.

4) **run.py** — The main entry point. Use this script; do not call evaluate.py, filter.py, or analyze.py directly.

---

## run.py

**What it does:** Runs one or more of evaluate, filter, and analyze. Evaluate must run first; filter and analyze both depend on its outputs and can run in any order.

**Expected input structure:** Input datasets must live under `datasets_root` in this layout:

```
{datasets_root}/
  synthetic/evaluation/synthetic_repeats_eval.csv
  identical/evaluation/identical_repeats_eval.csv
  approximate/evaluation/approximate_repeats_eval.csv
  baseline/evaluation/baseline.csv
```

**Output locations (per step):**

| Step | Location |
|------|----------|
| evaluate | `{results_root}/evaluation/{model_type}/synthetic/predictions.csv`, `identical/predictions.csv`, `approximate/predictions.csv`, `baseline/baseline_predictions.csv` |
| filter | `{datasets_root}/{repeat_type}/{model_type}/counterfactuals/` (e.g. `synthetic_eval_filtered.csv`, `approximate_eval_filtered_near_sub.csv`) |
| analyze | `{results_root}/evaluation/{model_type}/analysis/tasks_performance.csv` |

**Parameters:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_type` | Model to evaluate: `esm3` or `esm-c` | `esm3` |
| `--steps` | Which phases to run: `evaluate`, `filter`, `analyze` (can list multiple) | all three |
| `--datasets` | Which datasets to process: `synthetic`, `identical`, `approximate`, `baseline` | all four |
| `--datasets_root` | Root folder for input datasets | `{repo}/datasets` |
| `--results_root` | Root folder for predictions and analysis | `{repo}/results` |
| `--batch_size` | Batch size for evaluate step | `16` |

**Note:** If you include `analyze` in `--steps`, you must include all four datasets in `--datasets` (analyze needs all of them).

**Examples:**

```bash
# Full pipeline (evaluate → filter → analyze) for ESM3
python scripts/evaluation_experiment/run.py --model_type esm3

# Full pipeline for ESM-C
python scripts/evaluation_experiment/run.py --model_type esm-c

# Evaluate only (no filter, no analyze)
python scripts/evaluation_experiment/run.py --model_type esm3 --steps evaluate

# Evaluate and filter, skip analyze
python scripts/evaluation_experiment/run.py --model_type esm3 --steps evaluate filter

# Evaluate only synthetic and identical
python scripts/evaluation_experiment/run.py --model_type esm3 --steps evaluate --datasets synthetic identical

# Custom paths
python scripts/evaluation_experiment/run.py --model_type esm3 --datasets_root /path/to/datasets --results_root /path/to/results
```