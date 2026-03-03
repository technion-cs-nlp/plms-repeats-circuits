# Counterfactual Experiment

This experiment tests whether corrupting protein repeat regions causes language models to change their predictions. It applies counterfactual methods (mask, BLOSUM, permutation, etc.) to repeat regions and measures how often the model's top-1 prediction changes and how much the probability of the true token drops. Input datasets come from the evaluation experiment filter step.

**3 phases (use run.py):**

1) **evaluate.py** — This script takes an input dataset (proteins with repeats, filtered from the evaluation experiment) and applies one counterfactual method to each protein. It corrupts positions in the other repeat instances and tests whether the model's prediction at a masked position changes before and after corruption. It outputs one CSV per method containing key columns (`rep_id`, `repeat_key`) plus result columns (`masked_position`, `corrupted_positions`, `replacements`, `corrupted_sequence`, `masked_repeat_example`, `corrupted_repeat_example`, `true_token_prob_before`, `true_token_prob_after`, `is_corrupt_changed_argmax`, etc.). Run once per method (all methods are run automatically via run.py).

2) **filter.py** — This script takes the evaluate outputs across a set of selected methods and finds the intersection of positions where all methods caused the argmax prediction to change. For each method it then produces a filtered output file containing the source dataset rows for those common positions, merged with that method's result columns. Output goes to `datasets/{repeat_type}/{model}/circuit_discovery/`.

3) **analyze.py** — This script discovers the evaluate output CSVs, separates them into main and baseline methods, computes two metrics per method (fraction of positions where argmax changed, and mean probability drop of the correct token), and creates a grouped bar chart comparing Real vs Baseline Counterfactual. Output is saved as `.png` and `.pdf`. Can be run via `run.py` or interactively via `analyze.ipynb`.

4) **analyze.ipynb** — A notebook version of the analyze step. Useful for interactively exploring results or regenerating plots for a specific repeat type / model without re-running the full pipeline. Open the notebook from the repo root, set `input_dir` to the evaluate output directory (e.g. `results/counterfactual/identical/esm3`) and `output_dir` for where to save the plot, then run all cells.

5) **run.py** — The main entry point. Use this script; do not call evaluate.py, filter.py, or analyze.py directly.

---

## run.py

**What it does:** Runs one or more of evaluate, filter, and analyze. Evaluate must run first; filter and analyze both depend on its outputs.

**Expected input structure:** Input datasets must live under `datasets_root` in this layout:

```
{datasets_root}/
  identical/{model_type}/counterfactuals/<input>.csv
  approximate/{model_type}/counterfactuals/<input>.csv
  synthetic/{model_type}/counterfactuals/<input>.csv
```

These CSVs are produced by the evaluation experiment filter step.

**Output locations (per step):**

| Step | Location |
|------|----------|
| evaluate | `{results_root}/counterfactual/{repeat_type}/{model_type}/` — one CSV per method, e.g. `identical_counterfactual_mask.csv` |
| filter | `{datasets_root}/{repeat_type}/{model_type}/circuit_discovery/` — one filtered CSV per method |
| analyze | `{results_root}/counterfactual/{repeat_type}/{model_type}/analysis/<model>_<repeat_type>_counterfactual_comparison.png/.pdf` |

**Parameters:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_type` | Model to use: `esm3` or `esm-c` | `esm3` |
| `--steps` | Which phases to run: `evaluate`, `filter`, `analyze` (can list multiple) | all three |
| `--repeat_types` | Repeat types to process: `identical`, `approximate`, `synthetic` | all three |
| `--eval_methods` | Methods to run in evaluate: `all` or a list of method names (e.g. `mask blosum50`) | `all` |
| `--filter_methods` | Methods whose intersection is used in the filter step | `mask blosum100 blosum-opposite50 permutation` |
| `--main_only` | Run only main experiments (skip baseline methods) | off |
| `--dtype` | Floating-point precision for model: `float32`, `bfloat16`, `float16` | `float32` |
| `--random_seed` | Random seed for corruption sampling | `42` |
| `--datasets_root` | Root folder for input datasets | `{repo}/datasets` |
| `--results_root` | Root folder for results | `{repo}/results` |

**Examples:**

```bash
# Full pipeline (evaluate → filter → analyze) for ESM3
python scripts/counterfactual_experiment/run.py --model_type esm3

# Evaluate only for ESM-C (only BLOSUM 100% method — counterfactual analysis was run on ESM3 only;
# ESM-C uses blosum solely to produce circuit discovery datasets)
python scripts/counterfactual_experiment/run.py --model_type esm-c --steps evaluate --eval_methods blosum

# Evaluate only
python scripts/counterfactual_experiment/run.py --model_type esm3 --steps evaluate

# Evaluate only specific methods
python scripts/counterfactual_experiment/run.py --model_type esm3 --steps evaluate --eval_methods mask blosum50

# Filter and analyze (after evaluate has already run)
python scripts/counterfactual_experiment/run.py --model_type esm3 --steps filter analyze

# Run only on identical and approximate repeat types
python scripts/counterfactual_experiment/run.py --model_type esm3 --repeat_types identical approximate

# Custom paths
python scripts/counterfactual_experiment/run.py --model_type esm3 --datasets_root /path/to/datasets --results_root /path/to/results
```
