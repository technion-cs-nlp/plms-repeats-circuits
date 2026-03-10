# Neurons Analysis Experiment

Analyzes which concepts (repeat tokens, biochemical properties, special tokens) best predict neuron activations via AUROC. Uses circuit discovery datasets, component recurrence for neuron selection, and optionally RADAR for additional suspected repeats. Generates Figures 6, 9, 15, and 18 from the paper.

**Key output:** `{model_type}_neurons_to_best_concepts.csv` — best concept per neuron (layer, neuron_idx, component_id, concept_name, concept_category, final_tag, final_auroc, is_positive_direction, other_concept_names).

**Files (use run.py to call the full pipeline):**

**run_concept_experiment.py** — Lower-level entry point that runs the concept-matching experiment. Requires explicit `--csv_path` and `--neurons_csv`. Called by run.py during the run step.

**analyze.py** — Produces Concepts Categories bar chart, AUROC scatter by layer bin, layer×cluster heatmap, and biochemical comparison between models. Called by run.py during the analyze and compare steps.

**neurons_analysis.ipynb** — Interactive notebook that mirrors the analyze pipeline step by step. Includes an additional neuron distribution visualization when activations are saved. Use it to inspect concept results and generate figures.

**run.py** — The main entry point. Orchestrates the run step (concept experiment) and analyze step. Resolves paths from standard layout; run and analyze can be run on the same command line.

---

## run.py

**What it does:** Orchestrates three steps: **run** — concept-matching experiment; **analyze** — concepts bar chart, AUROC scatter, layer×cluster heatmap; **compare** — ESM-C vs. ESM-3 biochemical comparison. Steps can be combined (e.g. `--steps run analyze`).

### Prerequisites

- **Circuit discovery datasets** under `{datasets_root}/{repeat_type}/{model_type}/circuit_discovery/{repeat_type}_counterfactual_{method}.csv` (from [counterfactual experiment](../counterfactual_experiment/README.md) filter step)
- **Component recurrence (neurons)** under `{results_root}/component_recurrence/{model_type}/{counterfactual_type}/neurons_recurrence_{repeat_type}.csv` (from [component_recurrence_stats](../component_recurrence_stats/README.md))
- **RADAR dataset** (optional) under `{datasets_root}/{repeat_type}/{model_type}/radar/` — if present, merges and adds `additional_suspected_repeats` (see [datasets/README.md](../../datasets/README.md#radar-results-datasets))

### Expected outputs

| Step | Output location | What you get |
|------|-----------------|--------------|
| **run** | `{results_root}/neurons_analysis/{model_type}/{counterfactual_type}/{repeat_type}/concept_exp_results/` | **concepts.csv** — Concept metadata (concept_name, concept_category). **{model_type}_neurons_to_best_concepts.csv** — Best concept per neuron (layer, neuron_idx, component_id, concept_name, final_auroc, etc.). With `--save_activations`: **activations_with_concepts.pt** — Activations and concept indices for notebook neuron distribution viz. With `--save_neuron_stats`: **neuron_stats_with_repeats.csv** — Per-neuron statistics of MLP activation values (mean, variance, pct positive/negative, top-k max/min activations with sequence and token). With `--save_all_results`: **neuron_analysis_results_all_concepts.csv** — Full AUROC for all concepts per neuron. |
| **run** (baseline) | `.../concept_exp_results_Baseline/` | **concepts.csv**, **{model_type}_neurons_to_best_concepts.csv** only; only when `--run_baseline`. Uses shuffled concept positions (no --save_* flags). Analyze uses it for AUROC scatter baseline points. |
| **analyze** | `concept_exp_results/` | **concepts_categories** — Bar chart of concept counts by tag (Amino Acid, Repeat, Blosum62, etc.). **auroc_scatter_by_bin** — Scatter of AUROC by layer bin, colored by concept group; baseline points shown if present. **heatmap_layer_cluster_mean_score_neurons** — Heatmap of mean attribution scores across layers by cluster (Amino Acid, Biochemical Sim., Repeat); requires component_recurrence. Each saved as .png and .pdf. |
| **compare** | `{results_root}/neurons_analysis/comparison/{counterfactual_type}/{repeat_type}/` | **biochemical_concepts_comparison_between_models** — Side-by-side bar chart of ESM-C vs. ESM-3 biochemical concept counts. Saved as .png and .pdf. |

### Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--steps` | Pipeline stages: `run`, `analyze`, `compare`. | `run` |
| `--datasets_root` | Root for input datasets (circuit_discovery, RADAR). | `{repo}/datasets` |
| `--results_root` | Root for all outputs. | `{repo}/results` |
| `--repeat_type` | Repeat type for dataset and neurons: `identical`, `approximate`, `synthetic`. | `approximate` |
| `--model_types` | Model types to run. | `esm3 esm-c` |
| `--counterfactual_type` | Counterfactual method (e.g. blosum, mask). | `blosum` |
| `--ratio_threshold` | Min `{repeat_type}_ratio_in_graph` for neuron inclusion. | `0.8` |
| `--n_samples` | Number of protein sequences to process. | `5000` |
| `--seed` | Random seed. | `42` |
| `--min_samples` | Min concept size for AUROC (smaller concepts skipped). | `3000` |
| `--run_baseline` | Run baseline (shuffled concept positions) after regular run. | off |
| `--save_activations` | Save activations to .pt file; needed for neuron distribution viz in notebook. | off |
| `--save_neuron_stats` | Save neuron statistics to CSV. | off |
| `--save_all_results` | Save full AUROC results (all concepts per neuron) to CSV. | off |

### Examples

```bash
# Full pipeline (run + analyze + compare) for both models
python scripts/neurons_analysis_experiment/run.py \
  --repeat_type approximate --counterfactual_type blosum \
  --steps run analyze compare

# Run + analyze only (no ESM-C vs ESM-3 comparison)
python scripts/neurons_analysis_experiment/run.py \
  --repeat_type approximate --counterfactual_type blosum \
  --steps run analyze

# Run with activations (for neuron distribution viz in notebook)
python scripts/neurons_analysis_experiment/run.py \
  --repeat_type approximate --counterfactual_type blosum \
  --steps run analyze --save_activations

# Analyze only (after run has completed)
python scripts/neurons_analysis_experiment/run.py \
  --steps analyze --repeat_type approximate --counterfactual_type blosum

# Compare only (requires both esm3 and esm-c results)
python scripts/neurons_analysis_experiment/run.py \
  --steps compare --repeat_type approximate --counterfactual_type blosum

# Single model, custom paths
python scripts/neurons_analysis_experiment/run.py \
  --repeat_type approximate --model_types esm3 --counterfactual_type blosum \
  --datasets_root /path/to/datasets --results_root /path/to/results
```
