# Logit Lens Experiment

Compute and visualize logit lens for nodes (residual stream + attention/MLP clusters). Produces **logit lens across layers and cluster plot** (**paper figures 8 and 17**).

**Files:**

1) **run.py** — Main entry point. Orchestrates run_logit_lens and analyze. Use this script.

2) **run_logit_lens.py** — Computes logit lens metrics (top1, correct_prob) for nodes. Called by run.py.

3) **analyze.py** — Processes logit lens Results and produces logit lens across layers and cluster splot.

---

## run.py

**What it does:** Runs logit lens computation and/or analysis for the given repeat type and model(s).

### Prerequisites

- **Circuit discovery datasets** under `{datasets_root}/{repeat_type}/{model_type}/circuit_discovery/` (from [counterfactual experiment](../counterfactual_experiment/README.md) filter step)
- **Component recurrence** under `{results_root}/component_recurrence/{model_type}/{counterfactual_type}/nodes_recurrence_{repeat_type}.csv` (from [component_recurrence_stats](../component_recurrence_stats/README.md))
- **Attention heads clustering** under `{results_root}/attention_heads_clustering/{model_type}/{counterfactual_type}/clustering_results.csv` (from [attention_heads_clustering_experiments](../attention_heads_clustering_experiments/README.md)) — required for analyze step

### Expected outputs

| Output | Location | Description |
|--------|----------|-------------|
| **logit_lens_{repeat_type}_ratio{threshold}.csv** | `{results_root}/logit_lens/{model_type}/{counterfactual_type}/` | Per-component top1 and correct_prob. |
| **cluster_metrics_{repeat_type}.png/.pdf** | Same directory | Logit lens across layers and clusters plot (paper figures 8, 17). |

### Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--steps` | Steps: `run_logit_lens`, `analyze`. | both |
| `--model_types` | Model types: `esm3`, `esm-c`. | `esm3` `esm-c` |
| `--repeat_type` | Repeat type (one per run): `identical`, `approximate`, `synthetic`. | (required) |
| `--counterfactual_type` | Counterfactual method (e.g. `blosum`). | `blosum` |
| `--datasets_root` | Root for circuit discovery datasets. | `{repo}/datasets` |
| `--results_root` | Root for all results. | `{repo}/results` |
| `--n_samples` | Number of samples for logit lens. | `5000` |
| `--batch_size` | Batch size for logit lens. | `1` |
| `--random_state` | Random seed. | `42` |
| `--ratio_threshold` | Component recurrence ratio threshold. | `0.8` |

### Examples

```bash
# Full pipeline (run_logit_lens + analyze) for approximate, both models
python scripts/logit_lens_experiments/run.py --repeat_type approximate --counterfactual_type blosum

# Analyze only (after run_logit_lens has already run)
python scripts/logit_lens_experiments/run.py --steps analyze --repeat_type approximate --counterfactual_type blosum

# Single model, fewer samples (faster)
python scripts/logit_lens_experiments/run.py --repeat_type approximate --model_types esm-c --n_samples 100
```
