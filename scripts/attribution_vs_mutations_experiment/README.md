# Attribution vs Mutations Experiment

Measures how **node-level** EAP-IG attribution scores change as additional BLOSUM62 substitutions are introduced in the non-masked repeat (while the circuit-discovery counterfactual corrupts the other repeat). Mutations use the **highest-scoring BLOSUM62** substitution at each chosen position . **Nodes only** — no neurons, mean ablations, or separate mask-position ablations.

**Analysis output:** `plots/induction_heads_stress_test.png` — mean attribution of Induction Heads (positive `approximate_mean_score` only) vs. substitution rate (`n_mutations / repeat_length`).

**Defaults:** circuit-discovery **dataset** from **identical** repeats; **components** (recurrence filter + sign split) from **approximate** circuits.

**Pipeline steps (use `run.py`):**

1. **run_attribution_vs_mutations.py** — For each repeat length, run attribution patching at `n_mutations = 0 … repeat_length - 1` on a sampled subset of the circuit-discovery dataset. Writes `{model_type}_repeat_len_{L}_nodes.csv`.
2. **analyze.py** — Produces `induction_heads_stress_test.png`.
3. **run.py** — Orchestrates both steps. Do not call the step scripts directly unless debugging.

---

## Prerequisites

- **Circuit discovery dataset** — `{datasets_root}/{dataset_repeat_type}/{model_type}/circuit_discovery/` (default: `identical`)
- **Component recurrence (nodes)** — `{results_root}/component_recurrence/{model_type}/{counterfactual_type}/nodes_recurrence_{component_recurrence_repeat_type}.csv` (default: `approximate`)
- **Attention heads clustering** — `{results_root}/attention_heads_clustering/{model_type}/{counterfactual_type}/clustering_results.csv` (for analyze only)

---

## run.py

### Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--steps` | `run_attribution_vs_mutations`, `analyze` | both |
| `--model_types` | `esm3`, `esm-c` | `esm-c` |
| `--dataset_repeat_type` | Circuit-discovery CSV repeat type | `identical` |
| `--component_recurrence_repeat_type` | `nodes_recurrence_*` and sign-filter columns | `approximate` |
| `--counterfactual_type` | e.g. `blosum` | `blosum` |
| `--repeat_lengths` | Repeat lengths to run | `10 20 30` |
| `--n_samples` | Sequences per repeat length | `40` |
| `--component_recurrence_threshold` | Min `{component_recurrence_repeat_type}_ratio_in_graph` | `0.8` |
| `--eap_ig_steps` | EAP-IG steps | `5` |
| `--batch_size` | Batch size | `1` |
| `--aggregation` | `sum` or `pos_mean` | `sum` |
| `--metric` | `log_prob` or `logit_diff` | `log_prob` |
| `--random_state` | Random seed | `42` |
| `--datasets_root` | Dataset root | `{repo}/datasets` |
| `--results_root` | Results root | `{repo}/results` |

### Outputs

| Step | Location |
|------|----------|
| run_attribution_vs_mutations | `{results_root}/attribution_vs_mutations/{model_type}/{counterfactual_type}/{model_type}_repeat_len_{L}_nodes.csv` |
| analyze | `{results_root}/attribution_vs_mutations/{model_type}/{counterfactual_type}/plots/induction_heads_stress_test.{png,pdf}` |

### Examples

```bash
# Full pipeline (default: identical dataset, approximate components)
python scripts/attribution_vs_mutations_experiment/run.py \
  --model_types esm-c --counterfactual_type blosum

# Compute only (long-running; needs GPU)
python scripts/attribution_vs_mutations_experiment/run.py \
  --steps run_attribution_vs_mutations

# Plot only (after CSVs exist)
python scripts/attribution_vs_mutations_experiment/run.py \
  --steps analyze

# Single repeat length
python scripts/attribution_vs_mutations_experiment/run.py \
  --repeat_lengths 10
```
