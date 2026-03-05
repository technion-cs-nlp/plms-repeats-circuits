# Attribution Patching Experiment

This experiment runs circuit discovery via attribution patching (EAP-IG) on ESM3 and ESM-C. It discovers minimal circuits at the level of **edges**, **nodes**, or **neurons** for repeat-prediction tasks. The pipeline uses circuit_discovery datasets produced by the [counterfactual experiment](../counterfactual_experiment/README.md) (evaluate + filter).

**1 phase (use run.py):**

**discover** — Runs attribution patching for edges, nodes, or neurons. For each combination of repeat type, model, counterfactual method, and seed, it calls `run_attribution_patching_nodes_edges.py` (edges/nodes) or `run_attribution_patching_neurons.py` (neurons) with fixed hyperparameters. Neurons discovery requires nodes output to exist first.

**run.py** — The main entry point. Use this script; do not call the attribution scripts directly unless you need custom experiments.

For direct use of the attribution scripts (e.g. different metrics, custom circuit sizes, or other ideas), see [README_ATTRIBUTION_PATCHING.md](README_ATTRIBUTION_PATCHING.md).

---

## run.py

**What it does:** Runs the discover step: attribution patching for edges, nodes, or neurons. For each combination of repeat type, model, counterfactual method, and seed, it runs the appropriate attribution script with fixed hyperparameters (EAP-IG, log_prob, min circuit search, etc.).

**Expected input structure:** Circuit_discovery CSVs under `datasets_root`:

```
{datasets_root}/
  {identical|approximate|synthetic}/{esm3|esm-c}/circuit_discovery/{repeat_type}_counterfactual_{method}.csv
```

These are produced by the counterfactual experiment (filter step). Valid methods are those in `plms_repeats_circuits.utils.counterfactuals_config.COUNTERFACTUAL_METHODS`; only methods with existing CSVs are allowed.

**Output locations:**

| Step | Location |
|------|----------|
| discover | `{results_root}/circuit_discovery/{repeat}/{model_type}/{graph_type}/seed_{seed}/` |

**Parameters:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--graph_type` | Circuit granularity: `edges`, `nodes`, or `neurons` | required |
| `--methods` | Counterfactual methods (validated against datasets) | required |
| `--repeat_types` | Repeat types: `identical`, `approximate`, `synthetic` | all three |
| `--model_types` | Models: `esm3`, `esm-c` | `esm3` |
| `--seeds` | Random seeds for train/test split | `[42]` |
| `--n_examples` | Number of samples per run | `1000` |
| `--datasets_root` | Root for circuit_discovery CSVs | `{repo}/datasets` |
| `--results_root` | Root for outputs | `{repo}/results` |

**Examples (note: for the paper we ran phases 1–6):**
```bash
# 1. ESM3 nodes: all 4 methods, all repeat types, all seeds
python scripts/attribution_patching_experiment/run.py \
  --graph_type nodes --methods blosum mask blosum-opposite50 permutation \
  --repeat_types identical approximate synthetic --model_types esm3 --seeds 42 43 44 45 46

# 2. ESM3 neurons: approximate blosum, all seeds (requires nodes output first)
python scripts/attribution_patching_experiment/run.py \
  --graph_type neurons --methods blosum \
  --repeat_types approximate --model_types esm3 --seeds 42 43 44 45 46

# 3. ESM3 edges: approximate blosum, all seeds
python scripts/attribution_patching_experiment/run.py \
  --graph_type edges --methods blosum \
  --repeat_types approximate --model_types esm3 --seeds 42 43 44 45 46

# 4. ESM-C nodes: blosum, all repeat types, all seeds
python scripts/attribution_patching_experiment/run.py \
  --graph_type nodes --methods blosum \
  --repeat_types identical approximate synthetic --model_types esm-c --seeds 42 43 44 45 46

# 5. ESM-C neurons: approximate blosum, all seeds (requires nodes output first)
python scripts/attribution_patching_experiment/run.py \
  --graph_type neurons --methods blosum \
  --repeat_types approximate --model_types esm-c --seeds 42 43 44 45 46

# 6. ESM-C edges: approximate blosum, all seeds
python scripts/attribution_patching_experiment/run.py \
  --graph_type edges --methods blosum \
  --repeat_types approximate --model_types esm-c --seeds 42 43 44 45 46

# Custom paths
python scripts/attribution_patching_experiment/run.py \
  --graph_type nodes --methods blosum \
  --results_root /path/to/results --datasets_root /path/to/datasets
```
