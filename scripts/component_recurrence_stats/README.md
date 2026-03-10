# Component Recurrence Stats

Analyzes component recurrence across seeds for circuits (nodes or neurons). For a single repeat type, model, and counterfactual type, aggregates which components appear in minimal circuits across seeds.

**Purpose — component selection for downstream analysis:** These stats are used as inputs for the pipeline that interprets neurons and attention heads (Sections 4 and 5 in the paper). Each seed uses different dataset examples and discovers a different minimal circuit; components that recur across seeds are more likely to be important. The output CSVs  are used to **select which components to analyze** in the neuron- and attention-head interpretation experiments.

## Prerequisites

- Circuit discovery must have been run first (`scripts/circuit_discovery_experiment/run.py --steps discover`)
- Input layout: `{results_root}/circuit_discovery/{repeat_type}/{model_type}/{graph_type}/seed_{seed}/`

## Usage

### Commands we used

**ESM3**
```bash
# ESM3 nodes
python scripts/component_recurrence_stats/run_component_recurrence_stats.py \
  --graph_type nodes --repeat_type approximate --model_type esm3 --counterfactual_type blosum

# ESM3 neurons
python scripts/component_recurrence_stats/run_component_recurrence_stats.py \
  --graph_type neurons --repeat_type approximate --model_type esm3 --counterfactual_type blosum
```

**ESM-C**
```bash
# ESM-C nodes
python scripts/component_recurrence_stats/run_component_recurrence_stats.py \
  --graph_type nodes --repeat_type approximate --model_type esm-c --counterfactual_type blosum

# ESM-C neurons
python scripts/component_recurrence_stats/run_component_recurrence_stats.py \
  --graph_type neurons --repeat_type approximate --model_type esm-c --counterfactual_type blosum
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--graph_type` | yes | `nodes` or `neurons` |
| `--repeat_type` | yes | `identical`, `approximate`, or `synthetic` (one per run) |
| `--model_type` | yes | `esm3` or `esm-c` |
| `--counterfactual_type` | yes | Counterfactual method name (e.g. `blosum`, `mask`, `blosum-opposite50`, `permutation`) |
| `--seeds` | no | Space-separated seeds (default: 42 43 44 45 46) |
| `--results_root` | no | Root for circuit_discovery outputs (default: `{repo}/results`) |

## Output

- Location: `{results_root}/component_recurrence/{model_type}/{counterfactual_type}/{graph_type}_recurrence_{repeat_type}.csv`
- Example: `results/component_recurrence/esm-c/blosum/nodes_recurrence_approximate.csv`

## CSV Columns

- **Identity**: `component_id`, `component_type`, `layer`, `head` (attention only), `neuron_idx` (neurons only)
- **Per seed**: `{repeat}_{seed}_in_graph`, `{repeat}_{seed}_score`, `{repeat}_{seed}_score_sign`
- **Summary**: `{repeat}_total_circuits`, `{repeat}_in_graph_count`, `{repeat}_ratio_in_graph`, `{repeat}_mean_score`, `{repeat}_std_score`, `{repeat}_cv_score`

Scores are raw attribution scores (no normalization).
