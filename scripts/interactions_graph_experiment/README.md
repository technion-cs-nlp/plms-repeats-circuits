# Interactions Graph

Aggregate circuit edges by attention-head clusters and produce interaction heatmaps (Figure 7+16 in the paper). Loads circuit discovery edge graphs across seeds, aggregates edges by cluster (cluster1 → cluster2), filters by in_graph consistency, and plots positive/negative interaction heatmaps (Blues/Reds). Produces the cluster-interaction visualizations for the paper.

**Files:**

1) **run.py** — Main entry point. Loads circuit edges from circuit discovery, aggregates by cluster, filters by in_graph, and produces cluster_edges_signed.csv plus heatmap PDFs/PNGs. Runs over one or more model types in a loop.

2) **attention_heads_clustering_esm3.ipynb**, **attention_heads_clustering_esmc.ipynb** — Original notebooks that the run.py pipeline was refactored from. Use for reference or interactive exploration.

---

## run.py

**What it does:** Aggregates circuit edges by cluster across seeds, filters to consistent edges, and produces cluster interaction heatmaps (positive and negative).

### Prerequisites

- **Circuit discovery** (edges) under `{results_root}/circuit_discovery/{repeat_type}/{model_type}/edges/seed_{seed}/` — Circuit JSONs and circuit_info_edges.csv (from [circuit_discovery_experiment](../circuit_discovery_experiment/README.md) with graph_type=edges)
- **Attention heads clustering** under `{results_root}/attention_heads_clustering/{model_type}/{counterfactual_type}/clustering_results.csv` (from [attention_heads_clustering_experiments](../attention_heads_clustering_experiments/README.md))
- **Component recurrence** under `{results_root}/component_recurrence/{model_type}/{counterfactual_type}/nodes_recurrence_{repeat_type}.csv` (from [component_recurrence_stats](../component_recurrence_stats/README.md)) — Used to augment clustering with MLP nodes

### Expected outputs

| Output | Location | Description |
|--------|----------|-------------|
| **cluster_edges_signed.csv** | `{results_root}/interactions_graph/{model_type}/{counterfactual_type}/` | Per (cluster1, cluster2, score_sign): n_edges, mean_raw. |
| **interactions_map.pdf**, **interactions_map.png** | Same directory | Positive-interaction heatmap (Blues). |
| **interactions_map_neg.pdf**, **interactions_map_neg.png** | Same directory | Negative-interaction heatmap (Reds, absolute values). |

### Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--repeat_type` | Repeat type: `identical`, `approximate`, `synthetic`. | (required) |
| `--model_types` | Model types to run: `esm3`, `esm-c`. | `esm3` `esm-c` |
| `--counterfactual_type` | Counterfactual method (e.g. `blosum`, `mask`). | (required) |
| `--results_root` | Root for all outputs. | `{repo}/results` |
| `--seeds` | Seeds to aggregate over. | `42 43 44 45 46` |
| `--in_graph_ratio` | Min ratio of seeds where edge must be in_graph. | `0.8` |

### Examples

```bash
# Run for both models (default)
python scripts/interactions_graph_experiment/run.py \
  --repeat_type approximate --counterfactual_type blosum

# Single model
python scripts/interactions_graph_experiment/run.py \
  --repeat_type approximate --model_types esm-c --counterfactual_type blosum

# Custom seeds and in_graph threshold
python scripts/interactions_graph_experiment/run.py \
  --repeat_type approximate --counterfactual_type blosum \
  --seeds 42 43 44 --in_graph_ratio 0.9

# Custom results root
python scripts/interactions_graph_experiment/run.py \
  --repeat_type approximate --model_types esm3 esm-c --counterfactual_type blosum \
  --results_root /path/to/results
```
