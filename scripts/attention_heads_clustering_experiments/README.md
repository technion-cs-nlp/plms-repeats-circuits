# Attention Heads Clustering

Cluster attention heads by their behavioral features (copy score, pattern matching, position, amino-acid bias, etc.). Uses circuit discovery datasets and component recurrence to select which heads to analyze. Produces the attention-heads analysis in Section 4 of the paper.

**Files (use run.py to call the full pipeline):**

1) **run_collect_features.py** — Extracts per-head features (copy score, pattern matching, attention entropy, repeat focus, contribution to residual stream, aa bias, etc.) from the model over proteins in the circuit discovery dataset. Called by run.py during the collect_features step.

2) **analyze.py** — Performs attention-heads clustering analysis. Called by run.py during the analyze step.

3) **attention_heads_analysis.ipynb** — Interactive notebook that mirrors the analyze pipeline step by step. Use it to inspect clustering results.
4) **run.py** — The main entry point. Orchestrates collect_features and analyze. Runs over one or more model types in a loop.

---

## run.py

**What it does:** Runs collect_features and/or analyze for each model type. Collect_features extracts attention-head features from the model; analyze clusters them and produces summaries, ANOVA, outliers, and visualizations. Collect must run first (or use migrated or precomputed collected_features.csv); analyze depends on it.

### Prerequisites

- Circuit discovery datasets under `{datasets_root}/{repeat_type}/{model_type}/circuit_discovery/` (from the [counterfactual experiment](../counterfactual_experiment/README.md) filter step)
- Component recurrence CSVs under `{results_root}/component_recurrence/{model_type}/{counterfactual_type}/nodes_recurrence_{repeat_type}.csv` (from [component_recurrence_stats](../component_recurrence_stats/README.md))

### Expected outputs

| Step | Output location | What you get |
|------|-----------------|--------------|
| **collect_features** | `{results_root}/attention_heads_clustering/{model_type}/{counterfactual_type}/` | **collected_features.csv** — Per-example features for each attention head. |
| **analyze** | Same directory | **aggregated_features.csv** — Per-head features (mean over proteins). **kmeans_metrics.csv** — Inertia and silhouette for k=2..10. **inertia_silhouette_curves.png/.pdf** — Elbow and silhouette plots (Appendix). **cluster_summary_normalized.csv**, **cluster_summary_original.csv** — Per-cluster means (Appendix). **anova_results.csv** — Welch ANOVA p-values per feature (Appendix). **outlier_heads_repeat_focus.csv** — Heads with high repeat_focus (Appendix). **clustering_results.csv** — Per-head assignment (node_name, layer, head, cluster). **cluster_visualization.png/.pdf** — UMAP 2D with convex hulls (Figure 4). |

### Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--steps` | Pipeline stages: `collect_features`, `analyze`. | both |
| `--model_types` | Model types to run: `esm3`, `esm-c`. | `esm3` `esm-c` |
| `--counterfactual_type` | Counterfactual method (e.g. `blosum`, `mask`). | `blosum` |
| `--repeat_type` | Repeat type in the dataset: `identical`, `approximate`, `synthetic`. | `approximate` |
| `--datasets_root` | Root for circuit_discovery datasets. | `{repo}/datasets` |
| `--results_root` | Root for all outputs. | `{repo}/results` |
| `--batch_size` | Batch size for feature collection. | `1` |
| `--component_recurrence_threshold` | Min ratio_in_graph for a head to be included. | `0.8` |
| `--n_clusters` | Number of KMeans clusters. | `3` |

### Examples

```bash
# Full pipeline (collect + analyze) for both models
python scripts/attention_heads_clustering_experiments/run.py \
  --model_types esm3 esm-c --counterfactual_type blosum --repeat_type approximate

# Analyze only (after collect has run)
python scripts/attention_heads_clustering_experiments/run.py \
  --steps analyze --model_types esm3 esm-c

# Collect only, one model
python scripts/attention_heads_clustering_experiments/run.py \
  --steps collect_features --model_types esm3 --counterfactual_type blosum --repeat_type approximate

# Custom paths
python scripts/attention_heads_clustering_experiments/run.py \
  --model_types esm3 --datasets_root /path/to/datasets --results_root /path/to/results
```
