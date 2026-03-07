# Circuit Discovery, Evaluation, and Comparison

This experiment runs circuit discovery via attribution patching (EAP-IG) on ESM3 and ESM-C. It discovers minimal circuits at the level of **edges**, **nodes**, or **neurons** for repeat-prediction tasks. The pipeline uses circuit_discovery datasets produced by the [counterfactual experiment](../counterfactual_experiment/README.md) (evaluate + filter). After discovering circuits, we apply the circuit comparison pipeline to compare them using three metrics: IoU, Recall, and Cross-Task Faithfulness.

**Files (use run.py to call the full pipeline we used in the paper):**

1) **run_attribution_patching_nodes_edges.py** — Runs attribution patching for edges or nodes. For each combination of repeat type, model, counterfactual method, and seed, it discovers a minimal circuit (EAP-IG, log_prob). Called by run.py during the discover step. See [README_ATTRIBUTION_PATCHING.md](README_ATTRIBUTION_PATCHING.md) for direct use.

2) **run_attribution_patching_neurons.py** — Same for neurons. Requires nodes output to exist first. Called by run.py during the discover step. See [README_ATTRIBUTION_PATCHING.md](README_ATTRIBUTION_PATCHING.md) for direct use.

3) **run_iou_recall.py** — Computes IoU and recall between circuit pairs (across repeat types or across counterfactual methods). Called by run.py during the compare step when `iou` or `recall` are in `--compare_metrics`.

4) **run_cross_task.py** — Computes cross-task faithfulness (how well a circuit from one task transfers to another). Called by run.py during the compare step when `cross_task` is in `--compare_metrics`.

5) **analyze_faithfulness.py** — Produces faithfulness curves for nodes/edges/neuron graphs. Called by run.py during analyze when `faithfulness` or `all` is in `--analyze_types`. [For figures 2, 5, 10, 14, 20 in the paper]

6) **analyze_compare.py** — Produces heatmaps for circuit comparison. Called by run.py during analyze when `compare_heatmaps` or `all` is in `--analyze_types`. [For figures 3, 11, 21, 22 in the paper]

7) **analyze_utils.py** — Shared utilities for the analyze scripts.

8) **attribution_patching_utils.py** — Shared utilities for the circuit discovery scripts.

9) **plotting_config.json** — Plot settings (display order, font sizes, etc.).

10) **run.py** — The main entry point. Use this script; do not call the above scripts directly unless you need custom experiments.

---

## run.py

**What it does:** `run.py` orchestrates the full attribution-patching pipeline. You choose which steps to run: **discover** (find circuits), **compare** (compute metrics between circuits), and **analyze** (plot faithfulness and heatmaps). Discover must run first; compare and analyze depend on its outputs. Compare outputs feed into the analyze heatmaps.

**Pipeline flow:**

```
Input CSVs (from counterfactual experiment)
        ↓
   [discover]  →  Discovered circuits (per repeat type, model, method, seed)
        ↓
   [compare]   →  IOU, recall, cross_task CSV files
        ↓
   [analyze]   →  Faithfulness plots + heatmaps
```

### Prerequisites: Where Do Inputs Come From?

You **must** run the [counterfactual experiment](../counterfactual_experiment/README.md) first (at least the evaluate and filter steps). That experiment produces the input CSVs this pipeline expects. Each input CSV lists proteins and positions where counterfactual corruption changed the model’s prediction; attribution patching then finds the minimal circuit responsible for those predictions.

### Expected inputs

Circuit_discovery CSVs must live under `datasets_root` in this layout:

```
{datasets_root}/
  identical/{model_type}/circuit_discovery/{repeat_type}_counterfactual_{method}.csv
  approximate/{model_type}/circuit_discovery/{repeat_type}_counterfactual_{method}.csv
  synthetic/{model_type}/circuit_discovery/{repeat_type}_counterfactual_{method}.csv
```

### Expected outputs

| Step | Output location | What you get |
|------|-----------------|--------------|
| **discover** | `{results_root}/circuit_discovery/{repeat_type}/{model_type}/{graph_type}/seed_{seed}/` | One directory per (repeat, model, graph_type, seed). Each directory contains all circuits for that combination; the same directory can hold circuits from different counterfactual methods. Per circuit we save: **circuit JSON** (`{exp}_circuit{id}_*.json`) — the circuit graph (edges, nodes, or neurons) with actual attribution scores per component; used by compare and by the neurons script. **circuit_info CSV** (`circuit_info_{graph_type}.csv`) — the final minimal circuit(s) found by binary search (circuit_id, size, etc.); shared by all counterfactual methods. **faithfulness CSV** (`{exp}.csv` or `{exp}_neurons.csv`) — faithfulness scores for a list of circuit sizes, used for faithfulness curves.|
| **compare** | `{results_root}/circuit_discovery_compare/` | Subdirs `iou_recall/` and `cross_task/` (by mode: `across_repeats`, `across_counterfactual`). Each has CSVs with IOU, recall, or cross-task faithfulness scores. |
| **analyze (faithfulness)** | `{results_root}/faithfulness/` | Plots showing the trade-off between circuit size and faithfulness (one per model, repeat type, graph type). |
| **analyze (heatmaps)** | `{results_root}/circuit_discovery_compare/heatmaps/` | Single heatmaps per metric, combined all-metrics heatmaps per repeat type, and per-metric combined heatmaps for all repeat types. |

### Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--steps` | Pipeline stages to run. List one or more: `discover`, `compare`, `analyze`. Discover must run first; compare and analyze use its outputs. | `discover` |
| `--graph_type` | Circuit granularity: `edges`, `nodes`, or `neurons`. Compare step supports only `edges` and `nodes`. | required |
| `--methods` | Counterfactual methods that created the input datasets. Must match methods with CSVs under `datasets_root` (e.g. `blosum`, `mask`, `blosum-opposite50`, `permutation`). | required |
| `--repeat_types` | Repeat types in the dataset: `identical` (exact copies), `approximate` (similar sequences), `synthetic` (generated). Each has its own circuit_discovery CSV. | all three |
| `--model_types` | Language models to run: `esm3`, `esm-c`. | `esm3` |
| `--seeds` | Random seeds for the train/test split during discovery. Each seed yields a different split; results are typically averaged across seeds. | `42 43 44 45 46` |
| `--n_examples` | Max proteins per (repeat_type, model, method, seed) during discovery. Lower = faster; higher = more stable estimates. | `1000` |
| `--compare_modes` | **across_counterfactual**: compare circuits from different methods (e.g. blosum vs mask) on the same repeat type. **across_repeats**: compare circuits across repeat types (identical vs approximate vs synthetic) for the same method. Affects both the compare step and the analyze step (faithfulness and compare_heatmaps): only the modes you list are run. | both |
| `--compare_metrics` | **iou**: Intersection over Union of circuit components. **recall**: fraction of one circuit’s components found in another. **cross_task**: how well a circuit from task A predicts labels when applied to task B (faithfulness transfer). | all three |
| `--analyze_types` | **all**: faithfulness curves + heatmaps. **faithfulness**: only faithfulness vs circuit size plots. **compare_heatmaps**: only IOU/recall/cross_task heatmaps. | `all` |
| `--datasets_root` | Root folder for circuit_discovery CSVs (produced by the counterfactual experiment filter step). | `{repo}/datasets` |
| `--results_root` | Root folder for all outputs: discovered circuits, compare CSVs, faithfulness plots, heatmaps. | `{repo}/results` |

**General examples (to get started):**

```bash
# Discover only — nodes, one method, one repeat type (fastest way to try the pipeline)
python scripts/circuit_discovery_experiment/run.py \
  --graph_type nodes --methods blosum \
  --repeat_types approximate --model_types esm3

# Full pipeline — discover, compare, analyze (default compare_modes: both)
python scripts/circuit_discovery_experiment/run.py \
  --steps discover compare analyze \
  --graph_type nodes --methods blosum mask \
  --repeat_types identical approximate synthetic --model_types esm3

# Compare + analyze only (after discover has already run)
python scripts/circuit_discovery_experiment/run.py \
  --steps compare analyze \
  --graph_type nodes --methods blosum \
  --repeat_types identical approximate synthetic --model_types esm3 \
  --analyze_types compare_heatmaps

# Neurons (requires nodes output first)
python scripts/circuit_discovery_experiment/run.py \
  --steps discover --graph_type neurons \
  --methods blosum --repeat_types approximate --model_types esm3

# Custom paths
python scripts/circuit_discovery_experiment/run.py \
  --graph_type nodes --methods blosum \
  --results_root /path/to/results --datasets_root /path/to/datasets
```

**Pipeline we ran for the paper:**

**Section 3: Nodes circuits and comparisons**

**ESM3:** We first discover circuits for all 4 counterfactual methods, compare them across methods, and produce faithfulness curves and heatmaps. This corresponds to the Appendix (counterfactual comparison).

```bash
python scripts/circuit_discovery_experiment/run.py \
  --steps discover compare analyze \
  --graph_type nodes \
  --methods blosum mask blosum-opposite50 permutation \
  --repeat_types identical approximate synthetic \
  --model_types esm3 \
  --seeds 42 43 44 45 46 \
  --compare_modes across_counterfactual \
  --analyze_types all
```

After choosing blosum (see Appendix), we compare circuits across repeat types for the main Section 3 results:

```bash
python scripts/circuit_discovery_experiment/run.py \
  --steps compare analyze \
  --graph_type nodes \
  --methods blosum \
  --repeat_types identical approximate synthetic \
  --model_types esm3 \
  --seeds 42 43 44 45 46 \
  --compare_modes across_repeats \
  --analyze_types all
```

**ESM-C:** We use blosum only. Full pipeline (Appendix ESM-C results):

```bash
python scripts/circuit_discovery_experiment/run.py \
  --steps discover compare analyze \
  --graph_type nodes \
  --methods blosum \
  --repeat_types identical approximate synthetic \
  --model_types esm-c \
  --seeds 42 43 44 45 46 \
  --compare_modes across_repeats \
  --analyze_types all
```

**Section 4: Neurons graph** — You must run nodes circuits first.

```bash
python scripts/circuit_discovery_experiment/run.py \
  --steps discover \
  --graph_type neurons \
  --methods blosum \
  --repeat_types approximate \
  --model_types esm3 esm-c \
  --seeds 42 43 44 45 46
```

**Section 5: Edges graph** — Interactions between components.

```bash
python scripts/circuit_discovery_experiment/run.py \
  --steps discover \
  --graph_type edges \
  --methods blosum \
  --repeat_types approximate \
  --model_types esm3 esm-c \
  --seeds 42 43 44 45 46
```
