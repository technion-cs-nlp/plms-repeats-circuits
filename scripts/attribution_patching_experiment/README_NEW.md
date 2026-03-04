# Attribution Patching Scripts

This document describes the attribution patching scripts used in our experiments. These scripts are invoked from `run.py`. Here we give mor explanation on circuit dicovery scripts. 

---

## Background: What Is Attribution Patching?

**Attribution patching** identifies which parts of a language model contribute most to its behavior on a task. It:

1. Assigns a score to each component (edge, node, or neuron) based on how much it affects the output
2. Ranks components by score and selects the top ones (this subset is the "circuit")
3. Evaluates how well the circuit alone reproduces the full model (this is "faithfulness")

The model is represented as a graph: **nodes** are attention/MLP blocks, **edges** connect nodes, and **neurons** are individual units inside MLP layers. We can discover circuits at each level.

---

## Which Script Do I Use?

**`run_attribution_patching_nodes_edges.py`** — Use this when you want to find important edges or nodes. Run this first for most experiments. It produces a circuit graph saved as a JSON file. You can choose `--graph_type edges` (to score connections between nodes) or `--graph_type nodes` (to score whole attention/MLP blocks).

**`run_attribution_patching_neurons.py`** — Use this when you want to go finer-grained and find important neurons inside MLP layers. You must run the nodes script first: this script needs the nodes graph JSON and the number of nodes to keep. It then scores individual neurons within those nodes and evaluates neuron-level circuits.

**`run_cross_task.py`** — Use this when you want to test whether a circuit discovered on one task still works on another task. You need an existing circuit JSON from the edges or nodes script. The script evaluates the source circuit on target data and appends the results to a CSV.

**Typical workflow:** Run `run_attribution_patching_nodes_edges.py` with `--graph_type nodes` to get a nodes graph (JSON). Then run `run_attribution_patching_neurons.py` with `--nodes_graph_json` pointing at that file and `--n_nodes` set to how many nodes you want to keep. The neurons script will find the most important neurons within those nodes.

---

## Input Data

All scripts expect a CSV in **circuit_discovery format**. Required columns: `seq`, `cluster_id`, `rep_id`, `repeat_key`, `corrupted_sequence`, `masked_position`. For the `logit_diff` metric you also need `corrupted_amino_acid_type` or `replacements`.

---

## 1. run_attribution_patching_nodes_edges.py

Finds circuits at the level of **edges** (connections) or **nodes** (attention/MLP blocks). Use this script first for most experiments.

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--csv_path` | str | required | Path to input CSV. |
| `--output_dir` | str | required | Directory for outputs (logs, CSVs, circuit JSON). |
| `--total_n_samples` | int | required | Number of samples to use from CSV. |
| `--model_type` | str | required | `esm3` or `esm-c`. |
| `--graph_type` | str | edges | `edges` or `nodes`. |
| `--attribution_patching_method` | str | EAP | Attribution method: `EAP` or `EAP-IG`. |
| `--circuit_selection_method` | str | greedy_abs | Edges: `greedy`, `greedy_abs`, `top_n`, `top_n_abs`, `none`. Nodes: `top_n`, `top_n_abs` only. |
| `--n_components_in_circuit_list` | float, nargs='+' | [0.001, 0.002, …] | Circuit sizes to evaluate. Values &lt;1 = fraction of total; ≥1 = absolute count. |
| `--metric` | str | logit_diff | Evaluation metric: `logit_diff` or `log_prob`. |
| `--abs_score` | flag | False | Use absolute attribution scores per position before aggregation. |
| `--aggregation_method` | str | sum | How to aggregate per-position scores: `sum` or `pos_mean`. |
| `--circuit_json_path` | str | None | Load graph from JSON (skip attribution if file exists). |
| `--EAP_IG_steps` | int | 5 | Number of integrated-gradients steps for EAP-IG. |
| `--batch_size` | int | 1 | Dataloader batch size. |
| `--random_state` | int | 42 | Random seed for train/test split and sampling. |
| `--train_ratio` | float | 0.5 | Fraction of data for train (rest for test). |
| `--exp_prefix` | str | "" | Prefix for experiment name in filenames. |
| `--enable_min_circuit_search` | flag | False | Run binary search for minimal circuit within performance threshold. |
| `--min_performance_threshold` | float | 0.8 | Min faithfulness for minimal circuit search. |
| `--max_performance_threshold` | float | 0.85 | Max faithfulness for minimal circuit search. |
| `--min_circuit_size` | float | -1 | Min circuit size for search. ≥1=count, &lt;1=fraction, -1=infer from results. |
| `--max_circuit_size` | float | -1 | Max circuit size for search. Same rules as min. |
| `--max_search_steps` | int | 20 | Max binary-search iterations. |
| `--save_scores_per_example_npy` | flag | False | Save per-example attribution scores to NPY file. Relevant only when `--graph_type nodes`. |

### Output Files

| File | Description |
|------|-------------|
| `{experiment_name}.log` | Log file |
| `{experiment_name}.csv` | Faithfulness scores for each circuit size (input size, actual size, clean/corrupted/circuit means, faithfulness) |
| `{experiment_name}.json` | Circuit graph — **required as input for the neurons script** |
| `circuit_info_edges.csv` / `circuit_info_nodes.csv` | Minimal circuit found by binary search (only if `--enable_min_circuit_search`) |

### Example

```bash
python run_attribution_patching_nodes_edges.py \
  --csv_path datasets/circuit_discovery/mask.csv \
  --output_dir results/nodes \
  --total_n_samples 500 \
  --model_type esm3 \
  --graph_type nodes \
  --circuit_selection_method top_n_abs
```

---

## 2. run_attribution_patching_neurons.py

Finds circuits at the **neuron** level (individual units inside MLP layers). Must be run **after** the nodes script: it needs the nodes graph JSON and the number of nodes to keep.

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--csv_path` | str | required | Path to input CSV. |
| `--output_dir` | str | required | Output directory. |
| `--total_n_samples` | int | required | Number of samples. |
| `--model_type` | str | required | `esm3` or `esm-c`. |
| `--nodes_graph_json` | str | required | Path to nodes graph JSON (from run_attribution_patching_nodes_edges). |
| `--n_nodes` | int | required | Number of nodes to use (must match circuit in nodes_graph_json). |
| `--attribution_patching_method` | str | EAP | Attribution method: `EAP` or `EAP-IG`. |
| `--circuit_selection_method` | str | top_n_abs | `top_n` or `top_n_abs`. |
| `--n_neurons_in_circuit_list` | float, nargs='+' | [0.001, 0.002, …] | Neuron circuit sizes to evaluate. Values &lt;1 = fraction; ≥1 = absolute count. |
| `--min_performance_threshold_neurons` | float | 0.89 | Min faithfulness for neuron binary search. |
| `--max_performance_threshold_neurons` | float | 0.9 | Max faithfulness for neuron binary search. |
| `--is_per_layer_neurons` | flag | False | Select neurons per MLP layer (instead of globally). |
| `--metric` | str | logit_diff | Evaluation metric: `logit_diff` or `log_prob`. |
| `--abs_score` | flag | False | Use absolute attribution scores before aggregation. |
| `--aggregation_method` | str | sum | How to aggregate attribution scores: `sum` or `pos_mean`. |
| `--EAP_IG_steps` | int | 5 | IG steps for EAP-IG. |
| `--batch_size` | int | 1 | Dataloader batch size. |
| `--random_state` | int | 42 | Random seed for train/test split. |
| `--train_ratio` | float | 0.5 | Fraction of data for train. |
| `--exp_prefix` | str | "" | Prefix for experiment name in filenames. |
| `--enable_min_circuit_search` | flag | False | Binary search for minimal neuron circuit within threshold. |
| `--min_circuit_size` | float | -1 | Min circuit size for neuron search. ≥1=count, &lt;1=fraction, -1=infer. |
| `--max_circuit_size` | float | -1 | Max circuit size for neuron search. Same rules as min. |
| `--max_search_steps` | int | 20 | Max binary-search iterations. |

### Output Files

| File | Description |
|------|-------------|
| `{experiment_name}.log` | Log file |
| `{experiment_name}_neurons.csv` | Faithfulness by neuron circuit size |
| `{experiment_name}.json` | Neurons graph (saved after attribution and node selection) |
| `circuit_info_neurons.csv` | Minimal neuron circuit from binary search (only if `--enable_min_circuit_search`) |

### Example

```bash
# First run nodes to get the graph
python run_attribution_patching_nodes_edges.py \
  --csv_path datasets/circuit_discovery/mask.csv \
  --output_dir results/nodes \
  --total_n_samples 500 \
  --model_type esm3 \
  --graph_type nodes \
  --circuit_selection_method top_n_abs

# Then run neurons with the nodes graph (replace EXPERIMENT_NAME with the actual JSON filename from step 1)
python run_attribution_patching_neurons.py \
  --csv_path datasets/circuit_discovery/mask.csv \
  --output_dir results/neurons \
  --total_n_samples 500 \
  --model_type esm3 \
  --nodes_graph_json results/nodes/EXPERIMENT_NAME.json \
  --n_nodes 100
```

---

## 3. run_cross_task.py

Evaluates a **source** circuit (from one task) on **target** data (a different task). Used for cross-task faithfulness: does a circuit trained on task A still work on task B?

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source_task_name` | str | required | Source task name. |
| `--source_circuit_id` | str | required | Source circuit ID. |
| `--source_circuit_json` | str | required | Path to source circuit JSON. |
| `--target_task_name` | str | required | Target task name. |
| `--target_circuit_id` | str | required | Target circuit ID. |
| `--target_csv_path` | str | required | Path to target CSV data. |
| `--save_results_csv` | str | required | Path for cross-task results (appended). |
| `--output_dir` | str | required | Output directory for intermediate files. |
| `--total_n_samples` | int | required | Number of samples. |
| `--model_type` | str | required | `esm3` or `esm-c`. |
| `--graph_type` | str | nodes | `edges` or `nodes` (must match source circuit). |
| `--attribution_patching_method` | str | EAP | Pass-through to nodes_edges: `EAP` or `EAP-IG`. |
| `--circuit_selection_method` | str | greedy_abs | Pass-through to nodes_edges. |
| `--n_components_in_circuit_list` | float, nargs='+' | [0.001, …] | Circuit sizes to evaluate. |
| `--metric` | str | logit_diff | Evaluation metric: `logit_diff` or `log_prob`. |
| `--abs_score` | flag | False | Use absolute attribution scores. |
| `--aggregation_method` | str | sum | How to aggregate: `sum` or `pos_mean`. |
| `--EAP_IG_steps` | int | 5 | IG steps for EAP-IG. |
| `--batch_size` | int | 1 | Dataloader batch size. |
| `--random_state` | int | 42 | Random seed. |
| `--train_ratio` | float | 0.5 | Fraction of data for train. |
| `--exp_prefix` | str | "" | Prefix for experiment name. |
| `--delete_per_experiment_csv` | flag | False | Delete per-experiment CSV after run. |

### Output

Appends rows to `save_results_csv` with circuit identifiers and faithfulness scores.

### Example

```bash
python run_cross_task.py \
  --source_task_name task_a \
  --source_circuit_id uuid-1 \
  --source_circuit_json results/task_a/exp_nodes.json \
  --target_task_name task_b \
  --target_circuit_id uuid-2 \
  --target_csv_path datasets/task_b/mask.csv \
  --save_results_csv results/cross_task.csv \
  --output_dir results/cross_task_tmp \
  --total_n_samples 500 \
  --model_type esm3
```