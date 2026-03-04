# Attribution Patching Experiment

Run circuit discovery via EAP/EAP-IG on a protein repeat prediction task. The script loads a dataset, attributes edges/nodes/neurons, selects circuits, and evaluates faithfulness.

## Pipeline

1. Load CSV dataset and create induction task (clean vs corrupted, masked position, labels).
2. Split into train/test.
3. Build graph from model, run attribution (EAP or EAP-IG) on train.
4. For each circuit size in `n_edges_in_circuit_list`, select circuit and evaluate faithfulness on test.
5. Optionally: binary search for minimal circuit, two-step neuron search, or cross-task faithfulness.

## Input

- **CSV dataset**: Expects circuit_discovery format. Required columns: `seq`, `cluster_id`, `rep_id`, `repeat_key`, `corrupted_sequence`, `masked_position`. For `logit_diff` metric: `corrupted_amino_acid_type` or `replacements`.
- Data should be pre-filtered (e.g. max seq length 400).

---

## Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--csv_path` | str | Path to input CSV. |
| `--output_dir` | str | Directory for outputs (logs, CSVs, circuit JSON). |
| `--total_n_samples` | int | Number of samples to use. |
| `--task_name` | str | Name for the experiment (used in filenames). |
| `--model_type` | str | `esm3` or `esm-c`. |

---

## Dataset & Sampling

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--random_state` | int | 42 | Seed for sampling. |
| `--train_ratio` | float | 0.5 | Fraction of data for train (rest for test). |
| `--batch_size` | int | 1 | Batch size for dataloader. |

---

## Attribution

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--attribution_patching_method` | str | EAP | `EAP` or `EAP-IG`. |
| `--EAP_IG_steps` | int | 5 | IG steps for EAP-IG. |
| `--metric` | str | logit_diff | `logit_diff` or `log_prob`. |
| `--aggregation_method` | str | sum | `sum` or `pos_mean`. |
| `--abs_score` | flag | False | Use absolute scores before aggregation. |

---

## Circuit Selection

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--graph_type` | str | edges | `edges` or `nodes`. |
| `--post_attribution_patching_processing` | str | greedy_abs | How to select circuit. Edges: `greedy`, `greedy_abs`, `top_n`, `top_n_abs`, `none`. Nodes/neurons: `top_n`, `top_n_abs`. |
| `--n_edges_in_circuit_list` | float, nargs='+' | [0.001, 0.002, …] | Circuit sizes to evaluate. Values ≥1 = absolute count; <1 = fraction of total. |

---

## Output & Caching

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--circuit_json_path` | str | None | Load/save circuit from JSON. If provided and file exists, skips attribution. |
| `--save_scores_per_example_npy` | flag | False | Save per-example scores to NPY (via `all_examples_scores_npy_path`). |

---

## Neurons Graph

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--neurons_graph` | flag | False | Use NeuronGraph (neuron-level circuit). |
| `--find_neurons_circuits_two_steps` | flag | False | Two-step: first select nodes, then neurons within those nodes. Requires `post_attribution_patching_processing` in {top_n, top_n_abs}. |
| `--min_performance_threshold_neurons` | float | 0.89 | Min faithfulness for neuron circuit (two-step). |
| `--max_performance_threshold_neurons` | float | 0.9 | Max faithfulness for neuron circuit (two-step). |
| `--is_per_layer_neurons` | flag | False | Select neurons per MLP layer (two-step). |
| `--two_steps_neuron_circuit_node_circuit_list_sizes` | float, nargs='+' | [0, 50, 100, …] | Node counts for first step in two-step neuron search. |

---

## Minimal Circuit Search

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable_min_circuit_search` | flag | False | Binary search for smallest circuit in a performance window. |
| `--min_circuit_csv` | str | None | CSV path for min-circuit results. Default: `{output_dir}/min_circuits_{experiment_name}.csv`. |
| `--min_performance_threshold` | float | 0.8 | Lower bound of target faithfulness. |
| `--max_performance_threshold` | float | 0.85 | Upper bound of target faithfulness. |
| `--min_circuit_size` | float | -1 | Min circuit size for search. ≥1 = count, <1 = fraction, -1 = use evaluation results. |
| `--max_circuit_size` | float | -1 | Max circuit size for search. Same as above. |
| `--max_search_steps` | int | 20 | Max binary-search iterations. |

---

## Cross-Task Faithfulness

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--is_cross_task_faithfulness` | flag | False | Evaluate circuits from one task on another. |
| `--source_task_name` | str | None | Source task name. |
| `--source_circuit_id` | str | None | Source circuit ID. |
| `--target_task_name` | str | None | Target task name. |
| `--target_circuit_id` | str | None | Target circuit ID. |
| `--save_results_csv` | str | None | Path for cross-task results. |

Not supported with `find_neurons_circuits_two_steps`.

---

## Example

```bash
python run_attribution_patching.py \
  --csv_path datasets/synthetic/esm3/circuit_discovery/mask.csv \
  --output_dir results/attribution \
  --total_n_samples 500 \
  --task_name synthetic_mask \
  --model_type esm3 \
  --graph_type nodes \
  --post_attribution_patching_processing top_n_abs \
  --n_edges_in_circuit_list 0.01 0.05 0.1 0.5 1.0 \
  --attribution_patching_method EAP-IG \
  --EAP_IG_steps 5
```

---

## Output Files

- `{experiment_name}.log` – Log file.
- `{experiment_name}.csv` – Faithfulness by circuit size.
- `{experiment_name}.json` – Circuit graph (edges/nodes/neurons, scores, in_graph).
- With `--save_scores_per_example_npy`: `{experiment_name}_scores_per_example.npy`.
- With `--enable_min_circuit_search`: `min_circuits_{experiment_name}.csv`.
