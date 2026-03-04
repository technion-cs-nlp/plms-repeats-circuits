This module implements **Edge Attribution Patching (EAP)** and **EAP with Integrated Gradients (EAP-IG)** for circuit discovery in protein language models (ESM3, ESM-C). The code is based on [EAP-IG](https://github.com/hannamw/EAP-IG/tree/main). The same methods extend to nodes and neurons (we still call them EAP and EAP-IG), but those operate on nodes/neurons rather than edges. 
## Overview

The pipeline typically is:

1. Build a computational **graph** from the model
2. **Attribute** edges/nodes/neurons to get importance scores
3. Apply a threshold or top-k to define a **circuit**
4. **Evaluate** the circuit by patching out non-circuit components

---

## graph.py

Defines the computational graph representation: nodes (input, attention heads, MLPs, logits) and edges (connections via the residual stream).

### Classes

| Class | Description |
|-------|-------------|
| `Node` | Base node (input, attention, MLP, logits) |
| `InputNode` | Embedding layer output |
| `AttentionNode` | Single attention head (e.g. `a0.h3` = layer 0, head 3) |
| `MLPNode` | MLP block output (e.g. `m0`) |
| `MLPWithNeuronNode` | MLP with per-neuron granularity |
| `LogitNode` | Final logits |
| `Edge` | Connection between two nodes (Q/K/V for attention) |
| `Graph` | Full graph with nodes and edges |
| `NeuronGraph` | Graph with neuron-level nodes (no edges) |
| `GraphType` | Enum: `Edges` or `Nodes` |

### Main methods

- **`Graph.from_model(model_or_config, graph_type=GraphType.Edges)`** – Build graph from HookedESM3, HookedESMC, or config.
- **`NeuronGraph.from_model(model_or_config, graph_type=GraphType.Nodes)`** – Build neuron-level graph.
- **`graph.apply_threshold(threshold, absolute)`** – Keep edges/nodes with score ≥ threshold.
- **`graph.apply_topn(n, absolute)`** – Keep top-n edges or nodes.
- **`graph.apply_greedy(n_edges, reset, absolute)`** – Greedy circuit from logits (edges only).
- **`graph.prune_dead_nodes(prune_childless, prune_parentless)`** – Remove unreachable nodes.
- **`graph.to_json(filename)`** / **`Graph.from_json(filename)`** – Save/load graph state.

---

## attribute.py

Computes attribution scores (EAP and EAP-IG) for edges, nodes, or neurons.

### Main entry point

```python
attribute(model, graph, dataloader, metric, device, aggregation='sum',
          method='EAP', quiet=False, abs_per_pos=False, are_clean_logits_needed=False,
          eap_ig_steps=None, all_examples_scores_npy_path=None, use_mean_ablations=False, mean_ablations_dir=None,
          separate_mask_positions=False)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | HookedESM3 \| HookedESMC | Hooked model to attribute (must support forward hooks). |
| `graph` | Graph \| NeuronGraph | Computational graph to score. Use `Graph` with `GraphType.Edges` or `GraphType.Nodes`, or `NeuronGraph` for neuron-level attribution. |
| `dataloader` | DataLoader | Batches of `(clean, corrupted, masked_positions, labels, clean_id_names)`. See Dataloader format below. |
| `metric` | Callable | Objective for attribution. Signature: `metric(logits, clean_logits, masked_positions_tensor, labels_tensor)` → scalar. Gradients flow through this; typically a loss over logits at masked positions (e.g. logit diff, log prob, see plms_repeats_circuits/utils/patching_metrics.py). |
| `device` | torch.device | Device for model and tensors. |
| `aggregation` | str | How to aggregate attribution scores across sequence positions, then average over all examples: `'sum'` = sum over positions per example; `'pos_mean'` = mean over positions per example. |
| `method` | str | `'EAP'` = Edge Attribution Patching (edges) or Attribution Patching (nodes/neurons). `'EAP-IG'` = Integrated Gradients version. |
| `quiet` | bool | If True, disable tqdm progress bars. Default: False. |
| `abs_per_pos` | bool | If True, take absolute value of per-position scores before aggregation (magnitude-only importance). Default: False. |
| `are_clean_logits_needed` | bool | If True, run an extra forward pass to get clean logits and pass them to the metric (e.g. for faithfulness). If False, `clean_logits` is None and the metric must not require it. Default: False. |
| `eap_ig_steps` | int | Number of interpolation steps for EAP-IG. Higher = more accurate, slower. Default: 30. |
| `all_examples_scores_npy_path` | str | If set, save per-example attribution scores to this NPZ path (nodes/neurons only; not used for edges). |
| `use_mean_ablations` | bool | If True, use mean activations as the “corrupted” baseline instead of patching with corrupted inputs. Requires precomputed ablations via `compute_mean_activations_per_node`. |
| `mean_ablations_dir` | str | Directory with `.pt` files from `compute_mean_activations_per_node`. Required when `use_mean_ablations=True`. |
| `separate_mask_positions` | bool | When `use_mean_ablations=True`, use different mean activations for masked vs non-masked positions (`exclude_mask` / `only_mask`). Requires ablations computed with those modes. Default: False. |

### Other utilities

- **`tokenize_plus(model, inputs)`** – Tokenize protein sequences for ESM3/ESM-C.
- **`load_mean_ablations(mean_ablations_dir, device, separate_mask_positions)`** – Load mean ablation cache.

---

## evaluate.py

Evaluates a circuit by running the model with non-circuit components patched (corrupted) and measuring metrics.

### Functions

| Function | Description |
|----------|-------------|
| **`evaluate_graph`** | Evaluate edge-level circuit (Graph with GraphType.Edges) |
| **`evaluate_graph_node`** | Evaluate node-level circuit (Graph with GraphType.Nodes) |
| **`evaluate_graph_neurons`** | Evaluate neuron-level circuit (NeuronGraph) |
| **`evaluate_baseline`** | Run clean/corrupted model without circuit patching |
| **`compute_faithfulness`** | Faithfulness = (target − corrupted) / (clean − corrupted) |

### Parameters (common)

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | HookedESM3 \| HookedESMC | Model |
| `graph` | Graph \| NeuronGraph | Circuit to evaluate |
| `dataloader` | DataLoader | Same format as for attribution |
| `metrics` | Callable or list | One or more metric functions |
| `device` | torch.device | Device |
| `prune` | bool | Call `prune_dead_nodes` before eval (default True) |
| `calc_clean_logits` | bool | Pass clean logits to metric (e.g. for faithfulness) |

---

## circuit_selection.py

Circuit selection: choose which edges, nodes, or neurons form the circuit from attribution scores. Each function takes an optional `log` parameter: if `logging.Logger` is passed, messages go to the logger; otherwise they are printed.

### Functions

| Function | Description |
|----------|-------------|
| **`select_circuit_edges`** | Edge-level graph: select edges for the circuit using `greedy`, `greedy_abs`, `top_n`, `top_n_abs`, or `none`; prunes dead nodes. |
| **`select_circuit_nodes`** | Node-level graph: select nodes for the circuit using `top_n` or `top_n_abs`. |
| **`select_circuit_neurons`** | NeuronGraph: select neurons for the circuit using `top_n` or `top_n_abs`; supports per-layer and two-step (nodes first, then neurons). |

### Parameters (common)

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph` | Graph \| NeuronGraph | Attributed graph |
| `selection_method` | str | See selection methods per function below. |
| `log` | logging.Logger \| None | Logger for progress messages; if None, uses print. Default: None. |

### Selection methods (per function)

| Function | `selection_method` values |
|----------|---------------------------|
| `select_circuit_edges` | `'greedy'`, `'greedy_abs'`, `'top_n'`, `'top_n_abs'`, `'none'` |
| `select_circuit_nodes` | `'top_n'`, `'top_n_abs'` |
| `select_circuit_neurons` | `'top_n'`, `'top_n_abs'` |

### `select_circuit_neurons` (special parameters)

Neuron selection uses a two-step process: first select nodes, then select neurons within those nodes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_neurons` | int | Number of neurons to keep. If `is_per_layer=True`, this many per MLP layer; otherwise, total across the circuit. |
| `n_nodes_as_first_step_for_neurons` | int | In the first step, keep this many top-scored nodes (attention + MLP). Neurons are then selected only within those nodes. |
| `is_per_layer` | bool | If True, select `n_neurons` per MLP layer (`apply_topn_neurons_per_layer`). If False, select `n_neurons` total across the circuit (`apply_topn_only_neurons`). |

### Example

```python
from plms_repeats_circuits.EAP.circuit_selection import select_circuit_edges, select_circuit_nodes

# Edges: keep top 100 by absolute score
select_circuit_edges(graph, "top_n_abs", n_edges=100, log=logging.getLogger())

# Nodes: keep top n
select_circuit_nodes(graph, "top_n_abs", n_nodes=50, log=logging.getLogger())
```

---

## ablations.py

Computes mean activations for use as alternative “corrupted” baselines (mean ablation instead of patching with corrupted inputs).

### Main function

```python
compute_mean_activations_per_node(model, graph, dataloader, device, quiet=False,
                                  output_dir=None, save_neurons=False, activation_mode='all')
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | HookedESM3 | Model |
| `graph` | Graph | Graph (edges) |
| `dataloader` | DataLoader | Batches of (clean, masked_positions, labels) – no corrupted |
| `output_dir` | str | Directory to save `.pt` files |
| `save_neurons` | bool | Also compute mean neuron activations |
| `activation_mode` | str | `'all'`, `'exclude_mask'`, or `'only_mask'` |

Output files: `mean_activation_mlp.pt`, `mean_activation_attention_heads.pt`, `mean_activation_input.pt`, and optionally `mean_neuron_activations.pt`.

---

## Dataloader format

Each batch should be a tuple:

- `clean`: List[str] – clean protein sequences  
- `corrupted`: List[str] – corrupted sequences  
- `masked_positions`: List of masked token positions  
- `labels`: List of true labels for the metric  
- `clean_id_names`: List[str] – example IDs (e.g. `cluster_id_rep_id_repeat_key`)

---

## Example usage

```python
from plms_repeats_circuits.EAP.graph import Graph, GraphType
from plms_repeats_circuits.EAP.attribute import attribute
from plms_repeats_circuits.EAP.evaluate import evaluate_graph

# 1. Build graph
graph = Graph.from_model(model, graph_type=GraphType.Nodes)

# 2. Attribute
attribute(model, graph, dataloader, metric, device, method='EAP-IG', eap_ig_steps=5)

# 3. Select circuit (e.g. top 100 edges)
graph.apply_topn(100, absolute=True)

# 4. Evaluate
results = evaluate_graph(model, graph, dataloader, metric, device, prune=True)
```
