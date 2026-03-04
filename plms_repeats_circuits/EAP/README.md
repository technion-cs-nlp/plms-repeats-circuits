# EAP: Edge Attribution Patching

This module implements **Edge Attribution Patching (EAP)** and **EAP with Integrated Gradients (EAP-IG)** for circuit discovery in protein language models (ESM3, ESM-C). The code is based on [EAP-IG](https://github.com/hannamw/EAP-IG/tree/main).

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
          eap_ig_steps=None, all_examples_scores_csv_path=None, use_clean_id_names=False,
          all_examples_scores_npy_path=None, use_mean_ablations=False, mean_ablations_dir=None,
          separate_mask_positions=False)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | HookedESM3 \| HookedESMC | Model to attribute |
| `graph` | Graph \| NeuronGraph | Graph (edges, nodes, or neurons) |
| `dataloader` | DataLoader | Batches of (clean, corrupted, masked_positions, labels, clean_id_names) |
| `metric` | Callable | `metric(logits, clean_logits, masked_positions_tensor, labels_tensor)` → scalar |
| `device` | torch.device | Device for computation |
| `aggregation` | str | `'sum'` or `'pos_mean'` over positions |
| `method` | str | `'EAP'` or `'EAP-IG'` |
| `eap_ig_steps` | int | IG steps for EAP-IG (default 30) |
| `all_examples_scores_csv_path` | str | Path to save per-example node scores (nodes only) |
| `use_clean_id_names` | bool | Use dataset IDs instead of example_0, example_1, … |
| `all_examples_scores_npy_path` | str | Path to save scores as NPZ (nodes/neurons) |
| `use_mean_ablations` | bool | Use mean ablation instead of corrupted activations |
| `mean_ablations_dir` | str | Directory with precomputed mean ablations (`.pt` files) |
| `separate_mask_positions` | bool | Use separate mean ablations for masked vs non-masked positions |

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
