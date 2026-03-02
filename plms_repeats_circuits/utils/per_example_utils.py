import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Union
from protein_circuits.EAP.graph import NeuronGraph, Graph, GraphType


def find_npz_files(base_dir: Path, seeds: List[int], per_layer: bool) -> List[Path]:
    suffix = "nodes_neurons_two_steps_per_layer" if per_layer else "nodes_neurons_two_steps"
    npz_files = []
    
    for seed in seeds:
        results_dir = base_dir / f"random_state_{seed}" / f"results_noise_entire_repeat_{suffix}"
        if results_dir.exists():
            npz_files.extend(results_dir.glob("*_scores_per_example.npy.npz"))
    
    return npz_files


def load_importance_scores(
    base_dir: str,
    seeds: List[int],
    component_type: str,
    per_layer: bool
) -> Tuple[np.ndarray, List[str]]:
    print(f"loading importance scores from {base_dir} for component type {component_type} and per layer {per_layer}")
    base_path = Path(base_dir)
    npz_files = find_npz_files(base_path, seeds, per_layer)
    
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {base_dir}")
    
    all_scores = []
    all_names = []
    
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        
        if component_type == "attention":
            scores = data.get("scores_nodes", data.get("scores"))
        elif component_type == "neuron":
            scores = data["scores_neurons"]
        else:
            raise ValueError(f"Unknown component_type: {component_type}")
        
        names = data.get("example_row_names", data.get("example_names"))
        if isinstance(names, np.ndarray):
            names = names.tolist()
        
        all_scores.append(scores)
        all_names.extend(names)
    
    scores_matrix = np.vstack(all_scores)
    return deduplicate_scores(scores_matrix, all_names)


def deduplicate_scores(scores: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str]]:
    seen = {}
    keep_indices = []
    
    for idx, name in enumerate(names):
        if name not in seen:
            seen[name] = idx
            keep_indices.append(idx)
    
    return scores[keep_indices], [names[i] for i in keep_indices]


def normalize_scores(
    scores: np.ndarray,
    component_type: str,
    per_layer: bool,
    model
) -> np.ndarray:
    if component_type == "attention":
        row_sums = np.sum(np.abs(scores), axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return scores / row_sums
    
    elif component_type == "neuron" and per_layer:
        graph = NeuronGraph.from_model(model, graph_type=GraphType.Nodes)
        normalized = scores.copy()
        
        for layer in range(model.cfg.n_layers):
            node = graph.nodes[f"m{layer}"]
            layer_slice = graph.forward_index(node, return_index_in_neurons_array=True)
            
            layer_scores = scores[:, layer_slice]
            row_sums = np.sum(np.abs(layer_scores), axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            normalized[:, layer_slice] = layer_scores / row_sums
        
        return normalized
    
    else:
        row_sums = np.sum(np.abs(scores), axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return scores / row_sums


def build_component_index_mapping(component_type: str, model) -> Dict[str, int]:
    if component_type == "attention":
        graph = Graph.from_model(model, graph_type=GraphType.Nodes)
        mapping = {}
        for node in graph.nodes.values():
            if hasattr(node, 'layer') and hasattr(node, 'head'):
                idx = graph.forward_index(node, attn_slice=False)
                mapping[node.name] = idx
        return mapping
    
    elif component_type == "neuron":
        graph = NeuronGraph.from_model(model, graph_type=GraphType.Nodes)
        mapping = {}
        for layer in range(model.cfg.n_layers):
            node = graph.nodes[f"m{layer}"]
            layer_slice = graph.forward_index(node, return_index_in_neurons_array=True)
            for neuron_idx in range(model.cfg.d_mlp):
                comp_name = f"m{layer}_n{neuron_idx}"
                mapping[comp_name] = layer_slice.start + neuron_idx
        return mapping
    
    else:
        raise ValueError(f"Unknown component_type: {component_type}")


def get_component_indices(components: List[str], component_index_mapping: Dict[str, int]) -> List[int]:
    indices = []
    for comp in components:
        if comp not in component_index_mapping:
            raise ValueError(f"Component {comp} not found in mapping")
        print(f"getting component indices: component {comp} found in mapping- index: {component_index_mapping[comp]}")
        indices.append(component_index_mapping[comp])
    return indices

