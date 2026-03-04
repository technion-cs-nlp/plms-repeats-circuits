# Based on: https://github.com/hannamw/EAP-IG/blob/0edbdd72e3683db69c363a23deb1775f44ec8376/eap/attribute.py

from typing import Callable, List, Union, Optional,Literal
from functools import partial
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedESM3, HookedESMC
from tqdm import tqdm
from einops import einsum
from .graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode, GraphType, MLPWithNeuronNode, NeuronGraph
from transformer_lens.hook_points import HookPoint
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
from plms_repeats_circuits.utils.protein_similiarity_utils import analyze_repeat_positions, RepeatPositionData
import os
import warnings
from plms_repeats_circuits.utils.esm_utils import load_tokenizer_by_model_type

def load_mean_ablations(mean_ablations_dir: str, device, separate_mask_positions: bool):
    """
    Load mean ablation tensors from disk into memory.
    
    Returns dict with keys:
        'neurons': dict with 'all', 'exclude_mask', 'only_mask' (if separate_mask_positions)
        'mlp': dict with 'all', 'exclude_mask', 'only_mask'
        'attention': dict with 'all', 'exclude_mask', 'only_mask'
        'input': dict with 'all', 'exclude_mask', 'only_mask'
    """
    cache = {}
    mean_ablations_path = Path(mean_ablations_dir)
    
    modes = ['all']
    if separate_mask_positions:
        modes.extend(['exclude_mask', 'only_mask'])
    
    for component in ['neurons', 'mlp', 'attention', 'input']:
        cache[component] = {}
        for mode in modes:
            suffix = "" if mode == 'all' else f"_{mode}"
            
            if component == 'neurons':
                filename = f"mean_neuron_activations{suffix}.pt"
            elif component == 'mlp':
                filename = f"mean_activation_mlp{suffix}.pt"
            elif component == 'attention':
                filename = f"mean_activation_attention_heads{suffix}.pt"
            elif component == 'input':
                filename = f"mean_activation_input{suffix}.pt"
            
            filepath = mean_ablations_path / filename
            if filepath.exists():
                cache[component][mode] = torch.load(filepath, map_location=device)
            else:
                warnings.warn(f"{filepath} not found, skipping...")
    
    return cache


def initialize_activation_difference_with_mean_ablations(
    batch_size: int,
    n_pos: int, 
    graph: NeuronGraph,
    model,
    device,
    mean_ablations_cache: dict,
    is_neurons: bool = True,
    use_mode: str = 'all'
):
    """
    Initialize activation difference tensor with mean ablations instead of zeros.
    
    For neurons: returns [batch, pos, n_forward_neurons] with mean ablations replicated
    For nodes: returns [batch, pos, n_forward, d_model] with mean ablations replicated
    
    Args:
        batch_size: batch size
        n_pos: sequence length
        graph: NeuronGraph
        model: HookedESM3/HookedESMC
        device: device
        mean_ablations_cache: dict loaded from load_mean_ablations()
        is_neurons: True for neurons, False for nodes
        use_mode: 'all', 'exclude_mask', or 'only_mask'
    """
    if is_neurons:
        # Initialize for neurons: [batch, pos, n_forward_neurons]
        activation_diff = torch.zeros((batch_size, n_pos, graph.n_forward_neurons), device=device, dtype=model.cfg.dtype)
        
        # Get mean neuron activations: [n_layers, d_mlp]
        mean_neurons = mean_ablations_cache['neurons'].get(use_mode)
        if mean_neurons is None:
            warnings.warn(f"mode '{use_mode}' not found in neurons cache, using zeros")
            return activation_diff
        
        # Ensure mean_neurons is on the correct device
        mean_neurons = mean_neurons.to(device=device, dtype=model.cfg.dtype)
        
        # Map each neuron in graph to its mean activation
        for node in graph.nodes.values():
            if isinstance(node, MLPWithNeuronNode):
                layer = node.layer
                # Get the slice of neurons for this layer in the forward index
                neurons_slice = graph.forward_index(node, attn_slice=False, return_index_in_neurons_array=True)
                
                # mean_neurons[layer] has shape [d_mlp]
                # Replicate across batch and positions: [batch, pos, d_mlp]
                mean_act = mean_neurons[layer].unsqueeze(0).unsqueeze(0)  # [1, 1, d_mlp]
                mean_act = mean_act.expand(batch_size, n_pos, -1)  # [batch, pos, d_mlp]
                
                # Assign to activation_diff
                activation_diff[:, :, neurons_slice] = mean_act
        
        return activation_diff
    else:
        # Initialize for nodes: [batch, pos, n_forward, d_model]
        activation_diff = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=device, dtype=model.cfg.dtype)
        
        # Get mean MLP, attention, and input activations
        mean_mlp = mean_ablations_cache['mlp'].get(use_mode)
        mean_attn = mean_ablations_cache['attention'].get(use_mode)
        mean_input = mean_ablations_cache['input'].get(use_mode)
        
        if mean_mlp is None or mean_attn is None or mean_input is None:
            warnings.warn(f"mode '{use_mode}' not found in mlp/attention/input cache, using zeros")
            return activation_diff
        
        # Ensure tensors are on the correct device
        mean_mlp = mean_mlp.to(device=device, dtype=model.cfg.dtype)
        mean_attn = mean_attn.to(device=device, dtype=model.cfg.dtype)
        mean_input = mean_input.to(device=device, dtype=model.cfg.dtype)
        
        # Map each node to its mean activation
        for node in graph.nodes.values():
            if isinstance(node, InputNode):
                # Input node: forward_index = 0, mean_input has shape [d_model]
                fwd_idx = graph.forward_index(node, attn_slice=False)  # Should be 0
                mean_act = mean_input.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
                mean_act = mean_act.expand(batch_size, n_pos, -1)
                activation_diff[:, :, fwd_idx] = mean_act
                    
            elif isinstance(node, MLPNode):
                layer = node.layer
                fwd_idx = graph.forward_index(node, attn_slice=False)
                
                # mean_mlp[layer] has shape [d_model]
                mean_act = mean_mlp[layer].unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
                mean_act = mean_act.expand(batch_size, n_pos, -1)
                activation_diff[:, :, fwd_idx] = mean_act
                
            elif isinstance(node, AttentionNode):
                layer = node.layer
                head = node.head
                fwd_idx = graph.forward_index(node, attn_slice=False)
                
                # mean_attn[layer] has shape [n_heads, d_model]
                mean_act = mean_attn[layer, head].unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
                mean_act = mean_act.expand(batch_size, n_pos, -1)
                activation_diff[:, :, fwd_idx] = mean_act
        
        return activation_diff


def overwrite_masked_positions_with_mean_ablations(
    activation_diff: torch.Tensor,
    batch_size: int,
    masked_positions: List[List[int]],
    graph: NeuronGraph,
    model,
    device,
    mean_ablations_cache: dict,
    is_neurons: bool = True,
    use_mode: str = 'only_mask'
):
    """
    Overwrite specific masked positions in activation_diff with mean ablations from a different mode.
    
    Args:
        activation_diff: Existing activation difference tensor to modify in-place
        batch_size: batch size
        masked_positions: List of lists, where masked_positions[b] contains masked position indices for batch item b
        graph: NeuronGraph
        model: HookedESM3/HookedESMC
        device: device
        mean_ablations_cache: dict loaded from load_mean_ablations()
        is_neurons: True for neurons, False for nodes
        use_mode: Mode to use for masked positions (typically 'only_mask')
    """
    if is_neurons:
        # Get mean neuron activations for masked positions: [n_layers, d_mlp]
        mean_neurons = mean_ablations_cache['neurons'].get(use_mode)
        if mean_neurons is None:
            warnings.warn(f"mode '{use_mode}' not found in neurons cache, not overwriting masked positions")
            return
        
        # Ensure on correct device
        mean_neurons = mean_neurons.to(device=device, dtype=activation_diff.dtype)
        
        # For each batch item, overwrite the masked positions
        for b in range(batch_size):
            masked_pos_list = masked_positions[b]  # List of position indices for this batch item
            
            # Ensure masked_pos_list is iterable (handle scalar case)
            if isinstance(masked_pos_list, (int, np.integer)):
                masked_pos_list = [masked_pos_list]
            elif not masked_pos_list:
                continue
            
            # Overwrite for each node
            for node in graph.nodes.values():
                if isinstance(node, MLPWithNeuronNode):
                    layer = node.layer
                    neurons_slice = graph.forward_index(node, attn_slice=False, return_index_in_neurons_array=True)
                    
                    # For each masked position in this batch item
                    for pos in masked_pos_list:
                        # Overwrite activation_diff[b, pos, neurons_slice] with mean_neurons[layer] 
                        activation_diff[b, pos, neurons_slice] = mean_neurons[layer]
    else:
        # Get mean activations for nodes
        mean_mlp = mean_ablations_cache['mlp'].get(use_mode)
        mean_attn = mean_ablations_cache['attention'].get(use_mode)
        mean_input = mean_ablations_cache['input'].get(use_mode)
        
        if mean_mlp is None or mean_attn is None or mean_input is None:
            warnings.warn(f"mode '{use_mode}' not found in mlp/attention/input cache, not overwriting masked positions")
            return
        
        # Ensure on correct device
        mean_mlp = mean_mlp.to(device=device, dtype=activation_diff.dtype)
        mean_attn = mean_attn.to(device=device, dtype=activation_diff.dtype)
        mean_input = mean_input.to(device=device, dtype=activation_diff.dtype)
        
        # For each batch item, overwrite the masked positions
        for b in range(batch_size):
            masked_pos_list = masked_positions[b]
            
            # Ensure masked_pos_list is iterable (handle scalar case)
            if isinstance(masked_pos_list, (int, np.integer)):
                masked_pos_list = [masked_pos_list]
            elif not masked_pos_list:
                continue
            
            for node in graph.nodes.values():
                if isinstance(node, InputNode):
                    # Input node: forward_index = 0
                    fwd_idx = graph.forward_index(node, attn_slice=False)
                    for pos in masked_pos_list:
                        activation_diff[b, pos, fwd_idx] = mean_input
                
                elif isinstance(node, MLPNode):
                    layer = node.layer
                    fwd_idx = graph.forward_index(node, attn_slice=False)
                    for pos in masked_pos_list:
                        activation_diff[b, pos, fwd_idx] = mean_mlp[layer]
                        
                elif isinstance(node, AttentionNode):
                    layer = node.layer
                    head = node.head
                    fwd_idx = graph.forward_index(node, attn_slice=False)
                    for pos in masked_pos_list:
                        activation_diff[b, pos, fwd_idx] = mean_attn[layer, head]

def tokenize_plus(model: HookedESM3 | HookedESMC, inputs: List[str]):
    model_name = model.cfg.model_name
    if model_name is None:
        if isinstance(model, HookedESM3):
            model_name = "esm3"
        elif isinstance(model, HookedESMC):
            model_name = "esm-c"
        else:
            raise ValueError(f"Model {model} is not supported")

    tokenizer = load_tokenizer_by_model_type(model_name)
    tokenized_info = tokenizer(inputs, padding=True, return_tensors='pt', add_special_tokens=True)
    tokens = tokenized_info.input_ids
    attention_mask = tokenized_info.attention_mask 
    n_pos = attention_mask.size(1)
    lengths = attention_mask.sum(dim=1) 
    return tokens, attention_mask, n_pos, lengths

def make_hooks_and_matrices(model: HookedESM3 | HookedESMC, graph: Graph, batch_size:int , n_pos:int, scores: torch.Tensor, device, detach=True, aggregation='sum', abs_per_pos=False, lengths=None):
    activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=device, dtype=model.cfg.dtype)

    processed_attn_layers = set()
    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []
    
    def activation_hook(index, activations, hook, add : bool = True):
        """Hook to add/subtract activations to/from the activation difference matrix
        Args:
            index ([type]): forward index of the node
            activations ([type]): activations to add
            hook ([type]): hook (unused)
            add (bool, optional): whether to add or subtract. Defaults to True."""
        acts = activations.detach() if detach else activations
        if not add:
            acts = -acts
        try:
            activation_difference[:, :, index] += acts
        except RuntimeError as e:
            raise RuntimeError(f"Activation hook error at {hook.name}: {e}") from e
    
    def gradient_hook(lengths, aggregation:str, abs_per_pos:bool, prev_index: Union[slice, int], bwd_index: Union[slice, int], gradients:torch.Tensor, 
                      hook:HookPoint):
        """Hook to multiply the gradients by the activations and add them to the scores matrix

        Args:
            prev_index (Union[slice, int]): index before which all nodes contribute to the present node
            bwd_index (Union[slice, int]): backward pass index of the node
            gradients (torch.Tensor): gradients of the node
            hook ([type]): hook
        """
        grads = gradients.detach()
        try:
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)            
            score_per_pos = einsum(activation_difference[:, :, :prev_index], grads,'batch pos forward hidden, batch pos backward hidden -> batch pos forward backward')
            if abs_per_pos:
                score_per_pos = score_per_pos.abs()
            if aggregation == 'sum':
                s = score_per_pos.sum(dim=(0, 1))
            elif aggregation == 'pos_mean':
                # average over positions for each example, then sum over batch
                # shape is [batch, pos, forward, backward]
                temp = score_per_pos.sum(dim=1)  # now [batch, forward, backward]
                temp = temp / lengths.view(-1, 1, 1)
                s = temp.sum(dim=0)  # [forward, backward]
            else:
                raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')

            s = s.squeeze(1)
            scores[:prev_index, bwd_index] += s
        except RuntimeError as e:
            raise RuntimeError(f"Gradient hook error at {hook.name}: {e}") from e

    for name, node in graph.nodes.items():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            else:
                processed_attn_layers.add(node.layer)

        # exclude logits from forward
        if not isinstance(node, LogitNode):
            fwd_index = graph.forward_index(node)
            fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
            fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False))) #(Corrupted - Clean)
        if not isinstance(node, InputNode):
            prev_index = graph.prev_index(node)
            if isinstance(node, AttentionNode):
                for i, letter in enumerate('qkv'):
                    bwd_index = graph.backward_index(node, qkv=letter)
                    bwd_hooks.append((node.qkv_inputs[i], partial(gradient_hook, lengths, aggregation, abs_per_pos, prev_index, bwd_index)))
            else:
                bwd_index = graph.backward_index(node)
                bwd_hooks.append((node.in_hook, partial(gradient_hook, lengths, aggregation, abs_per_pos, prev_index, bwd_index)))
    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference

def get_scores_eap_ig(model: HookedESM3 | HookedESMC, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], device, steps=30, quiet=False, aggregation='sum', abs_per_pos=False):

    scores = torch.zeros((graph.n_forward, graph.n_backward), device= device, dtype=model.cfg.dtype)
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted , masked_positions, labels, clean_id_names in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens, attention_mask_clean, n_pos , lengths_clean= tokenize_plus(model, clean) 
        corrupted_tokens, attention_mask_corrupted, _ , _= tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, device, True, aggregation, abs_per_pos, lengths_clean.to(device))
        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model.forward(
                        sequence_tokens=corrupted_tokens.to(device),
                        sequence_id=attention_mask_corrupted.to(device)
                    )

            input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()
       
            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model.forward(
                        sequence_tokens=clean_tokens.to(device),
                        sequence_id=attention_mask_clean.to(device)
                    )
                clean_logits= clean_logits.detach()

        input_activations_clean = input_activations_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes['input'])]

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (input_activations_clean - input_activations_corrupted) 
                new_input.requires_grad = True
                return new_input
            return hook_fn

        total_steps = 0
        for step in range(1, steps+1):
            total_steps += 1
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))], bwd_hooks=bwd_hooks):
                logits = model.forward(
                    sequence_tokens=clean_tokens.to(device),
                    sequence_id=attention_mask_clean.to(device)
                )
                masked_positions_tensor = torch.tensor(masked_positions, device=device)
                labels_tensor = torch.tensor(labels, device=device)
                metric_value = metric(logits, clean_logits, masked_positions_tensor, labels_tensor)
                metric_value.backward()
                model.zero_grad(set_to_none=True)

    assert total_steps == steps, f"total_steps {total_steps} is not equal to steps {steps}"
    scores /= total_items
    scores /= total_steps

    return scores

def get_scores_eap(model: HookedESM3 | HookedESMC, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor], device, quiet=False,
aggregation='sum', abs_per_pos=False, are_clean_logits_needed=False):
    scores = torch.zeros((graph.n_forward, graph.n_backward), device= device, dtype=model.cfg.dtype)
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted , masked_positions, labels, clean_id_names in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask_clean, n_pos , lengths_clean= tokenize_plus(model, clean) 
        corrupted_tokens, attention_mask_corrupted, _ , _= tokenize_plus(model, corrupted) 
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, device, True, aggregation, abs_per_pos, lengths_clean.to(device))
        
        with torch.no_grad():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model.forward(
                        sequence_tokens=corrupted_tokens.to(device),
                        sequence_id=attention_mask_corrupted.to(device)
                    )      
            if are_clean_logits_needed:
                clean_logits = model.forward(
                        sequence_tokens=clean_tokens.to(device),
                        sequence_id=attention_mask_clean.to(device)
                    )
                clean_logits= clean_logits.detach()
            else:
                clean_logits = None
            
        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model.forward(
                    sequence_tokens=clean_tokens.to(device),
                    sequence_id=attention_mask_clean.to(device)
                )

            masked_positions_tensor = torch.tensor(masked_positions, device=device)
            labels_tensor = torch.tensor(labels, device=device)
            metric_value = metric(logits, clean_logits, masked_positions_tensor, labels_tensor) 
            metric_value.backward()
            model.zero_grad(set_to_none=True)  # clear the gradients

    scores /= total_items #averaging over all scores

    return scores

allowed_aggregations = {'sum', 'pos_mean'}
def attribute(model: HookedESM3 | HookedESMC, graph: Graph | NeuronGraph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], device, aggregation='sum',
 method: Union[Literal['EAP', 'EAP-IG']]="EAP", quiet=False, abs_per_pos=False, are_clean_logits_needed=False, eap_ig_steps=None, all_examples_scores_csv_path:str=None,
 use_clean_id_names:bool=False, all_examples_scores_npy_path:str=None, 
 use_mean_ablations:bool=False, mean_ablations_dir:Optional[str]=None, separate_mask_positions:bool=False):
    
    # Load mean ablations if requested
    mean_ablations_cache = None
    if use_mean_ablations:
        if mean_ablations_dir is None:
            raise ValueError("mean_ablations_dir must be provided when use_mean_ablations=True")
        mean_ablations_cache = load_mean_ablations(mean_ablations_dir, device, separate_mask_positions)
    
    if isinstance(graph, NeuronGraph):
        attribute_neurons(model=model, graph=graph, dataloader=dataloader, metric=metric, device=device, aggregation=aggregation, method=method, 
                        quiet=quiet, abs_per_pos=abs_per_pos, are_clean_logits_needed=are_clean_logits_needed, eap_ig_steps=eap_ig_steps,
                        all_examples_scores_npy_path=all_examples_scores_npy_path, mean_ablations_cache=mean_ablations_cache, 
                        separate_mask_positions=separate_mask_positions)
    else:
        if graph.graph_type == GraphType.Edges:
            attribute_edges(model=model, graph=graph, dataloader=dataloader, metric=metric, device=device, aggregation=aggregation, method=method, 
                            quiet=quiet, abs_per_pos=abs_per_pos, are_clean_logits_needed=are_clean_logits_needed, eap_ig_steps=eap_ig_steps)
        elif graph.graph_type == GraphType.Nodes:
            attribute_nodes(model=model, graph=graph, dataloader=dataloader, metric=metric, device=device, aggregation=aggregation, method=method, 
                        quiet=quiet, abs_per_pos=abs_per_pos, are_clean_logits_needed=are_clean_logits_needed, eap_ig_steps=eap_ig_steps, 
                        all_examples_scores_csv_path=all_examples_scores_csv_path, use_clean_id_names=use_clean_id_names, all_examples_scores_npy_path=all_examples_scores_npy_path)
                       
    
def attribute_edges(model: HookedESM3, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], device, aggregation='sum',
 method: Union[Literal['EAP', 'EAP-IG']]="EAP", quiet=False, abs_per_pos=False, are_clean_logits_needed=False, eap_ig_steps=None):
    
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')

    if method == 'EAP':
        scores = get_scores_eap(model, graph, dataloader, metric, quiet=quiet, device=device, aggregation=aggregation, abs_per_pos=abs_per_pos, are_clean_logits_needed=are_clean_logits_needed)
    elif method == 'EAP-IG':
        scores = get_scores_eap_ig(model=model, graph=graph, dataloader=dataloader, metric=metric, device=device, steps=eap_ig_steps, quiet=quiet,aggregation=aggregation, abs_per_pos=abs_per_pos)
    else:
        raise ValueError(f"integrated_gradients must be in ['EAP'], but got {method}")

    scores = scores.cpu().numpy()
    for edge in tqdm(graph.edges.values(), total=len(graph.edges)):
        edge.score = scores[graph.forward_index(edge.parent, attn_slice=False), graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False)]*edge.weight
        
def attribute_nodes(model: HookedESM3 | HookedESMC, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], device, aggregation='sum',
 method: Union[Literal['EAP', 'EAP-IG']]="EAP", quiet=False, abs_per_pos=False, are_clean_logits_needed=False, eap_ig_steps=None, all_examples_scores_csv_path:str=None, 
 use_clean_id_names:bool=False, all_examples_scores_npy_path:str=None):
    
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')

    if method == 'EAP':
        scores = get_scores_eap_nodes(model, graph, dataloader, metric, quiet=quiet, device=device, aggregation=aggregation, abs_per_pos=abs_per_pos, are_clean_logits_needed=are_clean_logits_needed, 
        all_examples_scores_csv_path=all_examples_scores_csv_path, use_clean_id_names=use_clean_id_names, all_examples_scores_npy_path=all_examples_scores_npy_path)
    elif method == 'EAP-IG':
        scores = get_scores_eap_ig_nodes(model=model, graph=graph, dataloader=dataloader, metric=metric, device=device, steps=eap_ig_steps, quiet=quiet,aggregation=aggregation, abs_per_pos=abs_per_pos,
        all_examples_scores_csv_path=all_examples_scores_csv_path, use_clean_id_names=use_clean_id_names, all_examples_scores_npy_path=all_examples_scores_npy_path)
    else:
        raise ValueError(f"integrated_gradients must be in ['EAP'], but got {method}")
    

    scores = scores.cpu().numpy()  # this will detach the scores

    for node in tqdm(graph.nodes.values(), total=len(graph.nodes)):
        if isinstance(node, LogitNode):
            continue  # skip logits nodes cause it doesnt have gradient.
        node.score = scores[graph.forward_index(node, attn_slice=False)]

def make_hooks_and_matrices_nodes(model: HookedESM3 | HookedESMC, graph: Graph, batch_size:int, 
                                 n_pos:int, scores: torch.Tensor, device, detach=True, aggregation='sum', abs_per_pos=False, lengths=None,
                                 scores_nodes_all_examples:dict=None, node_forward_index_to_name:dict=None, items_until_now:int=0, save_scores_per_node:bool=False,
                                 clean_id_names=None, use_clean_id_names:bool=False, scores_examples_np_arr:np.ndarray=None, indicies_examples_torch_arr:torch.Tensor=None,
                                 mean_ablations_cache: Optional[dict]=None, masked_positions: Optional[List[List[int]]]=None, 
                                 separate_mask_positions: bool=False):
    
    # Initialize activation_difference - either with zeros or mean ablations
    if mean_ablations_cache is not None:
        # Use mean ablations as initialization (clean hooks will subtract from this)
        # Result: mean_ablations - clean (analogous to corrupted - clean)
        if separate_mask_positions and masked_positions is not None:
            # Initialize with 'exclude_mask' mode (non-masked positions)
            activation_difference = initialize_activation_difference_with_mean_ablations(
                batch_size=batch_size,
                n_pos=n_pos,
                graph=graph,
                model=model,
                device=device,
                mean_ablations_cache=mean_ablations_cache,
                is_neurons=False,
                use_mode='exclude_mask'
            )
            # Overwrite masked positions with 'only_mask' mode
            overwrite_masked_positions_with_mean_ablations(
                activation_diff=activation_difference,
                batch_size=batch_size,
                masked_positions=masked_positions,
                graph=graph,
                model=model,
                device=device,
                mean_ablations_cache=mean_ablations_cache,
                is_neurons=False,
                use_mode='only_mask'
            )
        else:
            # Use 'all' mode for all positions
            activation_difference = initialize_activation_difference_with_mean_ablations(
                batch_size=batch_size,
                n_pos=n_pos,
                graph=graph,
                model=model,
                device=device,
                mean_ablations_cache=mean_ablations_cache,
                is_neurons=False,
                use_mode='all'
            )
    else:
        activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=device, dtype=model.cfg.dtype)

    processed_attn_layers = set()
    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []
    
    def activation_hook(index, activations, hook, add : bool = True):
        """Hook to add/subtract activations to/from the activation difference matrix
        Args:
            index ([type]): forward index of the node
            activations ([type]): activations to add
            hook ([type]): hook (unused)
            add (bool, optional): whether to add or subtract. Defaults to True."""
        acts = activations.detach() if detach else activations
        if not add:
            acts = -acts
        try:
            activation_difference[:, :, index] += acts
        except RuntimeError as e:
            raise RuntimeError(f"Activation hook error at {hook.name}: {e}") from e
    
    def gradient_hook(lengths, aggregation:str, abs_per_pos:bool, fwd_index: Union[slice, int], gradients:torch.Tensor, hook):
        """Takes in a gradient and uses it and activation_difference 
        to compute an update to the score matrix"""

        grads = gradients.detach()
        try:
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)
            fwd_diff = activation_difference[:, :, fwd_index]
            if fwd_diff.ndim == 3:
                fwd_diff = fwd_diff.unsqueeze(2)

            if fwd_diff.shape[2] != grads.shape[2]:
                raise ValueError(f"fwd_diff shape {fwd_diff.shape} does not match grads shape {grads.shape} for fwd_index {fwd_index}")
            score_per_pos = einsum(fwd_diff, grads,'batch pos forward hidden, batch pos forward hidden -> batch pos forward')
            temp = None
            if abs_per_pos:
                score_per_pos = score_per_pos.abs()
            if aggregation == 'sum':
                s = score_per_pos.sum(dim=(0, 1))
            elif aggregation == 'pos_mean':
                # average over positions for each example, then sum over batch
                # shape is [batch, pos, forward]
                temp = score_per_pos.sum(dim=1)  # now [batch, forward]
                temp = temp / lengths.view(-1, 1)
                s = temp.sum(dim=0)  # [forward]
            else:
                raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')
                                    
            if scores[fwd_index].ndim < s.ndim:
                s = s.squeeze(-1)  # make sure s is the same shape as scores[fwd_index]
            scores[fwd_index] += s

            if save_scores_per_node:
                if use_clean_id_names:
                    assert clean_id_names is not None, "clean_id_names is required when save_scores_per_node is True"
                    assert len(clean_id_names) == batch_size, "clean_id_names must be the same length as batch_size"
                # Create scores per example for each node because temp is not always defined
                scores_per_example = temp.clone() if temp is not None else score_per_pos.clone().sum(dim=1)  # [batch, forward]
                
                # Determine which node each column of scores_per_example corresponds to
                if isinstance(fwd_index, int):
                    node_indices = [fwd_index]
                elif isinstance(fwd_index, slice):
                    node_indices = list(range(fwd_index.start or 0, fwd_index.stop, fwd_index.step or 1))
                elif hasattr(fwd_index, '__iter__'):
                    node_indices = list(fwd_index)
                else:
                    raise ValueError(f"Unsupported fwd_index type: {type(fwd_index)}")

                assert len(node_indices) == scores_per_example.shape[1], (
                    f"Number of node_indices ({len(node_indices)}) doesn't match number of columns ({scores_per_example.shape[1]})"
                )

                # Save batch scores before s aggregation (in temp).
                # Remember it's [batch, forward] and we want to create a list of scores for all examples
                for i, curr_fwd_index in enumerate(node_indices):
                    node_name = node_forward_index_to_name[curr_fwd_index]
                    for batch_idx in range(scores_per_example.shape[0]):
                        example_idx = items_until_now + batch_idx
                        example_name = f"example_{example_idx}" if not use_clean_id_names else clean_id_names[batch_idx]
                        score = scores_per_example[batch_idx, i].item()
                        scores_nodes_all_examples[node_name][example_name] += score  # sum over steps

            if scores_examples_np_arr is not None and indicies_examples_torch_arr is not None:
                scores_per_example = (temp.clone() if temp is not None else score_per_pos.clone().sum(dim=1)).cpu().numpy()  # [batch, forward]
                examples_idx = indicies_examples_torch_arr.cpu().numpy()

                if isinstance(fwd_index, int):
                    # squeeze because for single node, shape is [batch, 1]
                    scores_examples_np_arr[examples_idx, fwd_index] += scores_per_example.squeeze(1)
                elif isinstance(fwd_index, slice):

                    cols = np.arange(fwd_index.start or 0, fwd_index.stop, fwd_index.step or 1)
                    assert scores_per_example.shape[1] == len(cols), (
                    f"Neuron slice mismatch: got {scores_per_example.shape[1]} scores but {len(cols)} cols for {fwd_index}")

                    scores_examples_np_arr[np.ix_(examples_idx, cols)] += scores_per_example
                else:
                    raise ValueError(f"Unsupported fwd_index type: {type(fwd_index)}")


        except RuntimeError as e:
            raise RuntimeError(f"Gradient hook error at {hook.name}: {e}") from e

    for name, node in graph.nodes.items():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            else:
                processed_attn_layers.add(node.layer)

        # exclude logits from forward and backward
        if not isinstance(node, LogitNode):
            fwd_index = graph.forward_index(node)
            fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
            fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False))) #(Corrupted - Clean)
            bwd_hooks.append((node.out_hook, partial(gradient_hook, lengths, aggregation, abs_per_pos, fwd_index)))
       
            
    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference



def get_scores_eap_nodes(model: HookedESM3 | HookedESMC, graph: Graph,
                         dataloader:DataLoader, metric: Callable[[Tensor], Tensor], device, quiet=False,
                            aggregation='sum', abs_per_pos=False, are_clean_logits_needed=False, all_examples_scores_csv_path:Path=None, 
                            use_clean_id_names:bool=False, all_examples_scores_npy_path:str=None):
    
    scores = torch.zeros((graph.n_forward), device= device, dtype=model.cfg.dtype)
    total_items = 0
    total_examples = len(dataloader.dataset)  
    dataloader = dataloader if quiet else tqdm(dataloader)
    scores_nodes_all_examples = defaultdict(lambda: defaultdict(float))
    node_forward_index_to_name = {graph.forward_index(node, attn_slice=False): node.name for node in graph.nodes.values() if not isinstance(node, LogitNode)}

    scores_examples_np_arr = None
    idx_to_example_name_list = None
    if all_examples_scores_npy_path is not None:
        scores_examples_np_arr = np.zeros((total_examples, graph.n_forward), dtype=np.float32)
        idx_to_example_name_list = []

    for clean, corrupted , masked_positions, labels, clean_id_names in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask_clean, n_pos , lengths_clean= tokenize_plus(model, clean) 
        corrupted_tokens, attention_mask_corrupted, _ , _= tokenize_plus(model, corrupted) 

        indicies_examples_torch_arr = None
        if all_examples_scores_npy_path is not None:
            length_before = len(idx_to_example_name_list)
            idx_to_example_name_list.extend(clean_id_names)
            length_after = len(idx_to_example_name_list)
            indicies_examples_torch_arr = torch.arange(length_before, length_after)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices_nodes(model, graph, batch_size, n_pos, scores, device, True, aggregation, abs_per_pos, 
        lengths_clean.to(device), scores_nodes_all_examples, node_forward_index_to_name, total_items - batch_size, save_scores_per_node=all_examples_scores_csv_path is not None,
        clean_id_names=clean_id_names, use_clean_id_names=use_clean_id_names, scores_examples_np_arr=scores_examples_np_arr, indicies_examples_torch_arr=indicies_examples_torch_arr)
        
        with torch.no_grad():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model.forward(
                        sequence_tokens=corrupted_tokens.to(device),
                        sequence_id=attention_mask_corrupted.to(device)
                    )      
            if are_clean_logits_needed:
                clean_logits = model.forward(
                        sequence_tokens=clean_tokens.to(device),
                        sequence_id=attention_mask_clean.to(device)
                    )
                clean_logits= clean_logits.detach()
            else:
                clean_logits = None
            
        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model.forward(
                    sequence_tokens=clean_tokens.to(device),
                    sequence_id=attention_mask_clean.to(device)
                )

            masked_positions_tensor = torch.tensor(masked_positions, device=device)
            labels_tensor = torch.tensor(labels, device=device)
            metric_value = metric(logits, clean_logits, masked_positions_tensor, labels_tensor) 
            metric_value.backward()
            model.zero_grad(set_to_none=True)  # clear the gradients

    scores /= total_items #averaging over all scores

    if all_examples_scores_csv_path is not None:
        try:
            node_names = sorted(scores_nodes_all_examples.keys())
            example_names = sorted({ex for node in scores_nodes_all_examples.values() for ex in node})

            data = []
            for node_name in node_names:
                row = [scores_nodes_all_examples[node_name][ex] for ex in example_names]
                data.append(row)
            df = pd.DataFrame(data, index=node_names, columns=example_names)
            df.to_csv(all_examples_scores_csv_path)
        except Exception as e:
            warnings.warn(f"Failed to save CSV to {all_examples_scores_csv_path}: {e}")

    if all_examples_scores_npy_path is not None:
        save_path = Path(all_examples_scores_npy_path)
        np.savez_compressed(save_path,
         scores=scores_examples_np_arr,
         example_row_names=np.array(idx_to_example_name_list))

    return scores


def get_scores_eap_ig_nodes(model: HookedESM3 | HookedESMC, graph: Graph, dataloader: DataLoader,
                            metric: Callable[[Tensor], Tensor], device, steps=30, quiet=False, aggregation='sum', abs_per_pos=False, all_examples_scores_csv_path:Path=None, 
                            use_clean_id_names:bool=False, all_examples_scores_npy_path:str=None):

    scores = torch.zeros((graph.n_forward), device= device, dtype=model.cfg.dtype)
    total_items = 0
    total_examples = len(dataloader.dataset)  
    dataloader = dataloader if quiet else tqdm(dataloader)
    scores_nodes_all_examples = defaultdict(lambda: defaultdict(float))
    node_forward_index_to_name = {graph.forward_index(node, attn_slice=False): node.name for node in graph.nodes.values() if not isinstance(node, LogitNode)}

    scores_examples_np_arr = None
    idx_to_example_name_list = None
    if all_examples_scores_npy_path is not None:
        scores_examples_np_arr = np.zeros((total_examples, graph.n_forward), dtype=np.float32)
        idx_to_example_name_list = []

    for clean, corrupted , masked_positions, labels, clean_id_names in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens, attention_mask_clean, n_pos , lengths_clean= tokenize_plus(model, clean) 
        corrupted_tokens, attention_mask_corrupted, _ , _= tokenize_plus(model, corrupted)

        indicies_examples_torch_arr = None
        if all_examples_scores_npy_path is not None:
            length_before = len(idx_to_example_name_list)
            idx_to_example_name_list.extend(clean_id_names)
            length_after = len(idx_to_example_name_list)
            indicies_examples_torch_arr = torch.arange(length_before, length_after)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices_nodes(model, graph, batch_size, n_pos, scores, device, True, aggregation, abs_per_pos, lengths_clean.to(device), scores_nodes_all_examples, 
        node_forward_index_to_name, total_items - batch_size, save_scores_per_node=all_examples_scores_csv_path is not None, 
        clean_id_names=clean_id_names, use_clean_id_names=use_clean_id_names, scores_examples_np_arr=scores_examples_np_arr, indicies_examples_torch_arr=indicies_examples_torch_arr)
        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model.forward(
                        sequence_tokens=corrupted_tokens.to(device),
                        sequence_id=attention_mask_corrupted.to(device)
                    )

            input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()
       
            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model.forward(
                        sequence_tokens=clean_tokens.to(device),
                        sequence_id=attention_mask_clean.to(device)
                    )
                clean_logits= clean_logits.detach()

        input_activations_clean = input_activations_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes['input'])]

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (input_activations_clean - input_activations_corrupted) + activations * 0
                return new_input
            return hook_fn

        total_steps = 0
        for step in range(1, steps+1):
            total_steps += 1
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))], bwd_hooks=bwd_hooks):
                logits = model.forward(
                    sequence_tokens=clean_tokens.to(device),
                    sequence_id=attention_mask_clean.to(device)
                )
                masked_positions_tensor = torch.tensor(masked_positions, device=device)
                labels_tensor = torch.tensor(labels, device=device)
                metric_value = metric(logits, clean_logits, masked_positions_tensor, labels_tensor)
                metric_value.backward()
                model.zero_grad(set_to_none=True)  # clear the gradients

    assert total_steps == steps, f"total_steps {total_steps} is not equal to steps {steps}"
    scores /= total_items
    scores /= total_steps

 
    if all_examples_scores_csv_path is not None:
        try:
            node_names = sorted(scores_nodes_all_examples.keys())
            example_names = sorted({ex for node in scores_nodes_all_examples.values() for ex in node})

            data = []
            for node_name in node_names:
                row = [scores_nodes_all_examples[node_name][ex] for ex in example_names]
                data.append(row)
            df = pd.DataFrame(data, index=node_names, columns=example_names)
            df = df / total_steps  # <-- divide all values to get the average
            df.to_csv(all_examples_scores_csv_path)
        except Exception as e:
            warnings.warn(f"Failed to save CSV to {all_examples_scores_csv_path}: {e}")

    if all_examples_scores_npy_path is not None:
        save_path = Path(all_examples_scores_npy_path)
        scores_examples_np_arr = scores_examples_np_arr / total_steps
        np.savez_compressed(save_path,
         scores=scores_examples_np_arr,
         example_row_names=np.array(idx_to_example_name_list, dtype=object))

    return scores


def create_masks(
        tokenized_inputs: torch.Tensor,
        repeat_locations_list: List[List[List[int]]],
        sequences_list:  List[str],
        alignments_list: List[List[str]],
        tokenizer):
    """
    Creates several boolean masks for sequence analysis, returned as stacked tensors:
    - repeat_positions_masks: (batch, seq_len) mask for all repeat positions
    - identical_repeat_positions_masks: (batch, seq_len) mask for identical repeat positions
    - non_identical_repeat_positions_masks: (batch, seq_len) mask for non-identical repeat positions
    - non_repeat_positions_masks: (batch, seq_len) mask for positions that are not repeats
    - masked_positions_masks: (batch, seq_len) mask for externally masked positions
    """

    repeat_positions_masks = []
    identical_repeat_positions_masks = []
    non_identical_repeat_positions_masks = []
    non_repeat_positions_masks = []
    masked_positions_masks = []
    all_tokens_masks = []

    for i, (seq, repeat_locations, alignments) in enumerate(zip(sequences_list, repeat_locations_list, alignments_list)):
        result_abs_pos_to_repeat_info, _ = analyze_repeat_positions(
            protein_seq=seq,
            repeat_locations=repeat_locations,
            alignments=alignments)

        tokenized_input = tokenized_inputs[i]
        repeat_positions = []
        identical_repeat_positions = []
        non_identical_repeat_positions = []

        for j, (start, end) in enumerate(repeat_locations):
            for pos in range(start, end + 1):
                idx = pos + 1  # +1: tokenizer starts at 1
                repeat_positions.append(idx)
                info = result_abs_pos_to_repeat_info[j][pos]
                if info.is_aligned_matching_identical:
                    identical_repeat_positions.append(idx)
                elif info.aligned_matching_absolute_position_in_sequence is not None:
                    non_identical_repeat_positions.append(idx)

        repeat_positions_mask = torch.zeros(len(tokenized_input), dtype=torch.bool)
        if repeat_positions:
            repeat_positions_mask[repeat_positions] = True
        repeat_positions_masks.append(repeat_positions_mask)

        identical_repeat_positions_mask = torch.zeros(len(tokenized_input), dtype=torch.bool)
        if identical_repeat_positions:
            identical_repeat_positions_mask[identical_repeat_positions] = True

        non_identical_repeat_positions_mask = torch.zeros(len(tokenized_input), dtype=torch.bool)
        if non_identical_repeat_positions:
            non_identical_repeat_positions_mask[non_identical_repeat_positions] = True

        non_repeat_positions_mask = torch.ones(len(tokenized_input), dtype=torch.bool)
        if repeat_positions:
            non_repeat_positions_mask[repeat_positions] = False
        non_repeat_positions_masks.append(non_repeat_positions_mask)

        masked_pos = (tokenized_input == tokenizer.mask_token_id).nonzero(as_tuple=True)[0][0].item()
        masked_positions_mask = torch.zeros(len(tokenized_input), dtype=torch.bool)
        masked_positions_mask[masked_pos] = True
        masked_positions_masks.append(masked_positions_mask)

        identical_repeat_positions_mask = identical_repeat_positions_mask & (~masked_positions_mask)
        non_identical_repeat_positions_mask = non_identical_repeat_positions_mask & ( ~masked_positions_mask )

        identical_repeat_positions_masks.append(identical_repeat_positions_mask)
        non_identical_repeat_positions_masks.append(non_identical_repeat_positions_mask)

        all_tokens_mask = torch.ones(len(tokenized_input), dtype=torch.bool)
        all_tokens_masks.append(all_tokens_mask)

    mask_dict = {
        "repeat_positions": torch.stack(repeat_positions_masks, dim=0),
        "identical_repeat_positions": torch.stack(identical_repeat_positions_masks, dim=0),
        "non_identical_repeat_positions": torch.stack(non_identical_repeat_positions_masks, dim=0),
        "non_repeat_positions": torch.stack(non_repeat_positions_masks, dim=0),
        "masked_positions": torch.stack(masked_positions_masks, dim=0),
        "all_tokens": torch.stack(all_tokens_masks, dim=0)
    }

    # Stack to get (batch, seq_len) tensors and
    return mask_dict


def make_hooks_and_matrices_nodes_per_token(model: HookedESM3 | HookedESMC, graph: Graph, batch_size:int, 
                                 n_pos:int, scores_dict: dict, masks_dict: dict, attention_mask: torch.Tensor,
                                 device, detach=True, aggregation='sum', abs_per_pos=False):
    
    activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=device, dtype=model.cfg.dtype) 

    processed_attn_layers = set()
    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []
    
    def activation_hook(index, activations, hook, add : bool = True):
        """Hook to add/subtract activations to/from the activation difference matrix
        Args:
            index ([type]): forward index of the node
            activations ([type]): activations to add
            hook ([type]): hook (unused)
            add (bool, optional): whether to add or subtract. Defaults to True."""
        acts = activations.detach() if detach else activations
        if not add:
            acts = -acts
        try:
            activation_difference[:, :, index] += acts
        except RuntimeError as e:
            raise RuntimeError(f"Activation hook error at {hook.name}: {e}") from e
    
    def gradient_hook(aggregation:str, abs_per_pos:bool, fwd_index: Union[slice, int], gradients:torch.Tensor, hook):
        """Takes in a gradient and uses it and activation_difference 
        to compute an update to the score matrix"""

        grads = gradients.detach()
        try:
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)
            fwd_diff = activation_difference[:, :, fwd_index]
            if fwd_diff.ndim == 3:
                fwd_diff = fwd_diff.unsqueeze(2)

            if fwd_diff.shape[2] != grads.shape[2]:
                raise ValueError(f"fwd_diff shape {fwd_diff.shape} does not match grads shape {grads.shape} for fwd_index {fwd_index}")
            for key, mask in masks_dict.items():
                score_per_pos = einsum(fwd_diff, grads,'batch pos forward hidden, batch pos forward hidden -> batch pos forward') # [batch, pos, forward]
                temp = None
                
                combined_mask = mask & attention_mask
                combined_mask = combined_mask.unsqueeze(2)
                combined_mask = combined_mask.float().to(device)
                score_per_pos = score_per_pos * combined_mask
                if abs_per_pos:
                    score_per_pos = score_per_pos.abs()
                if aggregation == 'sum':
                    s = score_per_pos.sum(dim=(0, 1))
                elif aggregation == 'pos_mean':
                    # average over positions for each example, then sum over batch
                    # shape is [batch, pos, forward]
                    temp = score_per_pos.sum(dim=1)  # now [batch, forward]
                    non_masked_positions = combined_mask.squeeze(-1).sum(dim=-1) # [batch]
                    temp = temp / non_masked_positions.view(-1, 1)
                    s = temp.sum(dim=0)  # [forward]
                else:
                    raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')
                    
                if scores_dict[key][fwd_index].ndim < s.ndim:
                    s = s.squeeze(-1)  # make sure s is the same shape as scores[fwd_index]
                scores_dict[key][fwd_index] += s

        except RuntimeError as e:
            raise RuntimeError(f"Gradient hook error at {hook.name}: {e}") from e

    for name, node in graph.nodes.items():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            else:
                processed_attn_layers.add(node.layer)

        # exclude logits from forward and backward
        if not isinstance(node, LogitNode):
            fwd_index = graph.forward_index(node)
            fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
            fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False))) #(Corrupted - Clean)
            bwd_hooks.append((node.out_hook, partial(gradient_hook, aggregation, abs_per_pos, fwd_index)))
       
            
    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference

        
        
def get_scores_eap_ig_nodes_per_token(model: HookedESM3 | HookedESMC, graph: Graph, dataloader: DataLoader,
                            metric: Callable[[Tensor], Tensor], device, steps=30, quiet=False, aggregation='sum', abs_per_pos=False):

    scores_keys = ['repeat_positions', 'identical_repeat_positions', 'non_identical_repeat_positions', 'non_repeat_positions', 'masked_positions', 'all_tokens']
    scores_dict = {key: torch.zeros((graph.n_forward), device= device, dtype=model.cfg.dtype) for key in scores_keys}
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)


    for clean, corrupted , masked_positions, labels, repeat_locations_list, alignments_list,  sequences_list in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        model_name = model.cfg.model_name
        if model_name is None:
            model_name = "esm3" if isinstance(model, HookedESM3) else "esm-c" if isinstance(model, HookedESMC) else "esm3"
        tokenizer = load_tokenizer_by_model_type(model_name)
        clean_tokens, attention_mask_clean, n_pos , lengths_clean= tokenize_plus(model, clean) 
        corrupted_tokens, attention_mask_corrupted, _ , _= tokenize_plus(model, corrupted)

        masks_dict = create_masks(clean_tokens, repeat_locations_list, sequences_list, alignments_list, tokenizer)
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices_nodes_per_token(model, graph, batch_size, n_pos, scores_dict, masks_dict, attention_mask_clean, device, True, aggregation, abs_per_pos)
        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model.forward(
                        sequence_tokens=corrupted_tokens.to(device),
                        sequence_id=attention_mask_corrupted.to(device)
                    )

            input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()
       
            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model.forward(
                        sequence_tokens=clean_tokens.to(device),
                        sequence_id=attention_mask_clean.to(device)
                    )
                clean_logits= clean_logits.detach()

        input_activations_clean = input_activations_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes['input'])]

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (input_activations_clean - input_activations_corrupted) + activations * 0
                return new_input
            return hook_fn

        total_steps = 0
        for step in range(1, steps+1):
            total_steps += 1
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))], bwd_hooks=bwd_hooks):
                logits = model.forward(
                    sequence_tokens=clean_tokens.to(device),
                    sequence_id=attention_mask_clean.to(device)
                )
                masked_positions_tensor = torch.tensor(masked_positions, device=device)
                labels_tensor = torch.tensor(labels, device=device)
                metric_value = metric(logits, clean_logits, masked_positions_tensor, labels_tensor)
                metric_value.backward()
                model.zero_grad(set_to_none=True)  # clear the gradients

    assert total_steps == steps, f"total_steps {total_steps} is not equal to steps {steps}"
    for key in scores_keys:
        scores_dict[key] /= total_items
        scores_dict[key] /= total_steps
    return scores_dict
     

def attribute_nodes_per_token(model: HookedESM3 | HookedESMC, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], device, aggregation='sum',
 method: Union[Literal['EAP-IG']]="EAP-IG", quiet=False, abs_per_pos=False, eap_ig_steps=None, scores_csv_path:str=None):
    
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')

    if method == 'EAP-IG':
        scores_dict = get_scores_eap_ig_nodes_per_token(model, graph, dataloader, metric, device, steps=eap_ig_steps, quiet=quiet, aggregation=aggregation, abs_per_pos=abs_per_pos)
    else:
        raise ValueError(f"integrated_gradients must be in ['EAP'], but got {method}")
    
    for key in scores_dict.keys():
        scores_dict[key] = scores_dict[key].cpu().numpy()  # this will detach the scores

    if scores_csv_path is not None:
        out_dir = os.path.dirname(scores_csv_path)
        if out_dir:  # Avoid trying to create '' (empty string) as a directory
            os.makedirs(out_dir, exist_ok=True)
    rows = []
    for node in tqdm(graph.nodes.values(), total=len(graph.nodes)):
        if isinstance(node, LogitNode):
            continue  # skip logits nodes cause it doesnt have gradient.
        row = [node.name] + [scores_dict[key][graph.forward_index(node, attn_slice=False)] for key in scores_dict.keys()]
        rows.append(row)
    columns = ['node_name'] + list(scores_dict.keys())
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(scores_csv_path, index=False)



def make_hooks_and_matrices_nodes_and_neurons(model: HookedESM3 | HookedESMC, graph: NeuronGraph, batch_size:int, 
                                 n_pos:int, scores_neurons: torch.Tensor, device, detach=True, aggregation='sum', abs_per_pos=False, lengths=None,
                                 scores_examples_np_arr_neurons:np.ndarray=None, indicies_examples_torch_arr:torch.Tensor=None, 
                                 mean_ablations_cache: Optional[dict]=None, masked_positions: Optional[List[List[int]]]=None, 
                                 separate_mask_positions: bool=False):
    

    #scores_neurons is [n.forward_neurons]

    # Initialize neurons_activations_difference - either with zeros or mean ablations
    if mean_ablations_cache is not None:
        # Use mean ablations as initialization (clean hooks will subtract from this)
        # Result: mean_ablations - clean (analogous to corrupted - clean)
        if separate_mask_positions and masked_positions is not None:
            # Initialize with 'exclude_mask' mode (non-masked positions)
            neurons_activations_difference = initialize_activation_difference_with_mean_ablations(
                batch_size=batch_size,
                n_pos=n_pos,
                graph=graph,
                model=model,
                device=device,
                mean_ablations_cache=mean_ablations_cache,
                is_neurons=True,
                use_mode='exclude_mask'
            )
            # Overwrite masked positions with 'only_mask' mode
            overwrite_masked_positions_with_mean_ablations(
                activation_diff=neurons_activations_difference,
                batch_size=batch_size,
                masked_positions=masked_positions,
                graph=graph,
                model=model,
                device=device,
                mean_ablations_cache=mean_ablations_cache,
                is_neurons=True,
                use_mode='only_mask'
            )
        else:
            # Use 'all' mode for all positions
            neurons_activations_difference = initialize_activation_difference_with_mean_ablations(
                batch_size=batch_size,
                n_pos=n_pos,
                graph=graph,
                model=model,
                device=device,
                mean_ablations_cache=mean_ablations_cache,
                is_neurons=True,
                use_mode='all'
            )
    else:
        neurons_activations_difference = torch.zeros((batch_size, n_pos, graph.n_forward_neurons), device=device, dtype=model.cfg.dtype)

    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []
    
    def activation_hook(index, activations, hook, add : bool = True):
        """Hook to add/subtract activations to/from the activation difference matrix
        Args:
            index ([type]): forward index of the node
            activations ([type]): activations to add
            hook ([type]): hook (unused)
            add (bool, optional): whether to add or subtract. Defaults to True."""
        acts = activations.detach() if detach else activations #acts are [batch, pos, n_neurons_in_layer]
        if not add:
            acts = -acts
        try:
            neurons_activations_difference[:, :, index] += acts #[batch, pos, n_neurons_in_layer]
        except RuntimeError as e:
            raise RuntimeError(f"Activation hook error at {hook.name}: {e}") from e
    
    def gradient_hook(lengths, aggregation:str, abs_per_pos:bool, fwd_index: Union[slice, int], gradients:torch.Tensor, hook):
        """Takes in a gradient and uses it and activation_difference 
        to compute an update to the score matrix"""

        grads = gradients.detach()
        try:
            target_activations_diff = neurons_activations_difference[:, :, fwd_index] #[batch, pos, n_neurons_in_layer]
            if grads.ndim != 3 or target_activations_diff.ndim != 3:
                raise ValueError(f"grads shape {grads.shape} or target_activations_diff shape {target_activations_diff.shape} is not 3")
            if grads.shape[2] != target_activations_diff.shape[2]:
                raise ValueError(f"Mismatch between grads width {grads.shape[2]} and target_activations_diff width {target_activations_diff.shape[2]} for fwd_index {fwd_index}")


            score_per_pos = target_activations_diff * grads #[batch, pos, n_neurons_in_layer]

            if abs_per_pos:
                score_per_pos = score_per_pos.abs()
            
            if aggregation == 'sum':
                s = score_per_pos.sum(dim = (0,1)) #[n_neurons_in_layer]
                temp = None
            elif aggregation == 'pos_mean':
                # average over positions for each example, then sum over batch
                # shape is [batch, pos, n_neurons_in_layer]
                temp = score_per_pos.sum(dim=1)  # now [batch, n_neurons_in_layer]

                temp = temp / lengths.view(-1, 1)
                s = temp.sum(dim=0)  # [n_neurons_in_layer]
            else:
                raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')

            scores_neurons[fwd_index] += s

            if scores_examples_np_arr_neurons is not None and indicies_examples_torch_arr is not None:
                scores_per_example = (temp.clone() if temp is not None else score_per_pos.clone().sum(dim=1)).cpu().numpy()  # [batch, n_neurons_in_layer]
                examples_idx = indicies_examples_torch_arr.cpu().numpy()

                if isinstance(fwd_index, int):
                    raise ValueError(f"fwd_index for neurons should be a slice, but got int: {fwd_index}. Something has changed.")
                elif isinstance(fwd_index, slice):
                    cols = np.arange(fwd_index.start or 0, fwd_index.stop, fwd_index.step or 1)
                    assert scores_per_example.shape[1] == len(cols), (
                    f"Neuron slice mismatch: got {scores_per_example.shape[1]} scores but {len(cols)} cols for {fwd_index}")
                    scores_examples_np_arr_neurons[np.ix_(examples_idx, cols)] += scores_per_example
                else:
                    raise ValueError(f"Unsupported fwd_index type: {type(fwd_index)}")

        except RuntimeError as e:
            raise RuntimeError(f"Gradient hook error at {hook.name}: {e}") from e

    for layer in range(graph.cfg['n_layers']):

        requested_node = graph.nodes[f"m{layer}"]
        if not isinstance(requested_node, MLPWithNeuronNode):
            raise ValueError(f"requested_node {requested_node} is not a MLPWithNeuronNode")

        fwd_index = graph.forward_index(requested_node, attn_slice=False, return_index_in_neurons_array=True)
        fwd_hooks_corrupted.append((requested_node.neurons_out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((requested_node.neurons_out_hook, partial(activation_hook, fwd_index, add=False))) #(Corrupted - Clean)
        bwd_hooks.append((requested_node.neurons_out_hook, partial(gradient_hook, lengths, aggregation, abs_per_pos, fwd_index)))
       


    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), neurons_activations_difference




def get_scores_eap_neurons(model: HookedESM3 | HookedESMC, graph: NeuronGraph,
                        dataloader:DataLoader, metric: Callable[[Tensor], Tensor], device, quiet=False,
                        aggregation='sum', abs_per_pos=False, are_clean_logits_needed=False, all_examples_scores_npy_path:str=None, 
                        mean_ablations_cache: Optional[dict]=None, separate_mask_positions: bool=False):

    scores = torch.zeros((graph.n_forward), device= device, dtype=model.cfg.dtype)
    scores_neurons = torch.zeros((graph.n_forward_neurons), device= device, dtype=model.cfg.dtype)

    total_items = 0
    total_examples = len(dataloader.dataset)  
    dataloader = dataloader if quiet else tqdm(dataloader)
   
    scores_examples_np_arr = None
    scores_examples_np_arr_neurons = None
    idx_to_example_name_list = None
    if all_examples_scores_npy_path is not None:
        scores_examples_np_arr = np.zeros((total_examples, graph.n_forward), dtype=np.float32)
        scores_examples_np_arr_neurons = np.zeros((total_examples, graph.n_forward_neurons), dtype=np.float32)
        idx_to_example_name_list = []

    for clean, corrupted , masked_positions, labels, clean_id_names in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask_clean, n_pos , lengths_clean= tokenize_plus(model, clean) 
        corrupted_tokens, attention_mask_corrupted, _ , _= tokenize_plus(model, corrupted) 
        
        indicies_examples_torch_arr = None
        if all_examples_scores_npy_path is not None:
            length_before = len(idx_to_example_name_list)
            idx_to_example_name_list.extend(clean_id_names)
            length_after = len(idx_to_example_name_list)
            indicies_examples_torch_arr = torch.arange(length_before, length_after)
        
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices_nodes(model, graph, batch_size, n_pos, 
        scores, device, True, aggregation,abs_per_pos, lengths_clean.to(device), scores_examples_np_arr=scores_examples_np_arr, 
        indicies_examples_torch_arr=indicies_examples_torch_arr, mean_ablations_cache=mean_ablations_cache, 
        masked_positions=masked_positions, separate_mask_positions=separate_mask_positions)

        (fwd_hooks_corrupted_neurons, fwd_hooks_clean_neurons, bwd_hooks_neurons), neurons_activations_difference = make_hooks_and_matrices_nodes_and_neurons(model, graph, batch_size, n_pos, scores_neurons,
         device, True, aggregation,abs_per_pos, lengths_clean.to(device), scores_examples_np_arr_neurons=scores_examples_np_arr_neurons,
         indicies_examples_torch_arr=indicies_examples_torch_arr, mean_ablations_cache=mean_ablations_cache, 
         masked_positions=masked_positions, separate_mask_positions=separate_mask_positions)
        
        with torch.no_grad():
            if mean_ablations_cache is None:
                with model.hooks(fwd_hooks=(fwd_hooks_corrupted + fwd_hooks_corrupted_neurons)):
                    _ = model.forward(
                            sequence_tokens=corrupted_tokens.to(device),
                            sequence_id=attention_mask_corrupted.to(device)
                        )
            else:
                pass
            if are_clean_logits_needed:
                clean_logits = model.forward(
                        sequence_tokens=clean_tokens.to(device),
                        sequence_id=attention_mask_clean.to(device)
                    )
                clean_logits= clean_logits.detach()
            else:
                clean_logits = None
            
        with model.hooks(fwd_hooks=(fwd_hooks_clean + fwd_hooks_clean_neurons), bwd_hooks=(bwd_hooks + bwd_hooks_neurons)):
            logits = model.forward(
                    sequence_tokens=clean_tokens.to(device),
                    sequence_id=attention_mask_clean.to(device)
                )

            masked_positions_tensor = torch.tensor(masked_positions, device=device)
            labels_tensor = torch.tensor(labels, device=device)
            metric_value = metric(logits, clean_logits, masked_positions_tensor, labels_tensor) 
            metric_value.backward()
            model.zero_grad(set_to_none=True)  # clear the gradients

    scores /= total_items #averaging over all scores
    scores_neurons /= total_items #averaging over all scores
    
    if all_examples_scores_npy_path is not None:
        save_path = Path(all_examples_scores_npy_path)
        np.savez_compressed(save_path,
         scores_nodes=scores_examples_np_arr,
         scores_neurons=scores_examples_np_arr_neurons,
         example_row_names=np.array(idx_to_example_name_list))

    return scores, scores_neurons



def get_scores_eap_ig_neurons(model: HookedESM3 | HookedESMC, graph: NeuronGraph, dataloader: DataLoader,
                            metric: Callable[[Tensor], Tensor], device, steps=30, quiet=False, aggregation='sum', abs_per_pos=False,
                            all_examples_scores_npy_path:str=None, mean_ablations_cache: Optional[dict]=None, separate_mask_positions: bool=False):

    scores = torch.zeros((graph.n_forward), device= device, dtype=model.cfg.dtype)
    scores_neurons = torch.zeros((graph.n_forward_neurons), device= device, dtype=model.cfg.dtype)
    total_items = 0
    total_examples = len(dataloader.dataset)  
    dataloader = dataloader if quiet else tqdm(dataloader)
    
    scores_examples_np_arr = None
    scores_examples_np_arr_neurons = None
    idx_to_example_name_list = None
    if all_examples_scores_npy_path is not None:
        scores_examples_np_arr = np.zeros((total_examples, graph.n_forward), dtype=np.float32)
        scores_examples_np_arr_neurons = np.zeros((total_examples, graph.n_forward_neurons), dtype=np.float32)
        idx_to_example_name_list = []

    for clean, corrupted , masked_positions, labels, clean_id_names in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens, attention_mask_clean, n_pos , lengths_clean= tokenize_plus(model, clean) 
        corrupted_tokens, attention_mask_corrupted, _ , _= tokenize_plus(model, corrupted)
    
        indicies_examples_torch_arr = None
        if all_examples_scores_npy_path is not None:
            length_before = len(idx_to_example_name_list)
            idx_to_example_name_list.extend(clean_id_names)
            length_after = len(idx_to_example_name_list)
            indicies_examples_torch_arr = torch.arange(length_before, length_after)
    
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices_nodes(model, graph, batch_size, n_pos, scores,
         device, True, aggregation, abs_per_pos, lengths_clean.to(device), scores_examples_np_arr=scores_examples_np_arr,
         indicies_examples_torch_arr=indicies_examples_torch_arr, mean_ablations_cache=mean_ablations_cache, 
         masked_positions=masked_positions, separate_mask_positions=separate_mask_positions)

        (fwd_hooks_corrupted_neurons, fwd_hooks_clean_neurons, bwd_hooks_neurons), neurons_activations_difference = make_hooks_and_matrices_nodes_and_neurons(model, graph, batch_size, n_pos, scores_neurons,
         device, True, aggregation,abs_per_pos, lengths_clean.to(device), scores_examples_np_arr_neurons=scores_examples_np_arr_neurons,
         indicies_examples_torch_arr=indicies_examples_torch_arr, mean_ablations_cache=mean_ablations_cache, 
         masked_positions=masked_positions, separate_mask_positions=separate_mask_positions)
        
        with torch.inference_mode():
            if mean_ablations_cache is None:
                with model.hooks(fwd_hooks=(fwd_hooks_corrupted + fwd_hooks_corrupted_neurons)):
                    _ = model.forward(
                            sequence_tokens=corrupted_tokens.to(device),
                            sequence_id=attention_mask_corrupted.to(device)
                        )
            else:
                pass

            input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()
       
            with model.hooks(fwd_hooks=(fwd_hooks_clean + fwd_hooks_clean_neurons)):
                clean_logits = model.forward(
                        sequence_tokens=clean_tokens.to(device),
                        sequence_id=attention_mask_clean.to(device)
                    )
                clean_logits= clean_logits.detach()

        input_activations_clean = input_activations_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes['input'])]

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (input_activations_clean - input_activations_corrupted) + activations * 0
                return new_input
            return hook_fn

        total_steps = 0
        for step in range(1, steps+1):
            total_steps += 1
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))], bwd_hooks=(bwd_hooks + bwd_hooks_neurons)):
                logits = model.forward(
                    sequence_tokens=clean_tokens.to(device),
                    sequence_id=attention_mask_clean.to(device)
                )
                masked_positions_tensor = torch.tensor(masked_positions, device=device)
                labels_tensor = torch.tensor(labels, device=device)
                metric_value = metric(logits, clean_logits, masked_positions_tensor, labels_tensor)
                metric_value.backward()
                model.zero_grad(set_to_none=True)  # clear the gradients

    assert total_steps == steps, f"total_steps {total_steps} is not equal to steps {steps}"
    scores /= total_items
    scores /= total_steps

 
    scores_neurons /= total_items
    scores_neurons /= total_steps
    
    if all_examples_scores_npy_path is not None:
        save_path = Path(all_examples_scores_npy_path)
        scores_examples_np_arr = scores_examples_np_arr / total_steps
        scores_examples_np_arr_neurons = scores_examples_np_arr_neurons / total_steps
        np.savez_compressed(save_path,
         scores_nodes=scores_examples_np_arr,
         scores_neurons=scores_examples_np_arr_neurons,
         example_row_names=np.array(idx_to_example_name_list, dtype=object))

    return scores, scores_neurons



def attribute_neurons(model: HookedESM3 | HookedESMC, graph: NeuronGraph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], device, aggregation='sum',
 method: Union[Literal['EAP', 'EAP-IG']]="EAP", quiet=False, abs_per_pos=False, are_clean_logits_needed=False, eap_ig_steps=None,
 all_examples_scores_npy_path:str=None, mean_ablations_cache: Optional[dict]=None, separate_mask_positions: bool=False):
    
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')

    if method == 'EAP':
        scores, scores_neurons = get_scores_eap_neurons(model=model, graph=graph, dataloader=dataloader, metric=metric, device=device, quiet=quiet, aggregation=aggregation,
        abs_per_pos=abs_per_pos, are_clean_logits_needed=are_clean_logits_needed, all_examples_scores_npy_path=all_examples_scores_npy_path, 
        mean_ablations_cache=mean_ablations_cache, separate_mask_positions=separate_mask_positions)
    elif method == 'EAP-IG':
        scores, scores_neurons = get_scores_eap_ig_neurons(model=model, graph=graph, dataloader=dataloader, metric=metric, device=device, steps=eap_ig_steps, quiet=quiet, aggregation=aggregation, abs_per_pos=abs_per_pos,
        all_examples_scores_npy_path=all_examples_scores_npy_path, mean_ablations_cache=mean_ablations_cache, separate_mask_positions=separate_mask_positions)
    else:
        raise ValueError(f"integrated_gradients must be in ['EAP'], but got {method}")
    

    scores = scores.cpu().numpy()  # this will detach the scores
    scores_neurons = scores_neurons.cpu().numpy()  # this will detach the scores

    for node in tqdm(graph.nodes.values(), total=len(graph.nodes)):
        if isinstance(node, LogitNode):
            continue  # skip logits nodes cause it doesnt have gradient.
        node.score = scores[graph.forward_index(node, attn_slice=False)]
        if isinstance(node, MLPWithNeuronNode):
            neurons_slice = graph.forward_index(node, attn_slice=False, return_index_in_neurons_array=True)
            node.neurons_scores = torch.tensor(scores_neurons[neurons_slice])
