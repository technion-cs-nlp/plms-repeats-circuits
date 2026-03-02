from typing import Callable, List, Union
from functools import partial 
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedESM3, SupportedESM3Config, HookedTransformerConfig
from tqdm import tqdm
from einops import einsum 
from .graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode, Node, Edge
from .attribute import make_hooks_and_matrices, tokenize_plus

def create_hook_args(activation_differences, corrupted_edges_matrix, input_index, prev_index, backward_index):
    # activation_differences: 'batch pos previous hidden'
    # corrupted_edges_vec: 'previous' or 'previous x head'
    relevant_differences = activation_differences[:, :, :prev_index]
    relevant_edges = corrupted_edges_matrix[:prev_index, backward_index]
    activation_differences_no_scaling = relevant_differences[:, :, input_index]
    corrupted_edges_vector_no_scaling = relevant_edges[input_index]
    mask = torch.arange(len(relevant_edges)) != input_index 
    activation_differences_scaling =relevant_differences[:, :, mask]
    corrupted_edges_vector_scaling = relevant_edges[mask]
    return activation_differences_scaling, corrupted_edges_vector_scaling, activation_differences_no_scaling, corrupted_edges_vector_no_scaling


def make_input_construction_hook(
    activation_differences,
    corrupted_edges_matrix,
    input_index,
    prev_index,
    backward_index,
    scaling_factor,
    attn=False):
    # activation differences : for each forward component we store corruped - clean differences for all positions . dims: (batch_size, n_pos, n_forward, d_model)
    # in graph_vector_need_scaling: a vector of (prev_components_num, 1)/ or (prev_components_num,) For logits, MLP, a matrix (prev_components_num, num heads) for attn. 
    # Each cell i,j indicates if there is an corrupted edge from i to j (means not in the circuit). if there is an edge- remain with clean. if there is no edge we need to add the difference corrupted-clean for that edge.
    # difference should be scaled 
    # in graph_vector_no_scaling: : a vector of (prev_components_num, 1)/ or (prev_components_num,) For logits, MLP, a matrix (prev_components_num, num heads) for attn. 
    # Each cell i,j indicates if there isrrupted edge from i to j (means not in the circuit). if there is an edge- remain with clean. if there is no edge we need to add the difference corrupted-clean for that edge.
    # difference should not be scaled 
    #scaling_factor :float
    #attn: indicates if we need to perform the calculation for attention heads or not 
    def input_construction_hook(activations, hook):
        activation_differences_scaling, corrupted_edges_vector_scaling, activation_differences_no_scaling, corrupted_edges_vector_no_scaling = create_hook_args(activation_differences=activation_differences, corrupted_edges_matrix=corrupted_edges_matrix, input_index=input_index, prev_index=prev_index, backward_index=backward_index)
        update = torch.zeros_like(activations)
        if attn:
            if activation_differences_scaling.numel() !=0:
                update += einsum(activation_differences_scaling, corrupted_edges_vector_scaling,'batch pos m hidden, m head -> batch pos head hidden') / scaling_factor
            if corrupted_edges_vector_no_scaling.numel() !=0 and corrupted_edges_vector_no_scaling.sum() > 0 :
                if len(activation_differences_no_scaling.shape) == 3:
                    activation_differences_no_scaling=activation_differences_no_scaling.unsqueeze(-2)
                if len(corrupted_edges_vector_no_scaling.shape) == 1:
                    corrupted_edges_vector_no_scaling= corrupted_edges_vector_no_scaling.unsqueeze(0)
                input_contribution = einsum(activation_differences_no_scaling, corrupted_edges_vector_no_scaling,'batch pos k hidden, k head -> batch pos head hidden')
                update += input_contribution
        else:
            if activation_differences_scaling.numel() !=0:
                update += einsum(activation_differences_scaling, corrupted_edges_vector_scaling,'batch pos m hidden, m -> batch pos hidden')/scaling_factor
            if corrupted_edges_vector_no_scaling.numel() !=0 and corrupted_edges_vector_no_scaling.sum() > 0 :
                update += activation_differences_no_scaling

        activations += update
        return activations
    return input_construction_hook

def make_input_construction_hook_no_scaling(
    activation_differences,
    corrupted_edges_matrix,
    input_index,
    prev_index,
    backward_index,
    attn=False):
    # activation differences : for each forward component we store corruped - clean differences for all positions . dims: (batch_size, n_pos, n_forward, d_model)
    # in graph_vector_need_scaling: a vector of (prev_components_num, 1)/ or (prev_components_num,) For logits, MLP, a matrix (prev_components_num, num heads) for attn. 
    # Each cell i,j indicates if there is an corrupted edge from i to j (means not in the circuit). if there is an edge- remain with clean. if there is no edge we need to add the difference corrupted-clean for that edge.
    # difference should be scaled 
    # in graph_vector_no_scaling: : a vector of (prev_components_num, 1)/ or (prev_components_num,) For logits, MLP, a matrix (prev_components_num, num heads) for attn. 
    # Each cell i,j indicates if there isrrupted edge from i to j (means not in the circuit). if there is an edge- remain with clean. if there is no edge we need to add the difference corrupted-clean for that edge.
    # difference should not be scaled 
    #scaling_factor :float
    #attn: indicates if we need to perform the calculation for attention heads or not 
    def input_construction_hook(activations, hook):
        relevant_differences = activation_differences[:, :, :prev_index]
        relevant_edges = corrupted_edges_matrix[:prev_index, backward_index]
        if attn:
                update = einsum(relevant_differences, relevant_edges,'batch pos m hidden, m head -> batch pos head hidden')
        else:
                update = einsum(relevant_differences, relevant_edges,'batch pos m hidden, m -> batch pos hidden')
        activations += update
        return activations
    return input_construction_hook

def make_input_construction_hook_wrapper(
    activation_differences,
    corrupted_edges_matrix,
    input_index,
    prev_index,
    backward_index,
    scaling_factor,
    use_scaling,
    attn=False):
    if use_scaling:
        return make_input_construction_hook(activation_differences, corrupted_edges_matrix, input_index, prev_index, backward_index, scaling_factor, attn)
    else:
        return make_input_construction_hook_no_scaling(activation_differences, corrupted_edges_matrix, input_index, prev_index, backward_index, attn)

# we make input construction hooks for every node but InputNodes. 
# We can also skip nodes not in the graph; it doesn't matter what their outputs are, as they will always be corrupted / reconstructed when serving as inputs to other nodes.
 # AttentionNodes have 3 inputs to reconstruct
def make_input_construction_hooks(model, graph,activation_differences, corrupted_edges_matrix, scaling_factor, input_index, calc_input_for_nodes_not_in_graph, use_scaling):
    input_construction_hooks = []
    for layer in range(model.cfg.n_layers):
        # add attention hooks:
        if calc_input_for_nodes_not_in_graph or any(graph.nodes[f'a{layer}.h{head}'].in_graph for head in range(model.cfg.n_heads)):
            for i, letter in enumerate('qkv'):
                node = graph.nodes[f'a{layer}.h0']
                prev_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True) #we want all edges information beween all prev components and current qkv heads 
                hook =  make_input_construction_hook_wrapper(activation_differences=activation_differences, corrupted_edges_matrix=corrupted_edges_matrix, input_index=input_index, prev_index=prev_index, backward_index=bwd_index, scaling_factor=scaling_factor, use_scaling=use_scaling ,attn=True)
                input_construction_hooks.append((node.qkv_inputs[i], hook))
        # add MLP hook
        if calc_input_for_nodes_not_in_graph or graph.nodes[f'm{layer}'].in_graph:
            node = graph.nodes[f'm{layer}']
            prev_index = graph.prev_index(node)
            bwd_index = graph.backward_index(node)
            hook =  make_input_construction_hook_wrapper(activation_differences=activation_differences, corrupted_edges_matrix=corrupted_edges_matrix, input_index=input_index, prev_index=prev_index, backward_index=bwd_index, scaling_factor=scaling_factor, use_scaling=use_scaling, attn=False)
            input_construction_hooks.append((node.in_hook, hook))
    # Logit node input construction
    node = graph.nodes[f'logits']
    if calc_input_for_nodes_not_in_graph or node.in_graph:
        prev_index = graph.prev_index(node)
        bwd_index = graph.backward_index(node)
        hook =  make_input_construction_hook_wrapper(activation_differences=activation_differences, corrupted_edges_matrix=corrupted_edges_matrix, input_index=input_index, prev_index=prev_index, backward_index=bwd_index, scaling_factor=scaling_factor, use_scaling=use_scaling, attn=False)
        input_construction_hooks.append((node.in_hook, hook))
    return input_construction_hooks

def evaluate_graph(model: HookedESM3, graph: Graph, dataloader: DataLoader, metrics: Union[List[Callable[[Tensor], Tensor]],Callable[[Tensor], Tensor]],
device, prune:bool=True, quiet=False, calc_input_for_nodes_not_in_graph=False, debug_corrupted_construction=False, use_scaling=True):
    if prune:
        graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)

    empty_circuit = not graph.nodes['logits'].in_graph
    if empty_circuit:
        print("Warning: empty circuit")

    # we construct the in_graph matrix, which is a binary matrix indicating which edges are in the circuit
    # we invert it so that we add in the corrupting activation differences for edges not in the circuit
    in_graph_matrix = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)
    for edge in graph.edges.values():
        if edge.in_graph:
            in_graph_matrix[graph.forward_index(edge.parent, attn_slice=False), graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False)] = 1
            
    corrupted_edges_matrix = 1 - in_graph_matrix 
    input_index = graph.forward_index(graph.nodes[f'input'])
    scaling_factor = model.cfg.esm3_scaling_factor     
    metrics_list = True
    if not isinstance(metrics, list):
        metrics = [metrics]
        metrics_list = False
    results = [[] for _ in metrics]
    
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted , masked_positions, labels in dataloader:
        clean_tokens, attention_mask_clean, n_pos = tokenize_plus(model, clean) 
        corrupted_tokens, attention_mask_corrupted, _ = tokenize_plus(model, corrupted) 

        (fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = make_hooks_and_matrices(model=model, graph=graph, batch_size=len(clean), n_pos=n_pos, scores=None, device=device)
        
        input_construction_hooks = make_input_construction_hooks(model, graph, activation_difference, corrupted_edges_matrix, scaling_factor, input_index, calc_input_for_nodes_not_in_graph, use_scaling)
        with torch.inference_mode():
            with model.hooks(fwd_hooks_corrupted):
                corrupted_logits = model.forward(
                        sequence_tokens=corrupted_tokens.to(device),
                        sequence_id=attention_mask_corrupted.to(device)
                    )
            if empty_circuit and not calc_input_for_nodes_not_in_graph:
                logits = corrupted_logits
            else:
                with model.hooks(fwd_hooks_clean + input_construction_hooks):
                    logits = model.forward(
                        sequence_tokens=clean_tokens.to(device),
                        sequence_id=attention_mask_clean.to(device)
                )
        if debug_corrupted_construction:
            print(torch.max(torch.abs(logits-corrupted_logits)))
            print(corrupted_logits)
            print(logits)
            assert torch.allclose(logits, corrupted_logits, atol=1e-5)
            print("new line\n")

        for i, metric in enumerate(metrics):
            masked_positions_tensor = torch.tensor(masked_positions, device=device)
            labels_tensor = torch.tensor(labels, device=device)
            r = metric(logits, corrupted_logits, masked_positions_tensor, labels_tensor).cpu() #the metric for attribution patching expect clean and corrupted. for eap-ig this is different
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if not metrics_list:
        results = results[0]
    return results

def evaluate_baseline(model: HookedESM3, dataloader:DataLoader, metrics: Union[List[Callable[[Tensor], Tensor]],Callable[[Tensor], Tensor]], device, run_corrupted=False, quiet=False):
    """
    Evaluate a model with a dataloader and a list of metrics, using only the clean examples, and without considering which graph edges are in/out of the circuit
    Args:
        model: HookedTransformer, the model to evaluate.
        dataloader: DataLoader, the dataloader to use for evaluation.
        metrics: Union[List[Callable[[Tensor], Tensor]],Callable[[Tensor], Tensor]], the metric or metrics to evaluate.
        run_corrupted: bool, if True, run the model on corrupted inputs.
        quiet: bool, if True, do not display progress bars.
    Returns:
        List[Tensor], the results of the evaluation.
    """
    metrics_list = True
    if not isinstance(metrics, list):
        metrics = [metrics]
        metrics_list = False
    
    results = [[] for _ in metrics]
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted , masked_positions, labels in dataloader:
        clean_tokens, attention_mask_clean, n_pos = tokenize_plus(model, clean) 
        corrupted_tokens, attention_mask_corrupted, _ = tokenize_plus(model, corrupted) 

        with torch.inference_mode():
            corrupted_logits = model.forward(
                        sequence_tokens=corrupted_tokens.to(device),
                        sequence_id=attention_mask_corrupted.to(device)
                    )
            logits = model.forward(
                        sequence_tokens=clean_tokens.to(device),
                        sequence_id=attention_mask_clean.to(device)
                )
        for i, metric in enumerate(metrics):
            masked_positions_tensor = torch.tensor(masked_positions, device=device)
            labels_tensor = torch.tensor(labels, device=device)
            r = metric(logits, corrupted_logits, masked_positions_tensor, labels_tensor).cpu() #the metric for attribution patching expect clean and corrupted. for eap-ig this is different
            if run_corrupted:
                r = metric(corrupted_logits, logits, masked_positions_tensor, labels_tensor).cpu() #the metric for attribution patching expect clean and corrupted. for eap-ig this is different
            else:
                 r = metric(logits, corrupted_logits, masked_positions_tensor, labels_tensor).cpu() #the metric for attribution patching expect clean and corrupted. for eap-ig this is different
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if not metrics_list:
        results = results[0]
    return results

def compute_faithfulness(clean_baseline, corrupted_baseline, target_baseline, per_element_faithfulness=False):
    
    if per_element_faithfulness:
        return ((target_baseline-corrupted_baseline) / (clean_baseline-corrupted_baseline)).mean()
    else:
        return (target_baseline.mean() - corrupted_baseline.mean()) / (clean_baseline.mean() - corrupted_baseline.mean())



