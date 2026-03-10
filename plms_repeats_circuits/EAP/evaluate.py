#Based on: https://github.com/hannamw/EAP-IG/blob/0edbdd72e3683db69c363a23deb1775f44ec8376/eap/evaluate_graph.py

from typing import Callable, List, Union
from functools import partial 
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedESM3, SupportedESM3Config, HookedTransformerConfig, HookedESMC
from tqdm import tqdm
from einops import einsum 
from .graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode, Node, Edge, NeuronGraph, MLPWithNeuronNode
from .attribute import make_hooks_and_matrices, make_hooks_and_matrices_nodes, tokenize_plus, make_hooks_and_matrices_nodes_and_neurons

def make_input_construction_hook(
    activation_differences,
    corrupted_edges_matrix,
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

# we make input construction hooks for every node but InputNodes. 
# We can also skip nodes not in the graph; it doesn't matter what their outputs are, as they will always be corrupted / reconstructed when serving as inputs to other nodes.
 # AttentionNodes have 3 inputs to reconstruct
def make_input_construction_hooks(model, graph,activation_differences, corrupted_edges_matrix, calc_input_for_nodes_not_in_graph):
    input_construction_hooks = []
    for layer in range(model.cfg.n_layers):
        # add attention hooks:
        if calc_input_for_nodes_not_in_graph or any(graph.nodes[f'a{layer}.h{head}'].in_graph for head in range(model.cfg.n_heads)):
            for i, letter in enumerate('qkv'):
                node = graph.nodes[f'a{layer}.h0']
                prev_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True) #we want all edges information beween all prev components and current qkv heads 
                hook =  make_input_construction_hook(activation_differences=activation_differences, corrupted_edges_matrix=corrupted_edges_matrix, prev_index=prev_index, backward_index=bwd_index, attn=True)
                input_construction_hooks.append((node.qkv_inputs[i], hook))
        # add MLP hook
        if calc_input_for_nodes_not_in_graph or graph.nodes[f'm{layer}'].in_graph:
            node = graph.nodes[f'm{layer}']
            prev_index = graph.prev_index(node)
            bwd_index = graph.backward_index(node)
            hook =  make_input_construction_hook(activation_differences=activation_differences, corrupted_edges_matrix=corrupted_edges_matrix, prev_index=prev_index, backward_index=bwd_index, attn=False)
            input_construction_hooks.append((node.in_hook, hook))
    # Logit node input construction
    node = graph.nodes[f'logits']
    if calc_input_for_nodes_not_in_graph or node.in_graph:
        prev_index = graph.prev_index(node)
        bwd_index = graph.backward_index(node)
        hook =  make_input_construction_hook(activation_differences=activation_differences, corrupted_edges_matrix=corrupted_edges_matrix, prev_index=prev_index, backward_index=bwd_index, attn=False)
        input_construction_hooks.append((node.in_hook, hook))
    return input_construction_hooks

def evaluate_graph(model: HookedESM3 | HookedESMC, graph: Graph, dataloader: DataLoader, metrics: Union[List[Callable[[Tensor], Tensor]],Callable[[Tensor], Tensor]],
device, prune:bool=True, quiet=False, calc_input_for_nodes_not_in_graph=False, debug_corrupted_construction=False, calc_clean_logits=False):
    if prune:
        graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)

    empty_circuit = not graph.nodes['logits'].in_graph
    if empty_circuit:
        print("Warning: empty circuit")

    # we construct the in_graph matrix, which is a binary matrix indicating which edges are in the circuit
    # we invert it so that we add in the corrupting activation differences for edges not in the circuit
    corrupted_edges_matrix = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)
    for edge in graph.edges.values():
        if not edge.in_graph:
            corrupted_edges_matrix[graph.forward_index(edge.parent, attn_slice=False), graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False)] = edge.weight

    metrics_list = True
    if not isinstance(metrics, list):
        metrics = [metrics]
        metrics_list = False
    results = [[] for _ in metrics]
    
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted , masked_positions, labels, clean_id_names in dataloader:
        clean_tokens, attention_mask_clean, n_pos, _ = tokenize_plus(model, clean) 
        corrupted_tokens, attention_mask_corrupted, _ , _= tokenize_plus(model, corrupted) 

        (fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = make_hooks_and_matrices(model=model, graph=graph, batch_size=len(clean), n_pos=n_pos, scores=None, device=device)
        
        input_construction_hooks = make_input_construction_hooks(model, graph, activation_difference, corrupted_edges_matrix, calc_input_for_nodes_not_in_graph)
        with torch.inference_mode():
            if calc_clean_logits:
                clean_logits = model.forward(
                        sequence_tokens=clean_tokens.to(device),
                        sequence_id=attention_mask_clean.to(device)
                    )
            else:
                clean_logits = None
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
            r = metric(logits, clean_logits, masked_positions_tensor, labels_tensor).cpu() #the metric for attribution patching expect clean and corrupted. for eap-ig this is different
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if not metrics_list:
        results = results[0]
    return results





def make_input_corruption_hook_node(
    corrupted_activations,
    fwd_hook,
    attn_head_index=None):
    
    def input_corruption_hook(activations, hook):
        #print("hook name:", hook.name, "fwd_hook:", fwd_hook)
        #print("activations shape:", activations.shape)
        relevant_node_corrupted_activations = corrupted_activations[:, :, fwd_hook]
        if relevant_node_corrupted_activations.shape == activations.shape:
            #print(hook.name, "has the same shape as the activations, replacing with corrupted activations")
            # if the shape is the same, we can just replace with the corrupted activations
            activations = relevant_node_corrupted_activations
        else:
            #print(hook.name, "has a different shape than the activations, replacing in the specific attention head with corrupted activations")
            # if the shape is different, we need to replace only the relevant attention head
            if activations.ndim !=4:
                raise ValueError(f"Expected activations to have 4 dimensions, got {activations.ndim} dimensions")
            if relevant_node_corrupted_activations.ndim != 3:
                raise ValueError(f"Expected relevant_node_corrupted_activations to have 3 dimensions, got {relevant_node_corrupted_activations.ndim} dimensions")
            
            if attn_head_index is not None:
                #print(f"shapeactivations[attn_head_index] {activations[attn_head_index].shape} vs corrupted shape : {relevant_node_corrupted_activations.shape}")
                activations[attn_head_index] = relevant_node_corrupted_activations
            else:
                raise ValueError("attn_head_index must be provided for attention nodes")
          
        return activations
    
    return input_corruption_hook


def make_input_corruption_hook_neurons(
    neurons_activations_corrupted,
    fwd_hook,
    neurons_indicies_in_graph):
    
    def input_corruption_hook(activations, hook):

        relevant_node_corrupted_activations = neurons_activations_corrupted[:, :, fwd_hook] # should be (b, n_pos, n_neurons)
        if relevant_node_corrupted_activations.shape != activations.shape:
            raise ValueError(f"Expected relevant_node_corrupted_activations to have shape {activations.shape}, got {relevant_node_corrupted_activations.shape}")
        
        if neurons_indicies_in_graph is None:
            raise ValueError("neurons_indicies_in_graph must be provided")
        
        ret = relevant_node_corrupted_activations.clone()
        if len(neurons_indicies_in_graph) > 0:
            # check that neurons is 1 dim array
            if neurons_indicies_in_graph.ndim != 1:
                raise ValueError(f"Expected neurons_indicies_in_graph to have 1 dimension, got {neurons_indicies_in_graph.ndim} dimensions")
            ret[:, :, neurons_indicies_in_graph] = activations[:, :, neurons_indicies_in_graph] # we set ret to be corrupted but then restore the original activations in in_graph indicies
        return ret
    
    return input_corruption_hook


def make_input_corruption_hooks_node(model, graph, corrupted_activations):
    input_corruption_hooks = []
    for node in graph.nodes.values():
        if node.in_graph == False and isinstance(node, LogitNode)==False:
            fwd_index = graph.forward_index(node, attn_slice=False)
            if isinstance(node, AttentionNode):
                attn_head_index = node.index
            else:
                attn_head_index = None
            hook = make_input_corruption_hook_node(corrupted_activations=corrupted_activations, fwd_hook=fwd_index, attn_head_index=attn_head_index)
            input_corruption_hooks.append((node.out_hook, hook))
    return input_corruption_hooks

def make_input_corruption_hooks_neurons(model, graph, neurons_corrupted_activations, corrupted_activations):

    input_corruption_hooks = []
    for node in graph.nodes.values():
        if node.in_graph == False and isinstance(node, LogitNode)==False and isinstance(node, MLPWithNeuronNode)==False:
            fwd_index = graph.forward_index(node, attn_slice=False)
            if isinstance(node, AttentionNode):
                attn_head_index = node.index
            else:
                attn_head_index = None
            hook = make_input_corruption_hook_node(corrupted_activations=corrupted_activations, fwd_hook=fwd_index, attn_head_index=attn_head_index)
            input_corruption_hooks.append((node.out_hook, hook))
        
        if isinstance(node, MLPWithNeuronNode):
            fwd_index_neurons = graph.forward_index(node, attn_slice=False, return_index_in_neurons_array=True)
            neurons_indicies_in_graph = node.neurons_indicies_in_graph
            if len(neurons_indicies_in_graph) == node.neurons_num:
                continue # we don't need to corrupt the neurons if they are all in the graph
            hook = make_input_corruption_hook_neurons(neurons_activations_corrupted=neurons_corrupted_activations, fwd_hook=fwd_index_neurons, neurons_indicies_in_graph=neurons_indicies_in_graph)
            input_corruption_hooks.append((node.neurons_out_hook, hook))
    return input_corruption_hooks


def evaluate_graph_node(model: HookedESM3 | HookedESMC, graph: Graph, dataloader: DataLoader,
                        metrics: Union[List[Callable[[Tensor], Tensor]],Callable[[Tensor], Tensor]],
                        device, prune:bool=True, quiet=False, debug=False, calc_clean_logits=False):
    if prune:
        graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)

    empty_circuit = not graph.nodes['logits'].in_graph
    if empty_circuit:
        print("Warning: empty circuit")

    metrics_list = True
    if not isinstance(metrics, list):
        metrics = [metrics]
        metrics_list = False
    results = [[] for _ in metrics]
    
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted , masked_positions, labels, clean_id_names in dataloader:
        clean_tokens, attention_mask_clean, n_pos, _ = tokenize_plus(model, clean) 
        corrupted_tokens, attention_mask_corrupted, _ , _= tokenize_plus(model, corrupted) 

        (fwd_hooks_corrupted, _, _), corrupted_activations = make_hooks_and_matrices(model=model, graph=graph, batch_size=len(clean), n_pos=n_pos, scores=None, device=device)
        
        input_corruption_hooks = make_input_corruption_hooks_node(model, graph, corrupted_activations)
        all_nodes_in_graph = len([node for node in graph.nodes.values() if node.in_graph]) == len(graph.nodes)

        with torch.inference_mode():
            if calc_clean_logits or all_nodes_in_graph:
                clean_logits = model.forward(
                        sequence_tokens=clean_tokens.to(device),
                        sequence_id=attention_mask_clean.to(device)
                    )
            else:
                clean_logits = None
         
            with model.hooks(fwd_hooks_corrupted):
                corrupted_logits = model.forward(
                        sequence_tokens=corrupted_tokens.to(device),
                        sequence_id=attention_mask_corrupted.to(device)
                    )
            if empty_circuit and not debug:
                logits = corrupted_logits
            else:
                with model.hooks(input_corruption_hooks):
                    logits = model.forward(
                        sequence_tokens=clean_tokens.to(device),
                        sequence_id=attention_mask_clean.to(device)
                )
        if debug:
            if empty_circuit:
                print("Empty circuit debug mode")
                print(f"Corrupted logits:{corrupted_logits}, hooked logits:{logits}")
                print(f"Max difference: {torch.max(torch.abs(logits-corrupted_logits))}")
                assert torch.allclose(logits, corrupted_logits, atol=1e-5), "Logits and corrupted logits are not close enough"
                print("new line\n")
            elif all_nodes_in_graph:
                print("Debug mode with all nodes in graph")
                print(f"Clean logits:{clean_logits}, hooked logits:{logits}")
                print(f"Max difference: {torch.max(torch.abs(logits-clean_logits))}")
                assert torch.allclose(logits, clean_logits, atol=1e-5), "Logits and clean logits are not close enough"
                print("new line\n")

        for i, metric in enumerate(metrics):
            masked_positions_tensor = torch.tensor(masked_positions, device=device)
            labels_tensor = torch.tensor(labels, device=device)
            r = metric(logits, clean_logits, masked_positions_tensor, labels_tensor).cpu() #the metric for attribution patching expect clean and corrupted. for eap-ig this is different
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if not metrics_list:
        results = results[0]
    return results

def evaluate_graph_neurons(model: HookedESM3 | HookedESMC, graph: NeuronGraph, dataloader: DataLoader,
                        metrics: Union[List[Callable[[Tensor], Tensor]],Callable[[Tensor], Tensor]],
                        device, quiet=False, debug=False, calc_clean_logits=False):


    empty_circuit = not graph.nodes['logits'].in_graph
    if empty_circuit:
        print("Warning: empty circuit")

    metrics_list = True
    if not isinstance(metrics, list):
        metrics = [metrics]
        metrics_list = False
    results = [[] for _ in metrics]
    
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted , masked_positions, labels, clean_id_names in dataloader:
        clean_tokens, attention_mask_clean, n_pos, _ = tokenize_plus(model, clean) 
        corrupted_tokens, attention_mask_corrupted, _ , _= tokenize_plus(model, corrupted) 

        (fwd_hooks_corrupted, _, _), corrupted_activations = make_hooks_and_matrices_nodes(model=model, graph=graph, batch_size=len(clean), n_pos=n_pos, scores=None, device=device)
        (fwd_hooks_corrupted_neurons, _, _), corrupted_neurons_activations = make_hooks_and_matrices_nodes_and_neurons(model=model, graph=graph, batch_size=len(clean), n_pos=n_pos, scores_neurons=None, device=device)
        
        input_corruption_hooks = make_input_corruption_hooks_neurons(model=model, graph=graph, neurons_corrupted_activations=corrupted_neurons_activations, corrupted_activations=corrupted_activations)
        all_nodes_in_graph = graph.count_included_nodes() == graph.count_total_nodes()

        # print("fwd_hooks_corrupted_neurons:")
        # for hook in fwd_hooks_corrupted_neurons:
        #     print("  ", hook)

        # print("\nfwd_hooks_corrupted:")
        # for hook in fwd_hooks_corrupted:
        #     print("  ", hook)

        # print("\ninput_corruption_hooks:")
        # for hook in input_corruption_hooks:
        #     print("  ", hook)

        with torch.inference_mode():
            if calc_clean_logits or all_nodes_in_graph:
                clean_logits = model.forward(
                        sequence_tokens=clean_tokens.to(device),
                        sequence_id=attention_mask_clean.to(device)
                    )
            else:
                clean_logits = None
         
            with model.hooks(fwd_hooks=(fwd_hooks_corrupted_neurons + fwd_hooks_corrupted)):
                corrupted_logits = model.forward(
                        sequence_tokens=corrupted_tokens.to(device),
                        sequence_id=attention_mask_corrupted.to(device)
                    )
            if empty_circuit and not debug:
                logits = corrupted_logits
            else:
                with model.hooks(input_corruption_hooks):
                    logits = model.forward(
                        sequence_tokens=clean_tokens.to(device),
                        sequence_id=attention_mask_clean.to(device)
                )
        if debug:
            if empty_circuit:
                print("Empty circuit debug mode")
                print(f"Corrupted logits:{corrupted_logits} \n hooked logits:{logits}")
                #print space of a new line
                print("\n")
                print(f"Max difference: {torch.max(torch.abs(logits-corrupted_logits))}")
                assert torch.allclose(logits, corrupted_logits, atol=1e-5), "Logits and corrupted logits are not close enough"
                print("new line\n")
            elif all_nodes_in_graph:
                print("Debug mode with all nodes in graph")
                print(f"Clean logits:{clean_logits}, hooked logits:{logits}")
                print(f"Max difference: {torch.max(torch.abs(logits-clean_logits))}")
                assert torch.allclose(logits, clean_logits, atol=1e-5), "Logits and clean logits are not close enough"
                print("new line\n")

        for i, metric in enumerate(metrics):
            masked_positions_tensor = torch.tensor(masked_positions, device=device)
            labels_tensor = torch.tensor(labels, device=device)
            r = metric(logits, clean_logits, masked_positions_tensor, labels_tensor).cpu() #the metric for attribution patching expect clean and corrupted. for eap-ig this is different
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if not metrics_list:
        results = results[0]
    return results

def evaluate_baseline(model: HookedESM3 | HookedESMC, dataloader:DataLoader, metrics: Union[List[Callable[[Tensor], Tensor]],Callable[[Tensor], Tensor]], device, 
run_corrupted=False, quiet=False, calc_clean_logits_for_metric=False):
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
    # todo: unpack variables based on if it has 4 o r 5 variables
    for variables in dataloader:
        if len(variables) == 4:
            clean, corrupted , masked_positions, labels = variables
        elif len(variables) == 5:
            clean, corrupted , masked_positions, labels, _ = variables
        else:
            raise ValueError(f"Expected 4 or 5 variables, got {len(variables)}")

        clean_tokens, attention_mask_clean, n_pos, _ = tokenize_plus(model, clean) 
        corrupted_tokens, attention_mask_corrupted, _, _ = tokenize_plus(model, corrupted) 
        clean_logits = None
        with torch.inference_mode():
            if run_corrupted:
                corrupted_logits = model.forward(
                        sequence_tokens=corrupted_tokens.to(device),
                        sequence_id=attention_mask_corrupted.to(device)
                    )
            elif run_corrupted==False or calc_clean_logits_for_metric:
                clean_logits = model.forward(
                            sequence_tokens=clean_tokens.to(device),
                            sequence_id=attention_mask_clean.to(device)
                    )
            else:
                raise ValueError("run_corrupted must be True or False, or calc_clean_logits must be True")
            
            
            
        for i, metric in enumerate(metrics):
            masked_positions_tensor = torch.tensor(masked_positions, device=device)
            labels_tensor = torch.tensor(labels, device=device)
            if run_corrupted:
                r = metric(corrupted_logits, clean_logits, masked_positions_tensor, labels_tensor).cpu() 
            else:
                 r = metric(clean_logits, clean_logits, masked_positions_tensor, labels_tensor).cpu()
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


