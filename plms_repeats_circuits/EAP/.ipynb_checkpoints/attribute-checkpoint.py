from typing import Callable, List, Union, Optional,Literal
from functools import partial
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedESM3
from tqdm import tqdm
from einops import einsum
from .graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode
from esm.tokenization import get_esm3_model_tokenizers

def show_memory():
    # Total allocated memory
    allocated = torch.cuda.memory_allocated()
    print(f"Allocated Memory: {allocated / 1e9:.2f} GB")

    # Total reserved memory (including fragmentation)
    reserved = torch.cuda.memory_reserved()
    print(f"Reserved Memory: {reserved / 1e9:.2f} GB")

    # Free unallocated memory within reserved memory
    free = reserved - allocated
    print(f"Free Memory: {free / 1e9:.2f} GB")

    total_memory = torch.cuda.get_device_properties(0).total_memory  # Bytes
    print(f"Total GPU Memory: {total_memory / 1e9:.2f} GB")

def tokenize_plus(model: HookedESM3, inputs: List[str]):
    tokenizer = get_esm3_model_tokenizers().sequence
    tokenized_info = tokenizer(inputs, padding=True, return_tensors='pt', add_special_tokens=True)
    tokens = tokenized_info.input_ids
    attention_mask = tokenized_info.attention_mask
    n_pos = attention_mask.size(1)
    return tokens, attention_mask, n_pos

def make_hooks_and_matrices(model: HookedESM3, graph: Graph, batch_size:int , n_pos:int, scores: torch.Tensor, device, detach=True):
    activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=device, dtype=model.cfg.dtype) #a tensor to stor activation differences of each componenet from forward

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
            print("Activation Hook Error", hook.name, activation_difference[:, :, index].size(), acts.size(), index)
            raise e
    
    def gradient_hook(prev_index: Union[slice, int], bwd_index: Union[slice, int], gradients:torch.Tensor, hook):
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
                grads = grads.unsqueeze(2) #meaning we get [batch pos 1 hidden] in that case ? 
            s = einsum(activation_difference[:, :, :prev_index], grads,'batch pos forward hidden, batch pos backward hidden -> forward backward')
            #lets divide for cases to understand what is going one here:
                #if the node is logits node: we hook the residual post of the last layer (before layer normalization and all regression heads)
                #basically the input to the logit node is [batch,pos, hidden] so the gradient of the metric w.r.t the input is also [batch,pos, hidden]
                #so we will unsqueeze(2) and get [batch pos 1 hidden]. #prev index is size graph.n_forward, so we basically get activation diffrences for all nodes
                #we will compute the approximated metric diffrences for each forward component and logit node, summed over all positions and batches

                #if node is mlp node : we hook the residual before adding mlp result (input mlp component).
                #basically the input to the mlp node is [batch,pos, hidden] so the gradient of the metric w.r.t the input is also [batch,pos, hidden]
                #so we will unsqueeze(2) and get [batch pos 1 hidden]. #prev index is size all the components before that mlp node (input+attentions+mlps before), 
                #so we basically get activation diffrences for previous nodes to that mlp because we assume all that nodes enter our mlp
                #we will compute the approximated metric diffrences for each prev component and mlp node, summed over all positions and bathces

                #if node is attention node : we hook the q,k,v matrices seperatly so we get here one of them. q,k,v are inputs to attention component
                #basically the input q or k or v to the attention node is [batch, pos, n_heads, hidden] so the gradient of the metric w.r.t the input is also [batch, pos, n_heads, hidden]
                #so we won't unsqueeze(2)]. #prev index is size all the components before that attention node (input+attentions+mlps before), 
                #so we basically get activation diffrences for previous nodes to that attention node because we assume all that nodes enter our attention
                #we will compute the approximated metric diffrences for each prev component and *each mlp head*, summed over all positions and bathces
                #i forgot to mention that bwd_index is a slice of all attention heads indexs in that case!

            s = s.squeeze(1)#.to(scores.device) #is squeezing will work for the case backward >1??
            scores[:prev_index, bwd_index] += s #we basically were able to compute approximated metric diff the relation between all prev components to the currenct components
        except RuntimeError as e:
            print("Gradient Hook Error", hook.name, activation_difference.size(), grads.size(), prev_index, bwd_index)
            raise e

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
                    bwd_hooks.append((node.qkv_inputs[i], partial(gradient_hook, prev_index, bwd_index)))
            else:
                bwd_index = graph.backward_index(node)
                bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))
            
    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference





#TODO - change the function to get tokens so we won't need to tell which modality we provided
def get_scores_eap(model: HookedESM3, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor], device, quiet=False,can_ignore_corrupted_grad=True):
    # print("memory before scores creation")
    # show_memory()
    scores = torch.zeros((graph.n_forward, graph.n_backward), device= device, dtype=model.cfg.dtype) #to store scores of edges between all possible edges (even non existent one)
    # print("memory after scores creation")
    # show_memory()
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted , masked_positions, labels in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        # print(f"batch size:{batch_size}, memory before  tokens and attention mask")
        # show_memory()
        clean_tokens, attention_mask_clean, n_pos = tokenize_plus(model, clean) 
        corrupted_tokens, attention_mask_corrupted, _ = tokenize_plus(model, corrupted) 
        # print(f"batch size:{batch_size}, memory after  tokens and attention mask")
        # show_memory()
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, device) # returns the required hooks + activation differences array but its empty
        # print(f"batch size:{batch_size}, memory after  make_hooks_and_matrices")
        # show_memory()
        if can_ignore_corrupted_grad:
            with torch.no_grad():
                with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                    corrupted_logits = model.forward(
                            sequence_tokens=corrupted_tokens.to(device),
                            sequence_id=attention_mask_corrupted.to(device)
                        )
                    corrupted_logits= corrupted_logits.detach()
        else:
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                corrupted_logits = model.forward(
                        sequence_tokens=corrupted_tokens.to(device),
                        sequence_id=attention_mask_corrupted.to(device)
                    )
                # print(f"batch size:{batch_size}, memory after  corrupted_logits")
                # show_memory()
        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model.forward(
                    sequence_tokens=clean_tokens.to(device),
                    sequence_id=attention_mask_clean.to(device)
                )
            # print(f"batch size:{batch_size}, memory after clean_logits")
            # show_memory()
            masked_positions_tensor = torch.tensor(masked_positions, device=device)
            labels_tensor = torch.tensor(labels, device=device)
            metric_value = metric(logits, corrupted_logits, masked_positions_tensor, labels_tensor) #the metric for attribution patching expect clean and corrupted. for eap-ig this is different
            metric_value.backward()

    scores /= total_items #averaging over all scores

    return scores


allowed_aggregations = {'sum'}
def attribute(model: HookedESM3, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], device, aggregation='sum', method: Union[Literal['EAP']]="EAP", quiet=False, fix_scaling_factor_issue=True, can_ignore_corrupted_grad=True):
    
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')

    if method == 'EAP':
        scores = get_scores_eap(model, graph, dataloader, metric, quiet=quiet, device=device, can_ignore_corrupted_grad=can_ignore_corrupted_grad)
    else:
        raise ValueError(f"integrated_gradients must be in ['EAP'], but got {method}")
    
    # if aggregation == 'mean':
    #     scores /= model.cfg.d_model
    # elif aggregation == 'l2':
    #     scores = torch.linalg.vector_norm(scores, ord=2, dim=-1)

    scores = scores.cpu().numpy()  # this will detach the scores

    for edge in tqdm(graph.edges.values(), total=len(graph.edges)):
        #to compute the score of an edge nod1->nod2: we take score in location [node1_index, node2_index]
        #why its true ? because to compute the approx L(corrupted in component)-L(clean) = L(clean) + gradLclean(corrupted-clean)
        #the gradient of node 2 is w.r.t its correct input since:
        #We are computing the the gradient with respect to child node. the input to the child node is sum of all edges input = e_embedding +(e2+...+ek)*normalization factor
        # so the grad for the embedding layer is same grad as the node since the gradient w.r.t to embedding layer is effectively 1 and also the grad of each other edge is normalization factor
        #so our score is actually the activation difference of the edge gathered by parent node, so we just need the out activation value of a node and not an edge
        #multiplied by the gradient of the node which is almost equal to the gradient of the edge 

        scaling_factor =model.cfg.esm3_scaling_factor if not isinstance(edge.parent, InputNode) and fix_scaling_factor_issue==True else 1.0
        edge.score = scores[graph.forward_index(edge.parent, attn_slice=False), graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False)]/scaling_factor
