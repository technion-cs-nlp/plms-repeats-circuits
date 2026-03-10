
from typing import Callable, List, Union, Dict, Optional
from functools import partial 
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedESM3, SupportedESM3Config, HookedTransformerConfig
from tqdm import tqdm
from einops import einsum 
from .graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode, Node, Edge
from .attribute import make_hooks_and_matrices, tokenize_plus
import os
from pathlib import Path

def make_input_mean_activations_mlps(mean_activations_mlp, tokens_count_mlp, padding_mask, layer):
    # padding_mask: (B, T) with 1 for real tokens, 0 for PAD/BOS/EOS
    def hook_fn(activations, hook):
        assert activations.ndim == 3  # (B, T, D)

        # Broadcasted masking with unsqueeze (avoid boolean indexing)
        mask = padding_mask.unsqueeze(-1).to(dtype=activations.dtype, device=activations.device)  # (B, T, 1)
        copied = activations * mask

        tokens_count = int(padding_mask.sum().item())
        if tokens_count == 0:
            return activations

        cur_mean = mean_activations_mlp[layer].to(device=activations.device, dtype=activations.dtype).detach()  # (D,)
        cur_cnt  = int(tokens_count_mlp[layer].item())  # Convert tensor to int

        # Sum over batch and time → (D,)
        batch_sum = copied.sum(dim=(0, 1))

        # Numerically stable running-mean update:
        new_cnt  = cur_cnt + tokens_count
        delta    = (batch_sum - tokens_count * cur_mean) / new_cnt
        new_mean = cur_mean + delta

        mean_activations_mlp[layer] = new_mean
        tokens_count_mlp[layer] = new_cnt
        return activations
    return hook_fn


def make_input_mean_activations_attention_heads(mean_acts_heads, tokens_count_heads, padding_mask, layer):
    # padding_mask: (B, T), 1 for real tokens, 0 for PAD/BOS/EOS
    def hook_fn(activations, hook):
        # activations: (B, T, H, D)
        assert activations.ndim == 4

        # Broadcasted mask to (B, T, 1, 1)
        mask = padding_mask.unsqueeze(-1).unsqueeze(-1).to(
            dtype=activations.dtype, device=activations.device
        )
        copied = activations * mask

        tokens_count = int(padding_mask.sum().item())
        if tokens_count == 0:
            return activations

        cur_mean = mean_acts_heads[layer].to(
            device=activations.device, dtype=activations.dtype
        ).detach()  # (H, D)
        cur_cnt = int(tokens_count_heads[layer].item())  # Convert tensor to int

        # Sum over batch and time -> (H, D)
        batch_sum = copied.sum(dim=(0, 1))

        # Numerically stable update:
        new_cnt = cur_cnt + tokens_count
        delta = (batch_sum - tokens_count * cur_mean) / new_cnt  # broadcasts over (H, D)
        new_mean = cur_mean + delta

        mean_acts_heads[layer]   = new_mean
        tokens_count_heads[layer] = new_cnt
        return activations
    return hook_fn


def make_input_mean_ablation_hooks(model, graph, mean_activations_mlp, mean_activations_attention_heads, tokens_count_mlp, tokens_count_attention_heads, padding_mask):
    hooks = []

    for layer in range(model.cfg.n_layers):
        # add attention hooks:
        attn_node = graph.nodes[f'a{layer}.h0']
        out_hook_name = attn_node.out_hook
        hook = make_input_mean_activations_attention_heads(
            mean_activations_attention_heads, tokens_count_attention_heads, padding_mask, layer
        )
        hooks.append((out_hook_name, hook))
        
        # add MLP hooks:
        mlp_node = graph.nodes[f'm{layer}']
        out_hook_name = mlp_node.out_hook
        hook = make_input_mean_activations_mlps(
            mean_activations_mlp, tokens_count_mlp, padding_mask, layer
        )
        hooks.append((out_hook_name, hook))
        
    return hooks


def make_input_mean_activation_hook(mean_activation_input, tokens_count_input, padding_mask):
    """Hook function to compute running mean for input (embedding) activations."""
    def hook_fn(activations, hook):
        # activations: (B, T, d_model) - embedding output
        assert activations.ndim == 3
        
        # Broadcasted masking with unsqueeze
        mask = padding_mask.unsqueeze(-1).to(dtype=activations.dtype, device=activations.device)  # (B, T, 1)
        copied = activations * mask
        
        tokens_count = int(padding_mask.sum().item())
        if tokens_count == 0:
            return activations
        
        cur_mean = mean_activation_input.to(device=activations.device, dtype=activations.dtype).detach()  # (d_model,)
        cur_cnt = int(tokens_count_input[0].item())
        
        # Sum over batch and time → (d_model,)
        batch_sum = copied.sum(dim=(0, 1))
        
        # Numerically stable running-mean update:
        new_count = cur_cnt + tokens_count
        delta = (batch_sum - tokens_count * cur_mean) / new_count
        new_mean = cur_mean + delta
        
        mean_activation_input[:] = new_mean
        tokens_count_input[0] = new_count
        
        return activations
    return hook_fn


def compute_mean_activations_per_node(model: HookedESM3, graph: Graph, dataloader: DataLoader, device, quiet=False, output_dir=None, save_neurons=False, activation_mode='all'):

    n_layers = model.cfg.n_layers
    
    # Validate activation_mode parameter
    valid_modes = ['all', 'exclude_mask', 'only_mask']
    if activation_mode not in valid_modes:
        raise ValueError(f"activation_mode must be one of {valid_modes}, got '{activation_mode}'")
    
    # Initialize tensors with correct shapes and types
    mean_activation_mlp = torch.zeros((n_layers, model.cfg.d_model), device=device, dtype=model.cfg.dtype)
    tokens_count_mlp = torch.zeros((n_layers,), device=device, dtype=torch.long)  # Use long for counts
    
    mean_activation_attention_heads = torch.zeros(
        (n_layers, model.cfg.n_heads, model.cfg.d_model), device=device, dtype=model.cfg.dtype
    )
    tokens_count_attention_heads = torch.zeros((n_layers,), device=device, dtype=torch.long)  # Per layer, not per head
    
    # Initialize input (embedding) mean activation
    mean_activation_input = torch.zeros((model.cfg.d_model,), device=device, dtype=model.cfg.dtype)
    tokens_count_input = torch.zeros((1,), device=device, dtype=torch.long)

    # Initialize neuron activation tensors if requested
    if save_neurons:
        # Get actual d_mlp from the model (handles SwiGLU correctly)
        d_mlp = model.blocks[0].mlp.l2.in_features  # Input to second linear layer
        mean_neuron_activations = torch.zeros((n_layers, d_mlp), device=device, dtype=model.cfg.dtype)
        tokens_count_neurons = torch.zeros((n_layers,), device=device, dtype=torch.long)
    
    model.eval()  # Use eval() instead of inference_mode() for hooks
    
    batch_idx = 0
    for batch in tqdm(dataloader, desc="Computing mean activations per node", disable=quiet):
        print(f"Batch {batch_idx}")
        clean, masked_positions, labels = batch
        clean_tokens, attention_mask_clean, n_pos, lengths = tokenize_plus(model, clean)
        
        # Create mask based on activation_mode (applies to all components)
        if activation_mode != 'all':
            # masked_positions: (B,) indices of masked positions
            # Create a binary mask: (B, T) with 1 for masked positions, 0 otherwise
            batch_size, seq_len = clean_tokens.shape
            masked_positions_mask = torch.zeros(
                (batch_size, seq_len), 
                device=attention_mask_clean.device,  # Use same device as attention_mask_clean
                dtype=attention_mask_clean.dtype
            )
            for b in range(batch_size):
                masked_positions_mask[b, masked_positions[b]] = 1
            
            if activation_mode == 'exclude_mask':
                # Exclude masked positions: keep real tokens except masked ones
                active_mask = attention_mask_clean * (1 - masked_positions_mask)
            elif activation_mode == 'only_mask':
                # Only masked positions: keep only masked positions that are real tokens
                active_mask = attention_mask_clean * masked_positions_mask
        else:
            # Default: use all non-padding tokens
            active_mask = attention_mask_clean
        
        # Debug: Print token counts for this batch
        batch_token_count = int(active_mask.sum().item())
        print(f"\n[Batch {batch_idx}] Token count: {batch_token_count}")
        print(f"  Sequence lengths: {lengths.tolist()}")
        print(f"  Active tokens per sequence: {active_mask.sum(dim=1).tolist()}")
        if activation_mode != 'all':
            print(f"  Masked positions: {masked_positions}")  # Already a list
        
        hooks = make_input_mean_ablation_hooks(
            model, graph, 
            mean_activation_mlp, mean_activation_attention_heads, 
            tokens_count_mlp, tokens_count_attention_heads, 
            active_mask  # Use the mode-based mask for all components
        )
        
        # Add input (embedding) hook
        input_hook_fn = make_input_mean_activation_hook(
            mean_activation_input, tokens_count_input, active_mask
        )
        hooks.append(("hook_embed", input_hook_fn))
        
        # Add neuron activation hooks if requested
        if save_neurons:
            for layer in range(n_layers):
                hook_name = f"blocks.{layer}.mlp.hook_post"
                hook_fn = make_neuron_mean_activation_hook(
                    mean_neuron_activations, tokens_count_neurons, active_mask, layer  # Use same mask
                )
                hooks.append((hook_name, hook_fn))
        
        with model.hooks(hooks):
            with torch.no_grad():  # Add no_grad for efficiency
                _ = model.forward(
                    sequence_tokens=clean_tokens.to(device),
                    sequence_id=attention_mask_clean.to(device)
                )
        
        batch_idx += 1
    
    # Print final token count statistics
    print(f"\n{'='*60}")
    print(f"Final token count statistics (mode={activation_mode}):")
    print(f"{'='*60}")
    print(f"Input token count: {tokens_count_input[0].item()} tokens")
    if save_neurons:
        print(f"Neuron token counts per layer:")
        for layer in range(min(3, n_layers)):  # Show first 3 layers
            print(f"  Layer {layer}: {tokens_count_neurons[layer].item()} tokens")
        if n_layers > 3:
            print(f"  ...")
            print(f"  Layer {n_layers-1}: {tokens_count_neurons[n_layers-1].item()} tokens")
    print(f"MLP token counts per layer:")
    for layer in range(min(3, n_layers)):  # Show first 3 layers
        print(f"  Layer {layer}: {tokens_count_mlp[layer].item()} tokens")
    if n_layers > 3:
        print(f"  ...")
        print(f"  Layer {n_layers-1}: {tokens_count_mlp[n_layers-1].item()} tokens")
    print(f"Attention token counts per layer:")
    for layer in range(min(3, n_layers)):  # Show first 3 layers
        print(f"  Layer {layer}: {tokens_count_attention_heads[layer].item()} tokens")
    if n_layers > 3:
        print(f"  ...")
        print(f"  Layer {n_layers-1}: {tokens_count_attention_heads[n_layers-1].item()} tokens")
    print(f"{'='*60}\n")
    
    if output_dir is not None:
        output_dir = Path(output_dir)  # Ensure it's a Path object
        output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if needed
        
        # Add mode suffix to all filenames
        suffix = "" if activation_mode == 'all' else f"_{activation_mode}"
        
        torch.save(mean_activation_mlp.cpu(), output_dir / f"mean_activation_mlp{suffix}.pt")
        torch.save(mean_activation_attention_heads.cpu(), output_dir / f"mean_activation_attention_heads{suffix}.pt")
        torch.save(mean_activation_input.cpu(), output_dir / f"mean_activation_input{suffix}.pt")
        
        print(f"Saved mean activation MLP (mode={activation_mode}): shape {mean_activation_mlp.shape}")
        print(f"Saved mean activation attention heads (mode={activation_mode}): shape {mean_activation_attention_heads.shape}")
        print(f"Saved mean activation input (mode={activation_mode}): shape {mean_activation_input.shape}")
        # Save neuron activations if computed
        if save_neurons:
            torch.save(mean_neuron_activations.cpu(), output_dir / f"mean_neuron_activations{suffix}.pt")
            torch.save(tokens_count_neurons.cpu(), output_dir / f"neuron_tokens_count{suffix}.pt")
            print(f"Saved mean neuron activations (mode={activation_mode}): shape {mean_neuron_activations.shape}")
    
    if save_neurons:
        return mean_activation_mlp, tokens_count_mlp, mean_activation_attention_heads, tokens_count_attention_heads, mean_activation_input, tokens_count_input, mean_neuron_activations, tokens_count_neurons
    else:
        return mean_activation_mlp, tokens_count_mlp, mean_activation_attention_heads, tokens_count_attention_heads, mean_activation_input, tokens_count_input


def make_neuron_mean_activation_hook(mean_neuron_acts, tokens_count_neurons, padding_mask, layer):
    def hook_fn(activations, hook):
        # activations: (B, T, d_mlp) - MLP neuron activations after activation function
        assert activations.ndim == 3
        
        # Broadcasted masking with unsqueeze (avoid boolean indexing)
        mask = padding_mask.unsqueeze(-1).to(dtype=activations.dtype, device=activations.device)  # (B, T, 1)
        copied = activations * mask
        
        tokens_count = int(padding_mask.sum().item())
        if tokens_count == 0:
            return activations
        
        cur_mean = mean_neuron_acts[layer].to(device=activations.device, dtype=activations.dtype).detach()  # (d_mlp,)
        cur_cnt = int(tokens_count_neurons[layer].item())
        
        # Sum over batch and time → (d_mlp,)
        batch_sum = copied.sum(dim=(0, 1))
        
        # Numerically stable running-mean update:
        new_cnt = cur_cnt + tokens_count
        delta = (batch_sum - tokens_count * cur_mean) / new_cnt
        new_mean = cur_mean + delta
        
        mean_neuron_acts[layer] = new_mean
        tokens_count_neurons[layer] = new_cnt
        return activations
    
    return hook_fn