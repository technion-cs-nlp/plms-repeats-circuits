from typing import Dict, List, Tuple
from functools import partial
import torch
from transformer_lens import HookedESM3, HookedESMC

from activation_data import NeuronInfo


def create_neuron_activation_hooks(
    neuron_info_list: List[NeuronInfo],
    activation_storage: Dict[str, torch.Tensor]
) -> List[Tuple[str, callable]]:
    """
    Create hooks to extract activations for specific neurons.
    
    Args:
        neuron_info_list: List of neurons to extract activations for
        activation_storage: Dictionary to store activations, keyed by component_id
    
    Returns:
        List of (hook_name, hook_function) tuples in transformer_lens style
    """
    hooks = []
    
    # Group neurons by layer for efficient hooking
    neurons_by_layer: Dict[int, List[NeuronInfo]] = {}
    for neuron_info in neuron_info_list:
        layer = neuron_info.layer
        if layer not in neurons_by_layer:
            neurons_by_layer[layer] = []
        neurons_by_layer[layer].append(neuron_info)
    
    # Create a hook for each layer that has neurons we care about
    for layer, layer_neurons in neurons_by_layer.items():
        # Hook the MLP output: blocks.{layer}.mlp.hook_post
        hook_name = f"blocks.{layer}.mlp.hook_post"
        
        # Get neuron indices for this layer (as tensor for indexing)
        neuron_indices = torch.tensor([n.neuron_idx for n in layer_neurons], dtype=torch.long)
        
        # Create mapping from neuron_idx to NeuronInfo for this layer
        neuron_info_dict = {n.neuron_idx: n for n in layer_neurons}
        
        # Create hook function (transformer_lens style with partial)
        def activation_hook(neuron_indices_tensor: torch.Tensor, neuron_info_dict: Dict[int, NeuronInfo], 
                           activations: torch.Tensor, hook) -> torch.Tensor:
            """
            Hook function to extract neuron activations.
            
            Args:
                neuron_indices_tensor: Tensor of neuron indices to extract
                neuron_info_dict: Mapping from neuron_idx to NeuronInfo
                activations: Shape [batch, seq_len, hidden_width] - MLP output activations
                hook: Hook object (unused)
            
            Returns:
                Original activations (pass-through)
            """
            # activations shape: [batch, seq_len, hidden_width]
            # Extract activations for specific neurons: [batch, seq_len, n_neurons]
            neuron_acts = activations[:, :, neuron_indices_tensor]  # [batch, seq_len, n_neurons]
            
            # Store activations for each neuron
            for i, neuron_idx in enumerate(neuron_indices_tensor.tolist()):
                neuron_info = neuron_info_dict[neuron_idx]
                # Store as [batch, seq_len] - we'll flatten later
                activation_storage[neuron_info.component_id] = neuron_acts[:, :, i].detach().clone()
            
            return activations  # Pass through unchanged
        
        # Use partial to bind the layer-specific parameters (transformer_lens style)
        hooks.append((hook_name, partial(activation_hook, neuron_indices, neuron_info_dict)))
    
    return hooks


def extract_activations_for_sequence(
    model: HookedESM3 | HookedESMC,
    tokens: torch.Tensor,
    neuron_info_list: List[NeuronInfo],
    device: torch.device,
    attention_mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Extract activations for all neurons for a single sequence.
    
    Args:
        model: The HookedESM3 model
        tokens: Tokenized sequence [seq_len]
        neuron_info_list: List of neurons to extract activations for
        device: Device to run on
        attention_mask: Optional attention mask [seq_len]
    
    Returns:
        Activation tensor [seq_len, n_neurons] where columns correspond to neuron_info_list order
    """
    # Initialize storage for activations
    activation_storage: Dict[str, torch.Tensor] = {}
    
    # Create hooks (transformer_lens style)
    hooks = create_neuron_activation_hooks(neuron_info_list, activation_storage)
    
    # Prepare input - ensure batch dimension exists
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)  # Add batch dimension: [1, seq_len]
    elif tokens.dim() == 2:
        # Already has batch dimension: [batch, seq_len]
        pass
    else:
        raise ValueError(f"Expected tokens to be 1D or 2D, got {tokens.dim()}D with shape {tokens.shape}")
    
    if attention_mask is None:
        attention_mask = torch.ones_like(tokens)
    else:
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)  # Add batch dimension: [1, seq_len]
        elif attention_mask.dim() == 2:
            # Already has batch dimension: [batch, seq_len]
            pass
        else:
            raise ValueError(f"Expected attention_mask to be 1D or 2D, got {attention_mask.dim()}D with shape {attention_mask.shape}")
        
        # Ensure attention_mask matches tokens batch size
        if attention_mask.shape[0] != tokens.shape[0]:
            raise ValueError(f"attention_mask batch size {attention_mask.shape[0]} doesn't match tokens batch size {tokens.shape[0]}")
    
    # Run model with hooks (transformer_lens style)
    model.eval()
    with torch.no_grad():
        with model.hooks(hooks):
            _ = model.forward(
                sequence_tokens=tokens.to(device),
                sequence_id=attention_mask.to(device)
            )
    
    # Collect activations in the order of neuron_info_list
    activations_list = []
    for neuron_info in neuron_info_list:
        if neuron_info.component_id not in activation_storage:
            raise ValueError(f"Activation not found for neuron {neuron_info.component_id}")
        
        # Get activation: [batch, seq_len] -> [seq_len]
        neuron_activation = activation_storage[neuron_info.component_id][0]  # Remove batch dim
        activations_list.append(neuron_activation)
    
    # Stack to get [seq_len, n_neurons]
    # Each activation is [seq_len], stacking along dim=1 creates new dimension at position 1
    activations = torch.stack(activations_list, dim=1)  # [seq_len, n_neurons]
    
    return activations
