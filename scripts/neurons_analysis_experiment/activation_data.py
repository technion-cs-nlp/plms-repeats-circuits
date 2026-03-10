"""
Activation data structures for neuron concept analysis.

Handles building and managing activation tensors with per-neuron dimensions.
"""
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

import torch
import pandas as pd


@dataclass
class TopKSequence:
    """Represents a sequence in top k activations."""
    seq_name: str
    value: float
    token: str


@dataclass
class NeuronActivationStats:
    """Statistics for a single neuron's activations."""
    # Basic statistics
    mean: float
    median: float
    max: float
    min: float
    variance: float
    
    # Percentage statistics
    pct_positive: float  # Percentage of positive activations
    pct_negative: float  # Percentage of negative activations
    
    # Top k sequences
    top_k_max: List[TopKSequence]  # Top k sequences with highest max activation
    top_k_min: List[TopKSequence]  # Top k sequences with lowest min activation


@dataclass
class NeuronInfo:
    """Complete information about a neuron, including all metadata."""
    # Basic neuron info
    component_id: str
    layer: int
    neuron_idx: int  # Neuron index from the CSV
    
    # Activation statistics (initialized as None, populated during build)
    stats_with_repeats: Optional[NeuronActivationStats] = None


@dataclass
class ActivationData:
    """Container for activation tensors and their metadata."""
    # Activation matrices: [n_tokens, n_neurons]
    activations_with_repeats: torch.Tensor  # Shape: [n_tokens_with_repeats, n_neurons]
    
    # Token sequences
    tokens_with_repeats: torch.Tensor
    
    # Neuron metadata
    neuron_info_list: list[NeuronInfo]  # List of neuron info, ordered by position in activation matrix
    # Note: The index in this list corresponds to the column index in the activation matrix
    
    # Position mappings
    seq_to_range_with_repeats: Dict[str, Tuple[int, int]]
    


class ActivationDataBuilder:
    """
    Incrementally builds activation data structure during iteration.
    More efficient than pre-building since we iterate through examples anyway.
    """
    def __init__(self, device, activations_d_type: torch.dtype, neurons_df: pd.DataFrame, tokenizer=None, top_k: int = 10):
        self.device = device
        self.activations_d_type = activations_d_type
        self.top_k = top_k
        self.tokenizer = tokenizer
        
        # Counters
        self.n_tokens_with_repeats = 0
        
        # Accumulators (lists for efficient incremental concatenation)
        self.token_lists_with_repeats = []
        self.activation_lists_with_repeats = []  # List of [seq_len, n_neurons] tensors
        
        # Mappings
        self.seq_to_range_with_repeats: Dict[str, Tuple[int, int]] = {}
        
        # Neuron metadata
        self.neuron_info_list = []
        self.n_neurons = 0
        
        # Top k tracking: For each neuron, track top k max/min sequences
        self.top_k_max_with_repeats: List[List[Tuple[str, float, str]]] = []
        self.top_k_min_with_repeats: List[List[Tuple[str, float, str]]] = []
        
        if neurons_df is None:
            raise ValueError("neurons_df is required. Please provide a DataFrame with neuron information.")
        self._load_neuron_info(neurons_df)
    
    def _load_neuron_info(self, neurons_df: pd.DataFrame):
        """Load neuron information from CSV and create mappings."""
        # Filter to neurons only
        neurons_df = neurons_df[neurons_df['component_type'] == 'neuron'].copy()
        neurons_df = neurons_df.sort_values(['layer', 'neuron_idx']).reset_index(drop=True)
        
        for idx, row in neurons_df.iterrows():
            neuron_info = NeuronInfo(
                component_id=row['component_id'],
                layer=int(row['layer']),
                neuron_idx=int(row['neuron_idx'])
            )
            self.neuron_info_list.append(neuron_info)
        
        self.n_neurons = len(self.neuron_info_list)
        print(f"  Loaded {self.n_neurons} neurons from CSV", flush=True)
        
        # Initialize top k tracking lists for each neuron
        self.top_k_max_with_repeats = [[] for _ in range(self.n_neurons)]
        self.top_k_min_with_repeats = [[] for _ in range(self.n_neurons)]
    
    def add_sequence_with_repeats(self, tokens: torch.Tensor, name: str, activations: Optional[torch.Tensor] = None):
        """
        Add a sequence for proteins with repeats (clean_masked).
        
        Args:
            tokens: Token sequence tensor [seq_len]
            name: Sequence name
            activations: Optional activation tensor [seq_len, n_neurons]. If None, random activations will be generated.
        """
        # Ensure we have a 1D tensor
        if not isinstance(tokens, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(tokens)}")
        
        # Ensure 1D or 2D with batch_size=1, then flatten to 1D
        if tokens.dim() == 2:
            if tokens.shape[0] != 1:
                raise ValueError(f"Expected batch_size=1 for 2D tensor, got batch_size={tokens.shape[0]}")
            tokens = tokens.squeeze(0)  # Remove batch dimension: [1, seq_len] -> [seq_len]
        elif tokens.dim() != 1:
            raise ValueError(f"Expected 1D or 2D tensor, got {tokens.dim()}D tensor with shape {tokens.shape}")
        
        tokens = tokens.to(self.device)  # Ensure on correct device
        
        seq_len = len(tokens)
        start = self.n_tokens_with_repeats
        end = start + seq_len - 1  # Inclusive end: [start, end] includes both start and end
        
        self.seq_to_range_with_repeats[name] = (start, end)
        self.n_tokens_with_repeats += seq_len
        self.token_lists_with_repeats.append(tokens)
        
        # Handle activations
        if activations is None:
            # Generate random activations for testing: [seq_len, n_neurons]
            activations = torch.randn(seq_len, self.n_neurons, device=self.device, dtype=self.activations_d_type)
        else:
            # Ensure correct shape and device
            activations = activations.to(device=self.device, dtype=self.activations_d_type)
            if activations.dim() != 2:
                raise ValueError(f"Expected 2D activations tensor [seq_len, n_neurons], got {activations.dim()}D with shape {activations.shape}")
            if activations.shape[0] != seq_len:
                raise ValueError(f"Activations sequence length {activations.shape[0]} doesn't match tokens sequence length {seq_len}")
            if activations.shape[1] != self.n_neurons:
                raise ValueError(f"Activations shape {activations.shape} doesn't match n_neurons={self.n_neurons}")
        
        self.activation_lists_with_repeats.append(activations)
        
        # Update top k tracking for this sequence
        self._update_top_k_for_sequence(activations, tokens, name, self.top_k_max_with_repeats, self.top_k_min_with_repeats)

    def _update_top_k_for_sequence(
        self, 
        activations: torch.Tensor, 
        tokens: torch.Tensor,
        seq_name: str,
        top_k_max_list: List[List[Tuple[str, float, str]]],
        top_k_min_list: List[List[Tuple[str, float, str]]]
    ):
        """
        Update top k max/min tracking for a sequence.
        
        Args:
            activations: [seq_len, n_neurons] tensor
            tokens: [seq_len] tensor
            seq_name: Name of the sequence
            top_k_max_list: List of top k max lists (one per neuron)
            top_k_min_list: List of top k min lists (one per neuron)
        """
        # Compute per-neuron max and min for this sequence
        seq_max_vals, seq_max_indices = activations.max(dim=0)  # [n_neurons]
        seq_min_vals, seq_min_indices = activations.min(dim=0)  # [n_neurons]
        
        for neuron_idx in range(self.n_neurons):
            max_val = seq_max_vals[neuron_idx].item()
            min_val = seq_min_vals[neuron_idx].item()
            
            # Get tokens
            if self.tokenizer is not None:
                max_token_idx = seq_max_indices[neuron_idx].item()
                min_token_idx = seq_min_indices[neuron_idx].item()
                
                # Decode token if tokenizer is available
                # Handle potential index out of bounds if tokens tensor is shorter/mismatched (shouldn't happen if validated)
                if max_token_idx < len(tokens):
                    max_token_id = tokens[max_token_idx].item()
                    # Use only the token ID, decode it
                    max_token_str = self.tokenizer.decode([max_token_id])
                else:
                    raise IndexError(f"Max token index {max_token_idx} out of bounds for tokens tensor of length {len(tokens)}")
                    
                if min_token_idx < len(tokens):
                    min_token_id = tokens[min_token_idx].item()
                    min_token_str = self.tokenizer.decode([min_token_id])
                else:
                    raise IndexError(f"Min token index {min_token_idx} out of bounds for tokens tensor of length {len(tokens)}")
            else:
                raise ValueError("Tokenizer is required to decode tokens but is not provided.")
            
            # Update top k max
            top_k_max = top_k_max_list[neuron_idx]
            top_k_max.append((seq_name, max_val, max_token_str))
            top_k_max.sort(key=lambda x: x[1], reverse=True)  # Sort descending by value
            if len(top_k_max) > self.top_k:
                top_k_max.pop()  # Remove lowest if over limit
            
            # Update top k min
            top_k_min = top_k_min_list[neuron_idx]
            top_k_min.append((seq_name, min_val, min_token_str))
            top_k_min.sort(key=lambda x: x[1])  # Sort ascending by value
            if len(top_k_min) > self.top_k:
                top_k_min.pop()  # Remove highest if over limit

    def build(self) -> ActivationData:
        """Finalize and return the ActivationData object."""
        tokens_with_repeats = torch.cat(self.token_lists_with_repeats, dim=0) if self.token_lists_with_repeats else torch.empty(0, dtype=torch.long, device=self.device)

        if self.activation_lists_with_repeats:
            activations_with_repeats = torch.cat(self.activation_lists_with_repeats, dim=0)
        else:
            activations_with_repeats = torch.zeros(self.n_tokens_with_repeats, self.n_neurons, device=self.device, dtype=self.activations_d_type)

        neuron_stats_with_repeats = self._compute_neuron_stats(
            activations_with_repeats,
            self.top_k_max_with_repeats,
            self.top_k_min_with_repeats
        )

        for idx, neuron_info in enumerate(self.neuron_info_list):
            neuron_info.stats_with_repeats = neuron_stats_with_repeats[idx]

        return ActivationData(
            activations_with_repeats=activations_with_repeats,
            tokens_with_repeats=tokens_with_repeats,
            neuron_info_list=self.neuron_info_list,
            seq_to_range_with_repeats=self.seq_to_range_with_repeats
        )
    
    def _compute_neuron_stats(
        self,
        activations: torch.Tensor,
        top_k_max_list: List[List[Tuple[str, float, str]]],
        top_k_min_list: List[List[Tuple[str, float, str]]]
    ) -> List[NeuronActivationStats]:
        """
        Compute statistics for each neuron from the activation tensor.
        
        Args:
            activations: [n_tokens, n_neurons] tensor
            top_k_max_list: List of top k max sequences per neuron
            top_k_min_list: List of top k min sequences per neuron
        
        Returns:
            List of NeuronActivationStats, one per neuron
        """
        stats_list = []
        
        for neuron_idx in range(self.n_neurons):
            # Extract activations for this neuron: [n_tokens]
            neuron_activations = activations[:, neuron_idx]
            
            # Compute basic statistics
            n_total = neuron_activations.numel()
            if n_total == 0:
                # Handle empty tensor case
                mean = 0.0
                median = 0.0
                max_val = 0.0
                min_val = 0.0
                variance = 0.0
                pct_positive = 0.0
                pct_negative = 0.0
            else:
                mean = neuron_activations.mean().item()
                median = neuron_activations.median().item()
                max_val = neuron_activations.max().item()
                min_val = neuron_activations.min().item()
                variance = neuron_activations.var().item()
                
                # Compute percentage statistics
                n_positive = (neuron_activations > 0).sum().item()
                n_negative = (neuron_activations < 0).sum().item()
                pct_positive = (n_positive / n_total * 100) if n_total > 0 else 0.0
                pct_negative = (n_negative / n_total * 100) if n_total > 0 else 0.0
            
            # Convert top k lists to TopKSequence objects
            top_k_max = [
                TopKSequence(seq_name=name, value=val, token=token)
                for name, val, token in top_k_max_list[neuron_idx]
            ]
            top_k_min = [
                TopKSequence(seq_name=name, value=val, token=token)
                for name, val, token in top_k_min_list[neuron_idx]
            ]
            
            stats = NeuronActivationStats(
                mean=mean,
                median=median,
                max=max_val,
                min=min_val,
                variance=variance,
                pct_positive=pct_positive,
                pct_negative=pct_negative,
                top_k_max=top_k_max,
                top_k_min=top_k_min
            )
            stats_list.append(stats)
        
        return stats_list

