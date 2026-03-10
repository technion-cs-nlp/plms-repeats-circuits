"""
Optimized AUROC Analyzer for neuron-concept relationships.

Key optimizations:
1. Vectorized computation across all neurons simultaneously
2. TorchMetrics for fast AUROC calculation
3. Batched statistics computation
4. Eliminated redundant sampling loops
"""
import math
import numpy as np
import pandas as pd
import torch
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from torchmetrics.functional.classification import binary_auroc

from activation_data import ActivationData
from concepts.neuron_concept_maker import ConceptGlobalIndicies
from concepts.concept import Concept


@dataclass
class ConceptNeuronMetadata:
    """Metadata for a concept-neuron pair."""
    # Concept group statistics
    concept_mean: float
    concept_var: float
    concept_max: float
    concept_min: float
    concept_mean_abs: float
    concept_var_abs: float
    concept_pct_positive: float
    concept_pct_negative: float
    
    # Non-concept group statistics
    non_concept_mean: float
    non_concept_var: float
    non_concept_max: float
    non_concept_min: float
    non_concept_mean_abs: float
    non_concept_var_abs: float
    non_concept_pct_positive: float
    non_concept_pct_negative: float

    # Combined variance
    combined_variance: float
    
    # Group sizes and frequencies
    concept_group_size: int
    non_concept_group_size: int
    invalid_group_size: int
    concept_frequency: float
    non_concept_frequency: float

    # AUROC info
    auroc: float
    one_minus_auroc: float
    best_auroc: float  # max(auroc, one_minus_auroc)

    used_absolute_activations: bool = False


class AUROCAnalyzerOptimized:
    """
    Optimized analyzer for computing AUROC between concepts and neuron activations.
    
    Key improvements over original:
    - Vectorized computation across all neurons
    - TorchMetrics for fast AUROC
    - Single-pass statistics computation
    - Eliminated redundant sampling
    """
    
    def __init__(
        self,
        activation_data: ActivationData,
        concepts_global_indicies_dict: Dict[str, ConceptGlobalIndicies],
        concepts_array_proteins_with_repeats: List[Concept],
        use_absolute_activations: bool = False,
        min_samples: int = 48,
        device: torch.device = None,
        seed: int = 42
    ):
        """Optimized AUROC analyzer - vectorized, uses full data."""
        self.activation_data = activation_data
        self.concepts_global_indicies_dict = concepts_global_indicies_dict
        self.concepts_array_proteins_with_repeats = concepts_array_proteins_with_repeats
        self.use_absolute_activations = use_absolute_activations  # If True, use abs(activations) for all computations
        self.min_samples = min_samples
        self.device = device if device is not None else torch.device('cpu')
        self.seed = seed
        
        self.n_neurons = len(activation_data.neuron_info_list)
        
        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        print(f"[AUROC] TorchMetrics | seed={seed} | use_absolute_activations={use_absolute_activations}")
    
    def analyze(self) -> pd.DataFrame:
        """
        Analyze all concepts and return results as a pandas DataFrame.
        
        Returns:
            DataFrame where each row represents one concept-neuron pair
        """
        print(f"\n[AUROC Analysis] Starting analysis for {len(self.concepts_array_proteins_with_repeats)} concepts x {self.n_neurons} neurons")
        
        all_results = []
        
        # Analyze concepts for proteins with repeats only
        for concept in self.concepts_array_proteins_with_repeats:
            concept_name = concept.concept_name
            start_time = time.time()
            
            concept_results = self._analyze_concept_vectorized(concept)
            if concept_results is not None:
                all_results.extend(concept_results)
            
            elapsed_time = time.time() - start_time
            print(f"  ✓ '{concept_name}': {elapsed_time:.3f}s")
        
        return pd.DataFrame(all_results)
    
    def _analyze_concept_vectorized(self, concept: Concept) -> Optional[List[Dict]]:
        """
        Analyze a single concept across ALL neurons at once (vectorized).
        
        Returns:
            List of result dictionaries, one per neuron, or None if skipped
        """
        concept_name = concept.concept_name
        concept_activations_matrix, non_concept_activations_matrix = self._get_activation_matrices_repeats_only(concept)
        
        if concept_activations_matrix is None:
            return None
        
        if self.use_absolute_activations:
            concept_activations_matrix = torch.abs(concept_activations_matrix)
            non_concept_activations_matrix = torch.abs(non_concept_activations_matrix)
        
        invalid_positions = self._get_invalid_positions_repeats_only(concept)
        invalid_group_size = invalid_positions.shape[0] if invalid_positions.numel() > 0 else 0
        
        # Compute statistics on FULL data - VECTORIZED
        concept_stats = self._compute_stats_vectorized(concept_activations_matrix)
        non_concept_stats = self._compute_stats_vectorized(non_concept_activations_matrix)
        
        # Compute combined variance and frequencies
        combined_activations = torch.cat([concept_activations_matrix, non_concept_activations_matrix], dim=0)
        combined_variance = torch.var(combined_activations, dim=0) if combined_activations.shape[0] > 0 else torch.zeros(self.n_neurons, device=self.device)
        total_valid = concept_activations_matrix.shape[0] + non_concept_activations_matrix.shape[0]
        concept_frequency = concept_activations_matrix.shape[0] / total_valid if total_valid > 0 else 0.0
        non_concept_frequency = non_concept_activations_matrix.shape[0] / total_valid if total_valid > 0 else 0.0
        
        all_auroc_results = self._compute_all_aurocs(
            concept_activations_matrix,
            non_concept_activations_matrix,
            concept_name
        )
        
        auroc_scores = all_auroc_results['auroc']
        auroc_one_minus = all_auroc_results['one_minus_auroc']
        best_auroc_scores = all_auroc_results['best_auroc']
        
        results = []
        for neuron_idx in range(self.n_neurons):
            neuron_info = self.activation_data.neuron_info_list[neuron_idx]
            metadata = ConceptNeuronMetadata(
                concept_mean=concept_stats['mean'][neuron_idx].item(),
                concept_var=concept_stats['var'][neuron_idx].item(),
                concept_max=concept_stats['max'][neuron_idx].item(),
                concept_min=concept_stats['min'][neuron_idx].item(),
                concept_mean_abs=concept_stats['mean_abs'][neuron_idx].item(),
                concept_var_abs=concept_stats['var_abs'][neuron_idx].item(),
                concept_pct_positive=concept_stats['pct_positive'][neuron_idx].item(),
                concept_pct_negative=concept_stats['pct_negative'][neuron_idx].item(),
                non_concept_mean=non_concept_stats['mean'][neuron_idx].item(),
                non_concept_var=non_concept_stats['var'][neuron_idx].item(),
                non_concept_max=non_concept_stats['max'][neuron_idx].item(),
                non_concept_min=non_concept_stats['min'][neuron_idx].item(),
                non_concept_mean_abs=non_concept_stats['mean_abs'][neuron_idx].item(),
                non_concept_var_abs=non_concept_stats['var_abs'][neuron_idx].item(),
                non_concept_pct_positive=non_concept_stats['pct_positive'][neuron_idx].item(),
                non_concept_pct_negative=non_concept_stats['pct_negative'][neuron_idx].item(),
                auroc=auroc_scores[neuron_idx].item(),
                one_minus_auroc=auroc_one_minus[neuron_idx].item(),
                best_auroc=best_auroc_scores[neuron_idx].item(),
                concept_group_size=concept_activations_matrix.shape[0],
                non_concept_group_size=non_concept_activations_matrix.shape[0],
                invalid_group_size=invalid_group_size,
                concept_frequency=concept_frequency,
                non_concept_frequency=non_concept_frequency,
                combined_variance=combined_variance[neuron_idx].item(),
                used_absolute_activations=self.use_absolute_activations
            )
            
            results.append({
                'concept_name': concept_name,
                'activation_data_idx': neuron_idx,
                'neuron_idx': neuron_info.neuron_idx,
                'component_id': neuron_info.component_id,
                'layer': neuron_info.layer,
                **{k: v for k, v in metadata.__dict__.items()}
            })
        
        return results
    
    def _compute_single_auroc(self, preds: torch.Tensor, target: torch.Tensor) -> float:
        """Compute AUROC for a single neuron (curve created separately to manage memory)."""
        auroc = binary_auroc(preds, target.long(), thresholds=None, validate_args=False)
        result = float('nan') if (torch.isnan(auroc) or torch.isinf(auroc)) else auroc.item()
        del auroc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    
    def _compute_all_aurocs(
        self,
        concept_activations: torch.Tensor,
        non_concept_activations: torch.Tensor,
        concept_name: str
    ) -> Dict[str, torch.Tensor]:
        n_concept, n_non_concept = concept_activations.shape[0], non_concept_activations.shape[0]
        target = torch.cat([torch.ones(n_concept, device=self.device), torch.zeros(n_non_concept, device=self.device)])
        auroc_scores = torch.zeros(self.n_neurons, device=self.device)
        for neuron_idx in range(self.n_neurons):
            preds = torch.cat([concept_activations[:, neuron_idx], non_concept_activations[:, neuron_idx]])
            auroc = self._compute_single_auroc(preds, target)
            del preds
            if not math.isnan(auroc):
                auroc_scores[neuron_idx] = auroc
            else:
                neuron_info = self.activation_data.neuron_info_list[neuron_idx]
                raise ValueError(f"Invalid AUROC for '{concept_name}' neuron {neuron_info.component_id}")
        
        del target
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        one_minus_auroc = 1.0 - auroc_scores
        best_auroc_scores = torch.maximum(auroc_scores, one_minus_auroc)
        
        return {
            'auroc': auroc_scores,
            'one_minus_auroc': one_minus_auroc,
            'best_auroc': best_auroc_scores,
        }
    
    def _compute_stats_vectorized(self, activations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute statistics for all neurons at once.
        
        Args:
            activations: [n_samples, n_neurons]
        
        Returns:
            Dictionary with statistics, each shape [n_neurons]
        """
        if activations.shape[0] == 0:
            return {
                'mean': torch.zeros(self.n_neurons, device=self.device),
                'var': torch.zeros(self.n_neurons, device=self.device),
                'max': torch.zeros(self.n_neurons, device=self.device),
                'min': torch.zeros(self.n_neurons, device=self.device),
                'mean_abs': torch.zeros(self.n_neurons, device=self.device),
                'var_abs': torch.zeros(self.n_neurons, device=self.device),
                'pct_positive': torch.zeros(self.n_neurons, device=self.device),
                'pct_negative': torch.zeros(self.n_neurons, device=self.device),
            }
        
        abs_activations = torch.abs(activations)
        
        return {
            'mean': torch.mean(activations, dim=0),  # [n_neurons]
            'var': torch.var(activations, dim=0),  # [n_neurons]
            'max': torch.max(activations, dim=0)[0],  # [n_neurons]
            'min': torch.min(activations, dim=0)[0],  # [n_neurons]
            'mean_abs': torch.mean(abs_activations, dim=0),  # [n_neurons]
            'var_abs': torch.var(abs_activations, dim=0),  # [n_neurons]
            'pct_positive': (activations > 0).float().mean(dim=0),  # [n_neurons]
            'pct_negative': (activations < 0).float().mean(dim=0),  # [n_neurons]
        }
    
    def _get_activation_matrices_repeats_only(
        self, concept: Concept
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        concept_name = concept.concept_name
        if concept_name not in self.concepts_global_indicies_dict:
            print(f"  ⚠ Skipping '{concept_name}': not found in concepts_global_indicies_dict")
            return None, None
        concept_global_indices = self.concepts_global_indicies_dict[concept_name]
        concept_positions = concept_global_indices.global_valid_positions_for_concept_on_proteins_with_repeats
        invalid_positions = concept_global_indices.global_invalid_positions_for_concept_on_proteins_with_repeats
        all_positions = torch.arange(
            self.activation_data.activations_with_repeats.shape[0],
            device=self.device,
            dtype=torch.long
        )
        combined = torch.cat([concept_positions, invalid_positions]) if concept_positions.numel() > 0 or invalid_positions.numel() > 0 else torch.empty(0, dtype=torch.long, device=self.device)
        non_concept_positions = all_positions[~torch.isin(all_positions, combined)] if combined.numel() > 0 else all_positions
        concept_activations_matrix = self.activation_data.activations_with_repeats[concept_positions, :]
        non_concept_activations_matrix = self.activation_data.activations_with_repeats[non_concept_positions, :]
        if concept_activations_matrix.shape[0] < self.min_samples or non_concept_activations_matrix.shape[0] < self.min_samples:
            print(f"  ⚠ Skipping '{concept_name}': insufficient samples (concept={concept_activations_matrix.shape[0]}, non_concept={non_concept_activations_matrix.shape[0]}, min={self.min_samples})")
            return None, None
        return concept_activations_matrix, non_concept_activations_matrix
    
    def _get_invalid_positions_repeats_only(self, concept: Concept) -> torch.Tensor:
        """Get invalid positions for a concept that only applies to proteins with repeats."""
        concept_name = concept.concept_name
        if concept_name not in self.concepts_global_indicies_dict:
            return torch.empty(0, dtype=torch.long, device=self.device)
        
        concept_global_indices = self.concepts_global_indicies_dict[concept_name]
        invalid_positions = concept_global_indices.global_invalid_positions_for_concept_on_proteins_with_repeats
        return invalid_positions if invalid_positions.numel() > 0 else torch.empty(0, dtype=torch.long, device=self.device)

