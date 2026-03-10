import torch
from pathlib import Path
from typing import Dict

from activation_data import ActivationData
from concepts.neuron_concept_maker import ConceptGlobalIndicies


def save_activation_data(
    activation_data: ActivationData,
    concepts_global_indicies_dict: Dict[str, ConceptGlobalIndicies],
    output_path: Path
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'activations_with_repeats': activation_data.activations_with_repeats,
        'tokens_with_repeats': activation_data.tokens_with_repeats,
        'neuron_info_list': activation_data.neuron_info_list,
        'seq_to_range_with_repeats': activation_data.seq_to_range_with_repeats,
        'concepts_global_indicies_dict': concepts_global_indicies_dict,
        'n_sequences': len(activation_data.seq_to_range_with_repeats),
        'n_tokens': activation_data.activations_with_repeats.shape[0],
        'n_neurons': activation_data.activations_with_repeats.shape[1],
    }
    
    torch.save(data, output_path)
    print(f"Saved activation data to {output_path}")
