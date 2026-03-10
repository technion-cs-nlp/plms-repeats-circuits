from typing import Dict, List
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerFast

from .concept import Concept, SequenceMetadata
from .repeat_concepts import make_repeat_concepts
from .biochemical_concepts import (
    make_biochemical_properties_concepts,
)
from .special_token_concepts import make_special_token_concepts

@dataclass
class ConceptGlobalIndicies:
    concept_name: str
    global_valid_positions_for_concept_on_proteins_with_repeats: torch.Tensor
    global_invalid_positions_for_concept_on_proteins_with_repeats: torch.Tensor

@dataclass
class TagSequenceWithRepeatsConceptsInput:
    sequence_metadata: SequenceMetadata
    tokenizer: PreTrainedTokenizerFast
    concepts_array_proteins_with_repeats: List[Concept]
    concepts_global_indicies_dict: Dict[str, ConceptGlobalIndicies]
    tokenized_seq_start_point_index_in_activations_tensor_for_proteins_with_repeats: int
    sequence_name: str = ""

class ConceptsMaker:

    def make_concepts_array(self, tokenizer: PreTrainedTokenizerFast):
        
        repeat_concepts = make_repeat_concepts()
        concepts_array_proteins_with_repeats = []
        concepts_array_proteins_with_repeats.extend(repeat_concepts)

        biochemical_properties_concepts = make_biochemical_properties_concepts(tokenizer)
        concepts_array_proteins_with_repeats.extend(biochemical_properties_concepts)

        special_token_concepts = make_special_token_concepts(tokenizer)
        concepts_array_proteins_with_repeats.extend(special_token_concepts)

        return concepts_array_proteins_with_repeats

    def _insert_concept_to_dict(self, concepts_global_indicies_dict: Dict[str, ConceptGlobalIndicies], concept_name: str, valid_positions_tensor: torch.Tensor, invalid_positions_tensor: torch.Tensor):
        if concept_name in concepts_global_indicies_dict:
            dict_obj = concepts_global_indicies_dict[concept_name]
            curr_valid = dict_obj.global_valid_positions_for_concept_on_proteins_with_repeats
            curr_invalid = dict_obj.global_invalid_positions_for_concept_on_proteins_with_repeats
            curr_valid = valid_positions_tensor if curr_valid is None else torch.unique(torch.cat([curr_valid, valid_positions_tensor]), sorted=True)
            curr_invalid = invalid_positions_tensor if curr_invalid is None else torch.unique(torch.cat([curr_invalid, invalid_positions_tensor]), sorted=True)
            dict_obj.global_valid_positions_for_concept_on_proteins_with_repeats = curr_valid
            dict_obj.global_invalid_positions_for_concept_on_proteins_with_repeats = curr_invalid
        else:
            obj = ConceptGlobalIndicies(
                concept_name=concept_name,
                global_valid_positions_for_concept_on_proteins_with_repeats=torch.unique(valid_positions_tensor, sorted=True),
                global_invalid_positions_for_concept_on_proteins_with_repeats=torch.unique(invalid_positions_tensor, sorted=True),
            )
            concepts_global_indicies_dict[concept_name] = obj

    def tag_sequence_with_repeats(self, tag_input: TagSequenceWithRepeatsConceptsInput):
        for concept in tag_input.concepts_array_proteins_with_repeats:
            try:
                valid_positions_tensor, invalid_positions_tensor = concept.concept_function_tagger.tag_sequence(tag_input.sequence_metadata, tag_input.tokenizer)
            except ValueError as e:
                raise ValueError(f"Error tagging concept '{concept.concept_name}' for sequence '{tag_input.sequence_name}': {e}")
            shift_valid = valid_positions_tensor + tag_input.tokenized_seq_start_point_index_in_activations_tensor_for_proteins_with_repeats
            shift_invalid = invalid_positions_tensor + tag_input.tokenized_seq_start_point_index_in_activations_tensor_for_proteins_with_repeats
            self._insert_concept_to_dict(tag_input.concepts_global_indicies_dict, concept.concept_name, shift_valid, shift_invalid)

