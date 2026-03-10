from typing import Tuple

import torch
from transformers import PreTrainedTokenizerFast

from .concept import (
    ConceptFunctionTagger,
    SequenceMetadata,
    Concept,
    ConceptCategory,
    create_general_invalid_positions_tensor,
    validate_concept_positions
)


class RepeatConceptFunctionTagger(ConceptFunctionTagger):
    def __init__(self, fields_names_for_positions_where_concept_is_true, fields_names_for_positions_where_concept_is_invalid, to_exclude_eos, to_exclude_bos, to_exclude_mask, concept_name):
        self.fields_names_for_positions_where_concept_is_true = fields_names_for_positions_where_concept_is_true
        self.fields_names_for_positions_where_concept_is_invalid = fields_names_for_positions_where_concept_is_invalid
        self.to_exclude_eos = to_exclude_eos
        self.to_exclude_bos = to_exclude_bos
        self.to_exclude_mask = to_exclude_mask
        self.concept_name = concept_name

    def tag_sequence(self, sequence_metadata: SequenceMetadata, tokenizer: PreTrainedTokenizerFast) -> Tuple[torch.Tensor, torch.Tensor]:
        invalid_positions_tensor = create_general_invalid_positions_tensor(sequence_metadata, self.to_exclude_eos, self.to_exclude_bos, self.to_exclude_mask)
        valid_tensors_to_combine = []
        if sequence_metadata.is_seq_contain_repeat and sequence_metadata.repeats_info_about_sequence is not None:
            for field_name in self.fields_names_for_positions_where_concept_is_true:
                valid_tensors_to_combine.append(getattr(sequence_metadata.repeats_info_about_sequence, field_name))
        if len(valid_tensors_to_combine) == 0:
            valid_positions_tensor = torch.empty(0, dtype=torch.long, device=sequence_metadata.tokenized_seq.device)
        else:
            valid_positions_tensor = torch.cat(valid_tensors_to_combine)
            valid_positions_tensor = torch.unique(valid_positions_tensor, sorted=True)
        invalid_tensors_to_combine = [invalid_positions_tensor]
        if sequence_metadata.is_seq_contain_repeat and sequence_metadata.repeats_info_about_sequence is not None:
            for field_name in self.fields_names_for_positions_where_concept_is_invalid:
                invalid_tensors_to_combine.append(getattr(sequence_metadata.repeats_info_about_sequence, field_name))
        invalid_positions_tensor = torch.cat(invalid_tensors_to_combine)
        invalid_positions_tensor = torch.unique(invalid_positions_tensor, sorted=True)
        validate_concept_positions(sequence_metadata.tokenized_seq, valid_positions_tensor, invalid_positions_tensor, self.concept_name)
        return valid_positions_tensor, invalid_positions_tensor


def make_repeat_concepts() -> list:
    repeat_concepts = []
    repeat_concept_on_two_instances = None

    repeat_concept_on_two_instances = Concept(
        concept_name="repeat_tokens",
        concept_category=ConceptCategory.REPEAT,
        concept_function_tagger=RepeatConceptFunctionTagger(
            fields_names_for_positions_where_concept_is_true=["repeat_positions_tensor"],
            fields_names_for_positions_where_concept_is_invalid=[
                "near_repeat_positions_tensor",
                "problematic_positions_in_sequence_tensor_related_to_suspected_repeat"
            ],
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=False,
            concept_name="repeat_tokens"
        )
    )
    
    repeat_concepts.append(repeat_concept_on_two_instances)

    repeat_concepts.append(Concept(
        concept_name="aligned_token_masked_pos",
        concept_category=ConceptCategory.REPEAT,
        concept_function_tagger=RepeatConceptFunctionTagger(
            fields_names_for_positions_where_concept_is_true=["aligned_token_to_masked_position_tensor"],
            fields_names_for_positions_where_concept_is_invalid=[
                "near_repeat_positions_tensor",
                "problematic_positions_in_sequence_tensor_related_to_suspected_repeat"
            ],
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=False,
            concept_name="aligned_token_masked_pos"
        )
    ))

    return repeat_concepts

