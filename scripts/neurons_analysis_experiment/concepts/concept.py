from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerFast

from .concept_utils import (
    RepeatsInfoAboutSequence,
    _to_long_tensor,
    compare_seq_positions_to_token_values,
)

class ConceptCategory(Enum):
    REPEAT = 1
    SPECIAL_TOKENS = 2
    BIOLOGY = 3


@dataclass
class SequenceMetadata:
    name: str
    is_seq_contain_repeat: bool
    tokenized_seq: torch.Tensor
    masked_position_after_tokenization: int
    tokenization_shift: int
    repeats_info_about_sequence: Optional[RepeatsInfoAboutSequence]
    bos_position_after_tokenization: torch.Tensor
    eos_position_after_tokenization: torch.Tensor
    unk_positions: torch.Tensor
    pad_positions: torch.Tensor
    original_sequence: Optional[str] = None


class ConceptFunctionTagger(ABC):
    @abstractmethod
    def tag_sequence(self, sequence_metadata: SequenceMetadata, tokenizer: PreTrainedTokenizerFast) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class Concept:
    def __init__(
        self,
        concept_name: str,
        concept_category: ConceptCategory,
        concept_function_tagger: ConceptFunctionTagger
    ):
        self.concept_name: str = concept_name
        self.concept_category: ConceptCategory = concept_category
        self.concept_function_tagger = concept_function_tagger


def validate_concept_positions(tokenized_seq: torch.Tensor, valid_positions_tensor: torch.Tensor, invalid_positions_tensor: torch.Tensor, concept_name: str = ""):
    seq_len = len(tokenized_seq)
    all_positions = torch.arange(seq_len, device=tokenized_seq.device)
    
    if valid_positions_tensor.numel() > 0:
        max_valid = valid_positions_tensor.max().item()
        min_valid = valid_positions_tensor.min().item()
        if min_valid < 0 or max_valid >= seq_len:
            raise ValueError(
                f"{concept_name}: Valid positions out of bounds. "
                f"Sequence length: {seq_len}, valid positions range: [{min_valid}, {max_valid}]"
            )
        valid_not_in_seq = valid_positions_tensor[~torch.isin(valid_positions_tensor, all_positions)]
        if valid_not_in_seq.numel() > 0:
            raise ValueError(
                f"{concept_name}: Valid positions contain values not in sequence: {valid_not_in_seq.tolist()}"
            )
    
    if invalid_positions_tensor.numel() > 0:
        max_invalid = invalid_positions_tensor.max().item()
        min_invalid = invalid_positions_tensor.min().item()
        if min_invalid < 0 or max_invalid >= seq_len:
            raise ValueError(
                f"{concept_name}: Invalid positions out of bounds. "
                f"Sequence length: {seq_len}, invalid positions range: [{min_invalid}, {max_invalid}]"
            )
        invalid_not_in_seq = invalid_positions_tensor[~torch.isin(invalid_positions_tensor, all_positions)]
        if invalid_not_in_seq.numel() > 0:
            raise ValueError(
                f"{concept_name}: Invalid positions contain values not in sequence: {invalid_not_in_seq.tolist()}"
            )
    
    if valid_positions_tensor.numel() > 0 and invalid_positions_tensor.numel() > 0:
        intersection = valid_positions_tensor[torch.isin(valid_positions_tensor, invalid_positions_tensor)]
        if intersection.numel() > 0:
            raise ValueError(
                f"{concept_name}: Valid and invalid positions have non-empty intersection: {intersection.tolist()}"
            )


def create_general_invalid_positions_tensor(sequence_metadata, to_exclude_eos, to_exclude_bos, to_exclude_mask) -> torch.Tensor:
    device = sequence_metadata.pad_positions.device

    tensors_to_combine = [sequence_metadata.pad_positions, sequence_metadata.unk_positions]

    if to_exclude_eos:
        eos_position = _to_long_tensor(sequence_metadata.eos_position_after_tokenization, device)
        tensors_to_combine.append(eos_position)
    if to_exclude_bos:
        bos_position = _to_long_tensor(sequence_metadata.bos_position_after_tokenization, device)
        tensors_to_combine.append(bos_position)
    if to_exclude_mask:
        masked_position = _to_long_tensor(sequence_metadata.masked_position_after_tokenization, device)
        tensors_to_combine.append(masked_position)
    
    if len(tensors_to_combine) == 0:
        invalid_positions = torch.empty(0, dtype=torch.long, device=device)
    else:
        invalid_positions = torch.cat(tensors_to_combine)
    
    invalid_positions = torch.unique(invalid_positions, sorted=True)

    return invalid_positions


class TokenConceptFunctionTagger(ConceptFunctionTagger):
    def __init__(
        self,
        token_values: List[int],
        concept_name: str,
        to_exclude_eos: bool = True,
        to_exclude_bos: bool = True,
        to_exclude_mask: bool = True
    ):
        self.token_values = token_values
        self.concept_name = concept_name
        self.to_exclude_eos = to_exclude_eos
        self.to_exclude_bos = to_exclude_bos
        self.to_exclude_mask = to_exclude_mask

    def tag_sequence(
        self,
        sequence_metadata: SequenceMetadata,
        tokenizer: PreTrainedTokenizerFast
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        invalid_positions_tensor = create_general_invalid_positions_tensor(
            sequence_metadata,
            to_exclude_eos=self.to_exclude_eos,
            to_exclude_bos=self.to_exclude_bos,
            to_exclude_mask=self.to_exclude_mask
        )
        valid_positions_tensor = compare_seq_positions_to_token_values(
            self.token_values,
            sequence_metadata.tokenized_seq
        )
        if valid_positions_tensor.numel() > 0 and invalid_positions_tensor.numel() > 0:
            valid_positions_tensor = valid_positions_tensor[~torch.isin(valid_positions_tensor, invalid_positions_tensor)]
            valid_positions_tensor = torch.unique(valid_positions_tensor, sorted=True)
        validate_concept_positions(sequence_metadata.tokenized_seq, valid_positions_tensor, invalid_positions_tensor, self.concept_name)
        return valid_positions_tensor, invalid_positions_tensor