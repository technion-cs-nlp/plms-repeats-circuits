import argparse
import ast
import csv
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from esm.utils.constants import esm3 as C
from plms_repeats_circuits.utils.esm_utils import (
    get_probs_from_logits,
    load_model,
    mask_protein_short,
    load_tokenizer_by_model_type,
)
from plms_repeats_circuits.utils.model_utils import get_device
from plms_repeats_circuits.utils.experiment_utils import set_random_seed
from plms_repeats_circuits.utils.protein_similiarity_utils import analyze_repeat_positions

import random
from Bio.Align.substitution_matrices import load


class CounterfactualCreator:
    def __init__(self, default_amino_acid='X', random_seed=42):
        self.blosum62 = load("BLOSUM62")
        self.standard_amino_acids = set("ARNDCEQGHILKMFPSTWYV")
        self.default_amino_acid = default_amino_acid
        self.mask_sampler = random.Random(random_seed)
        self.corrupt_sampler = random.Random(random_seed)

    def _get_top_blosum_substitution(self, original_aa):
        """
        Find the best substitution for the given amino acid using BLOSUM62.
        """
        substitutions = []

        for (aa1, aa2), score in self.blosum62.items():
            if aa1 == original_aa and aa2 in self.standard_amino_acids and aa2 != original_aa:
                substitutions.append((aa2, score))
            elif aa2 == original_aa and aa1 in self.standard_amino_acids and aa1 != original_aa:
                substitutions.append((aa1, score))

        if not substitutions:
            logging.warning(f"No BLOSUM substitution found for {original_aa}. Using default '{self.default_amino_acid}'.")
            return None

        substitutions.sort(key=lambda x: x[1], reverse=True)
        logging.debug(f"BLOSUM substitution for {original_aa} is {substitutions[0][0]} with score {substitutions[0][1]}")
        return substitutions[0][0]

    def _get_lowest_blosum_substitution(self, original_aa):
        """
        Find the worst substitution for the given amino acid using BLOSUM62.
        """
        substitutions = []

        for (aa1, aa2), score in self.blosum62.items():
            if aa1 == original_aa and aa2 in self.standard_amino_acids and aa2 != original_aa:
                substitutions.append((aa2, score))
            elif aa2 == original_aa and aa1 in self.standard_amino_acids and aa1 != original_aa:
                substitutions.append((aa1, score))

        if not substitutions:
            logging.warning(f"No BLOSUM substitution found for {original_aa}. Using default '{self.default_amino_acid}'.")
            return None

        substitutions.sort(key=lambda x: x[1])  # Sort by score ascending (lowest first)
        logging.debug(f"Lowest BLOSUM substitution for {original_aa} is {substitutions[0][0]} with score {substitutions[0][1]}")
        return substitutions[0][0]
    
    def sample_repeat_and_position_to_mask(self, repeat_times, repeat_length, repeat_locations):
        """
        Randomly select one repeat and one position inside it to mask.
        """
        target_repeat_index = self.mask_sampler.randint(0, repeat_times - 1)
        target_start, target_end = repeat_locations[target_repeat_index]
        target_length_index = self.mask_sampler.randint(0, repeat_length - 1)
        target_mask_pos = target_start + target_length_index
        return target_mask_pos, target_repeat_index, target_length_index
    
    
    def sample_identical_position_to_mask_similiar_repeats(self, protein_str, repeat_locations, repeat_alignments, high_confidence_positions):
        if len(repeat_locations) != 2:
            raise Exception("sample_identical_position_to_mask_similiar_repeats support only for repeat times = 2")
        
        # Make sure we have positions to sample from
        if not high_confidence_positions or len(high_confidence_positions) == 0:
            raise Exception("No high confidence positions provided")
        
        target_mask_pos = self.mask_sampler.choice(high_confidence_positions)
        
        target_repeat_index = None
        for i, location in enumerate(repeat_locations):
            start, end = location
            if start <= target_mask_pos <= end:
                target_repeat_index = i
                break
        
        if target_repeat_index is None:
            raise Exception(f"Target mask position {target_mask_pos} is not part of any repeat: {repeat_locations}")
        
        result_abs_pos_to_repeat_info, _ = analyze_repeat_positions(protein_str, repeat_locations, repeat_alignments)
        
        info_target = result_abs_pos_to_repeat_info[target_repeat_index][target_mask_pos]
        if info_target.aligned_matching_absolute_position_in_sequence is None:
            raise Exception("High confidence position does not have an aligned residue - illegal")
        
        other_repeat_abs_aligned_pos = info_target.aligned_matching_absolute_position_in_sequence
        other_repeat_idx = 1 - target_repeat_index  # assuming 2 repeats
        
        try:
            info_other_relative = result_abs_pos_to_repeat_info[other_repeat_idx][other_repeat_abs_aligned_pos]
            target_length_index = info_other_relative.relative_position_in_repeat
            other_start, other_end = repeat_locations[other_repeat_idx]
            assert protein_str[other_start + target_length_index] == protein_str[target_mask_pos], f"Amino acids don't match: {protein_str[other_start + target_length_index]} vs {protein_str[target_mask_pos]}"
        except KeyError:
            raise Exception(f"Could not find aligned position {other_repeat_abs_aligned_pos} in repeat {other_repeat_idx}")
        
        return target_mask_pos, target_repeat_index, target_length_index


    def sample_repeat_and_position_to_mask_mode_baseline_experiment(self, protein_str, repeat_times, repeat_length, repeat_locations, min_space):
        """
        Randomly select one repeat and one position inside it to mask.
        """
        if len(repeat_locations) > 2:
            raise ValueError("This sampling method only supports repeats with 2 occurrences.")
        valid_repeat_for_baseline_experiment = []
        for i, (start, end) in enumerate(repeat_locations):
            if i ==0:
                fake_repeat_start, fake_repeat_end = self._create_repeat_from_the_left(protein_str, start, min_space, repeat_length)
                if fake_repeat_start is not None and fake_repeat_end is not None:
                    valid_repeat_for_baseline_experiment.append(i)
            elif i == 1:
                fake_repeat_start, fake_repeat_end = self._create_repeat_from_the_right(protein_str, end, min_space, repeat_length)
                if fake_repeat_start is not None and fake_repeat_end is not None:
                    valid_repeat_for_baseline_experiment.append(i)
        
        if len(valid_repeat_for_baseline_experiment) == 0:
            return None, None, None
        target_repeat_index = self.mask_sampler.choice(valid_repeat_for_baseline_experiment)
        target_start, target_end = repeat_locations[target_repeat_index]
        target_length_index = self.mask_sampler.randint(0, repeat_length - 1)
        target_mask_pos = target_start + target_length_index
        return target_mask_pos, target_repeat_index, target_length_index

    def _create_repeat_from_the_right(self, protein_str, curr_repeat_end, min_space, repeat_length):
            fake_repeat_start = curr_repeat_end + min_space
            fake_repeat_end = fake_repeat_start + repeat_length - 1
            if fake_repeat_end >= len(protein_str):
                return None, None
            return fake_repeat_start, fake_repeat_end
    def _create_repeat_from_the_left(self, protein_str, curr_repeat_start, min_space, repeat_length):
        fake_repeat_end = curr_repeat_start - min_space
        fake_repeat_start = fake_repeat_end - repeat_length + 1
        if fake_repeat_start < 0:
            return None, None
        return fake_repeat_start, fake_repeat_end
    def _get_indicies_for_fake_repeat(self, protein_str, curr_repeat_start, curr_repeat_end, other_start, other_end, repeat_length, min_space):
        if curr_repeat_start > other_end: # curr is after other. so other is in the left side of the curr [other, curr]
            fake_repeat_start, fake_repeat_end = self._create_repeat_from_the_right(protein_str, curr_repeat_end, min_space, repeat_length)
            if fake_repeat_start is None or fake_repeat_end is None:
                 fake_repeat_start, fake_repeat_end = self._create_repeat_from_the_left(protein_str, other_start, min_space, repeat_length)
        else:
            fake_repeat_start, fake_repeat_end = self._create_repeat_from_the_left(protein_str, curr_repeat_start, min_space, repeat_length)
            if fake_repeat_start is None or fake_repeat_end is None:
                fake_repeat_start, fake_repeat_end = self._create_repeat_from_the_right(protein_str, other_end, min_space, repeat_length)
        return fake_repeat_start, fake_repeat_end
    
    def _validate_position_for_baseline(self, protein_str, repeat_locations, repeat_alignments, target_mask_pos, min_space):
        # Find which repeat the position belongs to
        target_repeat_index = None
        for i, location in enumerate(repeat_locations):
            start, end = location
            if start <= target_mask_pos <= end:
                target_repeat_index = i
                break
        
        if target_repeat_index is None:
            return None
        
        result_abs_pos_to_repeat_info, _ = analyze_repeat_positions(protein_str, repeat_locations, repeat_alignments)
        try:
            info_target = result_abs_pos_to_repeat_info[target_repeat_index][target_mask_pos]
            if info_target.aligned_matching_absolute_position_in_sequence is None:
                return None  
            
            other_repeat_abs_aligned_pos = info_target.aligned_matching_absolute_position_in_sequence
            other_repeat_idx = 1 - target_repeat_index 
            
            info_other_relative = result_abs_pos_to_repeat_info[other_repeat_idx][other_repeat_abs_aligned_pos]
            target_length_index = info_other_relative.relative_position_in_repeat
            
            other_start, other_end = repeat_locations[other_repeat_idx]
            assert protein_str[other_start + target_length_index] == protein_str[target_mask_pos], f"Amino acids don't match: {protein_str[other_start + target_length_index]} vs {protein_str[target_mask_pos]}"
            
            target_start, target_end = repeat_locations[target_repeat_index]
            repeat_length = other_end - other_start + 1
            
            fake_repeat_start, fake_repeat_end = self._get_indicies_for_fake_repeat(
                protein_str, target_start, target_end, other_start, other_end, 
                repeat_length=repeat_length, min_space=min_space
            )
            
            if fake_repeat_start is None or fake_repeat_end is None:
                return None 
            
            if target_length_index >= (fake_repeat_end - fake_repeat_start + 1) or target_length_index >= repeat_length:
                return None 
            
            return target_mask_pos, target_repeat_index, target_length_index
        
        except (KeyError, Exception) as e:
            logging.debug(f"Validation failed for position {target_mask_pos}: {str(e)}")
            return None

    def sample_identical_position_to_mask_similiar_baseline(self, protein_str, repeat_locations, repeat_alignments, high_confidence_positions, min_space):
        if len(repeat_locations) != 2:
            raise Exception("sample_identical_position_to_mask_baseline supports only for repeat times = 2")
        
        if not high_confidence_positions or len(high_confidence_positions) == 0:
            raise Exception("No high confidence positions provided")
        
        # Try positions in random order
        shuffled_positions = self.mask_sampler.sample(high_confidence_positions, len(high_confidence_positions))
        
        for pos in shuffled_positions:
            result = self._validate_position_for_baseline(
                protein_str, repeat_locations, repeat_alignments, pos, min_space
            )
            if result is not None:
                return result
        
        # If we've tried all positions and none work, raise an exception

        logging.warning("Could not find any valid high confidence position for baseline experiment")
        return None, None, None
        

    def corrupt_by_replacing_a_similar_length_part_with_char(
        self,
        protein_str: str,
        true_aa: str,
        target_repeat_index,
        repeat_length,
        repeat_locations: list[tuple[int, int]],
        corrupted_char_1: str,
        corrupted_char_2: str,
        min_space: int
    ):
        if len(repeat_locations) > 2:
            raise ValueError("This corruption method only supports repeats with 2 occurrences.")
        target_start, target_end = repeat_locations[target_repeat_index]
        other_repeat_index = 1 - target_repeat_index
        other_start, other_end = repeat_locations[other_repeat_index]
        fake_repeat_start, fake_repeat_end = self._get_indicies_for_fake_repeat(protein_str, target_start, target_end, other_start, other_end, repeat_length, min_space)
        if fake_repeat_start is None or fake_repeat_end is None:
            return None, None, None
        if repeat_length != (other_end-other_start+1):
            repeat_length = other_end-other_start+1

        relative_positions_inside_repeat = list(range(0, repeat_length))
        fake_repeat_locations = [(fake_repeat_start, fake_repeat_end), repeat_locations[target_repeat_index]]
        fake_repeat_locations = sorted(fake_repeat_locations, key=lambda x: (x[0], x[1]))
        new_target_repeat_index = 0
        for i, loc in enumerate(fake_repeat_locations):
            if loc == repeat_locations[target_repeat_index]:
                new_target_repeat_index = i
                break
        return self.corrupt_by_replacing_given_positions_with_char(
            protein_str=protein_str,
            true_aa=true_aa,
            target_repeat_index=new_target_repeat_index,
            repeat_locations=fake_repeat_locations,
            corrupted_char_1=corrupted_char_1,
            corrupted_char_2=corrupted_char_2,
            mode="relative",
            relative_positions_inside_repeat=relative_positions_inside_repeat
        )

    def corrupt_by_replacing_one_token_in_same_space_with_char(
        self,
        protein_str: str,
        true_aa: str,
        target_repeat_index,
        target_length_index,
        repeat_length,
        repeat_locations: list[tuple[int, int]],
        corrupted_char_1: str,
        corrupted_char_2: str,
        min_space: int
    ):
        if len(repeat_locations) > 2:
            raise ValueError("This corruption method only supports repeats with 2 occurrences.")
        target_start, target_end = repeat_locations[target_repeat_index]
        other_repeat_index = 1 - target_repeat_index
        other_start, other_end = repeat_locations[other_repeat_index]
        fake_repeat_start, fake_repeat_end = self._get_indicies_for_fake_repeat(protein_str, target_start, target_end, other_start, other_end, repeat_length, min_space)
        if fake_repeat_start is None or fake_repeat_end is None:
            return None, None, None
        
        relative_positions_inside_repeat = [target_length_index]
        fake_repeat_locations = [(fake_repeat_start, fake_repeat_end), repeat_locations[target_repeat_index]]
        fake_repeat_locations = sorted(fake_repeat_locations, key=lambda x: (x[0], x[1]))
        new_target_repeat_index = 0
        for i, loc in enumerate(fake_repeat_locations):
            if loc == repeat_locations[target_repeat_index]:
                new_target_repeat_index = i
                break
        return self.corrupt_by_replacing_given_positions_with_char(
            protein_str=protein_str,
            true_aa=true_aa,
            target_repeat_index=new_target_repeat_index,
            repeat_locations=fake_repeat_locations,
            corrupted_char_1=corrupted_char_1,
            corrupted_char_2=corrupted_char_2,
            mode="relative",
            relative_positions_inside_repeat=relative_positions_inside_repeat
        )

    def corrupt_by_replacing_all_repeat_with_char(
        self,
        protein_str: str,
        true_aa: str,
        target_repeat_index,
        repeat_length,
        repeat_locations: list[tuple[int, int]],
        corrupted_char_1: str,
        corrupted_char_2: str
    ):
        """
        Corrupt all repeats (except the target) by replacing every position with corrupted_char_1 or corrupted_char_2.
        """
        relative_positions_inside_repeat = list(range(0, repeat_length))
        return self.corrupt_by_replacing_given_positions_with_char(
            protein_str=protein_str,
            true_aa=true_aa,
            target_repeat_index=target_repeat_index,
            repeat_locations=repeat_locations,
            corrupted_char_1=corrupted_char_1,
            corrupted_char_2=corrupted_char_2,
            mode="relative",
            relative_positions_inside_repeat=relative_positions_inside_repeat
        )

    def corrupt_by_replacing_pct_repeat_with_char(
        self,
        protein_str: str,
        true_aa: str,
        target_repeat_index,
        repeat_length,
        repeat_locations: list[tuple[int, int]],
        corrupted_char_1: str,
        corrupted_char_2: str,
        pct_to_corrupt: float,
    ):
        if pct_to_corrupt > 1.0 or pct_to_corrupt <= 0.0:
            raise Exception("Invalid pct_to_corrupt: must be between 0.0 and 1.0")
        

        positions_to_corrupt = int(repeat_length * pct_to_corrupt)
        if positions_to_corrupt == 0 and pct_to_corrupt > 0:
            positions_to_corrupt = 1  # Ensure at least one position is corrupted if pct > 0
        
        all_relative_positions = list(range(repeat_length))
        
        relative_positions_inside_repeat = self.corrupt_sampler.sample(all_relative_positions, positions_to_corrupt)
        
        return self.corrupt_by_replacing_given_positions_with_char(
            protein_str=protein_str,
            true_aa=true_aa,
            target_repeat_index=target_repeat_index,
            repeat_locations=repeat_locations,
            corrupted_char_1=corrupted_char_1,
            corrupted_char_2=corrupted_char_2,
            mode="relative",
            relative_positions_inside_repeat=relative_positions_inside_repeat
        )

    def _get_corrupt_positions(self, start, end, mode, relative_positions=None, offsets=None):
        """
        Helper to calculate actual corruption positions in repeats based on mode.
        Supports:
            - relative: positions from start
            - offset: positions counted back from end
        """
        if mode == "relative":
            positions = [start + pos for pos in relative_positions]
        elif mode == "offset":
            positions = [end - offset for offset in offsets]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Safety filter to avoid out-of-bounds issues
        new_positions = [pos for pos in positions if start <= pos <= end]
        if len(new_positions) != len(positions):
            raise ValueError("relative positions inside repeats are illegal!")
        return new_positions

    def get_replacement(self, curr_aa, corrupted_char_1):
        if corrupted_char_1 == "blosum":
            return self._get_top_blosum_substitution(curr_aa)
        elif corrupted_char_1 == "blosum_opposite":
            return self._get_lowest_blosum_substitution(curr_aa)
        else:
            return corrupted_char_1
    def corrupt_by_replacing_given_positions_with_char(
        self,
        protein_str: str,
        true_aa: str,
        target_repeat_index: int,
        repeat_locations: list[tuple[int, int]],
        corrupted_char_1: str,
        corrupted_char_2: str,
        mode: str,
        relative_positions_inside_repeat: list[int] = None,
        repeat_positions_offsets_from_end_to_corrupt: list[int] = None,
    ):
        """
        Flexible corruption method combining fixed-relative-position and offset logic.
        Use:
            - `mode='relative'` with `relative_positions_inside_repeat`
            - `mode='offset'` with `repeat_positions_offsets_from_end_to_corrupt`
        """
        assert len(repeat_locations) == 2, "currently support only 2 repeat times due to similiar repeats having different relative positions..."
        # Ensure empty lists if no positions provided
        if relative_positions_inside_repeat is None:
            relative_positions_inside_repeat = []
        if repeat_positions_offsets_from_end_to_corrupt is None:
            repeat_positions_offsets_from_end_to_corrupt = []

        assert ((corrupted_char_1 is not None and len(corrupted_char_1) == 1) and (corrupted_char_2 is not None and len(corrupted_char_2) == 1)) or \
                (corrupted_char_1 == "blosum" and corrupted_char_2 is None) or \
                (corrupted_char_1 == "blosum_opposite" and corrupted_char_2 is None), \
                "Invalid corrupted_char settings."

        assert len(true_aa) == 1, "true_aa should be exactly one character"

        protein_chars = list(protein_str)
        corrupted_positions = []
        replacements = []

        for i, (start, end) in enumerate(repeat_locations):
            if i != target_repeat_index:
                corrupt_positions = self._get_corrupt_positions(
                    start, end, mode,
                    relative_positions=relative_positions_inside_repeat,
                    offsets=repeat_positions_offsets_from_end_to_corrupt
                )
                for corrupt_index in corrupt_positions:
                    curr_aa = protein_chars[corrupt_index]
                    top_option = self.get_replacement(curr_aa, corrupted_char_1)
                    top_option = top_option if top_option is not None else self.default_amino_acid
                    second_option = corrupted_char_2 if corrupted_char_2 is not None else self.default_amino_acid

                    if protein_chars[corrupt_index] != top_option:
                        protein_chars[corrupt_index] = top_option
                        replacements.append(top_option)
                    else:
                        protein_chars[corrupt_index] = second_option
                        replacements.append(second_option)

                    corrupted_positions.append(corrupt_index)

        corrupted_protein_str = "".join(protein_chars)

        # Safety check to ensure replacements and positions match
        assert len(corrupted_positions) == len(replacements), \
            "Mismatch between corrupted positions and replacements."

        return corrupted_protein_str, corrupted_positions, replacements

    def corrupt_by_random_permutation(
        self,
        protein_str: str,
        repeat_locations: list[tuple[int, int]],
        target_repeat_index: int
    ):
        """
        Randomly shuffle the content of all repeats except the target repeat.
        """
        protein_chars = list(protein_str)
        corrupted_positions = []
        replacements = []

        for i, (start, end) in enumerate(repeat_locations):
            if i != target_repeat_index:
                sub_chars = protein_chars[start:end + 1]  # end is inclusive
                self.corrupt_sampler.shuffle(sub_chars)
                protein_chars[start:end + 1] = sub_chars
                corrupted_positions.extend(range(start, end + 1))
                replacements.extend(sub_chars)

        corrupted_protein_str = "".join(protein_chars)
        return corrupted_protein_str, corrupted_positions, replacements

    def corrupt_by_permute_similiar_length( self,
        protein_str: str,
        repeat_locations: list[tuple[int, int]],
        target_repeat_index: int,
        min_space: int,
        repeat_length: int):

        if len(repeat_locations) > 2:
            raise ValueError("This corruption method only supports repeats with 2 occurrences.")
        target_start, target_end = repeat_locations[target_repeat_index]
        other_repeat_index = 1 - target_repeat_index
        other_start, other_end = repeat_locations[other_repeat_index]
        fake_repeat_start, fake_repeat_end = self._get_indicies_for_fake_repeat(protein_str, target_start, target_end, other_start, other_end, repeat_length, min_space)
        if fake_repeat_start is None or fake_repeat_end is None:
            return None, None, None

        fake_repeat_locations = [(fake_repeat_start, fake_repeat_end), repeat_locations[target_repeat_index]]
        fake_repeat_locations = sorted(fake_repeat_locations, key=lambda x: (x[0], x[1]))
        new_target_repeat_index = 0
        for i, loc in enumerate(fake_repeat_locations):
            if loc == repeat_locations[target_repeat_index]:
                new_target_repeat_index = i
                break
        return self.corrupt_by_random_permutation(
            protein_str=protein_str,
            repeat_locations=fake_repeat_locations,
            target_repeat_index=new_target_repeat_index
        )


    def corrupt_by_replacing_a_pct_on_similar_length_part_with_char(
            self,
            protein_str: str,
            true_aa: str,
            target_repeat_index,
            repeat_length,
            repeat_locations: list[tuple[int, int]],
            corrupted_char_1: str,
            corrupted_char_2: str,
            min_space: int,
            pct_to_corrupt: float,
        ):
            if len(repeat_locations) > 2:
                raise ValueError("This corruption method only supports repeats with 2 occurrences.")

            if pct_to_corrupt > 1.0 or pct_to_corrupt <= 0.0:
                raise Exception("Invalid pct_to_corrupt: must be between 0.0 and 1.0")
        
            target_start, target_end = repeat_locations[target_repeat_index]
            other_repeat_index = 1 - target_repeat_index
            other_start, other_end = repeat_locations[other_repeat_index]
            fake_repeat_start, fake_repeat_end = self._get_indicies_for_fake_repeat(protein_str, target_start, target_end, other_start, other_end, repeat_length, min_space)
            if fake_repeat_start is None or fake_repeat_end is None:
                return None, None, None
            if repeat_length != (other_end-other_start+1):
                repeat_length = other_end-other_start+1

            positions_to_corrupt = int(repeat_length * pct_to_corrupt)
            if positions_to_corrupt == 0 and pct_to_corrupt > 0:
                positions_to_corrupt = 1  # Ensure at least one position is corrupted if pct > 0
            
            all_relative_positions = list(range(repeat_length))
            
            relative_positions_inside_repeat = self.corrupt_sampler.sample(all_relative_positions, positions_to_corrupt)

            fake_repeat_locations = [(fake_repeat_start, fake_repeat_end), repeat_locations[target_repeat_index]]
            fake_repeat_locations = sorted(fake_repeat_locations, key=lambda x: (x[0], x[1]))
            new_target_repeat_index = 0
            for i, loc in enumerate(fake_repeat_locations):
                if loc == repeat_locations[target_repeat_index]:
                    new_target_repeat_index = i
                    break
            return self.corrupt_by_replacing_given_positions_with_char(
                protein_str=protein_str,
                true_aa=true_aa,
                target_repeat_index=new_target_repeat_index,
                repeat_locations=fake_repeat_locations,
                corrupted_char_1=corrupted_char_1,
                corrupted_char_2=corrupted_char_2,
                mode="relative",
                relative_positions_inside_repeat=relative_positions_inside_repeat
            )



def _get_dtype(dtype_str):
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]


KEY_COLS = ["cluster_id", "rep_id", "repeat_key"]
RESULT_COLS = [
    "masked_position",
    "corrupted_positions",
    "replacements",
    "corrupted_sequence",
    "masked_repeat_example",
    "corrupted_repeat_example",
    "true_token_prob_before",
    "true_token_prob_after",
    "corrupted_token_prob_before",
    "corrupted_token_prob_after",
    "true_token_logits_before",
    "true_token_logits_after",
    "corrupted_token_logits_before",
    "corrupted_token_logits_after",
    "top5_labels_before",
    "top5_labels_after",
    "top5_probs_before",
    "top5_probs_after",
    "is_corrupt_changed_argmax",
]


def extract_repeats_examples(corr_protein, repeat_locations, target_repeat_index):
    """Extract masked and corrupted repeat substrings from corrupted protein."""
    start, end = repeat_locations[target_repeat_index]
    masked_repeat_example = corr_protein[start : end + 1]
    other_index = next((i for i in range(len(repeat_locations)) if i != target_repeat_index), None)
    if other_index is None:
        corrupted_repeat_example = "N/A"
    else:
        s2, e2 = repeat_locations[other_index]
        corrupted_repeat_example = corr_protein[s2 : e2 + 1]
    return masked_repeat_example, corrupted_repeat_example


def get_position_to_mask(sorted_repeats, repeat_occurence_to_predict, repeat_position_offset_from_end_to_predict):
    """Calculate the position to mask based on repeat data."""
    start, end = sorted_repeats[repeat_occurence_to_predict]
    absolute_masked_position = end - repeat_position_offset_from_end_to_predict
    return absolute_masked_position, absolute_masked_position - start


def create_counterfactual_example(protein, args, row, creator: CounterfactualCreator):
    """Create one counterfactual example; returns target_mask_pos, counterfactual_protein, positions, replacements, target_repeat_index."""
    repeat_times = row["repeat_times"]
    repeat_length = row["repeat_length"]
    repeat_locations = row["repeat_locations"]
    min_space = row["min_space"]

    if args.mode == "random":
        target_mask_pos, target_repeat_index, target_length_index = creator.sample_repeat_and_position_to_mask(
            repeat_times, repeat_length, repeat_locations
        )
    elif args.mode == "baseline_experiment_identical":
        target_mask_pos, target_repeat_index, target_length_index = creator.sample_repeat_and_position_to_mask_mode_baseline_experiment(
            protein, repeat_times, repeat_length, repeat_locations, min_space
        )
        if target_mask_pos is None or target_repeat_index is None or target_length_index is None:
            raise ValueError("No repeat found with enough space to mask")
    elif args.mode == "fixed":
        target_repeat_index = args.repeat_occurence_to_predict
        target_mask_pos, target_length_index = get_position_to_mask(
            repeat_locations, target_repeat_index, args.repeat_position_offset_from_end_to_predict
        )
    elif args.mode == "random_only_identical":
        if repeat_times != 2:
            raise ValueError("random_only_identical requires repeat_times=2")
        repeat_alignments = row["repeat_alignments"]
        high_confidence_positions = row["high_confidence_positions"]
        target_mask_pos, target_repeat_index, target_length_index = creator.sample_identical_position_to_mask_similiar_repeats(
            protein, repeat_locations, repeat_alignments, high_confidence_positions
        )
        other_repeat_idx = 1 - target_repeat_index
        other_start, other_end = repeat_locations[other_repeat_idx]
        repeat_length = other_end - other_start + 1
    else:  # baseline_experiment_similiar
        if repeat_times != 2:
            raise ValueError("baseline_experiment_similiar requires repeat_times=2")
        repeat_alignments = row["repeat_alignments"]
        high_confidence_positions = row["high_confidence_positions"]
        target_mask_pos, target_repeat_index, target_length_index = creator.sample_identical_position_to_mask_similiar_baseline(
            protein, repeat_locations, repeat_alignments, high_confidence_positions, min_space=min_space
        )
        if target_mask_pos is None or target_repeat_index is None or target_length_index is None:
            raise ValueError("No repeat found with enough space to mask")
        other_repeat_idx = 1 - target_repeat_index
        other_start, other_end = repeat_locations[other_repeat_idx]
        repeat_length = other_end - other_start + 1

    true_aa = protein[target_mask_pos]
    method = args.method

    method_handlers = {
        "corrupt_all_other_repeats_with_token": lambda: creator.corrupt_by_replacing_all_repeat_with_char(
            protein_str=protein, true_aa=true_aa, target_repeat_index=target_repeat_index, repeat_length=repeat_length,
            repeat_locations=repeat_locations, corrupted_char_1=args.corrupted_amino_acid_type1,
            corrupted_char_2=args.corrupted_amino_acid_type2
        ),
        "fixed_corrupted_positions": lambda: creator.corrupt_by_replacing_given_positions_with_char(
            protein_str=protein, true_aa=true_aa, target_repeat_index=target_repeat_index,
            repeat_locations=repeat_locations, corrupted_char_1=args.corrupted_amino_acid_type1,
            corrupted_char_2=args.corrupted_amino_acid_type2, mode="offset",
            repeat_positions_offsets_from_end_to_corrupt=args.repeat_positions_offsets_from_end_to_corrupt
        ),
        "same_as_masking_position": lambda: creator.corrupt_by_replacing_given_positions_with_char(
            protein_str=protein, true_aa=true_aa, target_repeat_index=target_repeat_index,
            repeat_locations=repeat_locations, corrupted_char_1=args.corrupted_amino_acid_type1,
            corrupted_char_2=args.corrupted_amino_acid_type2, mode="relative",
            relative_positions_inside_repeat=[target_length_index]
        ),
        "permutation": lambda: creator.corrupt_by_random_permutation(
            protein_str=protein, repeat_locations=repeat_locations, target_repeat_index=target_repeat_index
        ),
        "corrupt_similiar_length_with_char": lambda: creator.corrupt_by_replacing_a_similar_length_part_with_char(
            protein_str=protein, true_aa=true_aa, target_repeat_index=target_repeat_index,
            repeat_locations=repeat_locations, corrupted_char_1=args.corrupted_amino_acid_type1,
            corrupted_char_2=args.corrupted_amino_acid_type2, repeat_length=repeat_length, min_space=min_space
        ),
        "corrupt_by_replacing_a_pct_on_similar_length_part_with_char": lambda: creator.corrupt_by_replacing_a_pct_on_similar_length_part_with_char(
            protein_str=protein, true_aa=true_aa, target_repeat_index=target_repeat_index,
            repeat_locations=repeat_locations, corrupted_char_1=args.corrupted_amino_acid_type1,
            corrupted_char_2=args.corrupted_amino_acid_type2, repeat_length=repeat_length, min_space=min_space,
            pct_to_corrupt=args.pct_corruption
        ),
        "corrupt_by_replacing_one_token_in_same_space_with_char": lambda: creator.corrupt_by_replacing_one_token_in_same_space_with_char(
            protein_str=protein, true_aa=true_aa, target_repeat_index=target_repeat_index,
            target_length_index=target_length_index, repeat_locations=repeat_locations,
            corrupted_char_1=args.corrupted_amino_acid_type1, corrupted_char_2=args.corrupted_amino_acid_type2,
            repeat_length=repeat_length, min_space=min_space
        ),
        "corrupt_pct_other_repeats_with_token": lambda: creator.corrupt_by_replacing_pct_repeat_with_char(
            protein_str=protein, true_aa=true_aa, target_repeat_index=target_repeat_index,
            repeat_length=repeat_length, repeat_locations=repeat_locations,
            corrupted_char_1=args.corrupted_amino_acid_type1, corrupted_char_2=args.corrupted_amino_acid_type2,
            pct_to_corrupt=args.pct_corruption
        ),
        "corrupt_by_permute_similiar_length": lambda: creator.corrupt_by_permute_similiar_length(
            protein_str=protein, repeat_locations=repeat_locations, target_repeat_index=target_repeat_index,
            repeat_length=repeat_length, min_space=min_space
        ),
    }

    if method not in method_handlers:
        raise ValueError(f"Unsupported method: {method}")

    counterfactual_protein, counterfactual_positions, replacements = method_handlers[method]()
    if counterfactual_protein is None or counterfactual_positions is None or replacements is None:
        raise ValueError("No valid counterfactual produced")
    return target_mask_pos, counterfactual_protein, counterfactual_positions, replacements, target_repeat_index


def predict(model, proteins, mask_positions, true_labels, device, replacement_char, tokenizer):
    """Run model prediction; return acc, top5_probs, top5_labels, correct_probs, replacement_probs, correct_logits, replacement_logits."""
    for i, p in enumerate(proteins):
        proteins[i] = p.replace(C.MASK_STR_SHORT, tokenizer.mask_token)
    tokenized = tokenizer(proteins, return_tensors="pt", add_special_tokens=True, padding=True)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    tokenized_labels = tokenizer(true_labels, return_tensors="pt", add_special_tokens=False, padding=False)["input_ids"].squeeze(-1).to(device)
    tokenized_replacement = tokenizer(
        replacement_char, return_tensors="pt", add_special_tokens=False, padding=False
    )["input_ids"].squeeze(-1).to(device)
    with torch.no_grad():
        output = model.forward(sequence_tokens=input_ids, sequence_id=attention_mask)
    logits = output.sequence_logits
    mask_pos_t = torch.tensor(mask_positions, device=device) + 1
    batch_idx = torch.arange(logits.size(0), device=device)
    masked_logits = logits[batch_idx, mask_pos_t]
    probs = get_probs_from_logits(masked_logits, device, tokenizer, mask_logits_of_invalid_ids=False)
    correct_probs = probs[batch_idx, tokenized_labels].detach().tolist()
    replacement_probs = probs[batch_idx, tokenized_replacement].detach().tolist()
    correct_logits = masked_logits[batch_idx, tokenized_labels].detach().tolist()
    replacement_logits = masked_logits[batch_idx, tokenized_replacement].detach().tolist()
    predicted_labels = probs.argmax(dim=-1)
    acc = (tokenized_labels == predicted_labels).float().mean().item()
    top5_probs, top5_idx = torch.topk(probs, k=5, dim=-1)
    top5_labels = [tokenizer.decode(idx) for idx in top5_idx]
    return acc, top5_probs.detach().tolist(), top5_labels, correct_probs, replacement_probs, correct_logits, replacement_logits


def process_dataset(model, df, tokenizer, device, output_path, args, creator):
    """Process all rows; write key + all new result columns (like evaluation_experiment/evaluate.py)."""
    header = KEY_COLS + RESULT_COLS
    total = len(df)
    processed = 0
    skipped = 0
    corrupted_pred_changed_count = 0
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for index, row in df.iterrows():
            try:
                protein = row["seq"]
                target_mask_pos, counterfactual_protein, counterfactual_positions, replacements, target_repeat_index = create_counterfactual_example(
                    protein, args, row, creator
                )
                true_aa = protein[target_mask_pos]
                replacement_char = replacements[0]
                protein_before = mask_protein_short(protein, target_mask_pos, tokenizer)
                (
                    acc_before,
                    top5_probs_before,
                    top5_labels_before,
                    true_prob_before,
                    repl_prob_before,
                    true_logits_before,
                    repl_logits_before,
                ) = predict(model, [protein_before], [target_mask_pos], [true_aa], device, replacement_char, tokenizer)
                protein_after = mask_protein_short(counterfactual_protein, target_mask_pos, tokenizer)
                (
                    acc_after,
                    top5_probs_after,
                    top5_labels_after,
                    true_prob_after,
                    repl_prob_after,
                    true_logits_after,
                    repl_logits_after,
                ) = predict(model, [protein_after], [target_mask_pos], [true_aa], device, replacement_char, tokenizer)
                is_changed = int(acc_after < 1.0)
                masked_repeat_example, corrupted_repeat_example = extract_repeats_examples(
                    protein_after, row["repeat_locations"], target_repeat_index
                )
                top5_lbl_before = top5_labels_before[0]
                top5_lbl_after = top5_labels_after[0]
                if isinstance(top5_lbl_before, str):
                    top5_lbl_before = top5_lbl_before.split()
                if isinstance(top5_lbl_after, str):
                    top5_lbl_after = top5_lbl_after.split()
                prefix = [row["cluster_id"], row["rep_id"], row["repeat_key"]]
                writer.writerow(prefix + [
                    target_mask_pos,
                    str(counterfactual_positions),
                    str(replacements),
                    counterfactual_protein,
                    masked_repeat_example,
                    corrupted_repeat_example,
                    true_prob_before[0],
                    true_prob_after[0],
                    repl_prob_before[0],
                    repl_prob_after[0],
                    true_logits_before[0],
                    true_logits_after[0],
                    repl_logits_before[0],
                    repl_logits_after[0],
                    str(top5_lbl_before),
                    str(top5_lbl_after),
                    str(top5_probs_before[0]),
                    str(top5_probs_after[0]),
                    is_changed,
                ])
                if is_changed:
                    corrupted_pred_changed_count += 1
                processed += 1
            except (ValueError, AssertionError) as e:
                logging.info(f"Row {index} skipped: {e}")
                skipped += 1
            if (processed + skipped) % 10 == 0:
                logging.info(f"Processed {processed + skipped}/{total} rows so far.")
    success_rate = corrupted_pred_changed_count / processed if processed > 0 else 0
    logging.info(
        f"Processing complete. Total processed: {processed}, Skipped: {skipped}, "
        f"Corrupted predictions changed: {corrupted_pred_changed_count}, Success rate: {success_rate}"
    )


def main(args=None):
    parser = argparse.ArgumentParser(description="Counterfactual experiment evaluation")
    parser.add_argument("--input_file", type=Path, required=True, help="Source dataset CSV with seq, repeat_locations, etc.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for output CSV and logs")
    parser.add_argument("--output_filename", type=str, required=True, help="Output CSV filename stem (e.g. identical_counterfactual_mask)")
    parser.add_argument("--model_type", choices=["esm3", "esm-c"], default="esm3", help="Language model to use")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Model dtype (default float32);",
    )
    parser.add_argument(
        "--mode",
        choices=["random", "fixed", "baseline_experiment_identical", "random_only_identical", "baseline_experiment_similiar"],
        required=True,
        help="""How to select the masked position:
  random: sample a random repeat and random position within it
  fixed: use repeat_occurence_to_predict and repeat_position_offset_from_end_to_predict
  baseline_experiment_identical: (2 repeats) sample random position within a real repeat; only repeats with a valid fake repeat region are eligible
  random_only_identical: (2 repeats) sample from aligned identical positions between repeats
  baseline_experiment_similiar: (2 repeats) sample from aligned identical positions; only positions with a valid fake repeat region are eligible""",
    )
    parser.add_argument(
        "--method",
        required=True,
        help="""Corruption method - what to corrupt in the counterfactual:
  corrupt_all_other_repeats_with_token: replace entire other repeat(s) with type1/type2
  corrupt_pct_other_repeats_with_token: replace fraction (pct_corruption) of other repeat
  permutation: randomly permute the other repeat
  corrupt_similiar_length_with_char: replace similar-length region elsewhere (fake repeat; baseline)
  corrupt_by_replacing_a_pct_on_similar_length_part_with_char: replace fraction of fake region (baseline)
  corrupt_by_permute_similiar_length: permute similar-length fake region (baseline)
  corrupt_by_replacing_one_token_in_same_space_with_char: replace one aligned position in fake region (baseline)
  fixed_corrupted_positions: replace given offsets (repeat_positions_offsets_from_end_to_corrupt)
  same_as_masking_position: replace only the aligned position in the other repeat""",
    )
    parser.add_argument("--corrupted_amino_acid_type1", default="A", help="Primary replacement for corrupted positions (mask, blosum, blosum_opposite, or a specific amino acid)")
    parser.add_argument("--corrupted_amino_acid_type2", default=None, help="Fallback replacement when type1 equals original (or None)")
    parser.add_argument("--pct_corruption", type=float, default=-1, help="Fraction of positions to corrupt (0-1) for pct-based methods")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--repeat_occurence_to_predict", type=int, help="(mode=fixed) Which repeat occurrence (0-indexed) to predict")
    parser.add_argument("--repeat_position_offset_from_end_to_predict", type=int, help="(mode=fixed) Offset from end of repeat for mask position")
    parser.add_argument("--repeat_positions_offsets_from_end_to_corrupt", type=int, nargs="+", help="(method=fixed_corrupted_positions) Offsets from end of repeat for corruption positions")
    parsed = parser.parse_args(args)
    parsed.input_file = Path(parsed.input_file)
    parsed.output_dir = Path(parsed.output_dir)

    if parsed.corrupted_amino_acid_type1 == "mask":
        parsed.corrupted_amino_acid_type1 = C.MASK_STR_SHORT
    if parsed.corrupted_amino_acid_type2 == "mask":
        parsed.corrupted_amino_acid_type2 = C.MASK_STR_SHORT

    set_random_seed(parsed.random_seed)
    
    output_name = parsed.output_filename
    if not output_name.endswith(".csv"):
        output_name = f"{output_name}.csv"
    output_path = parsed.output_dir / output_name
    log_dir = parsed.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_stem = Path(output_name).stem
    logging.basicConfig(
        filename=log_dir / f"{log_stem}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Args: %s", vars(parsed))

    df = pd.read_csv(parsed.input_file)
    df["repeat_locations"] = df["repeat_locations"].apply(ast.literal_eval)
    if parsed.mode in ("random_only_identical", "baseline_experiment_similiar"):
        if not all(c in df.columns for c in ["repeat_alignments", "high_confidence_positions"]):
            logging.error("Mode requires repeat_alignments and high_confidence_positions")
            sys.exit(1)
        df["repeat_alignments"] = df["repeat_alignments"].apply(ast.literal_eval)
        df["high_confidence_positions"] = df["high_confidence_positions"].apply(ast.literal_eval)

    device = get_device()
    model = load_model(
        model_type=parsed.model_type,
        device=device,
        use_transformer_lens_model=False,
        cache_attention_activations=False,
        cache_mlp_activations=False,
        output_type="all",
        cache_attn_pattern=False,
        split_qkv_input=False,
    )
    model = model.to(_get_dtype(parsed.dtype))
    tokenizer = load_tokenizer_by_model_type(parsed.model_type)
    creator = CounterfactualCreator(random_seed=parsed.random_seed)
    parsed.output_dir.mkdir(parents=True, exist_ok=True)
    process_dataset(model, df, tokenizer, device, output_path, parsed, creator)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
