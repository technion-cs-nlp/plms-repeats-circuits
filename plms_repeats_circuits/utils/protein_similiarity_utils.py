from Bio.Align.substitution_matrices import load
from functools import lru_cache
from typing import List, Dict, Optional,Tuple
from dataclasses import dataclass


blosum62 = load("BLOSUM62")
similar_groups = [
    set("GAVLI"),  # Aliphatic
    set("FYW"),    # Aromatic
    set("CM"),     # Sulfur-containing
    set("ST"),     # Hydroxylic
    set("KRH"),    # Basic
    set("DENQ"),   # Acidic and amidic
    set("P")       # Imino
]

aa_to_group = {aa: i for i, group in enumerate(similar_groups) for aa in group}

@lru_cache(maxsize=None)
def get_blosum_score(a, b):
    """
    Get the BLOSUM62 similarity score between two amino acids.
    """
    try:
        return blosum62[(a.upper(), b.upper())]
    except KeyError:
        return blosum62.get((b.upper(), a.upper()), -100)

def are_amino_acids_in_same_group(aa1, aa2):
    """
    Check if two amino acids belong to the same similarity group.
    """
    if aa1 == aa2:
        return True
    
    aa1, aa2 = aa1.upper(), aa2.upper()
    return aa_to_group.get(aa1) == aa_to_group.get(aa2) and aa_to_group.get(aa1) is not None

def are_amino_acids_blosum_similar(aa1, aa2, blosum_similarity_threshold: int = 0):
    """
    Check if two amino acids have a non-negative BLOSUM62 score.
    """
    if not aa1 or not aa2:
        return False
    
    score = get_blosum_score(aa1.upper(), aa2.upper())
    return score >= blosum_similarity_threshold

def compute_alignment_metrics(alignments) -> Dict:
    """
    Compute similarity metrics between aligned sequences.
    
    Args:
        alignments: List of aligned sequences
        
    Returns:
        Dictionary with computed metrics
    """
    result = {
        'mutation_percentage': 0.0,
        'indels_count': 0,
        'substitutions_count': 0,
        'indels_percentage': 0.0,
        'substitutions_percentage': 0.0,
        'identity_percentage': 0.0,
        'similarity_percentage': 0.0,
        'blosum_similarity_percentage': 0.0,
        'indels_score_percentage': 0.0,
        'substitutions_score_percentage': 0.0
    }
    
    try:
        alignments = [seq.upper() for seq in alignments]
        mutation_scores, ident_scores, sim_scores, blosum_scores = [], [], [], []
        substitutions_scores, indels_scores = [], []
        indels_count = substitutions_count = 0

        for i in range(len(alignments)):
            for j in range(i + 1, len(alignments)):
                seq1, seq2 = alignments[i], alignments[j]
                assert len(seq1) == len(seq2)

                aligned = ident = sim = blosum = 0
                indels = substitutions = mutations = 0
                alignment_length = 0

                for a, b in zip(seq1, seq2):
                    # Skip positions where both sequences have gaps
                    if (a == '.' and b == '.') or (a == '-' and b == '-'):
                        continue
                    
                    # Count this position in the alignment length
                    alignment_length += 1
                    
                    if a == '.' or b == '.' or a == '-' or b == '-':
                        indels += 1
                        mutations += 1
                        continue
                    
                    aligned += 1
                    if a != b:
                        substitutions += 1
                        mutations += 1
                    if a == b:
                        ident += 1
                        sim += 1
                    elif aa_to_group.get(a) == aa_to_group.get(b):
                        sim += 1
                    if get_blosum_score(a, b) >= 0:
                        blosum += 1

                # Use alignment_length (excludes double-gap positions) as denominator
                mutation_scores.append(mutations / alignment_length * 100 if alignment_length else 0)
                ident_scores.append(ident / alignment_length * 100 if alignment_length else 0)
                sim_scores.append(sim / alignment_length * 100 if alignment_length else 0)
                blosum_scores.append(blosum / alignment_length * 100 if alignment_length else 0)
                indels_count += indels
                substitutions_count += substitutions
                substitutions_scores.append(substitutions / alignment_length * 100 if alignment_length else 0)
                indels_scores.append(indels / alignment_length * 100 if alignment_length else 0)

        result['mutation_percentage'] = sum(mutation_scores) / len(mutation_scores) if mutation_scores else 0
        result['indels_count'] = indels_count
        result['substitutions_count'] = substitutions_count
        
        total_mut = indels_count + substitutions_count
        result['indels_percentage'] = (indels_count / total_mut) * 100 if total_mut else 0
        result['substitutions_percentage'] = (substitutions_count / total_mut) * 100 if total_mut else 0
        
        result['identity_percentage'] = sum(ident_scores) / len(ident_scores) if ident_scores else 0
        result['similarity_percentage'] = sum(sim_scores) / len(sim_scores) if sim_scores else 0
        result['blosum_similarity_percentage'] = sum(blosum_scores) / len(blosum_scores) if blosum_scores else 0
        
        result['indels_score_percentage'] = sum(indels_scores) / len(indels_scores) if indels_scores else 0
        result['substitutions_score_percentage'] = sum(substitutions_scores) / len(substitutions_scores) if substitutions_scores else 0

    except Exception as e:
        print(f"⚠️ Error computing alignment metrics: {e}")
    
    return result


@dataclass
class RepeatPositionData:
    identity: str  # Amino acid at this position
    absolute_location_in_sequence: int  # Location in the full sequence (0-based)
    relative_position_in_repeat: int  # Position within the repeat segment (0-based)
    relative_position_in_alignment: int  # Position within the alignment (0-based)
    non_aligned_matching_absolute_position: Optional[int]  # Corresponding position in other repeat (0-based), might not exist for longer repeat in the case the shorter repeat is at the end 
    non_aligned_matching_identity: Optional[str]  # Amino acid at the same relative position in the other repeat
    is_non_aligned_part_of_other_repeat: bool #is non aligned part of the second repeat
    is_nonaligned_matching_identical: bool  # True if current aa identity == matching non aligned aa identity
    is_non_aligned_matching_similiar:bool #True if current aa identity and matching non alined aa has positive blosum score
    aligned_matching_absolute_position_in_sequence: Optional[int]  # Position of matched AA in alignment (0-based)
    aligned_matching_identity: str  # Amino acid at matching position in alignment - can be aa or dot
    is_aligned_matching_identical: bool  # True if identity == alignment_matching_identity
    is_aligned_matching_similiar:bool #True if current aa identity and alined match aa has positive blosum score
    is_aligned_and_unaligned_match_same_position: bool  # True if matched AA in alignment is also matched in unaligned
    is_aligned_and_unaligned_match_same_identity_different_pos: bool  # True if same AA matched bot for non aligned and aligned match but at different positions
    is_near_indel: bool = False
    is_near_sub: bool= False

def _classify_position_based_on_near_mutation(repeat_locations, result_abs_pos_to_repeat_info, result_aligned_pos_to_repeat_info):
    for rep_idx, (start, end) in enumerate(repeat_locations):
        for pos in range(start, end + 1):
            curr_info = result_abs_pos_to_repeat_info[rep_idx][pos]
            
            # Check previous position
            is_prev_pos_sub = False
            is_prev_pos_indel = False
            if pos > start:  
                prev_pos = pos - 1
                if prev_pos in result_abs_pos_to_repeat_info[rep_idx]:
                    prev_info = result_abs_pos_to_repeat_info[rep_idx][prev_pos]
                    is_prev_pos_sub = (prev_info.aligned_matching_absolute_position_in_sequence is not None and 
                                      not prev_info.is_aligned_matching_identical)
                    is_prev_pos_indel = prev_info.aligned_matching_absolute_position_in_sequence is None
            
            # Check next position
            is_next_pos_sub = False
            is_next_pos_indel = False
            if pos < end:  
                next_pos = pos + 1
                if next_pos in result_abs_pos_to_repeat_info[rep_idx]:
                    next_info = result_abs_pos_to_repeat_info[rep_idx][next_pos]
                    is_next_pos_sub = (next_info.aligned_matching_absolute_position_in_sequence is not None and 
                                      not next_info.is_aligned_matching_identical)
                    is_next_pos_indel = next_info.aligned_matching_absolute_position_in_sequence is None
            
            # Update current position information
            result_abs_pos_to_repeat_info[rep_idx][pos].is_near_indel = is_next_pos_indel or is_prev_pos_indel
            result_abs_pos_to_repeat_info[rep_idx][pos].is_near_sub = is_next_pos_sub or is_prev_pos_sub
            
            # Update alignment-based information
            relative_position_in_alignment = curr_info.relative_position_in_alignment
            if relative_position_in_alignment is not None and relative_position_in_alignment in result_aligned_pos_to_repeat_info[rep_idx]:
                result_aligned_pos_to_repeat_info[rep_idx][relative_position_in_alignment].is_near_indel = is_next_pos_indel or is_prev_pos_indel
                result_aligned_pos_to_repeat_info[rep_idx][relative_position_in_alignment].is_near_sub = is_next_pos_sub or is_prev_pos_sub
    
    return result_abs_pos_to_repeat_info, result_aligned_pos_to_repeat_info



def analyze_repeat_positions(
    protein_seq: str,
    repeat_locations: List[List[int]],
    alignments: List[str],
    blosum_similarity_threshold: int = 0
) -> Tuple[Dict[int, Dict[int, RepeatPositionData]], Dict[int, Dict[int, RepeatPositionData]]]:
    """
    Analyze repeat positions in a protein, mapping original positions to alignment.
    
    Args:
        protein_seq: The full protein sequence
        repeat_locations: List of [start, end] positions for each repeat
        alignments: List of aligned repeat sequences
        
    Returns:
        Dictionary with repeat index as key and list of RepeatPositionData for each position
    """
    if len(repeat_locations) != 2:
        return {}
    
    alignment_to_original = {}
    original_to_alignment = {}
    
    # Check if alignments are available
    if len(alignments) < 2:
        return {}
    
    for rep_idx, (start, end) in enumerate(repeat_locations):
        if rep_idx < len(alignments):
            aligned_seq = alignments[rep_idx]
            alignment_to_original[rep_idx] = {}
            original_to_alignment[rep_idx] = {}
            orig_pos = start
            for align_pos, curr_aa in enumerate(aligned_seq):
                if curr_aa != '.' and curr_aa != '-':
                   # Safety check to prevent index errors
                   assert orig_pos <= end, "alignment is shorter than the repeat"
                   assert curr_aa.upper() == protein_seq[orig_pos]
                   alignment_to_original[rep_idx][align_pos] = orig_pos
                   original_to_alignment[rep_idx][orig_pos] = align_pos
                   orig_pos += 1
                   
    
    result_abs_pos_to_repeat_info = {}
    result_aligned_pos_to_repeat_info= {}
    for rep_idx, (start, end) in enumerate(repeat_locations):
        result_abs_pos_to_repeat_info [rep_idx]= {}
        result_aligned_pos_to_repeat_info[rep_idx] = {}

        other_idx = 1 if rep_idx == 0 else 0  # Index of the other repeat
        other_start, other_end = repeat_locations[other_idx]
        
        for pos in range(start, end + 1):
            curr_rel_pos = pos - start
            assert pos < len(protein_seq) , "bug- illegal position in a repeat"
                
            curr_aa = protein_seq[pos].upper()

            # non align match might be none
            other_nonaligned_abs_matched_pos = other_start + curr_rel_pos
            other_nonaligned_abs_matched_pos = other_nonaligned_abs_matched_pos if 0 <= other_nonaligned_abs_matched_pos < len(protein_seq) else None
            other_nonaligned_abs_matched_aa = protein_seq[other_nonaligned_abs_matched_pos].upper() if other_nonaligned_abs_matched_pos is not None else None
            
            align_pos = original_to_alignment.get(rep_idx, {}).get(pos)
            assert align_pos is not None, f"aligned position - should exist for every amino acid within a repeat: repeat locations: {repeat_locations}, alignments:{alignments}, rep_idx:{rep_idx}, pos:{pos}, curr_aa:{curr_aa}, original_to_alignment: {original_to_alignment}, alignment_to_original:{alignment_to_original}, segment: {protein_seq[start:end+1]}, seq:{protein_seq} "
            align_match_aa = alignments[other_idx][align_pos].upper() #rader alignments of same length
            other_align_abs_pos = None
            if align_match_aa and align_match_aa not in ['.', '-']:
                other_align_abs_pos = alignment_to_original.get(other_idx, {}).get(align_pos)

            position_data = RepeatPositionData(
                identity=curr_aa.upper(),
                absolute_location_in_sequence=pos,
                relative_position_in_repeat=curr_rel_pos,
                relative_position_in_alignment=align_pos,
                non_aligned_matching_absolute_position=other_nonaligned_abs_matched_pos,
                non_aligned_matching_identity=other_nonaligned_abs_matched_aa if other_nonaligned_abs_matched_pos else None,
                is_non_aligned_part_of_other_repeat=other_start<=other_nonaligned_abs_matched_pos <= other_end if other_nonaligned_abs_matched_pos else False,
                is_nonaligned_matching_identical=(curr_aa == other_nonaligned_abs_matched_aa) if other_nonaligned_abs_matched_aa else False,
                is_non_aligned_matching_similiar= are_amino_acids_blosum_similar(curr_aa, other_nonaligned_abs_matched_aa, blosum_similarity_threshold) if other_nonaligned_abs_matched_aa else False,
                aligned_matching_absolute_position_in_sequence=other_align_abs_pos,
                aligned_matching_identity=align_match_aa.upper() if align_match_aa else None,
                is_aligned_matching_identical=(curr_aa == align_match_aa) if align_match_aa and align_match_aa not in ['.', '-'] else False,
                is_aligned_matching_similiar= are_amino_acids_blosum_similar(curr_aa, align_match_aa, blosum_similarity_threshold) if align_match_aa and align_match_aa not in ['.', '-'] else False,
                is_aligned_and_unaligned_match_same_position=(other_nonaligned_abs_matched_pos == other_align_abs_pos) if other_nonaligned_abs_matched_pos is not None and other_align_abs_pos is not None else False,
                is_aligned_and_unaligned_match_same_identity_different_pos=(align_match_aa == other_nonaligned_abs_matched_aa and other_align_abs_pos != other_nonaligned_abs_matched_pos) if other_align_abs_pos and other_nonaligned_abs_matched_pos and align_match_aa and align_match_aa not in ['.', '-'] and other_nonaligned_abs_matched_aa else False
            )
            
            result_abs_pos_to_repeat_info[rep_idx][pos] = position_data
            result_aligned_pos_to_repeat_info[rep_idx][align_pos]  = position_data
        
    
    result_abs_pos_to_repeat_info, result_aligned_pos_to_repeat_info = _classify_position_based_on_near_mutation(
        repeat_locations, result_abs_pos_to_repeat_info, result_aligned_pos_to_repeat_info)
    
    return result_abs_pos_to_repeat_info, result_aligned_pos_to_repeat_info

