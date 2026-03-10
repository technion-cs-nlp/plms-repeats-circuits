from pathlib import Path
from typing import List, Optional
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast

from plms_repeats_circuits.utils.model_utils import mask_protein
from plms_repeats_circuits.utils.protein_similiarity_utils import analyze_repeat_positions

from concepts.concept_utils import RepeatsInfoAboutSequence

def parse_repeat_locations(loc_str):
    try:
        if pd.isna(loc_str):
            return []
        if isinstance(loc_str, str):
            loc_list = ast.literal_eval(loc_str)
        else:
            loc_list = loc_str
        if isinstance(loc_list, list):
            tuples = [tuple(pair) if isinstance(pair, list) else pair for pair in loc_list]
            return sorted(tuples)
        return []
    except Exception:
        return []


def get_protein_id(row):
    return (str(row['cluster_id']), str(row['rep_id']), str(row['seq']))


def drop_duplicates_from_radar(df_radar):
    df_radar = df_radar.copy()
    df_radar['dedup_key'] = df_radar.apply(
        lambda row: (row['protein_id'], tuple(sorted(row['repeat_locations_parsed']))),
        axis=1
    )
    df_radar_dedup = df_radar.drop_duplicates(subset=['dedup_key'], keep='first').copy()
    return df_radar_dedup.drop(columns=['dedup_key'])


def _ranges_overlap(range1, range2):
    start1, end1 = range1
    start2, end2 = range2
    return not (end1 < start2 or start1 > end2)


def collect_suspected_repeats_non_overlapping(row):
    valid_locs = row['repeat_locations_parsed']
    radar_locs_list = row.get('radar_locations_parsed', [])
    if radar_locs_list is None or (isinstance(radar_locs_list, float) and pd.isna(radar_locs_list)) or not radar_locs_list:
        return []
    suspected_ranges = set()
    for radar_detection in radar_locs_list:
        for radar_rng in radar_detection:
            overlaps = False
            for valid_rng in valid_locs:
                if _ranges_overlap(radar_rng, valid_rng):
                    overlaps = True
                    break
            if not overlaps:
                suspected_ranges.add(radar_rng)
    return [list(r) for r in sorted(suspected_ranges)] if suspected_ranges else []


def prepare_dataset_df(df: pd.DataFrame, radar_path: str = None) -> pd.DataFrame:
    """
    Apply proteins_single_repeat filter and optionally merge RADAR for additional_suspected_repeats.
    """
    if 'repeat_key' not in df.columns or 'cluster_id' not in df.columns or 'rep_id' not in df.columns or 'seq' not in df.columns:
        return df
    df = df.copy()
    df['protein_id'] = df.apply(get_protein_id, axis=1)
    repeat_types_count = df.groupby('protein_id')['repeat_key'].nunique()
    proteins_single_repeat = repeat_types_count[repeat_types_count == 1].index
    df = df[df['protein_id'].isin(proteins_single_repeat)].copy()
    if radar_path and Path(radar_path).exists():
        print(f"Merging RADAR data from {radar_path}...", flush=True)
        df_radar = pd.read_csv(radar_path)
        df['repeat_locations_parsed'] = df['repeat_locations'].apply(parse_repeat_locations)
        df_radar['repeat_locations_parsed'] = df_radar['repeat_locations'].apply(parse_repeat_locations)
        df_radar['protein_id'] = df_radar.apply(get_protein_id, axis=1)
        df_radar = drop_duplicates_from_radar(df_radar)
        radar_grouped = df_radar.groupby('protein_id')['repeat_locations_parsed'].apply(list).reset_index()
        radar_grouped.columns = ['protein_id', 'radar_locations_parsed']
        df = df.merge(radar_grouped, on='protein_id', how='left')
        df['additional_suspected_repeats'] = df.apply(collect_suspected_repeats_non_overlapping, axis=1)
        df = df.drop(columns=['repeat_locations_parsed', 'protein_id', 'radar_locations_parsed'])
        print(f"  Added {len(df)} sequences with additional suspected repeats", flush=True)
    else:
        df = df.drop(columns=['protein_id'])
        df['additional_suspected_repeats'] = [[] for _ in range(len(df))]
    return df


def create_neurons_analysis_dataset_pandas(df, total_n_samples, random_state, tokenizer):
    # Apply ast.literal_eval BEFORE sampling
    df = df.copy()
    df['repeat_locations'] = df['repeat_locations'].apply(safe_literal_eval)
    if 'repeat_alignments' not in df.columns:
        df['repeat_alignments'] = [[] for _ in range(len(df))]
    else:
        df['repeat_alignments'] = df['repeat_alignments'].apply(safe_literal_eval)
    if 'additional_suspected_repeats' not in df.columns:
        df['additional_suspected_repeats'] = [[] for _ in range(len(df))]
    else:
        df['additional_suspected_repeats'] = df['additional_suspected_repeats'].apply(safe_literal_eval)

    if total_n_samples < len(df):
        sampled_df = df.sample(n=total_n_samples, random_state=random_state)
        print(f"Sampled {len(sampled_df)} samples from {len(df)} samples.")
    else:
        print(f"total_n_samples {total_n_samples} is greater than the number of samples in the dataframe {len(df)}. Using all samples, {len(df)} samples.")
        sampled_df = df

    def process_row(row):
        clean = row['seq']
        name = f"{row['cluster_id']}_{row['rep_id']}_{row['repeat_key']}"
        masked_position = int(row['masked_position'])
        clean_masked = mask_protein(clean, masked_position, tokenizer)
        repeat_alignments = row.repeat_alignments if "repeat_alignments" in row else []
        result = {
            'clean_masked': clean_masked,
            'masked_position_after_tokenization': masked_position + 1,
            'repeat_locations': row.repeat_locations,
            'repeat_alignments': repeat_alignments,
            'seq': row.seq,
            'name': name,
            'additional_suspected_repeats': row.additional_suspected_repeats,
            'cluster_id': row['cluster_id'],
            'rep_id': row['rep_id'],
        }
        return pd.Series(result)
    return sampled_df.apply(process_row, axis=1).reset_index(drop=True)

def collate_neurons_analysis(xs):
    transposed = zip(*xs)
    return tuple(list(column) for column in transposed)
    
def safe_literal_eval(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x



class NeuronsAnalysisDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def head(self, n: int):
        self.df = self.df.head(n).reset_index(drop=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        return (
            row['clean_masked'],
            row['masked_position_after_tokenization'],
            row['repeat_locations'],
            row['repeat_alignments'],
            row['seq'],
            row['name'],
            row['additional_suspected_repeats'],
            row['cluster_id'],
            row['rep_id'],
        )

        return base_items

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_neurons_analysis)


# ---- sequence_processing_utils (merged) ----
def _is_bos_or_eos_or_pad(token: int, tokenizer: PreTrainedTokenizerFast) -> bool:
    if tokenizer.bos_token_id == token:
        return True
    if tokenizer.eos_token_id == token:
        return True
    if tokenizer.pad_token_id == token:
        return True
    return False


def _shift_locations(locations: List[List[int]], shift: int) -> List[List[int]]:
    return [[start + shift, end + shift] for start, end in locations]


def _is_inside_repeat(token_position_in_sequence: int, repeat_locations: List[List[int]]) -> bool:
    for repeat_location in repeat_locations:
        if repeat_location[0] <= token_position_in_sequence <= repeat_location[1]:
            return True
    return False


def _get_positions_near_repeat(
    shifted_locations: List[List[int]],
    near_repeat_threshold: int,
    tokenized_sequence: List[int],
    tokenizer: PreTrainedTokenizerFast
) -> List[int]:
    seq_len = len(tokenized_sequence)
    near_positions = set()
    for start, end in shifted_locations:
        for i in range(1, near_repeat_threshold + 1):
            pos = start - i
            if 0 <= pos and not _is_bos_or_eos_or_pad(tokenized_sequence[pos], tokenizer):
                near_positions.add(pos)
        for i in range(1, near_repeat_threshold + 1):
            pos = end + i
            if pos < seq_len and not _is_bos_or_eos_or_pad(tokenized_sequence[pos], tokenizer):
                near_positions.add(pos)
    filtered = [p for p in sorted(near_positions) if not _is_inside_repeat(p, shifted_locations)]
    return sorted(filtered)


def get_repeats_info_about_sequence(
    seq, tokenized_sequence, repeat_locations, masked_pos_after_tokenization, alignments,
    near_repeat_threshold: int, tokenizer: PreTrainedTokenizerFast,
    suspected_repeat_locations: List[List[int]], tokenization_shift: int, device: torch.device
) -> RepeatsInfoAboutSequence:
    shift_repeat_locations = _shift_locations(repeat_locations, tokenization_shift)
    repeat_positions = sorted([p for start, end in shift_repeat_locations for p in range(start, end + 1)])
    repeat_positions_tensor = torch.tensor(repeat_positions, device=device, dtype=torch.long)
    near_positions = _get_positions_near_repeat(shift_repeat_locations, near_repeat_threshold, tokenized_sequence, tokenizer)
    near_positions_tensor = torch.tensor(near_positions, device=device, dtype=torch.long)
    repeat_index_of_masked_pos = None
    for i, (start, end) in enumerate(shift_repeat_locations):
        if start <= masked_pos_after_tokenization <= end:
            repeat_index_of_masked_pos = i
            break
    result_abs_pos_to_repeat_info, _ = analyze_repeat_positions(seq, repeat_locations, alignments)
    if repeat_index_of_masked_pos is None:
        raise ValueError("No aligned token to masked position found")
    pos_data = result_abs_pos_to_repeat_info[repeat_index_of_masked_pos][masked_pos_after_tokenization - tokenization_shift]
    if pos_data.aligned_matching_absolute_position_in_sequence is None:
        raise ValueError("Masked position aligns to a gap")
    aligned_token_to_masked_position = pos_data.aligned_matching_absolute_position_in_sequence + tokenization_shift
    aligned_token_to_masked_position_tensor = torch.tensor([aligned_token_to_masked_position], device=device, dtype=torch.long)
    shift_suspected = _shift_locations(suspected_repeat_locations, tokenization_shift)
    suspected_positions = [p for start, end in shift_suspected for p in range(start, end + 1)]
    near_suspected = _get_positions_near_repeat(shift_suspected, near_repeat_threshold, tokenized_sequence, tokenizer)
    original_set = set(p for start, end in shift_repeat_locations for p in range(start, end + 1))
    total_problematic = sorted(set(suspected_positions + near_suspected) - original_set)
    total_problematic_tensor = torch.tensor(total_problematic, device=device, dtype=torch.long)
    return RepeatsInfoAboutSequence(
        repeat_positions_tensor=repeat_positions_tensor,
        near_repeat_positions_tensor=near_positions_tensor,
        aligned_token_to_masked_position_tensor=aligned_token_to_masked_position_tensor,
        problematic_positions_in_sequence_tensor_related_to_suspected_repeat=total_problematic_tensor,
    )

def lower_except_first(s: str) -> str:
    if not s:
        return s
    return s[0] + s[1:].lower()

def assign_final_tag(concept_category, concept_name):
    s=str(concept_name)
    if "blosum_cluster" in s or "_cluster_" in s:
        return "Blosum62"
    if "positive_charged_amino_acid" in s or "negative_charged_amino_acid" in s or "polar_uncharged_amino_acid" in s:
        return "Charge"
    if "polar_amino_acid" in s or "non_polar_amino_acid" in s:
        return "Polarity"
    if "hydrophobic_amino_acid" in s or "neutral_amino_acid" in s or "hydrophilic_amino_acid" in s:
        return "Hydropathy"
    if any(x in s for x in ["aliphatic_amino_acid", "aromatic_amino_acid", "aromatic_ring_amino_acid", "sulfur_amino_acid", "hydroxyl_amino_acid", "basic_amino_acid", "acidic_amino_acid", "amide_amino_acid"]):
        return "Chemical"
    if "very_small_amino_acid" in s or "small_amino_acid" in s or "medium_amino_acid" in s or "large_amino_acid" in s or "very_large_amino_acid" in s:
        return "Volume"
    if "hydrogen_donor_amino_acid" in s or "hydrogen_acceptor_amino_acid" in s or "hydrogen_donor_and_acceptor_amino_acid" in s or "no_hydrogen_donor_acceptor_amino_acid" in s:
        return "Hydrogen_Donor_Acceptor"
    if "alpha_helix_former_amino_acid_propensity" in s or "beta_sheet_former_amino_acid_propensity" in s or "turn_former_amino_acid_propensity" in s or "helix_breakers_amino_acid_propensity" in s:
        return "Secondary_Structure"
    
    #amino acid concepts 
    STANDARD_AMINO_ACIDS = {
    'A','R','N','D','C','Q','E','G','H','I',
    'L','K','M','F','P','S','T','W','Y','V'
    }
    amino_acid_concepts_names = [f"{aa}_amino_acid" for aa in STANDARD_AMINO_ACIDS]
    if s in amino_acid_concepts_names:
        return "Amino_Acid"
    
    if "repeat_tokens" in s or "aligned_token_masked_pos" in s:
        return "Repeat"
    if "mask_token" in s or "bos_token" in s or "eos_token" in s or "bos_eos_token" in s:
        return "Special_Token"
    
    print(f"Unknown concept name: {s}")
    return lower_except_first(concept_category)

def apply_concept_restrictions(df, min_auroc=0.99):
    restricted = {"mask_token", "bos_token", "eos_token", "aligned_token_masked_pos", "bos_eos_token"}
    return df[(~df["concept_name"].isin(restricted)) | (df["final_auroc"] >= min_auroc)]


def pick_best_with_blosum_preference(df):
    max_score = df["final_auroc"].max()
    tied = df[df["final_auroc"] == max_score]
    blosum = tied[tied["final_tag"] == "Blosum62"]
    return blosum.iloc[0] if not blosum.empty else tied.iloc[0]


def get_best_concept_per_neuron_filtered(df, min_auroc=0.99):
    df = apply_concept_restrictions(df, min_auroc=min_auroc)
    return pd.DataFrame([pick_best_with_blosum_preference(g) for _, g in df.groupby("component_id")])


def postprocess_auroc_results(results_df, concepts_array):
    concepts_df = pd.DataFrame([
        {"concept_name": c.concept_name, "concept_category": c.concept_category.name}
        for c in concepts_array
    ])
    concepts_df["final_tag"] = concepts_df.apply(
        lambda r: assign_final_tag(r["concept_category"], r["concept_name"]),
        axis=1,
    )
    df = results_df.merge(concepts_df[["concept_name", "concept_category", "final_tag"]], on="concept_name", how="left")
    df["is_positive_direction"] = df["concept_pct_positive"] > df["concept_pct_negative"]
    df["final_auroc"] = df.apply(
        lambda r: r["auroc"] if r["is_positive_direction"] else r["one_minus_auroc"],
        axis=1,
    )
    df["other_concept_names"] = [[] for _ in range(len(df))]
    grouped = df.groupby(["component_id", "final_auroc", "is_positive_direction"])
    for _, g in grouped:
        if len(g) <= 1:
            continue
        concept_names = g["concept_name"].tolist()
        for idx, row_idx in enumerate(g.index):
            df.at[row_idx, "other_concept_names"] = [c for i, c in enumerate(concept_names) if i != idx]
    return df
