"""Shared helpers for attribution vs mutations experiment."""

from __future__ import annotations

import ast
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from Bio.Align.substitution_matrices import load as load_blosum_matrix

from circuit_discovery_imports import create_induction_dataset_pandas

BLOSUM62 = load_blosum_matrix("BLOSUM62")
STANDARD_AMINO_ACIDS = set("ARNDCEQGHILKMFPSTWYV")


def load_relevant_components(
    component_recurrence_csv: str,
    repeat_type: str,
    component_recurrence_threshold: float,
) -> pd.DataFrame:
    """Filter component recurrence by {repeat_type}_ratio_in_graph >= threshold."""
    df = pd.read_csv(component_recurrence_csv)
    ratio_col = f"{repeat_type}_ratio_in_graph"
    if ratio_col not in df.columns:
        raise ValueError(f"CSV must contain '{ratio_col}' column")
    filtered = df[df[ratio_col].fillna(0) >= component_recurrence_threshold].copy()
    logging.info(
        "Loaded %d total components, filtered to %d relevant components",
        len(df),
        len(filtered),
    )
    return filtered


def get_top_blosum_substitution(
    original_aa: str, exclude_aa: Optional[str] = None
) -> Optional[str]:
    """Return the highest-scoring BLOSUM62 substitution for original_aa."""
    substitutions: list[tuple[str, float]] = []
    original_aa = original_aa.upper()

    for (aa1, aa2), score in BLOSUM62.items():
        if aa1 == original_aa and aa2 in STANDARD_AMINO_ACIDS and aa2 != original_aa:
            if exclude_aa is None or aa2 != exclude_aa:
                substitutions.append((aa2, score))
        elif aa2 == original_aa and aa1 in STANDARD_AMINO_ACIDS and aa1 != original_aa:
            if exclude_aa is None or aa1 != exclude_aa:
                substitutions.append((aa1, score))

    if not substitutions:
        logging.warning("No BLOSUM substitution found for %s", original_aa)
        return None

    substitutions.sort(key=lambda x: x[1], reverse=True)
    ret = substitutions[0][0]
    assert original_aa != ret
    return ret


def get_non_masked_repeat_info(
    seq: str,
    repeat_locations: List[List[int]],
    masked_position: int,
) -> Tuple[int, int, int, List[int], Optional[int]]:
    """Return indices and mutable positions in the non-masked repeat."""
    masked_repeat_idx = None
    assert (repeat_locations[0][1] - repeat_locations[0][0]) == (
        repeat_locations[1][1] - repeat_locations[1][0]
    ), "repeat locations must be of the same length"

    for i, (start, end) in enumerate(repeat_locations):
        if start <= masked_position <= end:
            masked_repeat_idx = i
            break

    if masked_repeat_idx is None:
        raise ValueError(
            f"Masked position {masked_position} not in any repeat: {repeat_locations}"
        )

    start_masked_repeat, _end_masked_repeat = repeat_locations[masked_repeat_idx]
    relative_masked_position = masked_position - start_masked_repeat
    non_masked_repeat_idx = 1 - masked_repeat_idx
    start_non_masked_repeat, end_non_masked_repeat = repeat_locations[non_masked_repeat_idx]
    aligned_position_to_exclude = start_non_masked_repeat + relative_masked_position

    available_positions = list(range(start_non_masked_repeat, end_non_masked_repeat + 1))
    if aligned_position_to_exclude in available_positions:
        available_positions.remove(aligned_position_to_exclude)

    return (
        non_masked_repeat_idx,
        start_non_masked_repeat,
        end_non_masked_repeat,
        available_positions,
        aligned_position_to_exclude,
    )


def introduce_mutations(
    seq: str,
    positions: List[int],
    n_mutations: int,
) -> Tuple[str, List[int], List[str]]:
    """Introduce n_mutations using top BLOSUM62 substitutions at random positions."""
    if n_mutations > len(positions):
        n_mutations = len(positions)

    selected_positions = random.sample(positions, n_mutations)
    seq_chars = list(seq)
    mutated_positions: list[int] = []
    replacements: list[str] = []

    for pos in selected_positions:
        original_aa = seq_chars[pos]
        replacement = get_top_blosum_substitution(original_aa)
        if replacement is not None and replacement != original_aa:
            seq_chars[pos] = replacement
            mutated_positions.append(pos)
            replacements.append(replacement)

    return "".join(seq_chars), mutated_positions, replacements


def create_mutated_induction_dataset(
    original_df: pd.DataFrame,
    n_mutations: int,
    tokenizer,
    metric: str,
) -> pd.DataFrame:
    """Mutate clean seq in non-masked repeat, then build induction EAP rows."""
    copy_df = original_df.copy()
    for idx, row in copy_df.iterrows():
        seq = row["seq"]
        repeat_locations = (
            ast.literal_eval(row["repeat_locations"])
            if isinstance(row["repeat_locations"], str)
            else row["repeat_locations"]
        )
        pos_col = "masked_position" if "masked_position" in row else "masked_poistion"
        masked_position = int(row[pos_col])

        _non_masked_idx, _start, _end, available_positions, _aligned = (
            get_non_masked_repeat_info(
                seq=seq,
                repeat_locations=repeat_locations,
                masked_position=masked_position,
            )
        )
        mutated_seq, _, _ = introduce_mutations(
            seq=seq,
            positions=available_positions,
            n_mutations=n_mutations,
        )
        copy_df.at[idx, "seq"] = mutated_seq

    return create_induction_dataset_pandas(
        df=copy_df,
        total_n_samples=len(copy_df),
        random_state=None,
        tokenizer=tokenizer,
        metric=metric,
    )


def nodes_csv_path(
    output_dir: Path, model_type: str, repeat_length: int
) -> Path:
    return output_dir / f"{model_type}_repeat_len_{repeat_length}_nodes.csv"
