"""
Save neuron activation statistics to CSV (with_repeats only).
"""
import pandas as pd
from typing import List

from activation_data import ActivationData, NeuronInfo, NeuronActivationStats


def save_neuron_stats_to_csv(activation_data: ActivationData, output_prefix: str):
    """
    Save neuron statistics to CSV file.

    Args:
        activation_data: The ActivationData object containing neuron info and stats
        output_prefix: Prefix for output file (e.g., "neuron_stats" will create
                      "neuron_stats_with_repeats.csv")
    """
    rows_with_repeats = []

    for neuron_info in activation_data.neuron_info_list:
        base_row = {
            'component_id': neuron_info.component_id,
            'layer': neuron_info.layer,
            'neuron_idx': neuron_info.neuron_idx,
        }

        if neuron_info.stats_with_repeats is None:
            raise ValueError(f"stats_with_repeats is None for neuron {neuron_info.component_id} (idx {neuron_info.neuron_idx})")

        stats = neuron_info.stats_with_repeats
        row_with_repeats = base_row.copy()

        max_values = [s.value for s in stats.top_k_max]
        max_sequences = [s.seq_name for s in stats.top_k_max]
        max_tokens = [s.token for s in stats.top_k_max]
        min_values = [s.value for s in stats.top_k_min]
        min_sequences = [s.seq_name for s in stats.top_k_min]
        min_tokens = [s.token for s in stats.top_k_min]

        row_with_repeats.update({
            'mean': stats.mean,
            'median': stats.median,
            'max': stats.max,
            'min': stats.min,
            'variance': stats.variance,
            'std': stats.variance ** 0.5,
            'pct_positive': stats.pct_positive,
            'pct_negative': stats.pct_negative,
            'top_k_max_values': max_values,
            'top_k_max_sequences': max_sequences,
            'top_k_max_tokens': max_tokens,
            'top_k_min_values': min_values,
            'top_k_min_sequences': min_sequences,
            'top_k_min_tokens': min_tokens,
        })
        rows_with_repeats.append(row_with_repeats)

    if rows_with_repeats:
        df_with_repeats = pd.DataFrame(rows_with_repeats)
        output_path_with_repeats = f"{output_prefix}_with_repeats.csv"
        df_with_repeats.to_csv(output_path_with_repeats, index=False)
        print(f"Saved {len(df_with_repeats)} neuron stats to {output_path_with_repeats}", flush=True)
    else:
        raise ValueError("No rows with repeats to save")
