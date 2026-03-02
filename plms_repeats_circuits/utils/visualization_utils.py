import numpy as np
import pandas as pd
import ast
import random
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass, field
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import torch
import seaborn as sns
from transformer_lens import HookedESM3
from protein_circuits.EAP.graph import Graph, GraphType
from protein_circuits.utils.per_example_utils import (
    load_importance_scores,
    normalize_scores,
    build_component_index_mapping,
    get_component_indices
)


def get_colorblind_palette(n_colors: Optional[int] = None) -> List[str]:
    """
    Returns seaborn's colorblind palette as a list of RGB strings for Plotly.
    
    Args:
        n_colors: Number of colors to return. Defaults to the full palette (usually 10).
    """
    palette = sns.color_palette("colorblind", n_colors=n_colors)
    return [f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})" for r, g, b in palette]



@dataclass
class VisConfig:
    dataset_path: str
    base_results_dir: str
    component_type: str
    components: List[str]
    model_type: str
    
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    per_layer: bool = False
    component_descriptions: List[str] = None
    
    selection_mode: str = "random"
    k_examples: int = 5
    M_candidates: int = 20
    random_seed: int = 42
    manual_example_names: List[str] = None
    manual_sequences: List[str] = None
    sort_by: str = "abs"
    
    show_corrupted: bool = False
    show_repeats: bool = True
    is_formal_figure: bool = True
    
    figure_width: int = None
    figure_height: int = None
    
    neuron_single_row_mode: bool = False
    neuron_use_abs_value: bool = False
    neuron_split_sequence: bool = False  # Split sequence into two rows in single_row_mode
    show_axis_labels: bool = False
    show_legend: bool = False


def is_uniprot_format(rep_id: str) -> bool:
    if pd.isna(rep_id) or not isinstance(rep_id, str):
        return False
    if "UniRef" in rep_id or "UPI" in rep_id:
        return False
    if "_" not in rep_id:
        return False
    parts = rep_id.split("_")
    return len(parts) >= 2


def get_model_type(model) -> str:
    if isinstance(model, HookedESM3):
        return "esm3"
    else:
        return "esm-c"


def load_dataset(csv_path: str, filter_uniprot: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    list_columns = ["repeat_locations", "repeat_alignments", "additional_suspected_repeats"]
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else []
            )
    
    if "cluster_id" in df.columns and "rep_id" in df.columns and "repeat_key" in df.columns:
        df["name"] = df.apply(
            lambda row: f"{row['cluster_id']}_{row['rep_id']}_{row['repeat_key']}", axis=1
        )
    
    if filter_uniprot and "rep_id" in df.columns:
        df = df[df["rep_id"].apply(is_uniprot_format)].copy()
    
    return df


def filter_scored_examples(
    example_names: List[str],
    dataset_df: pd.DataFrame,
    filter_uniprot: bool = True
) -> List[str]:
    valid = []
    print(f"filtering scored examples: len example names before filtering: {len(example_names)}")
    for name in example_names:
        if name not in dataset_df["name"].values:
            continue
        
        row = dataset_df[dataset_df["name"] == name].iloc[0]
        
        if "additional_suspected_repeats" in dataset_df.columns:
            suspected = row["additional_suspected_repeats"]
            if isinstance(suspected, list) and len(suspected) > 0:
                continue
        
        if filter_uniprot and "rep_id" in dataset_df.columns:
            rep_id = row.get("rep_id", "")
            if not is_uniprot_format(rep_id):
                continue
        
        valid.append(name)
    print(f"filtering scored examples: len valid names after filtering: {len(valid)}")
    return valid


def get_valid_dataset_rows(
    dataset_df: pd.DataFrame,
    filter_uniprot: bool = True
) -> pd.DataFrame:
    filtered = dataset_df.copy()
    
    print(f"getting valid dataset rows: rows before filtering: {len(filtered)}")
    if "additional_suspected_repeats" in filtered.columns:
        filtered = filtered[
            filtered["additional_suspected_repeats"].apply(lambda x: isinstance(x, list) and len(x) == 0)
        ].copy()
    
    if filter_uniprot and "rep_id" in filtered.columns:
        filtered = filtered[filtered["rep_id"].apply(is_uniprot_format)].copy()
    
    print(f"getting valid dataset rows: rows after filtering: {len(filtered)}")
    return filtered


def select_examples(
    config: VisConfig,
    scores: np.ndarray,
    example_names: List[str],
    dataset_df: pd.DataFrame,
    component_index_mapping: Dict[str, int]
) -> Union[List[str], List[List[str]]]:
    
    component_indices = get_component_indices(config.components, component_index_mapping)
    
    if config.selection_mode == "random":
        filtered_df = get_valid_dataset_rows(dataset_df, filter_uniprot=True)
        np.random.seed(config.random_seed)
        selected_idx = np.random.choice(len(filtered_df), size=min(config.k_examples, len(filtered_df)), replace=False)
        return filtered_df.iloc[selected_idx]["name"].tolist()
    
    elif config.selection_mode == "manual":
        if config.manual_example_names:
            return config.manual_example_names
        elif config.manual_sequences:
            result = []
            for seq in config.manual_sequences:
                mask = dataset_df["seq"] == seq
                if mask.any():
                    result.append(dataset_df[mask].iloc[0]["name"])
                else:
                    result.append(seq)
            return result
        return []
    
    valid_names = filter_scored_examples(example_names, dataset_df, filter_uniprot=True)
    print(f"len valid names of scored examples: {len(valid_names)}")
    valid_indices = [i for i, name in enumerate(example_names) if name in valid_names]
    valid_scores = scores[valid_indices]
    component_scores = valid_scores[:, component_indices]
    
    if config.selection_mode == "top_k":
        avg_scores = component_scores.mean(axis=1)
        sorted_indices = get_sorted_indices(avg_scores, config.sort_by)
        top_M_indices = sorted_indices[:config.M_candidates]
        
        np.random.seed(config.random_seed)
        selected = np.random.choice(top_M_indices, size=min(config.k_examples, len(top_M_indices)), replace=False)
        return [valid_names[i] for i in selected]
    
    elif config.selection_mode == "most_important":
        selected_per_component = []
        
        for comp_idx in range(len(config.components)):
            comp_scores = component_scores[:, comp_idx]
            sorted_indices = get_sorted_indices(comp_scores, config.sort_by)
            selected_idx = sorted_indices[0]
            selected_per_component.append(valid_names[selected_idx])
        
        return selected_per_component
    
    elif config.selection_mode == "shortest":
        filtered_df = get_valid_dataset_rows(dataset_df, filter_uniprot=True)
        filtered_df = filtered_df.copy()
        filtered_df["seq_len"] = filtered_df["seq"].str.len()
        filtered_df = filtered_df.sort_values("seq_len")
        
        # Get M smallest sequence lengths
        unique_lengths = filtered_df["seq_len"].unique()[:config.M_candidates]
        candidates = []
        
        # For each length, pick the candidate with maximum min_space
        for seq_len in unique_lengths:
            length_group = filtered_df[filtered_df["seq_len"] == seq_len]
            if "min_space" in length_group.columns:
                # Pick the one with maximum min_space
                best_candidate = length_group.loc[length_group["min_space"].idxmax()]
            else:
                # If no min_space, just take the first one
                best_candidate = length_group.iloc[0]
            candidates.append(best_candidate["name"])
        
        # Randomly sample k_examples from candidates
        if len(candidates) <= config.k_examples:
            return candidates
        else:
            return random.sample(candidates, config.k_examples)
    
    else:
        raise ValueError(f"Unknown selection_mode: {config.selection_mode}")


def get_sorted_indices(scores: np.ndarray, sort_by: str) -> np.ndarray:
    if sort_by == "abs":
        return np.argsort(np.abs(scores))[::-1]
    elif sort_by == "positive":
        return np.argsort(scores)[::-1]
    elif sort_by == "negative":
        return np.argsort(scores)
    else:
        raise ValueError(f"Unknown sort_by: {sort_by}")


def get_figure_params(is_formal: bool) -> Dict:
    if is_formal:
        return {
            "attention_subplot_size": 150,
            "attention_font_size": 14,
            "attention_tick_font_size": 10,
            "attention_colorbar_width": 10,
            "neuron_char_size": 12,
            "neuron_font_size": 14,
            "neuron_aa_font_size": 14,
            "neuron_subplot_size": 300,
            "neuron_title_font_size": 16,
            "neuron_spacing": 5,
            "title_font_size": 16,
            "label_font_size": 14,
            "margin": dict(l=40, r=100, t=80, b=60),
        }
    else:
        return {
            "attention_subplot_size": 400,
            "attention_font_size": 16,
            "attention_tick_font_size": 14,
            "attention_colorbar_width": 20,
            "neuron_char_size": 18,
            "neuron_font_size": 16,
            "neuron_aa_font_size": 14,
            "neuron_subplot_size": 500,
            "neuron_title_font_size": 18,
            "neuron_spacing": 10,
            "title_font_size": 20,
            "label_font_size": 16,
            "margin": dict(l=60, r=120, t=100, b=80),
        }


def get_example_metadata(df: pd.DataFrame, example_identifier: str) -> Dict:
    if example_identifier in df["name"].values:
        row = df[df["name"] == example_identifier].iloc[0]
        return {
            "name": example_identifier,
            "rep_id": row.get("rep_id", ""),
            "seq": row.get("seq", ""),
            "corrupted_sequence": row.get("corrupted_sequence", ""),
            "repeat_locations": row.get("repeat_locations", []),
            "masked_poistion": row.get("masked_poistion", None),
            "repeat_alignments": row.get("repeat_alignments", []),
        }
    elif "seq" in df.columns and example_identifier in df["seq"].values:
        row = df[df["seq"] == example_identifier].iloc[0]
        return {
            "name": row.get("name", example_identifier),
            "rep_id": row.get("rep_id", ""),
            "seq": example_identifier,
            "corrupted_sequence": row.get("corrupted_sequence", ""),
            "repeat_locations": row.get("repeat_locations", []),
            "masked_poistion": row.get("masked_poistion", None),
            "repeat_alignments": row.get("repeat_alignments", []),
        }
    else:
        return {
            "name": example_identifier[:20] + "..." if len(example_identifier) > 20 else example_identifier,
            "rep_id": "",
            "seq": example_identifier,
            "corrupted_sequence": "",
            "repeat_locations": [],
            "masked_poistion": None,
            "repeat_alignments": [],
        }


def format_component_title(component: str, description: str = None) -> str:
    if description:
        return f"{component} ({description})"
    return component


def downsample_tick_labels(labels: List[str]) -> List[str]:
    n = len(labels)
    if n <= 50:
        step = 1
    elif n <= 100:
        step = 2
    elif n <= 300:
        step = 5
    elif n <= 600:
        step = 10
    else:
        step = 20
    
    return [label if i % step == 0 else "" for i, label in enumerate(labels)]


def add_repeat_boundary_shapes(fig, repeat_locations, seq_len, row, col):
    if not repeat_locations or len(repeat_locations) == 0:
        return
    
    for start, end in repeat_locations:
        start_adj = start + 1
        end_adj = end + 2
        
        for pos in (start_adj, end_adj):
            # Vertical line
            fig.add_shape(
                type="line",
                x0=pos - 0.5, x1=pos - 0.5,
                y0=-0.5, y1=seq_len + 1.5,
                line=dict(color="red", width=1, dash="dash"),
                layer="above",
                row=row,
                col=col
            )
            # Horizontal line
            fig.add_shape(
                type="line",
                x0=-0.5, x1=seq_len + 1.5,
                y0=pos - 0.5, y1=pos - 0.5,
                line=dict(color="red", width=1, dash="dash"),
                layer="above",
                row=row,
                col=col
            )


def add_neuron_repeat_borders(fig, repeat_locations, grid_width, grid_height, tokenized_seq_len, original_seq_len, row, col):
    if not repeat_locations or len(repeat_locations) == 0:
        return
    
    # Map original sequence positions to tokenized positions
    # Tokenized: CLS (0), seq[0] (1), seq[1] (2), ..., seq[n-1] (n), EOS (n+1)
    # Original: seq[0] (0), seq[1] (1), ..., seq[n-1] (n-1)
    # So original position pos maps to tokenized position pos + 1
    
    for start, end in repeat_locations:
        for orig_pos in range(start, min(end + 1, original_seq_len)):
            tokenized_pos = orig_pos + 1  # Map to tokenized position (skip CLS)
            
            if tokenized_pos >= tokenized_seq_len:
                continue
            
            grid_row = tokenized_pos // grid_width
            grid_col = tokenized_pos % grid_width
            
            if grid_row >= grid_height:
                continue
            
            fig.add_shape(
                type="rect",
                x0=grid_col - 0.5,
                x1=grid_col + 0.5,
                y0=grid_row - 0.5,
                y1=grid_row + 0.5,
                line=dict(color="green", width=2),
                layer="above",
                row=row,
                col=col,
                fillcolor="rgba(0,0,0,0)",
            )


def visualize(
    config: VisConfig,
    examples: Union[List[str], List[List[str]]],
    dataset_df: pd.DataFrame,
    model: HookedESM3
) -> go.Figure:
    if config.component_type == "attention":
        return visualize_attention_heads(config, examples, dataset_df, model)
    elif config.component_type == "neuron":
        return visualize_neurons(config, examples, dataset_df, model)
    else:
        raise ValueError(f"Unknown component_type: {config.component_type}")


def visualize_attention_heads(
    config: VisConfig,
    examples: Union[List[str], List[List[str]]],
    dataset_df: pd.DataFrame,
    model: HookedESM3
) -> go.Figure:
    params = get_figure_params(config.is_formal_figure)
    
    is_most_important = config.selection_mode == "most_important"
    
    if is_most_important:
        n_rows = 1
        n_cols = len(config.components)
    else:
        n_rows = len(examples)
        n_cols = len(config.components)
    
    subplot_size = params["attention_subplot_size"]
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        horizontal_spacing=0.12,
        vertical_spacing=0.25,
    )
    
    subplot_size = params["attention_subplot_size"]
    colorbar_width = params["attention_colorbar_width"] * n_rows
    total_width = config.figure_width or (subplot_size * n_cols + 200 + colorbar_width)
    total_height = config.figure_height or (subplot_size * n_rows + 200)
    
    for i in range(n_rows):
        cached_cache = None
        cached_tokens = None
        cached_tokenizer = None
        
        for j in range(n_cols):
            if is_most_important:
                example_name = examples[j]
            else:
                example_name = examples[i]
            
            example_data = get_example_metadata(dataset_df, example_name)
            
            component = config.components[j]
            desc = config.component_descriptions[j] if config.component_descriptions else None
            
            if not getattr(config, "show_axis_labels", False):
                add_component_title(fig, component, desc, i, j, n_rows, n_cols, params, config.is_formal_figure)
            
            if not config.is_formal_figure:
                add_example_label(fig, example_data.get("rep_id", ""), i, j, n_rows, n_cols, params)
            
            if is_most_important:
                attn_result = create_attention_heatmap(
                    model, example_data, component, params, config, config.show_legend, i, n_rows, j, n_cols,
                    cache=None, tokens=None, tokenizer=None
                )
                if attn_result.get("cache") is not None:
                    del attn_result["cache"]
                    del attn_result["tokens"]
                    attn_result["tokenizer"] = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                attn_result = create_attention_heatmap(
                    model, example_data, component, params, config, config.show_legend, i, n_rows, j, n_cols,
                    cache=cached_cache, tokens=cached_tokens, tokenizer=cached_tokenizer
                )
                
                if cached_cache is None:
                    cached_cache = attn_result.get("cache")
                    cached_tokens = attn_result.get("tokens")
                    cached_tokenizer = attn_result.get("tokenizer")
            
            fig.add_trace(attn_result["trace"], row=i+1, col=j+1)
            
            if attn_result["trace"].showscale:
                trace_idx = len(fig.data) - 1
                xaxis_name = f"xaxis{j+1}" if j > 0 else "xaxis"
                subplot_idx = i * n_cols + j
                yaxis_name = f"yaxis{subplot_idx+1}" if subplot_idx > 0 else "yaxis"
                
                if xaxis_name in fig.layout and yaxis_name in fig.layout:
                    x_domain = fig.layout[xaxis_name].domain
                    y_domain = fig.layout[yaxis_name].domain
                    y_center = (y_domain[0] + y_domain[1]) / 2
                    
                    fig.data[trace_idx].colorbar.x = x_domain[1] + 0.02
                    fig.data[trace_idx].colorbar.xref = "paper"
                    fig.data[trace_idx].colorbar.xanchor = "left"
                    fig.data[trace_idx].colorbar.y = y_center
                    fig.data[trace_idx].colorbar.yref = "paper"
                    fig.data[trace_idx].colorbar.yanchor = "middle"
                    fig.data[trace_idx].colorbar.len = (y_domain[1] - y_domain[0]) * 0.7
                    fig.data[trace_idx].colorbar.thickness = params["attention_colorbar_width"]
            
            seq_len = len(attn_result["tick_indices"])
            
            if attn_result["show_ticks"]:
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=attn_result["tick_indices"],
                    ticktext=attn_result["tick_labels"],
                    tickangle=45,
                    tickfont=dict(size=5),
                    range=[-0.5, seq_len - 0.5],
                    row=i+1,
                    col=j+1
                )
                yaxis_update = {
                    "tickmode": "array",
                    "tickvals": attn_result["tick_indices"],
                    "ticktext": attn_result["tick_labels"],
                    "tickfont": dict(size=5),
                    "range": [seq_len - 0.5, -0.5],
                    "row": i+1,
                    "col": j+1
                }
                if n_rows == 1:
                    yaxis_update["scaleanchor"] = f"x{j+1}" if j > 0 else "x"
                    yaxis_update["scaleratio"] = 1
        
                                        
                fig.update_yaxes(**yaxis_update)
            else:
                fig.update_xaxes(
                    showticklabels=False,
                    range=[-0.5, seq_len - 0.5],
                    row=i+1,
                    col=j+1
                )
                yaxis_update = {
                    "showticklabels": False,
                    "range": [seq_len - 0.5, -0.5],
                    "row": i+1,
                    "col": j+1
                }
                if n_rows == 1:
                    yaxis_update["scaleanchor"] = f"x{j+1}" if j > 0 else "x"
                    yaxis_update["scaleratio"] = 1
               
                fig.update_yaxes(**yaxis_update)
            
            if config.show_repeats:
                repeat_locs = example_data.get("repeat_locations", [])
                seq_len = len(example_data.get("seq", ""))
                add_repeat_boundary_shapes(fig, repeat_locs, seq_len, i+1, j+1)
            
            if getattr(config, "show_axis_labels", False):
                fig.update_xaxes(title_text="Key", title_font=dict(size=14), title_standoff=1, row=i+1, col=j+1)
                fig.update_yaxes(title_text="Query", title_font=dict(size=14), title_standoff=1, row=i+1, col=j+1)
        
        if not is_most_important and cached_cache is not None:
            del cached_cache
            del cached_tokens
            cached_tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"show_legend: {getattr(config, 'show_legend', False)}")
    fig.update_layout(
        width=total_width,
        height=total_height,
        font=dict(size=params["attention_font_size"], family="Arial"),
        margin=params["margin"],
        showlegend=getattr(config, "show_legend", False),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    
    return fig


def visualize_neurons(
    config: VisConfig,
    examples: Union[List[str], List[List[str]]],
    dataset_df: pd.DataFrame,
    model: HookedESM3
) -> go.Figure:
    params = get_figure_params(config.is_formal_figure)
    
    is_most_important = config.selection_mode == "most_important"
    
    # Check if we should use single row per neuron mode
    use_single_row_mode = config.neuron_single_row_mode and config.k_examples == 1
    
    if use_single_row_mode:
        # Each neuron gets its own row with just the visualization
        n_rows = len(config.components)
        n_cols = 1  # Just visualization, no description column
    elif is_most_important:
        n_rows = 1
        n_cols = len(config.components)
    else:
        n_rows = len(examples)
        n_cols = len(config.components)
    
    subplot_size = params.get("neuron_subplot_size", params["attention_subplot_size"] * 2)
    base_height = subplot_size * n_rows + 100
    
    split_sequence = getattr(config, 'neuron_split_sequence', False)
    
    if use_single_row_mode:
        # Wider layout for single row mode with square cells
        # For split sequence, use narrower width to make cells less wide
        if split_sequence:
            total_width = config.figure_width or 750  # Narrower for split mode
        else:
            total_width = config.figure_width or 1400
        # Adjust height: more space when split into two rows
        height_per_row = None
        if split_sequence:
            if len(config.components) == 1:
                height_per_row = 40
            else:
                height_per_row = 80
        else:
            height_per_row = 60
        total_height = config.figure_height or (height_per_row * n_rows + 80)
    else:
        total_width = config.figure_width or (base_height * 3)
        total_height = config.figure_height or base_height
    
    if use_single_row_mode:
        # Create subplots with just one column for visualization
        # More vertical spacing when splitting sequence into two rows
        v_spacing = 0.12 if split_sequence else 0.001
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            horizontal_spacing=0.01,
            vertical_spacing=v_spacing,
        )
    else:
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            horizontal_spacing=0.1,
            vertical_spacing=0.08,
        )
    
    cached_cache = None
    cached_tokens = None
    cached_tokenizer = None
    
    for i in range(n_rows):
        if use_single_row_mode:
            # In single row mode, each row is a different neuron, single example
            example_name = examples[0] if isinstance(examples, list) else examples
            component_idx = i
            example_data = get_example_metadata(dataset_df, example_name)
            
            component = config.components[component_idx]
            desc = config.component_descriptions[component_idx] if config.component_descriptions else None
            
            # Create single-row neuron heatmap (no colorbar in single row mode)
            neuron_result = create_neuron_single_row_heatmap(
                model, example_data, component, params, config, show_colorbar=False,
                cache=cached_cache, tokens=cached_tokens, tokenizer=cached_tokenizer
            )
            
            if cached_cache is None:
                cached_cache = neuron_result.get("cache")
                cached_tokens = neuron_result.get("tokens")
                cached_tokenizer = neuron_result.get("tokenizer")
            
            fig.add_trace(neuron_result["trace"], row=i+1, col=1)
            
            # Make cells square by setting scale ratio and limit to actual sequence length
            seq_len = neuron_result.get("seq_len", 0)
            grid_width = neuron_result.get("grid_width", seq_len)
            n_rows_grid = neuron_result.get("n_rows_grid", 1)
            split_sequence = neuron_result.get("split_sequence", False)
            
            # For split sequence, ensure both rows align by using same x-axis range
            # Don't use scaleanchor for split mode as it can cause misalignment
            if split_sequence:
                fig.update_xaxes(
                    showticklabels=False,
                    range=[-0.5, grid_width - 0.5],  # Same width for both rows
                    row=i+1,
                    col=1
                )
                fig.update_yaxes(
                    showticklabels=False,
                    autorange="reversed",
                    range=[-0.5, n_rows_grid - 0.5],  # 2 rows
                    row=i+1,
                    col=1
                )
            else:
                fig.update_xaxes(
                    showticklabels=False, 
                    scaleanchor=f"y{i+1}" if i > 0 else "y",
                    scaleratio=1,
                    range=[-0.5, grid_width - 0.5],
                    row=i+1, 
                    col=1
                )
                fig.update_yaxes(
                    showticklabels=False, 
                    autorange="reversed",
                    range=[-0.5, n_rows_grid - 0.5],
                    row=i+1, 
                    col=1
                )
            
            # Add black borders to all cells, and green borders for repeat regions
            repeat_locs = neuron_result.get("repeat_locs", [])
            original_seq_len = neuron_result.get("original_seq_len", 0)
            
            # Check if we should show repeat borders for this neuron
            should_show_repeats = False
            if config.show_repeats and desc:
                should_show_repeats = "repeat" in desc.lower()
            
            # Add borders for all cells
            for token_pos in range(seq_len):
                orig_pos = token_pos - 1  # Tokenized position to original sequence position
                
                # Calculate grid position based on split or single row mode
                if split_sequence:
                    half_len = (seq_len + 1) // 2
                    row_idx = 0 if token_pos < half_len else 1
                    col_idx = token_pos if token_pos < half_len else token_pos - half_len
                else:
                    row_idx = 0
                    col_idx = token_pos
                
                # Check if this position is in a repeat region
                in_repeat = False
                if should_show_repeats and repeat_locs and 0 <= orig_pos < original_seq_len:
                    in_repeat = any(start <= orig_pos <= end for start, end in repeat_locs)
                
                # Choose border style
                if in_repeat:
                    border_color = "green"
                    border_width = 4
                else:
                    border_color = "black"
                    border_width = 1
                
                fig.add_shape(
                    type="rect",
                    x0=col_idx - 0.5,
                    x1=col_idx + 0.5,
                    y0=row_idx - 0.5,
                    y1=row_idx + 0.5,
                    line=dict(color=border_color, width=border_width),
                    layer="above",
                    row=i+1,
                    col=1,
                    fillcolor="rgba(0,0,0,0)",
                )
        else:
            # Original mode logic
            for j in range(n_cols):
                if is_most_important:
                    example_name = examples[j]
                else:
                    example_name = examples[i]
                
                example_data = get_example_metadata(dataset_df, example_name)
                
                component = config.components[j]
                desc = config.component_descriptions[j] if config.component_descriptions else None
                
                add_component_title(fig, component, desc, i, j, n_rows, n_cols, params, config.is_formal_figure)
                if not config.is_formal_figure:
                    add_example_label(fig, example_data.get("rep_id", ""), i, j, n_rows, n_cols, params)
                
                if is_most_important:
                    neuron_result = create_neuron_heatmap(
                        model, example_data, component, params, config, True,
                        cache=None, tokens=None, tokenizer=None
                    )
                    if neuron_result.get("cache") is not None:
                        del neuron_result["cache"]
                        del neuron_result["tokens"]
                        neuron_result["tokenizer"] = None
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    neuron_result = create_neuron_heatmap(
                        model, example_data, component, params, config, True,
                        cache=cached_cache, tokens=cached_tokens, tokenizer=cached_tokenizer
                    )
                    
                    if cached_cache is None:
                        cached_cache = neuron_result.get("cache")
                        cached_tokens = neuron_result.get("tokens")
                        cached_tokenizer = neuron_result.get("tokenizer")
                
                fig.add_trace(neuron_result["trace"], row=i+1, col=j+1)
                
                if neuron_result["trace"].showscale:
                    trace_idx = len(fig.data) - 1
                    xaxis_name = f"xaxis{j+1}" if j > 0 else "xaxis"
                    subplot_idx = i * n_cols + j
                    yaxis_name = f"yaxis{subplot_idx+1}" if subplot_idx > 0 else "yaxis"
                    
                    if xaxis_name in fig.layout and yaxis_name in fig.layout:
                        x_domain = fig.layout[xaxis_name].domain
                        y_domain = fig.layout[yaxis_name].domain
                        y_center = (y_domain[0] + y_domain[1]) / 2
                        
                        fig.data[trace_idx].colorbar.x = x_domain[1] + 0.02
                        fig.data[trace_idx].colorbar.xref = "paper"
                        fig.data[trace_idx].colorbar.xanchor = "left"
                        fig.data[trace_idx].colorbar.y = y_center
                        fig.data[trace_idx].colorbar.yref = "paper"
                        fig.data[trace_idx].colorbar.yanchor = "middle"
                        fig.data[trace_idx].colorbar.len = (y_domain[1] - y_domain[0]) * 0.7
                        fig.data[trace_idx].colorbar.thickness = params["attention_colorbar_width"]
                
                # Only show repeats if description contains "repeat" (case insensitive)
                should_show_repeats = config.show_repeats and desc and "repeat" in desc.lower()
                if should_show_repeats and neuron_result.get("repeat_locs"):
                    add_neuron_repeat_borders(
                        fig,
                        neuron_result["repeat_locs"],
                        neuron_result["grid_width"],
                        neuron_result["grid_height"],
                        neuron_result["seq_len"],  # Tokenized sequence length
                        neuron_result["original_seq_len"],  # Original sequence length
                        i+1,
                        j+1
                    )
                
                grid_width = neuron_result["grid_width"]
                grid_height = neuron_result["grid_height"]
                actual_max_col = neuron_result.get("actual_max_col", grid_width - 1)
                
                fig.update_xaxes(
                    showticklabels=False,
                    row=i+1,
                    col=j+1
                )
                fig.update_yaxes(
                    showticklabels=False,
                    autorange="reversed",
                    row=i+1,
                    col=j+1
                )
                
                repeat_locs = neuron_result.get("repeat_locs", [])
                original_seq_len = neuron_result.get("original_seq_len", 0)
                seq_len = neuron_result.get("seq_len", 0)
                
                for row_idx in range(grid_height):
                    for col_idx in range(grid_width):
                        token_pos = row_idx * grid_width + col_idx
                        if token_pos >= seq_len:
                            continue
                        
                        orig_pos = token_pos - 1
                        in_repeat = False
                        if should_show_repeats and 0 <= orig_pos < original_seq_len:
                            in_repeat = any(start <= orig_pos <= end for start, end in repeat_locs)
                        
                        if in_repeat:
                            border_color = "#00AA00"  # Softer green (less bright)
                            border_width = 5  # Thicker border
                        else:
                            border_color = "black"
                            border_width = 1
                        
                        fig.add_shape(
                            type="rect",
                            x0=col_idx - 0.5,
                            x1=col_idx + 0.5,
                            y0=row_idx - 0.5,
                            y1=row_idx + 0.5,
                            line=dict(color=border_color, width=border_width),
                            layer="above",
                            row=i+1,
                            col=j+1,
                            fillcolor="rgba(0,0,0,0)",
                        )
            
            if not is_most_important and cached_cache is not None:
                del cached_cache
                del cached_tokens
                cached_tokenizer = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Clean up cache at the end if in single row mode
    if use_single_row_mode and cached_cache is not None:
        del cached_cache
        del cached_tokens
        cached_tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Use smaller margins for single row mode
    if use_single_row_mode:
        margins = dict(l=20, r=20, t=20, b=20)
    else:
        margins = params["margin"]
    
    fig.update_layout(
        width=total_width,
        height=total_height,
        font=dict(size=params["neuron_font_size"], family="Arial"),
        margin=margins,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    
    return fig


def add_component_title(fig, component, desc, row, col, n_rows, n_cols, params, is_formal_figure=True):
    # Get subplot domains to position title above
    xaxis_name = f"xaxis{col+1}" if col > 0 else "xaxis"
    subplot_idx = row * n_cols + col
    yaxis_name = f"yaxis{subplot_idx+1}" if subplot_idx > 0 else "yaxis"
    
    if xaxis_name in fig.layout and yaxis_name in fig.layout:
        x_domain = fig.layout[xaxis_name].domain
        y_domain = fig.layout[yaxis_name].domain
        
        x_center = (x_domain[0] + x_domain[1]) / 2
        y_top = y_domain[1]
        
        if n_rows == 1 and desc and is_formal_figure:
            # For single row in formal figure: description above, component name below
            fig.add_annotation(
                text=desc,
                xref="paper",
                yref="paper",
                x=x_center,
                y=y_top + 0.005,
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=params["title_font_size"])
            )
            fig.add_annotation(
                text=component,
                xref="paper",
                yref="paper",
                x=x_center,
                y=y_domain[0] - 0.005,
                xanchor="center",
                yanchor="top",
                showarrow=False,
                font=dict(size=params["title_font_size"])
            )
        else:
            # For multiple rows OR non-formal figure: component name and description on same line above
            title_text = format_component_title(component, desc)
            fig.add_annotation(
                text=title_text,
                xref="paper",
                yref="paper",
                x=x_center,
                y=y_top + 0.005,
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=params["title_font_size"])
            )


def add_example_label(fig, rep_id, row, col, n_rows, n_cols, params):
    if rep_id:
        # Calculate subplot number for yref
        subplot_num = row * n_cols + col + 1
        
        xref = f"x{col+1} domain" if col > 0 else "x domain"
        yref = f"y{subplot_num} domain" if subplot_num > 1 else "y domain"
        
        y_offset = -0.12
        
        fig.add_annotation(
            text=rep_id,
            xref=xref,
            yref=yref,
            x=0.5,
            y=y_offset,
            xanchor="center",
            yanchor="top",
            showarrow=False,
            font=dict(size=params["label_font_size"])
        )


def create_attention_heatmap(model, example_data, component, params, config, show_colorbar, row_num, n_rows, col_num=None, n_cols=None, cache=None, tokens=None, tokenizer=None):
    from protein_circuits.utils.esm3_utils import mask_protein_short, replace_short_mask_with_mask_token, load_tokenizer
    from protein_circuits.EAP.attribute import tokenize_plus
    
    if tokenizer is None:
        tokenizer = load_tokenizer(config.model_type, model)
    
    if tokens is None or cache is None:
        seq = example_data["seq"]
        masked_pos = example_data["masked_poistion"]
        if masked_pos is not None:
            clean_masked = mask_protein_short(seq, masked_pos, tokenizer)
            clean_masked = replace_short_mask_with_mask_token(clean_masked, tokenizer)
        else:
            clean_masked = seq
        device = next(model.parameters()).device
        tokens, attention_mask, _, _ = tokenize_plus(model, [clean_masked])
        
        if cache is None:
            with torch.no_grad():
                _, cache = model.run_with_cache(sequence_tokens=tokens.to(device), sequence_id=attention_mask.to(device))
    
    layer = int(component.split(".")[0].replace("a", "").replace("h", ""))
    head = int(component.split(".")[1].replace("h", ""))
    
    attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"].squeeze()[head].cpu().numpy()
    
    token_strings = tokenizer.convert_ids_to_tokens(tokens[0].tolist())
    token_labels = [f"{tok} ({i})" for i, tok in enumerate(token_strings)]
    sampled_labels = downsample_tick_labels(token_labels)
    
    # Create hover data
    seq_len = len(token_strings)
    customdata = np.empty((seq_len, seq_len), dtype=object)
    for i in range(seq_len):
        for j in range(seq_len):
            customdata[i, j] = (i, j, token_strings[i], token_strings[j])
    
    # Position colorbar for each subplot - will be positioned to the right in the main function
    if show_colorbar:
        colorbar = dict(
            len=0.7,
            thickness=params["attention_colorbar_width"]
        )
    else:
        colorbar = None
    
    return {
        "trace": go.Heatmap(
            z=attn_pattern,
            colorscale="Blues",
            showscale=show_colorbar,
            colorbar=colorbar,
            customdata=customdata,
            hovertemplate=(
                "Query (row) %{customdata[0]}: %{customdata[2]}<br>"
                "Key (col) %{customdata[1]}: %{customdata[3]}<br>"
                "Attention: %{z:.5f}<extra></extra>"
            ),
        ),
        "tick_labels": sampled_labels,
        "tick_indices": list(range(len(token_strings))),
        "show_ticks": not config.is_formal_figure,
        "cache": cache,
        "tokens": tokens,
        "tokenizer": tokenizer,
    }


def create_neuron_single_row_heatmap(model, example_data, component, params, config, show_colorbar, cache=None, tokens=None, tokenizer=None):
    """Create a single-row heatmap for neuron visualization."""
    from protein_circuits.utils.esm3_utils import mask_protein_short, replace_short_mask_with_mask_token, load_tokenizer
    from protein_circuits.EAP.attribute import tokenize_plus
    
    if tokenizer is None:
        tokenizer = load_tokenizer(config.model_type, model)
    
    if tokens is None or cache is None:
        seq = example_data["seq"]
        masked_pos = example_data["masked_poistion"]
        if masked_pos is not None:
            clean_masked = mask_protein_short(seq, masked_pos, tokenizer)
            clean_masked = replace_short_mask_with_mask_token(clean_masked, tokenizer)
        else:
            clean_masked = seq
        
        tokens, attention_mask, _, _ = tokenize_plus(model, [clean_masked])
        device = next(model.parameters()).device
        if cache is None:
            with torch.no_grad():
                _, cache = model.run_with_cache(sequence_tokens=tokens.to(device), sequence_id=attention_mask.to(device))
    
    layer = int(component.split("_")[0].replace("m", ""))
    neuron_idx = int(component.split("_")[1].replace("n", ""))
    
    neuron_acts = cache[f"blocks.{layer}.mlp.hook_post"][0, :, neuron_idx].cpu().numpy()
    activations = neuron_acts
    
    # Apply absolute value if requested
    if config.neuron_use_abs_value:
        activations = np.abs(activations)
    
    token_strings = tokenizer.convert_ids_to_tokens(tokens[0].tolist())
    seq = example_data["seq"]
    repeat_locs = example_data.get("repeat_locations", [])
    
    seq_len = len(token_strings)
    
    # Check if we should split into two rows
    split_sequence = getattr(config, 'neuron_split_sequence', False)
    
    if split_sequence:
        # Split sequence into two rows - ensure both rows have same width for alignment
        half_len = (seq_len + 1) // 2  # First row gets the extra token if odd
        n_rows_grid = 2
        # Use the larger of the two row lengths to ensure both rows align
        grid_width = half_len  # Both rows will have this width
        grid = np.full((n_rows_grid, grid_width), np.nan)
        text_grid = [['' for _ in range(grid_width)] for _ in range(n_rows_grid)]
        hover_text = [['' for _ in range(grid_width)] for _ in range(n_rows_grid)]
        
        for i in range(seq_len):
            row_idx = 0 if i < half_len else 1
            col_idx = i if i < half_len else i - half_len
            # Ensure col_idx doesn't exceed grid_width (shouldn't happen, but safety check)
            if col_idx >= grid_width:
                continue
            
            act = activations[i] if i < len(activations) else 0
            grid[row_idx, col_idx] = act
            
            token_str = token_strings[i]
            if token_str == tokenizer.cls_token or token_str == "<cls>":
                display_token = "<"
            elif token_str == tokenizer.eos_token or token_str == "<eos>":
                display_token = ">"
            elif token_str == tokenizer.mask_token or token_str == "<mask>":
                display_token = "?"
            else:
                display_token = token_str
            
            text_grid[row_idx][col_idx] = display_token
            
            orig_pos = i - 1
            in_repeat = False
            if 0 <= orig_pos < len(seq):
                in_repeat = any(start <= orig_pos <= end for start, end in repeat_locs)
            
            border_marker = " [R]" if (config.show_repeats and in_repeat) else ""
            hover_text[row_idx][col_idx] = f"Token Pos: {i}<br>Token: {token_str}<br>Act: {act:.5f}{border_marker}"
    else:
        # Create single row layout (1 x seq_len)
        n_rows_grid = 1
        grid_width = seq_len
        grid = np.full((1, seq_len), np.nan)
        text_grid = [['' for _ in range(seq_len)]]
        hover_text = [['' for _ in range(seq_len)]]
        
        for i in range(seq_len):
            act = activations[i] if i < len(activations) else 0
            grid[0, i] = act
            
            token_str = token_strings[i]
            if token_str == tokenizer.cls_token or token_str == "<cls>":
                display_token = "<"
            elif token_str == tokenizer.eos_token or token_str == "<eos>":
                display_token = ">"
            elif token_str == tokenizer.mask_token or token_str == "<mask>":
                display_token = "?"
            else:
                display_token = token_str
            
            text_grid[0][i] = display_token
            
            orig_pos = i - 1
            in_repeat = False
            if 0 <= orig_pos < len(seq):
                in_repeat = any(start <= orig_pos <= end for start, end in repeat_locs)
            
            border_marker = " [R]" if (config.show_repeats and in_repeat) else ""
            hover_text[0][i] = f"Token Pos: {i}<br>Token: {token_str}<br>Act: {act:.5f}{border_marker}"
    
    if len(activations) > 0:
        if config.neuron_use_abs_value:
            max_val = max(np.max(activations), 1e-8)
            min_val = 0
        else:
            max_abs = max(np.max(np.abs(activations)), 1e-8)
            max_val = max_abs
            min_val = -max_abs
    else:
        max_val = 1.0
        min_val = -1.0 if not config.neuron_use_abs_value else 0
    
    # Use different colorscale for absolute values (blue scale)
    if config.neuron_use_abs_value:
        colorscale_custom = [
            [0.0, 'rgb(255, 255, 255)'],
            [0.5, 'rgb(100, 150, 200)'],
            [1.0, 'rgb(33, 102, 172)'],
        ]
    else:
        colorscale_custom = [
            [0.0, 'rgb(33, 102, 172)'],
            [0.25, 'rgb(100, 150, 200)'],
            [0.5, 'rgb(255, 255, 255)'],
            [0.75, 'rgb(220, 100, 80)'],
            [1.0, 'rgb(178, 24, 43)'],
        ]
    
    colorbar_config = None
    if show_colorbar:
        if config.neuron_use_abs_value:
            tickvals = [0, max_val/2, max_val]
            ticktext = [f"{val:.1f}" for val in tickvals]
        else:
            tickvals = [min_val, min_val/2, 0, max_val/2, max_val]
            ticktext = [f"{val:.1f}" for val in tickvals]
        colorbar_config = dict(
            len=0.3,
            thickness=params["attention_colorbar_width"],
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            tickfont=dict(size=10),
        )
    
    trace_kwargs = {
        'z': grid,
        'text': text_grid,
        'texttemplate': '%{text}',
        'textfont': dict(
            size=params.get("neuron_aa_font_size", 16),  # Bigger font for single row mode
            color='black',
            family='monospace',
        ),
        'hovertext': hover_text,
        'hoverinfo': 'text',
        'colorscale': colorscale_custom,
        'zmin': min_val,
        'zmax': max_val,
        'showscale': show_colorbar,
        'colorbar': colorbar_config,
        'xgap': 1,  # Small gap between cells creates border effect
        'ygap': 1,
    }
    
    # Only add zmid for diverging colorscale (non-absolute values)
    if not config.neuron_use_abs_value:
        trace_kwargs['zmid'] = 0
    
    trace = go.Heatmap(**trace_kwargs)
    
    return {
        "trace": trace,
        "cache": cache,
        "tokens": tokens,
        "tokenizer": tokenizer,
        "seq_len": seq_len,
        "original_seq_len": len(seq),
        "repeat_locs": repeat_locs,  # Always return repeat_locs
        "grid_width": grid_width,
        "n_rows_grid": n_rows_grid,
        "split_sequence": split_sequence,
    }


def create_neuron_heatmap(model, example_data, component, params, config, show_colorbar, cache=None, tokens=None, tokenizer=None):
    from protein_circuits.utils.esm3_utils import mask_protein_short, replace_short_mask_with_mask_token, load_tokenizer
    from protein_circuits.EAP.attribute import tokenize_plus
    
    if tokenizer is None:
        tokenizer = load_tokenizer(config.model_type, model)
    
    if tokens is None or cache is None:
        seq = example_data["seq"]
        masked_pos = example_data["masked_poistion"]
        if masked_pos is not None:
            clean_masked = mask_protein_short(seq, masked_pos, tokenizer)
            clean_masked = replace_short_mask_with_mask_token(clean_masked, tokenizer)
        else:
            clean_masked = seq
        
        tokens, attention_mask, _, _ = tokenize_plus(model, [clean_masked])
        device = next(model.parameters()).device
        if cache is None:
            with torch.no_grad():
                _, cache = model.run_with_cache(sequence_tokens=tokens.to(device), sequence_id=attention_mask.to(device))
    
    layer = int(component.split("_")[0].replace("m", ""))
    neuron_idx = int(component.split("_")[1].replace("n", ""))
    
    neuron_acts = cache[f"blocks.{layer}.mlp.hook_post"][0, :, neuron_idx].cpu().numpy()
    activations = neuron_acts
    token_strings = tokenizer.convert_ids_to_tokens(tokens[0].tolist())
    seq = example_data["seq"]
    repeat_locs = example_data.get("repeat_locations", [])
    
    seq_len = len(token_strings)
    
    grid_width = int(np.ceil(np.sqrt(seq_len * 1.8)))
    grid_height = int(np.ceil(seq_len / grid_width))
    
    if grid_height * grid_width < seq_len:
        grid_height += 1
    
    grid = np.full((grid_height, grid_width), np.nan)
    text_grid = [['' for _ in range(grid_width)] for _ in range(grid_height)]
    hover_text = [['' for _ in range(grid_width)] for _ in range(grid_height)]
    
    for i in range(seq_len):
        row = i // grid_width
        col = i % grid_width
        act = activations[i] if i < len(activations) else 0
        grid[row, col] = act
        
        token_str = token_strings[i]
        if token_str == tokenizer.cls_token or token_str == "<cls>":
            display_token = "<"
        elif token_str == tokenizer.eos_token or token_str == "<eos>":
            display_token = ">"
        elif token_str == tokenizer.mask_token or token_str == "<mask>":
            display_token = "_"
        else:
            display_token = token_str
        
        text_grid[row][col] = display_token
        
        orig_pos = i - 1
        in_repeat = False
        if 0 <= orig_pos < len(seq):
            in_repeat = any(start <= orig_pos <= end for start, end in repeat_locs)
        
        border_marker = " [R]" if (config.show_repeats and in_repeat) else ""
        hover_text[row][col] = f"Token Pos: {i}<br>Token: {token_str}<br>Act: {act:.5f}{border_marker}"
    
    actual_max_col = (seq_len - 1) % grid_width if seq_len > 0 else 0
    
    if len(activations) > 0:
        max_abs = max(np.max(np.abs(activations)), 1e-8)
    else:
        max_abs = 1.0
    
    colorscale_custom = [
        [0.0, 'rgb(33, 102, 172)'],
        [0.25, 'rgb(100, 150, 200)'],
        [0.5, 'rgb(255, 255, 255)'],
        [0.75, 'rgb(220, 100, 80)'],
        [1.0, 'rgb(178, 24, 43)'],
    ]
    
    colorbar_config = None
    if show_colorbar:
        tickvals = [-max_abs, -max_abs/2, 0, max_abs/2, max_abs]
        ticktext = [f"{val:.1f}" for val in tickvals]
        colorbar_config = dict(
            len=0.3,
            thickness=params["attention_colorbar_width"],
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            tickfont=dict(size=10),
        )
    
    trace = go.Heatmap(
        z=grid,
        text=text_grid,
        texttemplate='%{text}',
        textfont=dict(
            size=params.get("neuron_aa_font_size", 14),
            color='black',
            family='monospace',
        ),
        hovertext=hover_text,
        hoverinfo='text',
        colorscale=colorscale_custom,
        zmid=0,
        zmin=-max_abs,
        zmax=max_abs,
        showscale=show_colorbar,
        colorbar=colorbar_config,
        xgap=0,
        ygap=5,
    )
    
    return {
        "trace": trace,
        "cache": cache,
        "tokens": tokens,
        "tokenizer": tokenizer,
        "grid_width": grid_width,
        "grid_height": grid_height,
        "actual_max_col": actual_max_col,
        "seq_len": seq_len,
        "original_seq_len": len(seq),
        "repeat_locs": repeat_locs if config.show_repeats else [],
    }



