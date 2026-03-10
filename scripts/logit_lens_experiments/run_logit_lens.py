from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformer_lens import HookedESM3, HookedESMC

from plms_repeats_circuits.utils.model_utils import get_device, get_masked_token_logits, mask_protein, tokenize_input
from plms_repeats_circuits.utils.esm_utils import (
    get_probs_from_logits,
    load_model,
    load_tokenizer_by_model_type,
)


def create_logit_lens_dataset_pandas(
    df: pd.DataFrame,
    total_n_samples: int,
    random_state: int,
    tokenizer,
) -> pd.DataFrame:
    """Create logit lens dataset from circuit discovery CSV. Supports masked_position or masked_poistion."""
    pos_col = "masked_position" if "masked_position" in df.columns else "masked_poistion"
    if pos_col not in df.columns:
        raise ValueError("Dataset must have 'masked_position' or 'masked_poistion'")

    if total_n_samples < len(df):
        sampled_df = df.sample(n=total_n_samples, random_state=random_state)
    else:
        sampled_df = df

    def process_row(row):
        clean = row["seq"]
        name = f"{row['cluster_id']}_{row['rep_id']}_{row['repeat_key']}"
        masked_pos = int(row[pos_col])
        clean_masked = mask_protein(clean, masked_pos, tokenizer)
        label = row["seq"][masked_pos]
        return pd.Series({
            "clean_masked": clean_masked,
            "masked_position_after_tokenization": masked_pos + 1,
            "seq": clean,
            "label": label,
            "name": name,
        })

    return sampled_df.apply(process_row, axis=1).reset_index(drop=True)


def collate_logit_lens(xs):
    transposed = zip(*xs)
    return tuple(list(col) for col in transposed)


class LogitLensDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        return (
            row["clean_masked"],
            row["masked_position_after_tokenization"],
            row["seq"],
            row["label"],
            row["name"],
        )

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_logit_lens)


def load_relevant_components(
    component_recurrence_csv: str,
    repeat_type: str,
    ratio_threshold: float,
) -> pd.DataFrame:
    df = pd.read_csv(component_recurrence_csv)
    ratio_col = f"{repeat_type}_ratio_in_graph"
    if ratio_col not in df.columns:
        raise ValueError(f"CSV must contain '{ratio_col}' column")
    filtered = df[df[ratio_col].fillna(0) >= ratio_threshold].copy()
    logging.info(f"Loaded {len(df)} total components, filtered to {len(filtered)} relevant components")
    return filtered


def extract_component_activation(
    cache: Dict[str, torch.Tensor],
    component_id: str,
    component_type: str,
    layer: int,
    head: Optional[int] = None,
) -> torch.Tensor:
    if component_type == "attention":
        if head is None:
            raise ValueError("Head is required for attention components")
        hook_name = f"blocks.{layer}.attn.hook_result"
        activation = cache[hook_name]
        B, L, H, D = activation.shape
        activation = activation[:, :, head, :]
        return activation
    elif component_type == "mlp":
        hook_name = f"blocks.{layer}.hook_mlp_out"
        activation = cache[hook_name]
        return activation
    elif component_type == "resid_post":
        hook_name = f"blocks.{layer}.hook_resid_post"
        activation = cache[hook_name]
        return activation
    else:
        raise ValueError(f"Unknown component type: {component_type}")


def compute_logit_lens_metrics(
    model: HookedESM3 | HookedESMC,
    tokenizer,
    component_activation: torch.Tensor,
    masked_positions: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Compute only top1 and correct_prob."""
    B, L, D = component_activation.shape
    effective_masked_positions = masked_positions

    if isinstance(model, HookedESM3):
        regression_head = model.unembed.output_heads.sequence_head
    elif isinstance(model, HookedESMC):
        regression_head = model.unembed
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    normalized = model.ln_final(component_activation)
    logits = regression_head(normalized).to(device)
    probs = get_probs_from_logits(logits, device, tokenizer, mask_logits_of_invalid_ids=True)
    masked_position_probs = get_masked_token_logits(probs, effective_masked_positions)
    correct_token_ids = labels.to(device)
    batch_indices = torch.arange(B, device=device)
    correct_token_probs = masked_position_probs[batch_indices, correct_token_ids]
    top_k = min(1, masked_position_probs.shape[-1])
    topk_probs, topk_ids = masked_position_probs.topk(top_k, dim=-1)
    in_top1 = (topk_ids[:, 0] == correct_token_ids).float()
    return {
        "top1": in_top1.mean().item(),
        "correct_prob": correct_token_probs.mean().item(),
    }


def process_batch(
    batch: Tuple,
    model: HookedESM3 | HookedESMC,
    tokenizer,
    relevant_components: pd.DataFrame,
    device: torch.device,
) -> List[Dict]:
    """Process a batch and compute top1/correct_prob for nodes only."""
    clean_masked, masked_positions, sequences, labels, names = batch
    B = len(sequences)
    clean_tokens, clean_attention_mask = tokenize_input(clean_masked, tokenizer)
    clean_tokens = clean_tokens.to(device)
    clean_attention_mask = clean_attention_mask.to(device)
    masked_positions_tensor = torch.tensor(masked_positions).to(device)
    assert masked_positions_tensor.shape == (B,), (
        f"Expected masked_positions shape ({B},), got {masked_positions_tensor.shape}"
    )
    tokenized_labels = tokenizer(
        labels, padding=False, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    if tokenized_labels.dim() == 1:
        tokenized_labels = tokenized_labels.unsqueeze(-1)
    assert tokenized_labels.dim() == 2 and tokenized_labels.shape[0] == B, (
        f"Expected tokenized_labels 2D [B, L], got {tokenized_labels.shape}"
    )

    _, cache = model.run_with_cache(
        sequence_tokens=clean_tokens,
        sequence_id=clean_attention_mask,
    )

    n_layers = len(model.blocks)
    component_rows = []

    for batch_idx in range(B):
        seq_name = names[batch_idx]
        single_masked_pos = masked_positions_tensor[batch_idx].unsqueeze(0)
        single_label = tokenized_labels[batch_idx]
        if single_label.dim() == 2:
            single_label = single_label.squeeze(-1)
        assert single_label.dim() == 1 and single_label.numel() >= 1, (
            f"Expected single_label 1D with >=1 token, got {single_label.shape}"
        )
        single_label = single_label[0:1]  # take first token (matches OLD: one token per label)

        for layer in range(n_layers):
            comp_activation = extract_component_activation(
                cache, f"resid_post_l{layer}", "resid_post", layer, None
            )
            comp_activation = comp_activation[batch_idx : batch_idx + 1]
            metrics = compute_logit_lens_metrics(
                model, tokenizer, comp_activation,
                single_masked_pos, single_label, device,
            )
            component_rows.append({
                "sequence": sequences[batch_idx],
                "name": seq_name,
                "component_id": f"resid_post_l{layer}",
                "component_type": "resid_post",
                "layer": layer,
                "head": None,
                "top1": metrics["top1"],
                "correct_prob": metrics["correct_prob"],
            })

        for _, comp_row in relevant_components.iterrows():
            component_id = comp_row["component_id"]
            if component_id == "input":
                logging.debug("Skipping input component %s", component_id)
                continue
            component_type = comp_row["component_type"]
            layer = int(comp_row["layer"])
            head = int(comp_row["head"]) if component_type == "attention" and not pd.isna(comp_row.get("head")) else None
            if component_type == "attention" and head is None:
                raise ValueError(f"Head required for attention component {component_id}")
            comp_activation = extract_component_activation(
                cache, component_id, component_type, layer, head
            )
            comp_activation = comp_activation[batch_idx : batch_idx + 1]
            metrics = compute_logit_lens_metrics(
                model, tokenizer, comp_activation,
                single_masked_pos, single_label, device,
            )
            component_rows.append({
                "sequence": sequences[batch_idx],
                "name": seq_name,
                "component_id": component_id,
                "component_type": component_type,
                "layer": layer,
                "head": head,
                "top1": metrics["top1"],
                "correct_prob": metrics["correct_prob"],
            })

    return component_rows


def run_logit_lens(
    dataset_csv: str,
    component_recurrence_csv: str,
    out_csv_path: str,
    model_type: str,
    repeat_type: str,
    n_samples: int = 5000,
    random_state: int = 42,
    batch_size: int = 1,
    ratio_threshold: float = 0.8,
) -> None:
    """Run logit lens computation for nodes. Saves CSV to out_csv_path."""
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    log_path = os.path.join(os.path.dirname(out_csv_path) or ".", "logit_lens.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")],
    )

    device = get_device()
    model = load_model(
        model_type=model_type,
        device=device,
        use_transformer_lens_model=True,
        cache_attention_activations=True,
        cache_mlp_activations=True,
        output_type="sequence",
        cache_attn_pattern=False,
        split_qkv_input=False,
    )
    tokenizer = load_tokenizer_by_model_type(model_type)

    df = pd.read_csv(dataset_csv)
    dataset = create_logit_lens_dataset_pandas(df, n_samples, random_state, tokenizer)
    torch_dataset = LogitLensDataset(dataset)
    dataloader = torch_dataset.to_dataloader(batch_size=batch_size)

    relevant_components = load_relevant_components(
        component_recurrence_csv, repeat_type, ratio_threshold
    )
    if len(relevant_components) == 0:
        raise ValueError("No relevant components found in recurrence CSV")

    all_rows = []
    for batch in tqdm(dataloader, desc="Processing batches"):
        rows = process_batch(batch, model, tokenizer, relevant_components, device)
        all_rows.extend(rows)

    if all_rows:
        results_df = pd.DataFrame(all_rows)
        results_df.to_csv(out_csv_path, index=False)
        logging.info(f"Saved {len(results_df)} rows to {out_csv_path}")
    else:
        logging.warning("No results to save")
