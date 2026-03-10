
import argparse
import ast
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from transformer_lens import HookedESM3, HookedESMC, HookedTransformerConfig

from plms_repeats_circuits.EAP.graph import Graph, GraphType, AttentionNode
from plms_repeats_circuits.utils.model_utils import get_device, tokenize_input, mask_protein
from plms_repeats_circuits.utils.esm_utils import (
    load_model,
    load_tokenizer_by_model_type,
    get_probs_from_logits,
    get_vocab_size_if_mask_invalid_ids,
)
from plms_repeats_circuits.utils.esm_utils import replace_short_mask_with_mask_token
from plms_repeats_circuits.utils.protein_similiarity_utils import analyze_repeat_positions


def _safe_literal_eval(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x


def create_attention_feature_dataset(
    df: pd.DataFrame,
    tokenizer,
) -> pd.DataFrame:
    """
    Build dataset for attention head feature extraction.

    Expects circuit_discovery CSV with: cluster_id, rep_id, repeat_key, seq,
    masked_position (or masked_poistion), corrupted_sequence, repeat_locations,
    repeat_alignments (optional).

    Returns DataFrame with one row per example.
    """
    df = df.copy()
    df["repeat_locations"] = df["repeat_locations"].apply(_safe_literal_eval)
    if "repeat_alignments" not in df.columns:
        df["repeat_alignments"] = [[] for _ in range(len(df))]
    else:
        df["repeat_alignments"] = df["repeat_alignments"].apply(_safe_literal_eval)

    pos_col = "masked_position" if "masked_position" in df.columns else "masked_poistion"
    if pos_col not in df.columns:
        raise ValueError(f"Dataset must have 'masked_position' or 'masked_poistion'")

    def process_row(row):
        clean = row["seq"]
        masked_pos = int(row[pos_col])
        name = f"{row['cluster_id']}_{row['rep_id']}_{row['repeat_key']}"
        clean_masked = mask_protein(clean, masked_pos, tokenizer)
        corrupted = row["corrupted_sequence"]
        corrupted_masked = mask_protein(corrupted, masked_pos, tokenizer)
        corrupted_masked = replace_short_mask_with_mask_token(corrupted_masked, tokenizer)
        return pd.Series(
            {
                "clean_masked": clean_masked,
                "masked_position_after_tokenization": masked_pos + 1,
                "repeat_locations": row["repeat_locations"],
                "repeat_alignments": row["repeat_alignments"],
                "seq": row["seq"],
                "label": row["seq"][masked_pos],
                "name": name,
                "corrupted_masked": corrupted_masked,
            }
        )

    return df.apply(process_row, axis=1).reset_index(drop=True)


def collate_interpret(xs):
    transposed = zip(*xs)
    return tuple(list(col) for col in transposed)


class AttentionFeatureDataset(Dataset):
    """PyTorch Dataset for attention head feature extraction."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return (
            row["clean_masked"],
            row["masked_position_after_tokenization"],
            row["repeat_locations"],
            row["repeat_alignments"],
            row["seq"],
            row["label"],
            row["name"],
            row["corrupted_masked"],
        )

    def to_dataloader(self, batch_size: int):
        from torch.utils.data import DataLoader

        return DataLoader(
            self, batch_size=batch_size, collate_fn=collate_interpret
        )



STANDARD_AA_LIST = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
        ]

class AttnFeatureCollector:
    def __init__(self, graph: Graph, model: HookedESM3| HookedESMC):
        self.g:Graph = graph
        self.model:HookedESM3| HookedESMC = model
        self.relu = nn.ReLU()
        model_name = None
        if isinstance(self.model, HookedESM3):
            model_name = "esm3"
        elif isinstance(self.model, HookedESMC):
            model_name = "esm-c"
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")
        from plms_repeats_circuits.utils.esm_utils import load_tokenizer_by_model_type
        self.tokenizer = load_tokenizer_by_model_type(model_name)
        self.aa_to_id = {aa: self.tokenizer.convert_tokens_to_ids(aa) for aa in STANDARD_AA_LIST}
        self.id_to_aa = {v: k for k, v in self.aa_to_id.items()}

    def verify_self_attn_dims(self, attn_pattern: torch.Tensor):
            if attn_pattern.dim() != 4:
                raise ValueError(f"Expected 4D tensor [B, H, L, L], but got shape {attn_pattern.shape}")
            
            B, H, Q, K = attn_pattern.shape
            if Q != K:
                raise ValueError(f"Expected square attention matrices, but got shape {Q} x {K}")

    def verify_self_attn_out_or_in(self, attn_result_by_head: torch.Tensor):
        cfg: HookedTransformerConfig = self.model.cfg
        n_heads = cfg.n_heads
        d_model = cfg.d_model

        if attn_result_by_head.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, H, L, D], but got shape {attn_result_by_head.shape}")

        B, L, H, D = attn_result_by_head.shape

        if H != n_heads:
            raise ValueError(f"Expected {n_heads} heads, but got {H}")
        if D != d_model:
            raise ValueError(f"Expected head out dim to match residual {d_model}, but got {D}")

    def _get_unique_tokens_per_sequence(self, tokenized_input: torch.Tensor) -> list[torch.Tensor]:
        """
        For each sequence in the batch, return sorted unique token IDs.

        Args:
            tokenized_input: Tensor of shape [B, L]

        Returns:
            List of B tensors, each of shape [N_i], sorted unique token IDs per sequence.
        """
        if tokenized_input.dim() != 2:
            raise ValueError(f"Expected tokenized input of shape [B, L], got {tokenized_input.shape}")

        pad_token_id = self.tokenizer.pad_token_id
        unique_per_sequence = []

        for seq in tokenized_input:
            unique_tokens = torch.unique(seq)
            sorted_tokens = torch.sort(unique_tokens).values
            unique_per_sequence.append(sorted_tokens)

        return unique_per_sequence

    def _compute_sequence_probs(self, attn_result_by_head: torch.Tensor, mean_normalize=True, mask_invalid_ids=True):
        org_dim = attn_result_by_head.dim()

        if org_dim == 3:
            cfg: HookedTransformerConfig = self.model.cfg
            n_heads = cfg.n_heads

            if attn_result_by_head.shape[1] == n_heads:
                attn_result_by_head = attn_result_by_head.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected shape for attn_result_by_head: {attn_result_by_head.shape}")

        final_layer_norm = self.model.ln_final
        if isinstance(self.model, HookedESM3):
            sequence_regression_head = self.model.unembed.output_heads.sequence_head
        elif isinstance(self.model, HookedESMC):
            sequence_regression_head = self.model.unembed
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

        normalized = final_layer_norm(attn_result_by_head)
        probs = sequence_regression_head(normalized)
        device = get_device()
        probs = probs.to(device)
        probs = get_probs_from_logits(probs, device, self.tokenizer, mask_logits_of_invalid_ids=mask_invalid_ids)
        if mean_normalize:
            probs = probs - torch.mean(probs, dim=-1, keepdim=True)
            probs = self.relu(probs)

        if org_dim == 3:
            probs = probs.squeeze(1)
            if probs.dim() != 3:
                raise RuntimeError("Unexpected logits shape after squeeze")

        return probs

    def compute_copy_score(
        self, 
        attn_result_by_head: torch.Tensor,
        attn_pattern: torch.Tensor,
        masked_positions: torch.Tensor,
        tokenized_input: torch.Tensor, 
    ):
        self.verify_self_attn_out_or_in(attn_result_by_head)
        self.verify_self_attn_dims(attn_pattern)

        B, L, H, D = attn_result_by_head.shape

        if masked_positions.dim() != 1 or masked_positions.shape[0] != B:
            raise ValueError(f"Expected masked token positions to be a 1D tensor of shape [B], but got shape {masked_positions.shape}")

        B2, H2, Q2, K2 = attn_pattern.shape
        if B2 != B or H2 != H or Q2 != L or K2 != L:
            raise ValueError(
                f"Shape mismatch between attention result and pattern: "
                f"attn_result [B={B}, L={L}, H={H}, D={D}] vs attn_pattern [B={B2}, H={H2}, Q={Q2}, K={K2}]"
            )

        if tokenized_input.dim() != 2 or tokenized_input.shape[0] != B or tokenized_input.shape[1] != L:
            raise ValueError(f"Expected tokenized input to be a 2D tensor of shape [{B}, {L}], but got {tokenized_input.shape}")
        
        device = attn_result_by_head.device
        attn_pattern = attn_pattern.to(device)
        masked_positions = masked_positions.to(device)
        tokenized_input = tokenized_input.to(device)

        batch_indices = torch.arange(B, device=device)
        head_indices = torch.arange(H, device=device)

        masked_outputs = attn_result_by_head[batch_indices, masked_positions]
        masked_position_attn_pattern = attn_pattern[batch_indices, :, masked_positions]

        normalized_probs = self._compute_sequence_probs(attn_result_by_head=masked_outputs, mean_normalize=True)
        unique_tokens_per_sequence_list = self._get_unique_tokens_per_sequence(tokenized_input)

        hard_copy_scores = []
        for i in batch_indices:
            heads_normalized_probs = normalized_probs[i]
            unique_token_ids = unique_tokens_per_sequence_list[i]

            gathered_unique_tokens_probs = heads_normalized_probs[:, unique_token_ids]
            sum_unique_tokens_probs = torch.sum(gathered_unique_tokens_probs, dim=-1)

            if torch.allclose(sum_unique_tokens_probs, torch.zeros_like(sum_unique_tokens_probs)):
                raise RuntimeError("Total probs increase is close to 0")

            heads_attn_pattern_for_masked = masked_position_attn_pattern[i]
            argmax_index_positions = torch.argmax(heads_attn_pattern_for_masked, dim=-1)
            tokens = tokenized_input[i][argmax_index_positions]
            argmax_probs = heads_normalized_probs[head_indices, tokens]
            copy_scores = argmax_probs / sum_unique_tokens_probs
            hard_copy_scores.append(copy_scores)

        hard_copy_scores = torch.stack(hard_copy_scores, dim=0)
        return {"copy_score": hard_copy_scores}

    def compute_pattern_matching_score(
        self, 
        attn_pattern: torch.Tensor,
        repeat_locations_list: List[List[List[int]]],
        sequences_list:  List[str],
        alignments_list: List[List[str]]):
        
        self.verify_self_attn_dims(attn_pattern)
        B, H, Q, K = attn_pattern.shape
        if B!= len(sequences_list) or B!=len(alignments_list) or B!=len(repeat_locations_list):
            raise ValueError("mismatch")

        results = []

        for i, (seq, repeat_locations, alignments) in enumerate(zip(sequences_list, repeat_locations_list, alignments_list)):
            result_abs_pos_to_repeat_info, dict2= analyze_repeat_positions(
                protein_seq=seq,
                repeat_locations=repeat_locations,
                alignments=alignments
            )
            positions_list = []
            for j , (start, end) in enumerate(repeat_locations):
                for pos in range(start, end + 1):
                    aligned = result_abs_pos_to_repeat_info[j][pos].aligned_matching_absolute_position_in_sequence
                    if aligned is None:
                        raise RuntimeError("invalid case")
                    
                    positions_list.append((pos+1, aligned+1))
            
            attn_pattern_seq = attn_pattern[i]
        
            q_tensor = torch.tensor([q for q, _ in positions_list], device=attn_pattern.device)
            k_tensor = torch.tensor([k for _, k in positions_list], device=attn_pattern.device)

            attn_values = attn_pattern_seq[:, q_tensor, k_tensor]
            mean_attention_per_head = attn_values.mean(dim=1)
            results.append(mean_attention_per_head)

        return {"pattern_matching_score": torch.stack(results, dim=0)}

    def compute_position_score(self, 
        attn_pattern: torch.Tensor,
        real_token_mask: torch.Tensor, 
        padding_mask: torch.Tensor,
        normalization_method: str = None):
        """
        Returns position_score [B, H] (hard relative position std).
        attn_pattern: [B, H, Q, K]
        real_token_mask: [B, Q], True for real tokens, False for pad/special
        padding_mask: [B, Q], True for padding, False for valid
        normalization_method: "sequence_length" or "None"
        """
        self.verify_self_attn_dims(attn_pattern)
        B, H, Q, K = attn_pattern.shape
        if Q != K:
            raise ValueError("Expected square attention")

        rel_pos = torch.arange(K).unsqueeze(0) - torch.arange(Q).unsqueeze(1)
        rel_pos = rel_pos.to(attn_pattern.device).float().unsqueeze(0).unsqueeze(0)

        if normalization_method == "sequence_length":
            lengths = (~padding_mask).sum(dim=-1)
            rel_pos_expanded = rel_pos.expand(B, H, Q, K)
            lengths_expanded = lengths.view(B, 1, 1, 1).expand(B, H, Q, K)
            rel_pos = rel_pos_expanded / lengths_expanded

        argmax_k = attn_pattern.argmax(dim=-1)
        query_ids = torch.arange(Q, device=attn_pattern.device).view(1, 1, Q)
        hard_rel_pos = argmax_k - query_ids
        if normalization_method == "sequence_length":
            lengths = (~padding_mask).sum(dim=-1).to(attn_pattern.device).float()
            lengths_expanded = lengths.view(B, 1, 1).expand(B, H, Q)
            hard_rel_pos = hard_rel_pos / lengths_expanded

        mask = real_token_mask.unsqueeze(1)
        hard_rel_pos = hard_rel_pos.float().masked_fill(~mask, float('nan'))
        hard_mean_sq = torch.nanmean(hard_rel_pos ** 2, dim=-1)
        hard_mean = torch.nanmean(hard_rel_pos, dim=-1)
        hard_var = hard_mean_sq - hard_mean ** 2
        hard_var = torch.clamp(hard_var, min=0)
        hard_std = torch.sqrt(hard_var)

        return {"position_score": hard_std}

    def compute_attn_entropy(
        self,
        attn_pattern: torch.Tensor,
        real_token_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        to_normalize: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Mean normalized attention entropy per head, over real token queries.
        Returns dict with "attn_entropy" [B, H].
        """
        self.verify_self_attn_dims(attn_pattern)
        B, H, L, _ = attn_pattern.shape

        eps = 1e-9
        attn_for_entropy = attn_pattern.clamp(min=eps)
        entropy_per_query = -(attn_for_entropy * torch.log(attn_for_entropy)).sum(dim=-1)  # [B, H, L]

        if to_normalize:
            valid_key_counts = (~padding_mask).sum(dim=-1).clamp(min=1)  # [B]
            log_k = torch.log(valid_key_counts.float()).unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            entropy_per_query = entropy_per_query / log_k

        entropy_per_query_masked = entropy_per_query.masked_fill(
            ~real_token_mask.unsqueeze(1), float("nan")
        )
        mean_entropy = torch.nanmean(entropy_per_query_masked, dim=-1)  # [B, H]
        return {"attn_entropy": mean_entropy}

    def compute_contribution_to_residual_stream(
        self,
        head_out: torch.Tensor,  # [B, L, H, D]
        pre_layer_tensor: torch.Tensor,  # [B, L, D]
        post_layer_tensor: torch.Tensor,  # [B, L, D]
        repeat_locations_list: List,
        use_scaling_factor: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Contribution of each head's output to the residual stream at repeat positions.
        Returns dict with "contribution_to_residual_stream" [B, H].
        """
        self.verify_self_attn_out_or_in(head_out)
        B, L, H, D = head_out.shape
        device = head_out.device
        eps = 1e-12

        if use_scaling_factor:
            scaling_factor = getattr(self.model.cfg, "esm3_scaling_factor", 1.0)
            head_out = head_out / scaling_factor

        all_contribs = []
        for i, repeat_locations in enumerate(repeat_locations_list):
            positions_list = []
            for start, end in repeat_locations:
                positions_list.append(torch.arange(start, end + 1, device=device) + 1)
            positions = torch.cat(positions_list, dim=0)

            pre = pre_layer_tensor[i, positions]
            post = post_layer_tensor[i, positions]
            head_seq = head_out[i, positions]

            norm_head = torch.norm(head_seq, dim=-1)
            layer_diff = post - pre
            norm_layer = torch.norm(layer_diff, dim=-1)
            denom = norm_layer.unsqueeze(1).clamp(min=eps)
            contrib_ratio = norm_head / denom

            head_score = torch.mean(contrib_ratio, dim=0)
            all_contribs.append(head_score)

        return {"contribution_to_residual_stream": torch.stack(all_contribs, dim=0)}

    def compute_vocab_entropy(
        self,
        attn_result_by_head: torch.Tensor,
        masked_positions: torch.Tensor,
        to_normalize: bool = True,
        mask_invalid_ids: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Entropy of output probs at masked position in vocab space.
        Returns dict with "vocab_entropy" [B, H].
        """
        self.verify_self_attn_out_or_in(attn_result_by_head)
        B, L, H, D = attn_result_by_head.shape
        device = attn_result_by_head.device

        batch_idx = torch.arange(B, device=device)
        head_out = attn_result_by_head[batch_idx, masked_positions]  # [B, H, D]
        probs = self._compute_sequence_probs(
            head_out, mean_normalize=False, mask_invalid_ids=mask_invalid_ids
        )  # [B, H, vocab_size] after squeeze

        entropy_per_head = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)  # [B, H]

        if to_normalize:
            vocab_size = (
                get_vocab_size_if_mask_invalid_ids(self.tokenizer)
                if mask_invalid_ids
                else len(self.tokenizer.all_token_ids)
            )
            entropy_per_head = entropy_per_head / torch.log(
                torch.tensor(vocab_size, device=device, dtype=torch.float32)
            )

        return {"vocab_entropy": entropy_per_head}

    def create_special_tokens_mask(
        self, 
        tokenized_input: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int
    ) -> torch.Tensor:
        """
        Returns mask [B, L]: True for real tokens, False for BOS/EOS/PAD.
        """
        special_token_ids = torch.tensor([bos_token_id, eos_token_id, pad_token_id], device=tokenized_input.device)
        is_special = (tokenized_input.unsqueeze(-1) == special_token_ids).any(-1)
        real_token_mask = ~is_special
        return real_token_mask
    
    def create_only_padding_mask(self, tokenized_input: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """
        Returns mask [B, L]: True for padding tokens, False for real tokens.
        """
        return tokenized_input == pad_token_id

    def compute_attn_to_bos_eos(
        self,
        attn_pattern: torch.Tensor,
        tokenized_input: torch.Tensor,
        real_token_mask: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        ) -> dict[str, torch.Tensor]:
        """
        Returns attn_to_bos and attn_to_eos [B, H] each.
        """
        B, H, L, _ = attn_pattern.shape
        device = attn_pattern.device
    
        bos_positions = (tokenized_input == bos_token_id).float().argmax(dim=1)
        eos_positions = (tokenized_input == eos_token_id).float().argmax(dim=1)
    
        batch_idx = torch.arange(B, device=device).unsqueeze(1).unsqueeze(2).expand(B, H, L)
        head_idx  = torch.arange(H, device=device).unsqueeze(0).unsqueeze(2).expand(B, H, L)
        query_idx = torch.arange(L, device=device).unsqueeze(0).unsqueeze(0).expand(B, H, L)
        bos_key_idx = bos_positions.unsqueeze(1).unsqueeze(2).expand(B, H, L)
        eos_key_idx = eos_positions.unsqueeze(1).unsqueeze(2).expand(B, H, L)
    
        attn_to_bos = attn_pattern[batch_idx, head_idx, query_idx, bos_key_idx]
        attn_to_eos = attn_pattern[batch_idx, head_idx, query_idx, eos_key_idx]
    
        attn_to_bos = attn_to_bos.masked_fill(~real_token_mask.unsqueeze(1), float('nan'))
        attn_to_eos = attn_to_eos.masked_fill(~real_token_mask.unsqueeze(1), float('nan'))
    
        mean_attn_to_bos = torch.nanmean(attn_to_bos, dim=2)
        mean_attn_to_eos = torch.nanmean(attn_to_eos, dim=2)
    
        return {
            "attn_to_bos": mean_attn_to_bos,
            "attn_to_eos": mean_attn_to_eos
        }


    def compute_aa_bias(
        self,
        attn_pattern: torch.Tensor,
        tokenized_input: torch.Tensor,
        real_token_mask: torch.Tensor,
        tokens_aggregation_method: str = "mean",
    ) -> dict:
        """
        Returns dict {aa_bias_{AA}: [B, H]} (NaN if AA absent in sequence).
        """
        if tokens_aggregation_method != "mean":
            raise ValueError(f"Only tokens_aggregation_method='mean' is supported, got {tokens_aggregation_method}")

        aa_to_id = {aa: self.tokenizer.convert_tokens_to_ids(aa) for aa in STANDARD_AA_LIST}
        aa_ids = torch.tensor([aa_to_id[aa] for aa in STANDARD_AA_LIST], device=tokenized_input.device)
        B, H, L, _ = attn_pattern.shape

        key_aa_mask = (tokenized_input.unsqueeze(1) == aa_ids.unsqueeze(0).unsqueeze(2))
        attn_to_aa = torch.einsum('bhql,bal->bhqa', attn_pattern, key_aa_mask.float())

        real_token_mask_q = real_token_mask.unsqueeze(1).unsqueeze(-1)
        attn_to_aa = attn_to_aa.masked_fill(~real_token_mask_q, float('nan'))

        bias = torch.nanmean(attn_to_aa, dim=2)

        aa_present = key_aa_mask.any(dim=-1)
        aa_present_expand = aa_present.unsqueeze(1).expand(-1, H, -1)
        bias = bias.masked_fill(~aa_present_expand, float('nan'))

        result = {}
        for i, aa in enumerate(STANDARD_AA_LIST):
            result[f"aa_bias_{aa}"] = bias[..., i]
        return result


    def compute_repeat_focus(
        self,
        attn_pattern: torch.Tensor,
        repeat_locations_list: List[List[List[int]]],
        real_token_mask: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        log_ratio_eps: float = 1e-9,
    ) -> dict[str, torch.Tensor]:
        """
        Returns repeat_focus [B, H] (max peaks log ratio, repeat vs non-repeat).
        """
        self.verify_self_attn_dims(attn_pattern)
        B, H, Q, K = attn_pattern.shape

        if B != len(repeat_locations_list):
            raise ValueError("Mismatch between batch size and repeat_locations_list length")

        log_ratios = []

        for b, repeat_locations in enumerate(repeat_locations_list):
            repeat_positions = []
            for start, end in repeat_locations:
                repeat_positions.extend(range(start, end + 1))
            if len(repeat_positions) == 0:
                raise ValueError("No repeat positions found")

            repeat_positions = torch.tensor(repeat_positions, device=attn_pattern.device) + 1

            attn_b = attn_pattern[b]
            real_query_mask_b = real_token_mask[b]

            valid_non_repeat_mask = torch.ones(Q, dtype=torch.bool, device=attn_pattern.device)
            valid_non_repeat_mask[repeat_positions] = False
            valid_non_repeat_mask &= real_query_mask_b

            key_mask = real_query_mask_b
            attn_b_masked = attn_b.masked_fill(
                ~key_mask.unsqueeze(0).unsqueeze(0),
                float('-inf')
            )
            max_values = attn_b_masked.max(dim=-1).values

            max_repeat = max_values[:, repeat_positions].mean(dim=-1)
            max_non_repeat = max_values[:, valid_non_repeat_mask].mean(dim=-1)
            log_ratio = torch.log(max_repeat + log_ratio_eps) - torch.log(max_non_repeat + log_ratio_eps)
            log_ratios.append(log_ratio)

        return {"repeat_focus": torch.stack(log_ratios, dim=0)}

    def compute_difference_between_clean_and_corrupted_head_patterns(
        self,
        attn_pattern_clean: torch.Tensor,
        attn_pattern_corrupted: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        KL divergence from clean to corrupted attention patterns.
        Returns dict with "diff_corrupted_clean" [B, H].
        """
        self.verify_self_attn_dims(attn_pattern_clean)
        self.verify_self_attn_dims(attn_pattern_corrupted)

        if attn_pattern_clean.shape != attn_pattern_corrupted.shape:
            raise ValueError(
                f"Shape mismatch: {attn_pattern_clean.shape} vs {attn_pattern_corrupted.shape}"
            )

        eps = 1e-12
        P = attn_pattern_clean.clamp(min=eps)
        Q = attn_pattern_corrupted.clamp(min=eps)
        kl_per_q = (P * (P.log() - Q.log())).sum(dim=-1)
        kl_per_head = kl_per_q.mean(dim=-1)
        return {"diff_corrupted_clean": kl_per_head}

    def _get_attn_heads_in_graph_indicies(self, layer):
        attn_heads_in_layer = [node for node in self.g.nodes.values() if node.layer == layer and node.in_graph and isinstance(node, AttentionNode)]
        attn_heads_in_layer_indices = torch.tensor([node.head for node in attn_heads_in_layer])
        return attn_heads_in_layer , attn_heads_in_layer_indices


    def compute_all_scores_for_nodes(
    self, clean_cache, 
    clean_tokens,
    masked_positions,
    repeat_locations,
    repeat_alignments,
    clean_sequences,
    tokenized_labels,
    corrupted_cache,
    corrupted_tokens,
    names):

        device = get_device()
        all_rows = []

        for layer in range(self.model.cfg.n_layers):
            attn_heads_in_layer, attn_heads_in_layer_indices = self._get_attn_heads_in_graph_indicies(layer)
            if len(attn_heads_in_layer) == 0:
                continue

            attn_result_by_head = clean_cache[f'blocks.{layer}.attn.hook_result'].detach().to(device)
            attn_pattern = clean_cache[f'blocks.{layer}.attn.hook_pattern'].detach().to(device)

            attn_pattern_corrupted = corrupted_cache[f'blocks.{layer}.attn.hook_pattern'].detach().to(device)
            attn_result_by_head_corrupted = corrupted_cache[f'blocks.{layer}.attn.hook_result'].detach().to(device)

            mask = self.create_special_tokens_mask(
                clean_tokens,
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id
            ).to(device)

            only_padding_mask = self.create_only_padding_mask(clean_tokens, self.tokenizer.pad_token_id).to(device)

            copy_scores = self.compute_copy_score(
                attn_result_by_head=attn_result_by_head,
                attn_pattern=attn_pattern,
                masked_positions=masked_positions,
                tokenized_input=clean_tokens,
            )
            pattern_matching = self.compute_pattern_matching_score(
                attn_pattern=attn_pattern,
                repeat_locations_list=repeat_locations,
                sequences_list=clean_sequences,
                alignments_list=repeat_alignments,
            )
            relpos_normalized = self.compute_position_score(attn_pattern=attn_pattern, real_token_mask=mask, padding_mask=only_padding_mask, normalization_method="sequence_length")

            attn_entropy = self.compute_attn_entropy(
                attn_pattern=attn_pattern,
                real_token_mask=mask,
                padding_mask=only_padding_mask,
                to_normalize=True,
            )

            contribution_to_residual = self.compute_contribution_to_residual_stream(
                head_out=attn_result_by_head,
                pre_layer_tensor=clean_cache[f"blocks.{layer}.hook_resid_pre"].detach().to(device),
                post_layer_tensor=clean_cache[f"blocks.{layer}.hook_resid_mid"].detach().to(device),
                repeat_locations_list=repeat_locations,
                use_scaling_factor=True,
            )

            vocab_entropy = self.compute_vocab_entropy(
                attn_result_by_head=attn_result_by_head,
                masked_positions=masked_positions,
                to_normalize=True,
                mask_invalid_ids=True,
            )

            bos_eos = self.compute_attn_to_bos_eos(
                attn_pattern=attn_pattern,
                tokenized_input=clean_tokens,
                real_token_mask=mask,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            per_aa_bias = self.compute_aa_bias(
                attn_pattern=attn_pattern,
                tokenized_input=clean_tokens,
                real_token_mask=mask,
                tokens_aggregation_method="mean",
            )

            repeat_focus = self.compute_repeat_focus(
                attn_pattern=attn_pattern,
                repeat_locations_list=repeat_locations,
                real_token_mask=mask,
                padding_mask=only_padding_mask,
                log_ratio_eps=1e-9,
            )

            diff_corrupted_clean = self.compute_difference_between_clean_and_corrupted_head_patterns(
                attn_pattern_clean=attn_pattern,
                attn_pattern_corrupted=attn_pattern_corrupted,
            )

            for node, head_idx in zip(attn_heads_in_layer, attn_heads_in_layer_indices):
                for b in range(attn_pattern.shape[0]):
                    row = {
                        "seq_name": names[b],
                        "layer": layer,
                        "head": int(head_idx),
                        "node_name": getattr(node, "name", None),
                        "copy_score": float(copy_scores["copy_score"][b, head_idx].item()),
                        "pattern_matching_score": float(pattern_matching["pattern_matching_score"][b, head_idx].item()),
                        "attn_to_bos": float(bos_eos["attn_to_bos"][b, head_idx].item()),
                        "attn_to_eos": float(bos_eos["attn_to_eos"][b, head_idx].item()),
                        "position_score": float(relpos_normalized["position_score"][b, head_idx].item()),
                        "repeat_focus": float(repeat_focus["repeat_focus"][b, head_idx].item()),
                        "diff_corrupted_clean": float(diff_corrupted_clean["diff_corrupted_clean"][b, head_idx].item()),
                        "attn_entropy": float(attn_entropy["attn_entropy"][b, head_idx].item()),
                        "vocab_entropy": float(vocab_entropy["vocab_entropy"][b, head_idx].item()),
                        "contribution_to_residual_stream_mean_tokens": float(contribution_to_residual["contribution_to_residual_stream"][b, head_idx].item()),
                    }
                    for col, bias in per_aa_bias.items():
                        row[col] = float(bias[b, head_idx].item()) if torch.isfinite(bias[b, head_idx]) else float('nan')

                    all_rows.append(row)

        return all_rows




def run_experiment(
    dataloader,
    tokenizer,
    feature_collector: AttnFeatureCollector,
    model,
    device: torch.device,
    out_csv_path: str,
) -> None:
    output_dir = os.path.dirname(out_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    first_batch = True
    total_rows = 0

    for batch_num, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
        (
            clean_masked,
            masked_positions,
            repeat_locations,
            repeat_alignments,
            sequences,
            labels,
            names,
            corrupted_masked,
        ) = batch

        clean_tokens, clean_attention_mask = tokenize_input(clean_masked, tokenizer)
        clean_tokens = clean_tokens.to(device)
        clean_attention_mask = clean_attention_mask.to(device)
        masked_positions_tensor = torch.tensor(masked_positions, dtype=torch.long).to(device)
        tokenized_labels = (
            tokenizer(labels, padding=False, return_tensors="pt", add_special_tokens=False)
            .input_ids.to(device)
        )
        _, clean_cache = model.run_with_cache(
            sequence_tokens=clean_tokens,
            sequence_id=clean_attention_mask,
        )
        corrupted_tokens, corrupted_attention_mask = tokenize_input(corrupted_masked, tokenizer)
        corrupted_tokens = corrupted_tokens.to(device)
        corrupted_attention_mask = corrupted_attention_mask.to(device)

        _, corrupted_cache = model.run_with_cache(
            sequence_tokens=corrupted_tokens,
            sequence_id=corrupted_attention_mask,
        )

        rows = feature_collector.compute_all_scores_for_nodes(
            clean_cache,
            clean_tokens,
            masked_positions_tensor,
            repeat_locations,
            repeat_alignments,
            sequences,
            tokenized_labels.squeeze(-1),
            corrupted_cache,
            corrupted_tokens,
            names,
        )

        df = pd.DataFrame(rows)

        if first_batch:
            df.to_csv(out_csv_path, index=False, mode="w")
            first_batch = False
            logging.info(f"Wrote {len(df)} rows (header included) to {out_csv_path} (batch {batch_num + 1})")
        else:
            df.to_csv(out_csv_path, index=False, header=False, mode="a")
            logging.info(f"Appended {len(df)} rows to {out_csv_path} (batch {batch_num + 1})")

        total_rows += len(df)

    logging.info(f"Extraction complete. Total rows written: {total_rows}")


def assign_nodes_list_in_graph(graph: Graph, nodes_list: List[str]) -> None:
    for node in graph.nodes.values():
        node.in_graph = node.name in nodes_list

    n_attn = graph.count_attention_nodes(filter_by_in_graph=True)
    if n_attn != len(nodes_list):
        raise ValueError(f"Number of nodes in graph {n_attn} does not match nodes list {len(nodes_list)}")
    logging.info(f"Number of nodes in graph {n_attn} matches nodes list {len(nodes_list)}")


def collect_features(
    dataset_csv: str,
    out_csv_path: str,
    model_type: str,
    component_recurrence_csv: str,
    batch_size: int,
    repeat_type: str,
    component_recurrence_threshold: float = 0.8,
) -> None:
    """Run feature collection. Uses all rows in dataset (no n_samples, n_splits)."""
    device = get_device()
    model = load_model(
        model_type=model_type,
        device=device,
        use_transformer_lens_model=True,
        cache_attention_activations=True,
        cache_mlp_activations=False,
        output_type="sequence",
        cache_attn_pattern=True,
        split_qkv_input=False,
    )
    tokenizer = load_tokenizer_by_model_type(model_type)

    df = pd.read_csv(dataset_csv)
    processed_df = create_attention_feature_dataset(df, tokenizer)
    torch_dataset = AttentionFeatureDataset(processed_df)
    dataloader = torch_dataset.to_dataloader(batch_size=batch_size)

    comp_df = pd.read_csv(component_recurrence_csv)
    ratio_col = f"{repeat_type}_ratio_in_graph"
    if ratio_col not in comp_df.columns:
        raise ValueError(f"Component recurrence CSV must have column '{ratio_col}'")
    comp_df = comp_df[
        (comp_df[ratio_col] >= component_recurrence_threshold)
        & (comp_df.component_type == "attention")
    ]
    nodes_in_graph = comp_df.component_id.tolist()

    g = Graph.from_model(model_or_config=model, graph_type=GraphType.Nodes)
    assign_nodes_list_in_graph(g, nodes_in_graph)

    feature_collector = AttnFeatureCollector(graph=g, model=model)

    run_experiment(
        dataloader=dataloader,
        tokenizer=tokenizer,
        feature_collector=feature_collector,
        model=model,
        device=device,
        out_csv_path=out_csv_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=str, required=True, help="Path to circuit_discovery dataset CSV")
    parser.add_argument("--out_csv_path", type=str, required=True, help="Path to output features CSV")
    parser.add_argument("--model_type", type=str, required=True, choices=["esm3", "esm-c"], help="Model type")
    parser.add_argument(
        "--component_recurrence_csv",
        type=str,
        required=True,
        help="Path to component recurrence CSV (nodes_recurrence_{repeat_type}.csv)",
    )
    parser.add_argument("--repeat_type", type=str, required=True, help="Repeat type (identical, approximate, synthetic)")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument(
        "--component_recurrence_threshold",
        type=float,
        default=0.8,
        help="Minimum ratio_in_graph for a component to be included (default: 0.8)",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed (unused, kept for compatibility)")
    args = parser.parse_args()

    if not 0 <= args.component_recurrence_threshold <= 1:
        raise ValueError("component_recurrence_threshold must be between 0 and 1")

    log_dir = os.path.dirname(args.out_csv_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "collect_features.log") if log_dir else "collect_features.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="a")],
    )
    logging.info("Arguments:\n" + json.dumps(vars(args), indent=2))

    if not os.path.exists(args.component_recurrence_csv):
        raise ValueError(f"component_recurrence_csv not found: {args.component_recurrence_csv}")

    collect_features(
        dataset_csv=args.dataset_csv,
        out_csv_path=args.out_csv_path,
        model_type=args.model_type,
        component_recurrence_csv=args.component_recurrence_csv,
        batch_size=args.batch_size,
        repeat_type=args.repeat_type,
        component_recurrence_threshold=args.component_recurrence_threshold,
    )


if __name__ == "__main__":
    main()
