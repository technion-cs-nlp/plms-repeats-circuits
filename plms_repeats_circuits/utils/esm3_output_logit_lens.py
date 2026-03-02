from transformer_lens import HookedESM3, SupportedESM3Config, HookedTransformerConfig
from protein_circuits.EAP.graph import Graph, GraphType, Node, Edge, AttentionNode, MLPNode
import torch
from protein_circuits.utils.model_utils import mask_protein
from protein_circuits.utils.esm3_utils import get_probs_from_logits, replace_short_mask_with_mask_token
import pandas as pd
import ast
from esm.tokenization import get_esm3_model_tokenizers
from dataclasses import dataclass
from typing import Optional
import einops
from esm.utils.sampling import sample_function_logits, sample_residue_annotation_logits
from esm.utils.function.encode_decode import decode_function_tokens, decode_residue_annotation_tokens
from esm.pretrained import ESM3_function_decoder_v0
from esm.utils.constants import esm3 as C
SUPPORTED_TRACKS = {"sequence", "structure", "secondary_structure", "sasa", "function"}
MAX_RESIDUE_ANNOTATIONS = C.MAX_RESIDUE_ANNOTATIONS

# ======================================================
# Dataclasses
# ======================================================
@dataclass
class LogitsLensResults:
    decoded_vocab_tokens: list[str]
    logits: torch.Tensor
    probabilities: torch.Tensor
    normalized_entropy_per_position: torch.Tensor  # [B, L] or [B, L, H]
    argmax_token_ids: torch.Tensor  # [B, L] or [B, L, H]
    argmax_token_logits: torch.Tensor  # [B, L] or [B, L, H]
    argmax_token_probabilities: torch.Tensor  # [B, L] or [B, L, H]
    argmax_token_texts: list  # decoded tokens preserving sequence/position/head structure
    topk_token_ids: torch.Tensor  # [..., top_k]
    topk_token_probabilities: torch.Tensor  # [..., top_k]
    topk_token_texts: list  # decoded tokens for top-k predictions


@dataclass
class ExtendedSequenceLensResults(LogitsLensResults):
    masked_position_logits: Optional[torch.Tensor]
    masked_position_probabilities: Optional[torch.Tensor]
    correct_token_logits: Optional[torch.Tensor]
    correct_token_probabilities: Optional[torch.Tensor]


@dataclass
class ResidueAnnotationLensResults:
    decoded_vocab_tokens: list[str]
    logits: torch.Tensor  # [B, L, MAX_RES] or [B, L, H, MAX_RES]
    probabilities: torch.Tensor  # [B, L, MAX_RES] or [B, L, H, MAX_RES]
    entropy_per_annotation: torch.Tensor  # [B, L, MAX_RES] or [B, L, H, MAX_RES]
    entropy_per_token_mean: torch.Tensor  # [B, L] or [B, L, H]
    normalized_entropy_per_annotation: torch.Tensor  # [B, L, MAX_RES] or [B, L, H, MAX_RES]
    normalized_entropy_per_token_mean: torch.Tensor  # [B, L] or [B, L, H]
    count_above_threshold: torch.Tensor  # [B, L] or [B, L, H]
    topk_annotation_ids: torch.Tensor  # [B, L, K] or [B, L, H, K] (K = MAX_RES)
    topk_annotation_log_probs: torch.Tensor  # same shape as ids
    topk_annotation_probabilities: torch.Tensor  # same shape as ids
    topk_annotation_texts: list  # decoded annotation labels per position


# ======================================================
# Helpers
# ======================================================
def _decode_ids_with_fallback(tokenizer, token_ids, track_name: str = "sequence"):
    """Convert token ids to strings with graceful fallback across tokenizer capabilities."""
    if track_name == "structure":
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return [f"token_{token_id}" for token_id in token_ids]

    if track_name == "function":
        vocab_to_index = tokenizer.vocab_to_index
        # Properly invert the mapping: index → vocab string
        index_to_vocab = {v: k for k, v in vocab_to_index.items()}

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()

        return [index_to_vocab.get(token_id, f"token_{token_id}") for token_id in token_ids]

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()

    if isinstance(token_ids, int):
        token_ids_list = [token_ids]
    else:
        token_ids_list = list(token_ids)

    decoded_tokens = []
    has_decode = hasattr(tokenizer, "decode")

    for token_id in token_ids_list:
        token_str = None

        if has_decode:
            try:
                token_str = tokenizer.decode([token_id])
            except TypeError:
                token_str = tokenizer.decode(token_id)
            except Exception:
                token_str = None

        if token_str is None or token_str == "":
            token_str = f"token_{token_id}"

        decoded_tokens.append(token_str)

    return decoded_tokens


def _residue_annotation_ids_to_labels(tokenizer, annotation_ids):
    if isinstance(annotation_ids, torch.Tensor):
        annotation_ids = annotation_ids.detach().cpu().tolist()
    if isinstance(annotation_ids, int):
        annotation_ids_list = [annotation_ids]
    else:
        annotation_ids_list = list(annotation_ids)

    vocabulary = getattr(tokenizer, "vocabulary", [])
    vocab_size = len(vocabulary)

    return [
        vocabulary[idx] if idx < vocab_size else f"token_{idx}"
        for idx in annotation_ids_list
    ]


# ======================================================
# Core Functions
# ======================================================
def _compute_track_probs_and_logits(
    model,
    tokenizer,
    regression_head,
    component_out: torch.Tensor,
    device: torch.device,
    track_name: str = "sequence",
):
    """
    Compute per-token logits and probabilities for either attention or MLP components.
    """
    final_layer_norm = model.ln_final
    normalized = final_layer_norm(component_out)
    logits = regression_head(normalized).to(device)
    if track_name == "function":
        logits = einops.rearrange(logits, "... (k v) -> ... k v", k=8)
    probs = get_probs_from_logits(logits, device, tokenizer, mask_logits_of_invalid_ids=False)
    return probs, logits


def compute_track_lens_results(
    model: HookedESM3,
    tokenizers,
    component_out: torch.Tensor,
    device: torch.device,
    masked_positions: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    track_name: str = "sequence",
    top_k: int = 5,
):
    """
    Compute per-sequence lens metrics including entropy, argmax decoding,
    and correct-token probabilities for both MLP and attention components.
    """

    if track_name not in SUPPORTED_TRACKS:
        raise ValueError(
            f"Unsupported track '{track_name}'. Supported tracks: {sorted(SUPPORTED_TRACKS)}"
        )

    # ---------------------- shape checks ----------------------
    if component_out.dim() == 3:
        B, L, D = component_out.shape
    elif component_out.dim() == 4:
        B, L, H, D = component_out.shape
    else:
        raise ValueError(f"Expected component_out to have dim 3 or 4, but got {component_out.dim()}")

    if (masked_positions is None) != (labels is None):
        raise ValueError("masked_positions and labels must either both be provided or both be None")

    if masked_positions is not None and masked_positions.dim() != 1:
        raise ValueError(
            f"Expected masked_positions to be 1D (one mask per example), but got {masked_positions.shape}"
        )
    if labels is not None and labels.dim() != 1:
        raise ValueError(f"Expected labels to be 1D (one target id per example), but got {labels.shape}")

    if masked_positions is not None:
        assert B == masked_positions.shape[0] == labels.shape[0], "Batch size mismatch"

    try:
        tokenizer = getattr(tokenizers, track_name)
    except AttributeError as exc:
        raise ValueError(f"Tokenizers object is missing attribute '{track_name}'") from exc

    head_attr_name = "ss8_head" if track_name == "secondary_structure" else f"{track_name}_head"

    try:
        regression_head = getattr(model.unembed.output_heads, head_attr_name)
    except AttributeError as exc:
        raise ValueError(
            f"Model has no regression head '{head_attr_name}' for track '{track_name}'"
        ) from exc

    # ---------------------- compute probs/logits ----------------------
    idx = torch.arange(B, device=device)
    probs, logits = _compute_track_probs_and_logits(model, tokenizer, regression_head, component_out, device, track_name)

    # ---------------------- gather masked positions ----------------------
    masked_position_logits: Optional[torch.Tensor] = None
    masked_position_probs: Optional[torch.Tensor] = None
    correct_token_logits: Optional[torch.Tensor] = None
    correct_token_probs: Optional[torch.Tensor] = None

    if masked_positions is not None:
        masked_position_logits = logits[idx, masked_positions]
        masked_position_probs = probs[idx, masked_positions]

        correct_token_ids = labels.to(device)
        if component_out.dim() == 3:
            gather_ids = correct_token_ids.unsqueeze(-1)  # [B, 1]
            correct_token_logits = masked_position_logits.gather(-1, gather_ids).squeeze(-1)
            correct_token_probs = masked_position_probs.gather(-1, gather_ids).squeeze(-1)
        else:  # [B, L, H, D]
            gather_ids = correct_token_ids.unsqueeze(1).unsqueeze(-1).expand(
                -1, masked_position_logits.shape[1], -1
            )
            correct_token_logits = masked_position_logits.gather(-1, gather_ids).squeeze(-1)
            correct_token_probs = masked_position_probs.gather(-1, gather_ids).squeeze(-1)

    # ---------------------- compute entropy ----------------------
    vocab_size = probs.shape[-1]
    log_vocab_size = torch.log(torch.tensor(vocab_size, dtype=probs.dtype, device=device))
    safe_probs = probs.clamp_min(1e-9)
    entropy = -(safe_probs * safe_probs.log()).nan_to_num(0).sum(dim=-1)
    normalized_entropy_per_position = entropy / log_vocab_size

    argmax_token_ids = probs.argmax(dim=-1)
    gather_for_argmax = argmax_token_ids.unsqueeze(-1)
    argmax_token_logits = logits.gather(-1, gather_for_argmax).squeeze(-1)
    argmax_token_probabilities = probs.gather(-1, gather_for_argmax).squeeze(-1)
    ids_nested = argmax_token_ids.detach().cpu().tolist()
    if component_out.dim() == 3 and track_name != "function":
        argmax_token_texts = [
            _decode_ids_with_fallback(tokenizer, seq_ids, track_name) for seq_ids in ids_nested
        ]
    else:
        argmax_token_texts = [
            [_decode_ids_with_fallback(tokenizer, head_ids, track_name) for head_ids in pos_ids]
            for pos_ids in ids_nested
        ]

    top_k = max(1, min(top_k, vocab_size))
    topk_probs, topk_ids = probs.topk(top_k, dim=-1)
    topk_token_probabilities = topk_probs
    topk_token_ids = topk_ids

    ids_topk_nested = topk_token_ids.detach().cpu().tolist()
    if component_out.dim() == 3 and track_name != "function":
        topk_token_texts = [
            [
                _decode_ids_with_fallback(tokenizer, candidate_ids, track_name)
                for candidate_ids in position_ids
            ]
            for position_ids in ids_topk_nested
        ]
    else:
        topk_token_texts = [
            [
                [
                    _decode_ids_with_fallback(tokenizer, candidate_ids, track_name)
                    for candidate_ids in head_ids
                ]
                for head_ids in position_ids
            ]
            for position_ids in ids_topk_nested
        ]

    decoded_vocab_tokens = _decode_ids_with_fallback(tokenizer, list(range(vocab_size)), track_name)

    # ---------------------- return structured results ----------------------
    return ExtendedSequenceLensResults(
        decoded_vocab_tokens=decoded_vocab_tokens,
        logits=logits,
        probabilities=probs,
        normalized_entropy_per_position=normalized_entropy_per_position,
        argmax_token_ids=argmax_token_ids,
        argmax_token_logits=argmax_token_logits,
        argmax_token_probabilities=argmax_token_probabilities,
        argmax_token_texts=argmax_token_texts,
        topk_token_ids=topk_token_ids,
        topk_token_probabilities=topk_token_probabilities,
        topk_token_texts=topk_token_texts,
        masked_position_logits=masked_position_logits,
        masked_position_probabilities=masked_position_probs,
        correct_token_logits=correct_token_logits,
        correct_token_probabilities=correct_token_probs,
    )


def compute_residue_annotation_lens_results(
    model: HookedESM3,
    tokenizers,
    component_out: torch.Tensor,
    device: torch.device,
    annotation_threshold: float = 0.5,
) -> ResidueAnnotationLensResults:
    tokenizer = tokenizers.residue_annotations
    try:
        regression_head = getattr(model.unembed.output_heads, "residue_head")
    except AttributeError as exc:
        raise ValueError("Model has no regression head for residue annotations") from exc

    if component_out.dim() == 3:
        pass
    elif component_out.dim() == 4:
        pass
    else:
        raise ValueError(
            f"Expected component_out to have dim 3 or 4 for residue annotations, but got {component_out.dim()}"
        )

    normalized = model.ln_final(component_out)
    logits = regression_head(normalized).to(device)
    probabilities = torch.sigmoid(logits)
    if torch.isnan(probabilities).any():
        print("Found NaN in probabilities for residue annotations")
        print(probabilities)
    else:
        print("No NaN in probabilities for residue annotations")
    if torch.isinf(probabilities).any():
        print("Found Inf in probabilities for residue annotations")
        print(probabilities)
    else:
        print("No Inf in probabilities for residue annotations")
    safe_probs = probabilities.clamp(min=1e-6, max=1 - 1e-6)
    entropy_per_annotation = -(
        safe_probs * safe_probs.log() + (1 - safe_probs) * (1 - safe_probs).log()
    ) 
    if torch.isnan(entropy_per_annotation).any():
        print("Found NaN in entropy per annotation for residue annotations")
        print(entropy_per_annotation)
    else:
        print("No NaN in entropy per annotation for residue annotations")
    if torch.isinf(entropy_per_annotation).any():
        print("Found Inf in entropy per annotation for residue annotations")
        print(entropy_per_annotation)
    else:
        print("No Inf in entropy per annotation for residue annotations")
    log_two = torch.log(torch.tensor(2.0, dtype=entropy_per_annotation.dtype, device=device))
    normalized_entropy_per_annotation = entropy_per_annotation / log_two
    entropy_per_token_mean = entropy_per_annotation.mean(dim=-1)
    normalized_entropy_per_token_mean = normalized_entropy_per_annotation.mean(dim=-1)
    count_above_threshold = (probabilities >= annotation_threshold).sum(dim=-1)

    topk_annotation_ids, topk_annotation_log_probs = sample_residue_annotation_logits(
        logits=logits, annotation_threshold=annotation_threshold
    )
    topk_annotation_probabilities = topk_annotation_log_probs.exp()

    ids_topk_nested = topk_annotation_ids.detach().cpu().tolist()
    if topk_annotation_ids.dim() == 3:
        topk_annotation_texts = [
            [
                _residue_annotation_ids_to_labels(tokenizer, annotation_ids)
                for annotation_ids in position_ids
            ]
            for position_ids in ids_topk_nested
        ]
    else:
        topk_annotation_texts = [
            [
                [
                    _residue_annotation_ids_to_labels(tokenizer, annotation_ids)
                    for annotation_ids in head_ids
                ]
                for head_ids in position_ids
            ]
            for position_ids in ids_topk_nested
        ]

    decoded_vocab_tokens = list(getattr(tokenizer, "vocabulary", []))

    return ResidueAnnotationLensResults(
        decoded_vocab_tokens=decoded_vocab_tokens,
        logits=logits,
        probabilities=probabilities,
        entropy_per_annotation=entropy_per_annotation,
        entropy_per_token_mean=entropy_per_token_mean,
        normalized_entropy_per_annotation=normalized_entropy_per_annotation,
        normalized_entropy_per_token_mean=normalized_entropy_per_token_mean,
        count_above_threshold=count_above_threshold,
        topk_annotation_ids=topk_annotation_ids,
        topk_annotation_log_probs=topk_annotation_log_probs,
        topk_annotation_probabilities=topk_annotation_probabilities,
        topk_annotation_texts=topk_annotation_texts,
    )


def return_predicted_function_annotations(model, tokenizers, component_out: torch.Tensor, device: torch.device, function_token_decoder, annotation_min_length=5, decoder_annotation_threshold=0.3):
   
    if component_out.dim() != 3:
        raise ValueError(f"Expected component_out to have dim 3, but got {component_out.dim()}")

    if component_out.shape[0] != 1:
        raise ValueError(f"Expected component_out to have batch size 1, but got {component_out.shape[0]}")

    
    probs, logits = _compute_track_probs_and_logits(model, tokenizers.function, model.unembed.output_heads.function_head, component_out, device, "function")

    ids, _ = sample_function_logits(logits= logits, tokenizer= tokenizers.function)
    ids = ids.squeeze(0)
    print(ids.shape)
    res = decode_function_tokens(function_token_ids=ids, function_token_decoder=function_token_decoder, annotation_min_length=annotation_min_length, function_tokens_tokenizer=tokenizers.function, decoder_annotation_threshold=decoder_annotation_threshold)
    print(f"decoder_annotation_threshold: {decoder_annotation_threshold}")
    return res


def return_predicted_residue_annotations(
    model: HookedESM3,
    tokenizers,
    component_out: torch.Tensor,
    device: torch.device,
    annotation_threshold: float = 0.5,
    annotation_min_length: int | None = 5,
    annotation_gap_merge_max: int | None = 3,
):
    if component_out.dim() != 3:
        raise ValueError(
            f"Expected component_out to have dim 3 for residue annotations, but got {component_out.dim()}"
        )
    if component_out.shape[0] != 1:
        raise ValueError(
            f"Expected component_out to have batch size 1 for residue annotations, but got {component_out.shape[0]}"
        )

    try:
        regression_head = getattr(model.unembed.output_heads, "residue_head")
    except AttributeError as exc:
        raise ValueError("Model has no regression head for residue annotations") from exc

    normalized = model.ln_final(component_out)
    logits = regression_head(normalized).to(device)

    annotation_ids, _ = sample_residue_annotation_logits(
        logits=logits, annotation_threshold=annotation_threshold
    )
    annotation_ids = annotation_ids.squeeze(0)

    annotations = decode_residue_annotation_tokens(
        residue_annotations_token_ids=annotation_ids,
        residue_annotations_tokenizer=tokenizers.residue_annotations,
        annotation_min_length=annotation_min_length,
        annotation_gap_merge_max=annotation_gap_merge_max,
    )

    return annotations

