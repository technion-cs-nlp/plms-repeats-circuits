import torch
from transformer_lens import HookedESM3,SupportedESM3Config
import torch.nn.functional as F
from esm.tokenization import (
    TokenizerCollectionProtocol,
    get_invalid_tokenizer_ids,
)
from esm.utils.constants import esm3 as C

def get_probs_from_logits(logits, device, tokenizer, mask_logits_of_invalid_ids=True):
    logits1 = logits.clone()
    mask = torch.ones_like(logits1, dtype=torch.bool, device=device)
    if mask_logits_of_invalid_ids:
        valid_ids = list(
            (
                set(tokenizer.all_token_ids)
                - set(tokenizer.special_token_ids)
                - set(get_invalid_tokenizer_ids(tokenizer))
            )
        )
        mask[..., valid_ids] = False
        logits1[mask] = -torch.inf
    
    probs = F.softmax(logits1, dim=-1)
    return probs

def mask_protein_short(protein, mask_position, tokenizer):
    tokenizer = tokenizer
    protein_list = list(protein)
    protein_list[mask_position]= C.MASK_STR_SHORT
    masked_protein = "".join(protein_list)
    return masked_protein

def replace_short_mask_with_mask_token(protein, tokenizer):
    return protein.replace(C.MASK_STR_SHORT, tokenizer.mask_token)

def load_esm3_hooked_model(device, cache_activations=False, esm3_output_type="sequence"):
    config = SupportedESM3Config(
        use_attn_result=cache_activations,
        use_split_qkv_input=cache_activations,
        use_hook_mlp_in=cache_activations,
        use_attn_in=False,
        esm3_output_type=esm3_output_type,
        esm3_use_torch_layer_norm=True,
        esm3_use_torch_attention_calc=True,
        esm3_use_org_rotary=True,
        esm3_capture_activations_before_normalization=True
    )
    esm3_hooked = HookedESM3.from_pretrained(esm_cfg=config, device=device)
    return esm3_hooked