import torch
from transformer_lens import HookedESM3,SupportedESM3Config, HookedESMC, SupportedESMCConfig
import torch.nn.functional as F
from esm.tokenization import (
    TokenizerCollectionProtocol,
    get_invalid_tokenizer_ids,
)
from esm.utils.constants import esm3 as C
from esm.tokenization import get_esm3_model_tokenizers, get_esmc_model_tokenizers
from esm.models.esmc import ESMC
from esm.pretrained import (
    ESM3_sm_open_v0,
    ESMC_600M_202412,
    ESMC_300M_202412,
)

def get_vocab_size_if_mask_invalid_ids(tokenizer):
    valid_ids = list(
            (
                set(tokenizer.all_token_ids)
                - set(tokenizer.special_token_ids)
                - set(get_invalid_tokenizer_ids(tokenizer))
            )
        )
    return len(valid_ids)

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

def mask_protein_short_ranges(protein: str, mask_ranges: list[list[int]]) -> str:
    """
    Mask ranges of residues in a protein sequence with MASK_STR_SHORT.

    Args:
        protein (str): Protein sequence (string of amino acids).
        mask_ranges (list of [start, end]): List of (start, end) indices to mask (inclusive).

    Returns:
        str: Masked protein sequence.
    """
    protein_list = list(protein)

    for start, end in mask_ranges:
        if not (0 <= start <= end < len(protein_list)):
            raise ValueError(f"Invalid mask range [{start}, {end}] for sequence of length {len(protein_list)}.")

        for pos in range(start, end + 1):
            protein_list[pos] = C.MASK_STR_SHORT

    return "".join(protein_list)

def replace_short_mask_with_mask_token(protein, tokenizer):
    return protein.replace(C.MASK_STR_SHORT, tokenizer.mask_token)



def _load_esm3_model(use_transformer_lens_model, device, cache_attention_activations=False, cache_mlp_activations=False, output_type="sequence", cache_attn_pattern=False, split_qkv_input=True):
    if use_transformer_lens_model:
        config = SupportedESM3Config(
            use_attn_result=cache_attention_activations,
            use_split_qkv_input=cache_attention_activations and split_qkv_input,
            use_hook_mlp_in=cache_mlp_activations,
            use_attn_in=cache_attention_activations and not split_qkv_input,
            esm3_output_type=output_type,
            esm3_use_torch_layer_norm=True,
            esm3_use_torch_attention_calc=not cache_attn_pattern,
            esm3_use_org_rotary=True,
        )
        #print config params one by one
        print(f"_load_esm3_model config params:")
        if hasattr(config, '__dict__'):
            for param, value in config.__dict__.items():
                print(f"{param}: {value}")
        else:
            print(f"{config}")
        print(f"_load_esm3_model device: {device}")
        model = HookedESM3.from_pretrained(esm_cfg=config, device=device)
        model.eval()
    else:
        model = ESM3_sm_open_v0(device).to(device).to(torch.float32)
        model.eval()
    print(f"esm3 model loaded")
    return model

def _load_esm_c_model(model_type, use_transformer_lens_model, device, cache_attention_activations=False, cache_mlp_activations=False, cache_attn_pattern=False, split_qkv_input=True):
    if model_type == "esmc_600m" or model_type == "esmc" or model_type == "esm-c":
        model_name = "esmc_600m"
    elif model_type == "esmc_300m":
        model_name = "esmc_300m"
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    if use_transformer_lens_model:
        config = SupportedESMCConfig(
            use_attn_result=cache_attention_activations,
            use_split_qkv_input=cache_attention_activations and split_qkv_input,
            use_hook_mlp_in=cache_mlp_activations,
            use_attn_in=cache_attention_activations and not split_qkv_input,
            esmc_use_torch_layer_norm=True,
            esmc_use_torch_attention_calc=not cache_attn_pattern,
            esmc_use_org_rotary=True
        )
        #dump config and print it 
        #print params one by one
        print(f"_load_esm_c_model config params:")
        if hasattr(config, '__dict__'):
            for param, value in config.__dict__.items():
                print(f"{param}: {value}")
        else:
            print(f"{config}")
        print(f"_load_esm_c_model model_name: {model_name}")
        print(f"_load_esm_c_model device: {device}")
        model = HookedESMC.from_pretrained(esmc_cfg=config, model_name=model_name, device=device)
    else:
        model = ESMC_600M_202412(device=device) if model_name == "esmc_600m" else ESMC_300M_202412(device=device)
        model = model.to(device).to(torch.float32).eval()
    print(f"esm-c model loaded- {model_name}")
    return model

def load_model(model_type, device,use_transformer_lens_model=True, cache_attention_activations=False, cache_mlp_activations=False, output_type="sequence", cache_attn_pattern=False, split_qkv_input=True):
    if model_type == "esm3":
        return _load_esm3_model(use_transformer_lens_model, device, cache_attention_activations, cache_mlp_activations, output_type, cache_attn_pattern, split_qkv_input)
    elif model_type == "esm-c" or model_type == "esmc" or model_type == "esmc_600m" or model_type == "esmc_300m":
        return _load_esm_c_model(model_type, use_transformer_lens_model, device, cache_attention_activations, cache_mlp_activations, cache_attn_pattern, split_qkv_input)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def _load_esm3_tokenizer():
    print(f"loading esm3 tokenizer")
    return get_esm3_model_tokenizers().sequence

def _load_esm_c_tokenizer(model):
    print(f"loading esm-c tokenizer")
    return model.tokenizer

def load_tokenizer_by_model_type(model_type):
    if model_type == "esm3":
        print(f"loading esm3 tokenizer")
        return get_esm3_model_tokenizers().sequence
    elif model_type == "esm-c" or model_type == "esmc" or model_type == "esmc_600m" or model_type == "esmc_300m":
        print(f"loading esmc tokenizer")
        return get_esmc_model_tokenizers()
    else:
        raise ValueError(f"Invalid model type: {model_type}")