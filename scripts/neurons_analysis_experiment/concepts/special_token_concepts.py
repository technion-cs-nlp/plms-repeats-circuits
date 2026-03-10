from transformers import PreTrainedTokenizerFast

from .concept import Concept, ConceptCategory, TokenConceptFunctionTagger


def make_special_token_concepts(tokenizer: PreTrainedTokenizerFast) -> list:
    concepts = []
    
    bos_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None else None
    eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None else None
    mask_token_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None else None
    
    if bos_token_id is None:
        raise ValueError("BOS token ID not found, cannot create BOS concept")
    
    if eos_token_id is None:
        raise ValueError("EOS token ID not found, cannot create BOS concept")
    
    concepts.append(Concept(
        concept_name="bos_token",
        concept_category=ConceptCategory.SPECIAL_TOKENS,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=[bos_token_id],
            concept_name="bos_token",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=False
        )
    ))
    concepts.append(Concept(
        concept_name="eos_token",
        concept_category=ConceptCategory.SPECIAL_TOKENS,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=[eos_token_id],
            concept_name="eos_token",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=False
        )
    ))
    concepts.append(Concept(
        concept_name="bos_eos_token",
        concept_category=ConceptCategory.SPECIAL_TOKENS,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=[bos_token_id, eos_token_id],
            concept_name="bos_eos_token",
            to_exclude_eos=False,
            to_exclude_bos=False,
            to_exclude_mask=False
        )
    ))
    if mask_token_id is None:
        raise ValueError("MASK token ID not found, cannot create MASK concept")
    concepts.append(Concept(
        concept_name="mask_token",
        concept_category=ConceptCategory.SPECIAL_TOKENS,
        concept_function_tagger=TokenConceptFunctionTagger(
            token_values=[mask_token_id],
            concept_name="mask_token",
            to_exclude_eos=True,
            to_exclude_bos=True,
            to_exclude_mask=False
        )
    ))
    return concepts

