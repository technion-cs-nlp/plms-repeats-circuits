from typing import Any, Dict, List, Optional

COUNTERFACTUAL_METHODS: List[Dict[str, Any]] = [
    # 100% coverage
    {
        "name": "mask",
        "display_name": "100% Mask",
        "main_eval": {"method": "corrupt_all_other_repeats_with_token", "type1": "mask", "type2": "mask"},
        "baseline_eval": {"method": "corrupt_similiar_length_with_char", "type1": "mask", "type2": "mask"},
    },
    {
        "name": "blosum",
        "display_name": "100% BLOSUM",
        "main_eval": {"method": "corrupt_all_other_repeats_with_token", "type1": "blosum", "type2": None},
        "baseline_eval": {"method": "corrupt_similiar_length_with_char", "type1": "blosum", "type2": None},
    },
    {
        "name": "blosum-opposite",
        "display_name": "100% BLOSUM Opposite",
        "main_eval": {"method": "corrupt_all_other_repeats_with_token", "type1": "blosum_opposite", "type2": None},
        "baseline_eval": {"method": "corrupt_similiar_length_with_char", "type1": "blosum_opposite", "type2": None},
    },
    # 50% coverage
    {
        "name": "mask50",
        "display_name": "50% Mask",
        "main_eval": {"method": "corrupt_pct_other_repeats_with_token", "type1": "mask", "type2": "mask", "pct": 0.5},
        "baseline_eval": {"method": "corrupt_by_replacing_a_pct_on_similar_length_part_with_char", "type1": "mask", "type2": "mask", "pct": 0.5},
    },
    {
        "name": "blosum50",
        "display_name": "50% BLOSUM",
        "main_eval": {"method": "corrupt_pct_other_repeats_with_token", "type1": "blosum", "type2": None, "pct": 0.5},
        "baseline_eval": {"method": "corrupt_by_replacing_a_pct_on_similar_length_part_with_char", "type1": "blosum", "type2": None, "pct": 0.5},
    },
    {
        "name": "blosum-opposite50",
        "display_name": "50% BLOSUM Opposite",
        "main_eval": {"method": "corrupt_pct_other_repeats_with_token", "type1": "blosum_opposite", "type2": None, "pct": 0.5},
        "baseline_eval": {"method": "corrupt_by_replacing_a_pct_on_similar_length_part_with_char", "type1": "blosum_opposite", "type2": None, "pct": 0.5},
    },
    # 20% coverage
    {
        "name": "mask20",
        "display_name": "20% Mask",
        "main_eval": {"method": "corrupt_pct_other_repeats_with_token", "type1": "mask", "type2": "mask", "pct": 0.2},
        "baseline_eval": {"method": "corrupt_by_replacing_a_pct_on_similar_length_part_with_char", "type1": "mask", "type2": "mask", "pct": 0.2},
    },
    {
        "name": "blosum20",
        "display_name": "20% BLOSUM",
        "main_eval": {"method": "corrupt_pct_other_repeats_with_token", "type1": "blosum", "type2": None, "pct": 0.2},
        "baseline_eval": {"method": "corrupt_by_replacing_a_pct_on_similar_length_part_with_char", "type1": "blosum", "type2": None, "pct": 0.2},
    },
    {
        "name": "blosum-opposite20",
        "display_name": "20% BLOSUM Opposite",
        "main_eval": {"method": "corrupt_pct_other_repeats_with_token", "type1": "blosum_opposite", "type2": None, "pct": 0.2},
        "baseline_eval": {"method": "corrupt_by_replacing_a_pct_on_similar_length_part_with_char", "type1": "blosum_opposite", "type2": None, "pct": 0.2},
    },
    # Permutation
    {
        "name": "permutation",
        "display_name": "Permutation",
        "main_eval": {"method": "permutation", "type1": "A", "type2": None},
        "baseline_eval": {"method": "corrupt_by_permute_similiar_length", "type1": "A", "type2": None},
    },
]


def _main_pattern(name: str) -> str:
    """Substring to find main output file: {repeat_type}_counterfactual_{name}.csv"""
    return f"counterfactual_{name}"


def _baseline_pattern(name: str) -> str:
    """Substring to find baseline output file: {repeat_type}_counterfactual_{name}_baseline.csv"""
    return f"counterfactual_{name}_baseline"


# Convenience lookups (derived from name, no complex patterns stored)
METHOD_PATTERNS: Dict[str, str] = {m["name"]: _main_pattern(m["name"]) for m in COUNTERFACTUAL_METHODS}
MAIN_METHOD_PATTERNS: Dict[str, str] = {
    _main_pattern(m["name"]): m["display_name"] for m in COUNTERFACTUAL_METHODS
}
BASELINE_METHOD_PATTERNS: Dict[str, str] = {
    _baseline_pattern(m["name"]): m["display_name"]
    for m in COUNTERFACTUAL_METHODS
    if m.get("baseline_eval")
}
