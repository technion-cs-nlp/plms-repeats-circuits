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


def _best_pattern_match(stem: str, patterns: Dict[str, str]) -> Optional[str]:
    """Find all patterns that appear anywhere in *stem*, return the display_name
    of the longest one.  Longest-wins avoids 'blosum' swallowing 'blosum-opposite'
    or 'blosum-opposite50', etc., regardless of stem format."""
    matches = [(pat, name) for pat, name in patterns.items() if pat in stem]
    if not matches:
        return None
    return max(matches, key=lambda x: len(x[0]))[1]


def identify_result_file(stem: str) -> tuple[Optional[str], Optional[str]]:
    """Reverse lookup: given a filename stem (no extension), return (display_name, kind).

    Works for any file type (csv, json, …) — pass only the stem.
    kind is 'baseline' or 'main'.  Returns (None, None) if unrecognised.

    Example:
        identify_result_file('identical_counterfactual_blosum_baseline')
        → ('100% BLOSUM', 'baseline')
        identify_result_file('identical_counterfactual_blosum-opposite')
        → ('100% BLOSUM Opposite', 'main')
    """
    display = _best_pattern_match(stem, BASELINE_METHOD_PATTERNS)
    if display:
        return display, "baseline"
    display = _best_pattern_match(stem, MAIN_METHOD_PATTERNS)
    if display:
        return display, "main"
    return None, None


def find_file_for_method(
    method_name: str,
    input_dir: "Path",  # type: ignore[name-defined]
    kind: str = "main",
    ext: str = "csv",
) -> Optional["Path"]:  # type: ignore[name-defined]
    """Forward lookup: scan *input_dir* for a result file belonging to *method_name*.

    Args:
        method_name: name from COUNTERFACTUAL_METHODS (e.g. 'blosum').
        input_dir:   directory to search recursively.
        kind:        'main' or 'baseline'.
        ext:         file extension to search for, without leading dot (e.g. 'csv', 'json').

    Returns the first matching file in sorted order, or None if not found.
    """
    from pathlib import Path as _Path
    target_display = next(
        (m["display_name"] for m in COUNTERFACTUAL_METHODS if m["name"] == method_name),
        None,
    )
    if target_display is None:
        return None
    for f in sorted(_Path(input_dir).rglob(f"*.{ext}")):
        display, file_kind = identify_result_file(f.stem)
        if display == target_display and file_kind == kind:
            return f
    return None
