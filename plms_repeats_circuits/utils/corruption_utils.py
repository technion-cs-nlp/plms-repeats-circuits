import re

def get_corrupt_method_name(filename):
    lower_filename = filename.lower()
    method_name = ""
    token = ""

    # Method name
    if "corrupt_all" in lower_filename or "corrupt_similiar_length_with_char" in lower_filename:
        method_name = "Cover All"
    elif "same_as_masking" in lower_filename or  "corrupt_by_replacing_one_token_in_same_space_with_char" in lower_filename:
        method_name = "Replace Equivalent"
    elif "permutation" in lower_filename or "corrupt_by_permute_similiar_length" in lower_filename:
        method_name = "Permutation"
    elif "corrupt_pct" in lower_filename or "corrupt_by_replacing_a_pct_on_similar_length" in lower_filename:
        pattern = re.compile(r"_pct([0-9]+\.[0-9]+)")
        match = pattern.search(lower_filename)
        if match:
            pct_value = float(match.group(1))
            method_name = f"Cover {pct_value * 100:.0f}%"
        else:
            raise ValueError("No pct value found for pct method")

    else:
        raise ValueError(f"Unsupported method {lower_filename}")

    # Token
    if "blosum_opposite_none" in lower_filename:
        token = "BLOSUM OPPOSITE"
    elif "blosum_none" in lower_filename:
        token = "BLOSUM"
    elif "a_s" in lower_filename:
        token = "Alanine"
    elif "_____" in lower_filename:
        token = "Mask"
    else:
        pass

    # Check token
    if token == "" and method_name != "Permutation":
        raise ValueError("Method with no token")

    # Baseline
    baseline = "Baseline" if "baseline" in lower_filename else ""

    # Final string
    str = f"{method_name} {baseline}"
    if method_name !="Permutation":
        str += f" ({token})"
    return str


def get_sort_key(method_name):
    if method_name.startswith("Original"):
        return 0
    elif method_name.startswith("Replace Equivalent"):
        return 1
    elif method_name.startswith("Permutation"):
        return 2
    elif method_name.startswith("Cover All"):
        return 3
    elif method_name.startswith("Cover 50%"):
        return 4
    elif method_name.startswith("Cover 20%"):
        return 5
    else:
        # Put any unknown methods last
        return 100
