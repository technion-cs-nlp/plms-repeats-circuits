"""Concept-related utilities."""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch


def _to_long_tensor(x, device):
    """Convert various types to 1D long tensor on device."""
    if isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.integer):
            return torch.from_numpy(x).to(device=device).long().reshape(-1)
        elif np.issubdtype(x.dtype, np.floating):
            if not np.all(x == np.floor(x)):
                raise ValueError(f"NumPy array contains non-integer float values: {x}")
            return torch.from_numpy(x.astype(np.int64)).to(device=device).long().reshape(-1)
        else:
            raise ValueError(f"Unsupported numpy array dtype: {x.dtype}")
    elif isinstance(x, (np.integer, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        x = int(x)
    elif isinstance(x, (np.floating, np.float16, np.float32, np.float64)):
        if float(x).is_integer():
            x = int(x)
        else:
            raise ValueError(f"NumPy float {x} is not an integer")

    if isinstance(x, torch.Tensor):
        if x.dtype.is_floating_point:
            if not torch.all(x == torch.floor(x)):
                raise ValueError(f"Tensor contains non-integer float values: {x}")
        return x.to(device=device).long().reshape(-1)

    if isinstance(x, int):
        return torch.tensor([x], device=device, dtype=torch.long)

    if isinstance(x, float):
        if not x.is_integer():
            raise ValueError(f"Float {x} is not an integer")
        return torch.tensor([int(x)], device=device, dtype=torch.long)

    if isinstance(x, (list, tuple)):
        vals = []
        for el in x:
            if isinstance(el, (np.integer, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                el = int(el)
            elif isinstance(el, (np.floating, np.float16, np.float32, np.float64)):
                if float(el).is_integer():
                    el = int(el)
                else:
                    raise ValueError(f"Non-integer element {el} in {x}")
            if isinstance(el, int):
                vals.append(el)
            elif isinstance(el, float) and el.is_integer():
                vals.append(int(el))
            else:
                raise ValueError(f"Non-integer element {el} (type: {type(el)}) in {x}")
        return torch.tensor(vals, device=device, dtype=torch.long)

    raise ValueError(f"Unsupported type: {type(x)}")


@dataclass
class RepeatsInfoAboutSequence:
    repeat_positions_tensor: torch.Tensor
    near_repeat_positions_tensor: torch.Tensor
    aligned_token_to_masked_position_tensor: torch.Tensor
    problematic_positions_in_sequence_tensor_related_to_suspected_repeat: torch.Tensor


def compare_seq_positions_to_token_values(
    token_values: List[int], tokenized_seq: torch.Tensor
) -> torch.Tensor:
    assert isinstance(token_values, list), "token_values must be a list"
    assert isinstance(tokenized_seq, torch.Tensor), "tokenized_seq must be a torch.Tensor"

    if tokenized_seq.dim() == 2 and tokenized_seq.shape[0] == 1:
        tokenized_seq = tokenized_seq.squeeze(0)
    elif tokenized_seq.dim() == 1:
        pass
    else:
        raise ValueError(
            f"tokenized_seq must be shape (L,) or (1, L), but got {tuple(tokenized_seq.shape)}"
        )

    token_values_tensor = _to_long_tensor(token_values, tokenized_seq.device)
    mask = (tokenized_seq.unsqueeze(0) == token_values_tensor.unsqueeze(1)).any(dim=0)
    return torch.nonzero(mask, as_tuple=False).flatten().long()


__all__ = [
    "RepeatsInfoAboutSequence",
    "_to_long_tensor",
    "compare_seq_positions_to_token_values",
]
