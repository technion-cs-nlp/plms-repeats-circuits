import torch
from transformers import PreTrainedTokenizerFast

def get_masked_token_logits(logits: torch.Tensor, masked_positions: torch.Tensor):
    """
    Retrieves the logits for a single masked token per sample in a batch.

    Args:
        logits (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_dim)
        masked_positions (torch.Tensor): Tensor of shape (batch_size,) containing indices of masked tokens

    Returns:
        torch.Tensor: Tensor of shape (batch_size, hidden_dim) containing logits for the masked tokens.
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected logits to have shape (batch_size, seq_len, hidden_dim), but got {logits.shape}")
    
    batch_size = logits.size(0)

    if masked_positions.dim() != 1 or masked_positions.size(0) != batch_size:
        raise ValueError(f"Expected masked_positions to have shape ({batch_size},), but got {masked_positions.shape}")
    
    idx = torch.arange(batch_size, device=logits.device)  # Create batch indices (batch_size,)
    masked_logits = logits[idx, masked_positions]  # Extract logits for the masked tokens

    return masked_logits  # Shape: (batch_size, hidden_dim)


def get_masked_tokens_logits(logits: torch.Tensor, masked_positions: torch.Tensor):
    """
    Extracts the logits corresponding to the masked positions.

    Args:
        logits (torch.Tensor): Tensor of shape (batch_size, seq_len, hidden_dimension)
        masked_positions (torch.Tensor): Tensor of shape (batch_size, n_masked_positions)
                                          specifying masked positions in the sequence.

    Returns:
        torch.Tensor: Logits of shape (batch_size, n_masked_positions, hidden_dimension)
    """
    assert logits.dim() == 3, "Expected logits to have shape (batch_size, seq_len, hidden_dimension)"
    
    batch_size, seq_len, hidden_dim = logits.shape
    
    # Ensure masked_positions is always of shape (batch_size, n_masked_positions)
    if masked_positions.dim() == 1:
        masked_positions = masked_positions.unsqueeze(-1)  # Convert (batch_size,) → (batch_size, 1)
    
    # Create batch indices (batch_size, 1) to match masked_positions shape
    idx = torch.arange(batch_size, device=logits.device).unsqueeze(-1)  

    # Index logits using batch indices and masked positions
    masked_logits = logits[idx, masked_positions]  # Shape: (batch_size, n_masked_positions, hidden_dimension)

    return masked_logits


def mask_protein(protein, mask_position, tokenizer):
    return protein[:mask_position] + tokenizer.mask_token + protein[mask_position + 1:]

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def tokenize_input(inputs, tokenizer):
    tokenized_info = tokenizer(inputs, padding=True, return_tensors='pt', add_special_tokens=True)
    tokens = tokenized_info.input_ids
    attention_mask = tokenized_info.attention_mask
    return tokens, attention_mask

def tokenize_labels(labels,tokenizer):
    tokenized_labels = tokenizer(
            labels,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False
        )['input_ids'].tolist()
    return tokenized_labels