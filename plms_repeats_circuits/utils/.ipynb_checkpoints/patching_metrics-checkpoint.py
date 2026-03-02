import torch
from protein_circuits.utils.model_utils import get_masked_token_logits
from functools import partial
import torch

def logit_diff(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, masked_positions: torch.Tensor, labels: torch.Tensor, mean=True, prob=False, loss=False):
    masked_clean_logits = get_masked_token_logits(clean_logits, masked_positions)
    if masked_clean_logits.dim() != 2:
        raise ValueError(f"Expected masked_clean_logits to have shape (batch_size, hidden_dim), but got {masked_clean_logits.shape}")
    if labels.dim() != 2 or labels.size(1) != 2 or labels.size(0) != masked_clean_logits.size(0):
        raise ValueError(f"Expected labels to have shape (batch_size, 2), but got {labels.shape}")
    cleans = torch.softmax(masked_clean_logits, dim=-1) if prob else masked_clean_logits
    good_bad = torch.gather(cleans, -1, labels.to(cleans.device))
    if good_bad.dim() !=2 or good_bad.size(1) !=2 or good_bad.size(0) != masked_clean_logits.size(0):
        raise ValueError(f"Expected good_bad to have shape (batch_size, 2), but got {good_bad.shape}")
    results = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        # remember it's reversed to make it a loss
        results = -results
    if mean: 
        results = results.mean()
    return results

def log_prob(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, masked_positions: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    masked_clean_logits = get_masked_token_logits(clean_logits, masked_positions)
    if masked_clean_logits.dim() != 2:
        raise ValueError(f"Expected masked_clean_logits to have shape (batch_size, hidden_dim), but got {masked_clean_logits.shape}")
    if labels.dim() == 1:
        labels = labels.unsqueeze(-1)
    if labels.dim() != 2 or labels.size(1) != 1 or labels.size(0) != masked_clean_logits.size(0):
        raise ValueError(f"Expected labels to have shape (batch_size, 1), but got {labels.shape}")
    cleans = torch.log_softmax(masked_clean_logits, dim=-1)
    results = torch.gather(cleans, -1, labels.to(cleans.device))
    if results.dim() !=2 or results.size(1) !=1 or results.size(0) != masked_clean_logits.size(0):
        raise ValueError(f"Expected results to have shape (batch_size, 1), but got {results.shape}")
    results = results.squeeze(-1)
    if loss:
        # remember it's reversed to make it a loss
        results = -results
    if mean: 
        results = results.mean()
    return results

def create_loss_and_metric(metric_name):
    if metric_name == "logit_diff":
        loss=partial(logit_diff, loss=True, mean=True)
        metric=partial(logit_diff, loss=False, mean=False)
        return loss, metric
    if metric_name =="log_prob":
        loss=partial(log_prob, loss=True, mean=True)
        metric=partial(log_prob, loss=False, mean=False)
        return loss, metric
    else: 
        raise ValueError("Metric name Not Supported")