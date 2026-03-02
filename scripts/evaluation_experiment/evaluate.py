import pandas as pd
import torch
import argparse
import logging
import csv
import time
import random
import ast
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from plms_repeats_circuits.utils.esm_utils import get_probs_from_logits, load_model, load_tokenizer_by_model_type
from plms_repeats_circuits.utils.model_utils import get_device, mask_protein
from plms_repeats_circuits.utils.experiment_utils import set_random_seed


def predict_masked_tokens(model, tokenizer, masked_proteins, mask_positions, device):
    """Run forward pass and return probabilities at masked positions."""
    tokenized = tokenizer(
        masked_proteins, return_tensors="pt", add_special_tokens=True, padding=True
    )
    input_ids = tokenized["input_ids"].to(device)
    attn_mask = tokenized["attention_mask"].to(device)
    mask_pos = torch.tensor(mask_positions, device=device)

    with torch.no_grad():
        logits = model.forward(
            sequence_tokens=input_ids, sequence_id=attn_mask
        ).sequence_logits
        batch_idx = torch.arange(logits.size(0), device=device)
        masked_logits = logits[batch_idx, mask_pos]
        probs = get_probs_from_logits(
            masked_logits, device, tokenizer, mask_logits_of_invalid_ids=False
        )

    return probs


def tokenize_labels(tokenizer, labels, device):
    """Tokenize single-character amino acid labels to a 1-D tensor."""
    return tokenizer(
        labels, return_tensors="pt", add_special_tokens=False, padding=False
    )["input_ids"].squeeze(-1).to(device)



class MaskedRepeatDataset(Dataset):
    """One masked example per position in each repeat occurrence."""

    def __init__(self, protein, repeat_locations, tokenizer):
        self.examples = []
        for rep_num, (start, end) in enumerate(
            sorted(repeat_locations, key=lambda x: x[0]), start=1
        ):
            if end >= len(protein) or start > end:
                continue
            for i in range(end - start + 1):
                pos = start + i
                if pos >= len(protein):
                    continue
                self.examples.append({
                    "masked_protein": mask_protein(protein, pos, tokenizer),
                    "label": protein[pos],
                    "mask_position": pos + 1,  # +1 for BOS token
                    "relative_position": i + 1,
                    "relative_repeat_number": rep_num,
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def collate_fn(batch):
        return {key: [d[key] for d in batch] for key in batch[0]}


REPEAT_HEADER = [
    "cluster_id", "rep_id", "repeat_key", "masked_position",
    "true_label", "predicted_label", "true_label_probability", "predicted_label_probability",
    "is_correct", "relative_repeat_number",
]


def _get_dtype(dtype_str):
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]


def run_repeat_evaluation(args):
    """Run repeat prediction evaluation on all proteins in the dataset."""
    device = get_device()
    model = load_model(
        model_type=args.model_type, device=device, use_transformer_lens_model=False,
        cache_attention_activations=False, cache_mlp_activations=False,
        output_type="all", cache_attn_pattern=False, split_qkv_input=False,
    )
    model = model.to(_get_dtype(args.dtype))
    tokenizer = load_tokenizer_by_model_type(args.model_type)

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    dataset = pd.read_csv(input_path)
    if dataset.empty:
        logging.warning("Empty dataset, nothing to evaluate.")
        return

    dataset["repeat_locations"] = dataset["repeat_locations"].apply(ast.literal_eval)

    output_path = output_dir / "predictions.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(REPEAT_HEADER)

        for i, row in dataset.iterrows():
            try:
                ds = MaskedRepeatDataset(row["seq"], row["repeat_locations"], tokenizer)
                loader = DataLoader(
                    ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=MaskedRepeatDataset.collate_fn,
                )
                prefix = [row["cluster_id"], row["rep_id"], row["repeat_key"]]

                for batch in loader:
                    probs = predict_masked_tokens(
                        model, tokenizer, batch["masked_protein"],
                        batch["mask_position"], device,
                    )
                    labels = tokenize_labels(tokenizer, batch["label"], device)
                    batch_idx = torch.arange(len(labels), device=device)
                    true_probs = probs[batch_idx, labels]
                    pred_probs, pred_labels = probs.max(dim=-1)

                    for j in range(len(labels)):
                        writer.writerow(prefix + [
                            batch["mask_position"][j] - 1,
                            batch["label"][j],
                            tokenizer.decode(pred_labels[j].cpu().item()),
                            true_probs[j].cpu().item(),
                            pred_probs[j].cpu().item(),
                            pred_labels[j].cpu().item() == labels[j].cpu().item(),
                            batch["relative_repeat_number"][j],
                        ])

                if (i + 1) % 10 == 0:
                    logging.info(f"Processed {i + 1}/{len(dataset)} proteins")
            except Exception as e:
                logging.error(f"Error processing row {i}: {e}")


class BaselineDataset(Dataset):
    """Masks one random position per protein for baseline evaluation."""

    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.iloc[idx]["seq"]
        pos = random.randint(0, len(seq) - 1)
        masked = mask_protein(seq, pos, self.tokenizer)
        return seq, masked, pos + 1, seq[pos]  # +1 for BOS token


BASELINE_HEADER = [
    "sequence", "masked_position", "true_label", "true_label_probability", "is_correct",
]


def run_baseline_evaluation(args):
    """Run baseline masked-token prediction on random positions."""
    device = get_device()
    set_random_seed(args.random_seed)

    model = load_model(
        model_type=args.model_type, device=device, use_transformer_lens_model=False,
        cache_attention_activations=False, cache_mlp_activations=False,
        output_type="all", cache_attn_pattern=False, split_qkv_input=False,
    )
    model = model.to(_get_dtype(args.dtype))
    tokenizer = load_tokenizer_by_model_type(args.model_type)

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_path = output_dir / "baseline_predictions.csv"

    df = pd.read_csv(input_path)
    dataset = BaselineDataset(df, tokenizer)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda batch: tuple(map(list, zip(*batch))),
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(BASELINE_HEADER)

        processed, total_prob = 0, 0.0
        for sequences, masked_proteins, mask_positions, true_labels in loader:
            try:
                probs = predict_masked_tokens(
                    model, tokenizer, masked_proteins, mask_positions, device
                )
                labels = tokenize_labels(tokenizer, true_labels, device)
                batch_idx = torch.arange(len(labels), device=device)
                true_probs = probs[batch_idx, labels]

                total_prob += true_probs.sum().item()
                processed += len(sequences)

                pred_probs, pred_labels = probs.max(dim=-1)

                for j in range(len(sequences)):
                    writer.writerow([
                        sequences[j],
                        mask_positions[j] - 1,
                        true_labels[j],
                        true_probs[j].item(),
                        pred_labels[j].cpu().item() == labels[j].cpu().item(),
                    ])
                if processed % (args.batch_size * 10) == 0:
                    logging.info(f"Processed {processed}/{len(df)} proteins")
            except Exception as e:
                logging.error(f"Error processing batch: {e}")

        if processed > 0:
            logging.info(f"Average true label probability: {total_prob / processed:.4f}")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Evaluate masked-token prediction on protein sequences"
    )
    parser.add_argument("--input_file", required=True, help="Path to input CSV")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    parser.add_argument("--model_type", choices=["esm3", "esm-c"], default="esm3")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--mode", choices=["repeat", "baseline"], default="repeat",
        help="'repeat' evaluates on repeat positions; 'baseline' on random positions",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed (baseline mode)")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Model dtype (default float32); use bfloat16 for esm-c to match paper",
    )
    args = parser.parse_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_folder = output_dir / "logs"
    logs_folder.mkdir(parents=True, exist_ok=True)

    log_file = logs_folder / f"evaluate_{args.mode}.log"
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.info(f"Args: {vars(args)}")

    start = time.time()
    if args.mode == "repeat":
        run_repeat_evaluation(args)
    else:
        run_baseline_evaluation(args)
    logging.info(f"Completed in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
