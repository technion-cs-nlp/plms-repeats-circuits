import pandas as pd
import ast

from plms_repeats_circuits.utils.model_utils import mask_protein
from plms_repeats_circuits.utils.esm_utils import replace_short_mask_with_mask_token


def create_induction_dataset_pandas(df, total_n_samples, random_state, tokenizer, metric):
    """Process induction dataset. Expects df to be pre-filtered (e.g. max seq length 400)."""
    if len(df) < total_n_samples:
        raise ValueError(f"Not enough samples: {len(df)} rows available, but {total_n_samples} requested.")
    if total_n_samples < len(df):
        sampled_df = df.sample(n=total_n_samples, random_state=random_state)
    else:
        print(f"total_n_samples {total_n_samples} >= dataframe size {len(df)}. Using all {len(df)} samples.")
        sampled_df = df
    
    def process_row(row):
        clean = row['seq']
        name = f"{row['cluster_id']}_{row['rep_id']}_{row['repeat_key']}"
        corrupted = row['corrupted_sequence']
        masked_position = int(row['masked_position'])
        clean_masked = mask_protein(clean, masked_position, tokenizer)
        corrupted_masked = mask_protein(corrupted, masked_position, tokenizer)
        corrupted_masked = replace_short_mask_with_mask_token(corrupted_masked, tokenizer)
        labels = [clean[masked_position]]
        if metric == "logit_diff":
            if 'corrupted_amino_acid_type' in df.columns:
                labels.append(row['corrupted_amino_acid_type'])
            elif 'replacments' in df.columns:
                replacements = ast.literal_eval(row['replacments'])
                if len(replacements) != 1:
                    raise ValueError("got unexpected corrupted amino acids to logit diff metric")
                labels.append(replacements[0])
            else:
                raise ValueError("missing support for corrupted amino acid column")
        tokenized_labels = tokenizer(
            labels,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False
        )['input_ids'].squeeze(-1).tolist()
        return pd.Series({
            'clean_masked': clean_masked,
            'corrupted_masked': corrupted_masked,
            'masked_position_after_tokenization': masked_position + 1,
            'tokenized_labels': tokenized_labels,
            'clean_id_names': name
        })
    
    return sampled_df.apply(process_row, axis=1)
