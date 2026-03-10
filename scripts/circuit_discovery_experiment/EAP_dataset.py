from torch.utils.data import Dataset, DataLoader
import pandas as pd


def collate_EAP(xs):
    """Collate batches. EAP expects (clean, corrupted, positions, labels, clean_id_names)."""
    clean, corrupted, positions, labels, clean_id_names = zip(*xs)
    return (
        list(clean),
        list(corrupted),
        list(positions),
        list(labels),
        list(clean_id_names),
    )


class EAPDataset(Dataset):
    """Dataset for EAP. Always yields (clean_masked, corrupted_masked, masked_position, tokenized_labels, clean_id_names)."""

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        if "clean_id_names" not in self.df.columns:
            raise ValueError(
                "DataFrame must have 'clean_id_names' column. "
            )
        return (
            row["clean_masked"],
            row["corrupted_masked"],
            row["masked_position_after_tokenization"],
            row["tokenized_labels"],
            row["clean_id_names"],
        )
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)