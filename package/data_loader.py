import torch
from torch.utils.data import Dataset, DataLoader


class InteractionDataset(Dataset):
    """Dataset wrapping a list of interaction sequences (list of lists)."""

    def __init__(self, sequences: list[list[int]]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def create_dataloader(
    sequences: list[list[int]],
    batch_size: int = 32,
    shuffle: bool = True,
    collate_fn=None,
) -> DataLoader:
    dataset = InteractionDataset(sequences)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
