import torch
from torch.utils.data import Dataset, DataLoader


class InteractionDataset(Dataset):
    """A PyTorch Dataset that wraps a list of interaction sequences.

    Each interaction sequence is a variable-length list of integers,
    where each integer represents an interaction type/event.

    Example:
        >>> sequences = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
        >>> dataset = InteractionDataset(sequences)
        >>> dataset[0]
        [1, 2, 3]

    Args:
        sequences: A list of sequences, where each sequence is a list of
            integers representing interaction events.
            Shape conceptually: (num_samples, variable_length)
    """

    def __init__(self, sequences: list[list[int]]):
        # Store the raw sequences as-is (no tensor conversion yet,
        # because each sequence has different length)
        self.sequences = sequences

    def __len__(self):
        """Return the total number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """Return a single sequence (list of ints) at the given index.

        Args:
            idx: Index of the sequence to retrieve.

        Returns:
            A list of integers representing one interaction sequence.
        """
        return self.sequences[idx]


def create_dataloader(
    sequences: list[list[int]],
    batch_size: int = 32,
    shuffle: bool = True,
    collate_fn=None,
) -> DataLoader:
    """Create a DataLoader from a list of interaction sequences.

    The DataLoader handles batching and shuffling. The collate_fn controls
    how individual samples are combined into a batch (e.g., cropping + padding
    for contrastive learning).

    Example:
        >>> from package.collate import make_collate_fn
        >>> loader = create_dataloader(sequences, batch_size=16, collate_fn=make_collate_fn())
        >>> for x1, mask1, x2, mask2 in loader:
        ...     pass  # each is a batch of tensors

    Args:
        sequences: List of interaction sequences (list of list of ints).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data each epoch.
        collate_fn: Function that defines how to combine samples into a batch.
            If None, PyTorch's default collate is used (won't work well with
            variable-length sequences).

    Returns:
        A PyTorch DataLoader ready for iteration.
    """
    # Wrap raw sequences in our Dataset class
    dataset = InteractionDataset(sequences)

    # DataLoader handles batching, shuffling, and applies collate_fn to each batch
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
