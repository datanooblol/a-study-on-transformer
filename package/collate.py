import random

import torch

from package.pad_and_mask import pad_and_mask


def crop_pair(
    sequence: list[int], min_len: int = 3, max_len: int | None = None
) -> tuple[list[int], list[int]]:
    """Produce two random contiguous crops from a sequence (positive pair).

    Each crop has a random start and random length in [min_len, max_len],
    preserving the original order.

    Args:
        sequence: The source sequence.
        min_len: Minimum crop length.
        max_len: Maximum crop length (defaults to len(sequence) - 1).

    Returns:
        Two subsequences (crops) of the original sequence.
    """
    n = len(sequence)
    if max_len is None:
        max_len = max(min_len, n - 1)
    max_len = min(max_len, n)
    min_len = min(min_len, n)

    def _random_crop() -> list[int]:
        length = random.randint(min_len, max_len)
        start = random.randint(0, n - length)
        return sequence[start : start + length]

    return _random_crop(), _random_crop()


def contrastive_collate_fn(
    batch: list[list[int]], min_len: int = 3, max_len: int | None = None, pad_value: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function that crops positive pairs and pads them.

    Args:
        batch: List of sequences from the DataLoader.
        min_len: Minimum crop length.
        max_len: Maximum crop length.
        pad_value: Padding value.

    Returns:
        x1_padded, x1_mask, x2_padded, x2_mask
    """
    crops_1, crops_2 = [], []
    for seq in batch:
        c1, c2 = crop_pair(seq, min_len=min_len, max_len=max_len)
        crops_1.append(c1)
        crops_2.append(c2)

    x1_padded, x1_mask = pad_and_mask(crops_1, pad_value=pad_value)
    x2_padded, x2_mask = pad_and_mask(crops_2, pad_value=pad_value)
    return x1_padded, x1_mask, x2_padded, x2_mask


def make_collate_fn(min_len: int = 3, max_len: int | None = None, pad_value: int = 0):
    """Factory to create a collate_fn with fixed crop parameters."""

    def collate_fn(batch):
        return contrastive_collate_fn(batch, min_len=min_len, max_len=max_len, pad_value=pad_value)

    return collate_fn
