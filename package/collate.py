import random

import torch

from package.pad_and_mask import pad_and_mask


def crop_pair(
    sequence: list[int], min_len: int = 3, max_len: int | None = None
) -> tuple[list[int], list[int]]:
    """Produce two random contiguous crops from a sequence (positive pair).

    Contrastive learning needs pairs of "views" from the same sample.
    Here, each view is a random contiguous subsequence (crop) that preserves
    the original order of elements.

    Example:
        >>> sequence = [1, 2, 3, 4, 5, 6, 7]
        >>> c1, c2 = crop_pair(sequence, min_len=3, max_len=5)
        >>> # c1 might be [2, 3, 4, 5] (start=1, length=4)
        >>> # c2 might be [4, 5, 6] (start=3, length=3)
        >>> # Both preserve original order, both are contiguous

    Args:
        sequence: The source sequence to crop from.
        min_len: Minimum number of elements in each crop.
        max_len: Maximum number of elements in each crop.
            Defaults to len(sequence) - 1 so crops don't always cover everything.

    Returns:
        A tuple of two lists (crop1, crop2), each a contiguous subsequence.
    """
    n = len(sequence)

    # Default max_len to n-1 so we don't always get the full sequence
    if max_len is None:
        max_len = max(min_len, n - 1)

    # Clamp max_len and min_len to not exceed sequence length
    max_len = min(max_len, n)
    min_len = min(min_len, n)

    def _random_crop() -> list[int]:
        # Pick a random crop length between min_len and max_len (inclusive)
        length = random.randint(min_len, max_len)

        # Pick a random start index such that the crop fits within the sequence
        # start can range from 0 to (n - length) inclusive
        start = random.randint(0, n - length)

        # Slice the sequence — this preserves order
        return sequence[start : start + length]

    # Generate two independent random crops from the same sequence
    return _random_crop(), _random_crop()


def contrastive_collate_fn(
    batch: list[list[int]], min_len: int = 3, max_len: int | None = None, pad_value: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function that creates contrastive pairs from a batch of sequences.

    For each sequence in the batch:
    1. Crop two random subsequences (positive pair).
    2. Pad all crop-1's together and all crop-2's together.

    This is passed to DataLoader as collate_fn so each batch yields
    two sets of padded sequences ready for the transformer.

    Example:
        >>> batch = [[1,2,3,4,5], [6,7,8,9], [10,11,12,13,14,15]]
        >>> x1_padded, x1_mask, x2_padded, x2_mask = contrastive_collate_fn(batch)
        >>> x1_padded.shape  # (3, max_crop_len_in_crop1_group)
        >>> x2_padded.shape  # (3, max_crop_len_in_crop2_group)

    Args:
        batch: List of sequences from the DataLoader (one batch worth).
        min_len: Minimum crop length passed to crop_pair.
        max_len: Maximum crop length passed to crop_pair.
        pad_value: Value used for padding shorter crops.

    Returns:
        x1_padded: LongTensor (batch_size, max_len_1) — padded first crops.
        x1_mask: BoolTensor (batch_size, max_len_1) — mask for first crops.
        x2_padded: LongTensor (batch_size, max_len_2) — padded second crops.
        x2_mask: BoolTensor (batch_size, max_len_2) — mask for second crops.
    """
    crops_1, crops_2 = [], []

    # For each sequence in the batch, generate a positive pair of crops
    for seq in batch:
        c1, c2 = crop_pair(seq, min_len=min_len, max_len=max_len)
        crops_1.append(c1)
        crops_2.append(c2)

    # Pad each group of crops independently (they may have different max lengths)
    x1_padded, x1_mask = pad_and_mask(crops_1, pad_value=pad_value)
    x2_padded, x2_mask = pad_and_mask(crops_2, pad_value=pad_value)

    return x1_padded, x1_mask, x2_padded, x2_mask


def make_collate_fn(min_len: int = 3, max_len: int | None = None, pad_value: int = 0):
    """Factory that creates a collate_fn with fixed crop parameters.

    We need this because DataLoader's collate_fn only accepts one argument (batch).
    This factory "bakes in" the crop parameters and returns a function with
    the correct signature.

    Example:
        >>> collate_fn = make_collate_fn(min_len=3, max_len=8)
        >>> loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

    Args:
        min_len: Minimum crop length.
        max_len: Maximum crop length.
        pad_value: Padding value.

    Returns:
        A collate function compatible with PyTorch DataLoader.
    """

    def collate_fn(batch):
        return contrastive_collate_fn(batch, min_len=min_len, max_len=max_len, pad_value=pad_value)

    return collate_fn
