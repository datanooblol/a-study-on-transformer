import torch


def pad_and_mask(
    sequences: list[list[int]], pad_value: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences to equal length and produce an attention mask.

    Since sequences in a batch have different lengths, we need to:
    1. Pad shorter sequences with a dummy value (pad_value) so all have the same length.
    2. Create a mask that tells the model which positions are real vs padding.

    Example:
        >>> sequences = [[1, 2, 3], [4, 5]]
        >>> padded, mask = pad_and_mask(sequences)
        >>> padded
        tensor([[1, 2, 3],
                [4, 5, 0]])  # shape: (2, 3) — 0 is padding
        >>> mask
        tensor([[ True,  True,  True],
                [ True,  True, False]])  # shape: (2, 3)

    Args:
        sequences: List of variable-length sequences. Each is a list of ints.
            Conceptually: (batch_size, variable_length)
        pad_value: Integer value used to fill padding positions. Default 0.

    Returns:
        padded: LongTensor of shape (batch_size, max_len).
            All sequences padded to the length of the longest sequence.
        mask: BoolTensor of shape (batch_size, max_len).
            True = real token, False = padding. Used to tell the transformer
            which positions to ignore during attention.
    """
    # Get the length of each sequence
    lengths = [len(s) for s in sequences]

    # Find the longest sequence — all others will be padded to this length
    max_len = max(lengths)

    # Create a (batch_size, max_len) tensor filled entirely with pad_value
    # This is our "canvas" that we'll overwrite with real values
    padded = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)

    # Create a (batch_size, max_len) tensor of all False (padding by default)
    mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)

    for i, seq in enumerate(sequences):
        # Copy real token values into the first `lengths[i]` positions of row i
        # padded[i, :lengths[i]] selects positions 0..length-1 in row i
        padded[i, : lengths[i]] = torch.tensor(seq, dtype=torch.long)

        # Mark those same positions as True (real tokens, not padding)
        mask[i, : lengths[i]] = True

    return padded, mask
