import torch


def pad_and_mask(
    sequences: list[list[int]], pad_value: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences to the longest in the batch and produce an attention mask.

    Args:
        sequences: List of variable-length sequences.
        pad_value: Value used for padding.

    Returns:
        padded: (batch, max_len) LongTensor of padded sequences.
        mask: (batch, max_len) BoolTensor — True for real tokens, False for padding.
    """
    lengths = [len(s) for s in sequences]
    max_len = max(lengths)
    padded = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        padded[i, : lengths[i]] = torch.tensor(seq, dtype=torch.long)
        mask[i, : lengths[i]] = True
    return padded, mask
