import torch


def precompute_freqs(dim: int, max_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute rotation frequencies (cos and sin) for Rotary Position Embedding.

    RoPE encodes position by rotating pairs of dimensions in Q and K vectors.
    This function precomputes the rotation angles for all possible positions
    up to max_len, so we don't recompute them every forward pass.

    The formula for each frequency:
        theta_i = 1 / (base ^ (2i / dim))   for i = 0, 1, ..., dim/2 - 1

    Then for each position p:
        angle(p, i) = p * theta_i

    Example:
        >>> freqs = precompute_freqs(dim=16, max_len=100)
        >>> freqs.shape
        torch.Size([100, 8, 2])  # (max_len, dim//2, 2) where 2 = [cos, sin]

    Args:
        dim: Dimension of each attention head (head_dim). Must be even.
        max_len: Maximum sequence length to precompute for.
        base: Base for the frequency calculation. Higher = slower rotation.
            Default 10000.0 (from the original Transformer paper).

    Returns:
        Tensor of shape (max_len, dim//2, 2) containing [cos, sin] pairs
        for each position and each frequency band.
    """
    # theta_i = 1 / (base ^ (2i / dim)) for i = 0, 2, 4, ..., dim-2
    # torch.arange(0, dim, 2) gives [0, 2, 4, ..., dim-2] — shape: (dim//2,)
    # Dividing by dim and exponentiating gives the frequency for each pair
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    # theta shape: (dim//2,)

    # positions = [0, 1, 2, ..., max_len-1] — one entry per sequence position
    positions = torch.arange(max_len).float()
    # positions shape: (max_len,)

    # Outer product: each position multiplied by each frequency
    # angles[p, i] = position_p * theta_i
    angles = torch.outer(positions, theta)
    # angles shape: (max_len, dim//2)

    # Stack cos and sin along a new last dimension
    # freqs[p, i, 0] = cos(angle), freqs[p, i, 1] = sin(angle)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)
    # return shape: (max_len, dim//2, 2)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Embedding to Q or K tensor.

    RoPE rotates pairs of adjacent dimensions using position-dependent angles.
    For each pair (x0, x1) at position p:
        rotated_x0 = x0 * cos(angle_p) - x1 * sin(angle_p)
        rotated_x1 = x0 * sin(angle_p) + x1 * cos(angle_p)

    This is a 2D rotation matrix applied independently to each pair of dims.

    Example:
        >>> # Q after linear projection, reshaped to per-head
        >>> q = torch.randn(2, 10, 4, 16)  # (batch=2, seq_len=10, n_heads=4, head_dim=16)
        >>> freqs = precompute_freqs(dim=16, max_len=512)
        >>> q_rotated = apply_rope(q, freqs)
        >>> q_rotated.shape
        torch.Size([2, 10, 4, 16])  # same shape, but values are rotated by position

    Args:
        x: Input tensor. Expected shape: (batch, seq_len, n_heads, head_dim)
            The last dim (head_dim) must be even since we rotate in pairs.
        freqs: Precomputed frequencies of shape (max_len, head_dim//2, 2).

    Returns:
        Rotated tensor with the same shape as x.
    """
    # Get the sequence length from x to slice the precomputed freqs
    # x shape: (batch, seq_len, n_heads, head_dim)
    seq_len = x.shape[-2]
    # We only need freqs for positions 0..seq_len-1
    freqs = freqs[:seq_len].to(x.device)
    # freqs shape: (seq_len, head_dim//2, 2)

    # Split the last dimension into pairs: head_dim → (head_dim//2, 2)
    # e.g., head_dim=16 becomes 8 pairs of 2
    x_pairs = x.unflatten(-1, (-1, 2))
    # x_pairs shape: (batch, seq_len, n_heads, head_dim//2, 2)

    # Separate the two elements of each pair
    x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
    # x0 shape: (batch, seq_len, n_heads, head_dim//2)
    # x1 shape: (batch, seq_len, n_heads, head_dim//2)

    # Extract cos and sin from freqs
    cos = freqs[..., 0]  # (seq_len, head_dim//2)
    sin = freqs[..., 1]  # (seq_len, head_dim//2)
    # Broadcasting: cos/sin are (seq_len, head_dim//2)
    # x0/x1 are (batch, seq_len, n_heads, head_dim//2)
    # cos/sin will broadcast over batch and n_heads dimensions

    # Apply 2D rotation to each pair
    out0 = x0 * cos - x1 * sin  # (batch, seq_len, n_heads, head_dim//2)
    out1 = x0 * sin + x1 * cos  # (batch, seq_len, n_heads, head_dim//2)

    # Interleave the rotated pairs back: stack along last dim then flatten
    # stack → (batch, seq_len, n_heads, head_dim//2, 2)
    # flatten(-2) merges last two dims → (batch, seq_len, n_heads, head_dim)
    return torch.stack([out0, out1], dim=-1).flatten(-2)
    # return shape: (batch, seq_len, n_heads, head_dim) — same as input x
