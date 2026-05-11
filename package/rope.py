import torch


def precompute_freqs(dim: int, max_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute rotation frequencies for RoPE.

    Returns:
        freqs: (max_len, dim//2, 2) — cos and sin values.
    """
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_len).float()
    angles = torch.outer(positions, theta)  # (max_len, dim//2)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)  # (max_len, dim//2, 2)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to input tensor.

    Args:
        x: (batch, seq_len, n_heads, dim) or (batch, n_heads, seq_len, dim)
           Assumes last dim is the head dimension to rotate.
        freqs: precomputed (max_len, dim//2, 2)

    Returns:
        Rotated tensor, same shape as x.
    """
    seq_len = x.shape[-2]
    freqs = freqs[:seq_len].to(x.device)  # (seq_len, dim//2, 2)

    # Split x into pairs for rotation
    x_pairs = x.unflatten(-1, (-1, 2))  # (..., seq_len, dim//2, 2)
    x0, x1 = x_pairs[..., 0], x_pairs[..., 1]

    cos = freqs[..., 0]  # (seq_len, dim//2)
    sin = freqs[..., 1]

    # Rotate: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos

    return torch.stack([out0, out1], dim=-1).flatten(-2)
