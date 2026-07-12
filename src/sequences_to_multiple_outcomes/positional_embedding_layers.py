import torch
import torch.nn as nn
from enum import StrEnum

class RoPEStrategy(StrEnum):
    INTERLEAVED = "interleaved"
    HALF_SPLIT = "half_split"


class RoPE(nn.Module):
    """Rotary positional embedding with a choice of pairing convention.

    strategy="interleaved": pairs adjacent d_models (2i, 2i+1) -- the original RoFormer
        paper's own formulation, same as package/encoder_block.py in this repo.
    strategy="half_split": pairs d_model i with d_model i + d_model//2 -- the GPT-NeoX/LLaMA/
        HuggingFace convention (rotate_half).
    Both give the same relative-position-invariance property; they only differ in
    which two d_modelensions get rotated together.
    """

    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, d_model: int, max_len: int, strategy: RoPEStrategy = RoPEStrategy.INTERLEAVED):
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even to split into rotation pairs, got {d_model}"
        assert strategy in ("interleaved", "half_split"), f"unknown strategy {strategy!r}"
        self.strategy = strategy

        theta = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))   # (d_model//2,) -- one speed per pair
        positions = torch.arange(max_len).float()
        angles = torch.outer(positions, theta)                             # (max_len, d_model//2)

        if strategy == RoPEStrategy.INTERLEAVED:
            # each frequency repeats consecutively: [θ0, θ0, θ1, θ1, ...]
            emb = angles.repeat_interleave(2, dim=-1)                      # (max_len, d_model)
        else:
            # each frequency repeats once per half: [θ0, θ1, ..., θ0, θ1, ...]
            emb = torch.cat([angles, angles], dim=-1)                      # (max_len, d_model)

        self.register_buffer("cos", emb.cos())   # (max_len, d_model)
        self.register_buffer("sin", emb.sin())   # (max_len, d_model)

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        if self.strategy == RoPEStrategy.INTERLEAVED:
            x1 = x[..., 0::2]   # even-indexed elements of each pair
            x2 = x[..., 1::2]   # odd-indexed elements of each pair
            return torch.stack((-x2, x1), dim=-1).flatten(-2)
        else:
            x1, x2 = x.chunk(2, dim=-1)   # first half, second half
            return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_heads, head_d_model)
        seq_len = x.shape[1]
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(2)   # (1, seq_len, 1, d_model) -- broadcasts over batch & heads
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(2)
        return x * cos + self._rotate(x) * sin
