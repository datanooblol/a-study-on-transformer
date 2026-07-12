import torch
import torch.nn as nn
from typing import Callable

class MultiHeadAttention(nn.Module):
    def __init__(self, position_emb:Callable[[torch.Tensor], torch.Tensor], d_model:int, n_heads:int):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wQ = nn.Linear(d_model, d_model)
        self.wK = nn.Linear(d_model, d_model)
        self.wV = nn.Linear(d_model, d_model)
        self.wO = nn.Linear(d_model, d_model)
        self.position_emb = position_emb

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        # mask: True = real token, False = padding (matches collate_fn's convention).
        # Inverted here, at the one point it's actually needed -- masked_fill wants True = block this position.
        batch, seq_len, d_model = X.shape
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, head_dim)
        # write the example shape of n_heads/head_dim later
        q = self.wQ(X).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wK(X).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.wV(X).view(batch, seq_len, self.n_heads, self.head_dim)

        q = self.position_emb(q)
        k = self.position_emb(k)

        # (batch, seq_len, n_heads, head_dim) -> (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        normalized_term = self.head_dim ** 0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) / normalized_term
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)

        out = torch.matmul(attn, v)
        # (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, n_heads, head_dim)
        out = out.transpose(1, 2)
        # (batch, seq_len, n_heads, head_dim) -> (batch, seq_len, d_model(n_heads, head_dim))
        out = out.contiguous().view(batch, seq_len, d_model)

        return self.wO(out)

