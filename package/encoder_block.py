import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # converts token ids (batch, seq_len) → vectors (batch, seq_len, d_model)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


# RoPE is copied as-is into decoder_block.py — identical class, same logic
class RoPE(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        # one rotation speed per dimension pair — small index = fast, large index = slow
        theta = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # token positions in the sequence: [0, 1, 2, ..., max_len-1]
        positions = torch.arange(max_len).float()
        # rotation angle for every (position, dimension pair) combination
        angles = torch.outer(positions, theta)
        # store cos and sin side by side: (max_len, dim//2, 2)
        self.register_buffer("freqs", torch.stack([angles.cos(), angles.sin()], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, n_heads, head_dim)
        seq_len = x.shape[1]
        # slice to actual seq_len, unsqueeze batch and n_heads dims for broadcasting
        # (seq_len, head_dim//2, 2) → (1, seq_len, 1, head_dim//2, 2)
        freqs = self.freqs[:seq_len].unsqueeze(0).unsqueeze(2)
        # group last dim into pairs: head_dim → (head_dim//2, 2)
        x_pairs = x.unflatten(-1, (-1, 2))
        x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
        cos, sin = freqs[..., 0], freqs[..., 1]
        # apply 2D rotation formula to each pair
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        # interleave back and flatten to original shape
        return torch.stack([out0, out1], dim=-1).flatten(-2)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RoPE(self.head_dim, max_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # in the decoder this same logic is split into two classes:
        # MaskedSelfAttention (adds causal mask) and CrossAttention (Q from decoder, K/V from encoder)
        batch, seq_len, d_model = x.shape

        # project and split d_model into n_heads chunks of head_dim
        # (batch, seq_len, d_model) → (batch, seq_len, n_heads, head_dim)
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)

        # apply RoPE to Q and K only — shape unchanged
        q = self.rope(q)
        k = self.rope(k)

        # move n_heads next to batch so matmul runs all heads in parallel
        # (batch, seq_len, n_heads, head_dim) → (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # each head computes its own attention scores
        # (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            # mask is (batch, seq_len) — True means padding, set those to -inf
            # in the decoder this becomes two separate masks: tgt_pad_mask (padding) + causal_mask (future tokens)
            scores = scores.masked_fill(mask[:, None, None, :], float("-inf"))

        attn = scores.softmax(dim=-1)

        # weighted sum of V per head
        # (batch, n_heads, seq_len, head_dim) → (batch, seq_len, d_model)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        return self.out_proj(out)


# FeedForward is copied as-is into decoder_block.py — identical class, same logic
class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        # expand: (batch, seq_len, d_model) → (batch, seq_len, d_model * expansion)
        self.ff1 = nn.Linear(d_model, d_model * expansion)
        # contract: (batch, seq_len, d_model * expansion) → (batch, seq_len, d_model)
        self.ff2 = nn.Linear(d_model * expansion, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ReLU introduces non-linearity — without it ff1+ff2 collapse into one linear layer
        return self.ff2(F.relu(self.ff1(x)))


# EncoderBlock output (batch, src_len, d_model) is passed as enc_out into DecoderBlock.forward in decoder_block.py
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int, expansion: int = 4):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, max_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, expansion)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # attention + residual + normalize
        x = self.norm1(x + self.attn(x, mask))
        # FFN + residual + normalize
        x = self.norm2(x + self.ffn(x))
        return x
