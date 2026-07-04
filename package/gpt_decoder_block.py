import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RoPE ──────────────────────────────────────────────────────────────────────
# identical to encoder_block.py and decoder_block.py

class RoPE(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        theta = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_len).float()
        angles = torch.outer(positions, theta)
        self.register_buffer("freqs", torch.stack([angles.cos(), angles.sin()], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        freqs = self.freqs[:seq_len].unsqueeze(0).unsqueeze(2)
        x_pairs = x.unflatten(-1, (-1, 2))
        x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
        cos, sin = freqs[..., 0], freqs[..., 1]
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        return torch.stack([out0, out1], dim=-1).flatten(-2)


# ── Masked Self-Attention ──────────────────────────────────────────────────────
# identical to MaskedSelfAttention in decoder_block.py
# no CrossAttention here — GPT only attends to its own previous tokens, no encoder to look at

class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RoPE(self.head_dim, max_len)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)

        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        causal_mask = torch.ones(seq_len, seq_len, device=x.device).triu(diagonal=1).bool()
        scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))

        attn = scores.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        return self.out_proj(out)


# ── FeedForward ────────────────────────────────────────────────────────────────
# identical to encoder_block.py and decoder_block.py

class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.ff1 = nn.Linear(d_model, d_model * expansion)
        self.ff2 = nn.Linear(d_model * expansion, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff2(F.relu(self.ff1(x)))


# ── GPT Decoder Block ──────────────────────────────────────────────────────────
# same as EncoderBlock in encoder_block.py but with a causal mask added
# same as DecoderBlock in decoder_block.py but with CrossAttention + norm2 removed
# two sub-layers: masked self-attention + FFN (no cross-attention — nothing to cross-attend to)

class GPTDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int, expansion: int = 4):
        super().__init__()
        self.attn = MaskedSelfAttention(d_model, n_heads, max_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, expansion)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # masked self-attention: each token can only see itself and previous tokens
        x = self.norm1(x + self.attn(x, pad_mask))
        # FFN: position-wise transformation
        x = self.norm2(x + self.ffn(x))
        return x
