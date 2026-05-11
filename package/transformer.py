import torch
import torch.nn as nn
import torch.nn.functional as F

from package.rope import precompute_freqs, apply_rope


class RoPEAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            x: (batch, seq_len, d_model)
            freqs: precomputed RoPE frequencies
            mask: (batch, seq_len) — True for real tokens, False for padding
        """
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, S, n_heads, head_dim)

        # Apply RoPE to Q and K
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # Transpose to (B, n_heads, S, head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            # mask: (B, S) → (B, 1, 1, S) — False positions get -inf
            attn_mask = ~mask[:, None, None, :]
            attn = attn.masked_fill(attn_mask, float("-inf"))

        attn = self.dropout(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = RoPEAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: torch.Tensor | None = None):
        x = x + self.dropout(self.attn(self.norm1(x), freqs, mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class ContrastiveTransformerEncoder(nn.Module):
    """Transformer encoder for producing sequence embeddings.

    Architecture:
        Token embedding → N x (RoPE attention + FFN) → Mean pool → Projection head
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        proj_dim: int = 32,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer("freqs", precompute_freqs(d_model // n_heads, max_len))
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim),
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode sequences into embeddings (without projection head).

        Use this for clustering after training.
        """
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, self.freqs, mask)
        x = self.norm(x)

        # Mean pool over real tokens
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with projection head (for contrastive training)."""
        x = self.encode(x, mask)
        x = self.proj_head(x)
        return F.normalize(x, dim=-1)
