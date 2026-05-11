import torch
import torch.nn as nn
import torch.nn.functional as F

from package.rope import precompute_freqs, apply_rope


class RoPEAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embedding (RoPE).

    Instead of adding positional encodings to the input, RoPE rotates
    the Query and Key vectors inside attention. This gives the model
    awareness of relative positions between tokens.

    Example:
        >>> attn = RoPEAttention(d_model=64, n_heads=4)
        >>> x = torch.randn(2, 10, 64)       # (batch=2, seq_len=10, d_model=64)
        >>> freqs = precompute_freqs(16, 512) # head_dim=64//4=16
        >>> mask = torch.ones(2, 10, dtype=torch.bool)
        >>> out = attn(x, freqs, mask)
        >>> out.shape
        torch.Size([2, 10, 64])  # same as input

    Args:
        d_model: Total model dimension (will be split across heads).
        n_heads: Number of attention heads. d_model must be divisible by n_heads.
        dropout: Dropout rate applied to attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # dimension per head

        # Single linear layer that produces Q, K, V all at once (more efficient)
        # Input: d_model → Output: 3 * d_model (one d_model each for Q, K, V)
        self.qkv = nn.Linear(d_model, 3 * d_model)

        # Final projection after concatenating all heads back together
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: torch.Tensor | None = None):
        """Forward pass of multi-head attention with RoPE.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            freqs: Precomputed RoPE frequencies (max_len, head_dim//2, 2).
            mask: Boolean mask of shape (batch, seq_len).
                True = real token, False = padding (will be ignored in attention).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        B, S, D = x.shape
        # B = batch size, S = sequence length, D = d_model

        # Project input to Q, K, V simultaneously
        # (B, S, D) → (B, S, 3*D)
        # Then reshape to separate the 3 components and split into heads
        # (B, S, 3*D) → (B, S, 3, n_heads, head_dim)
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)

        # Unbind along dim=2 to get Q, K, V separately
        # Each has shape: (B, S, n_heads, head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Apply RoPE rotation to Q and K (not V — V carries content, not position)
        # Shape stays: (B, S, n_heads, head_dim)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # Transpose to (B, n_heads, S, head_dim) for batched matrix multiply
        # This groups all positions together per head for efficient attention
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        # q, k, v shape: (B, n_heads, S, head_dim)

        # Compute attention scores: Q @ K^T / sqrt(head_dim)
        # (B, n_heads, S, head_dim) @ (B, n_heads, head_dim, S) → (B, n_heads, S, S)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # attn shape: (B, n_heads, S, S)
        # attn[b, h, i, j] = how much token i attends to token j in head h

        if mask is not None:
            # mask shape: (B, S) — need to expand to (B, 1, 1, S)
            # so it broadcasts across n_heads and query positions
            # ~mask inverts: True (real) → False (don't mask), False (pad) → True (mask out)
            attn_mask = ~mask[:, None, None, :]
            # attn_mask shape: (B, 1, 1, S)

            # Set attention scores to -inf where mask is True (padding positions)
            # After softmax, -inf becomes 0 — so padding tokens get zero attention
            attn = attn.masked_fill(attn_mask, float("-inf"))

        # Softmax over the last dim (key positions) to get attention weights
        # Then apply dropout for regularization
        attn = self.dropout(F.softmax(attn, dim=-1))
        # attn shape: (B, n_heads, S, S) — each row sums to 1

        # Multiply attention weights by V to get weighted combination
        # (B, n_heads, S, S) @ (B, n_heads, S, head_dim) → (B, n_heads, S, head_dim)
        out = attn @ v

        # Transpose back: (B, n_heads, S, head_dim) → (B, S, n_heads, head_dim)
        # Then reshape to concatenate all heads: (B, S, n_heads * head_dim) = (B, S, D)
        out = out.transpose(1, 2).reshape(B, S, D)

        # Final linear projection
        # (B, S, D) → (B, S, D)
        return self.out_proj(out)


class EncoderBlock(nn.Module):
    """A single transformer encoder block (pre-norm style).

    Architecture:
        x → LayerNorm → RoPE Attention → Dropout → Add residual
          → LayerNorm → Feed-Forward (FFN) → Dropout → Add residual

    Pre-norm means we normalize BEFORE the sublayer (attention/FFN),
    which tends to train more stably than post-norm.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        ff_dim: Hidden dimension of the feed-forward network (typically 2-4x d_model).
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = RoPEAttention(d_model, n_heads, dropout)
        # Feed-forward network: expand → activate → compress
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),   # expand: d_model → ff_dim
            nn.GELU(),                     # activation (smooth ReLU variant)
            nn.Linear(ff_dim, d_model),   # compress back: ff_dim → d_model
        )
        self.norm1 = nn.LayerNorm(d_model)  # normalize before attention
        self.norm2 = nn.LayerNorm(d_model)  # normalize before FFN
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: torch.Tensor | None = None):
        """Forward pass through one encoder block.

        Args:
            x: Input of shape (batch, seq_len, d_model).
            freqs: RoPE frequencies.
            mask: Padding mask of shape (batch, seq_len).

        Returns:
            Output of shape (batch, seq_len, d_model).
        """
        # Sublayer 1: Attention with residual connection
        # norm → attend → dropout → add back original x
        x = x + self.dropout(self.attn(self.norm1(x), freqs, mask))

        # Sublayer 2: FFN with residual connection
        # norm → FFN → dropout → add back original x
        x = x + self.dropout(self.ff(self.norm2(x)))

        return x
        # output shape: (batch, seq_len, d_model) — same as input


class ContrastiveTransformerEncoder(nn.Module):
    """Transformer encoder that produces fixed-size embeddings for contrastive learning.

    Architecture:
        Token Embedding → N x EncoderBlock (with RoPE) → LayerNorm → Mean Pool → Projection

    Two entry points:
        - forward(): Used during TRAINING. Includes projection head + L2 normalization.
        - encode(): Used during INFERENCE/CLUSTERING. Returns raw embeddings without projection.

    Why two entry points?
        The projection head maps embeddings to a smaller space optimized for the
        contrastive loss. But for downstream tasks (clustering), the richer
        representation BEFORE projection works better.

    Example:
        >>> model = ContrastiveTransformerEncoder(vocab_size=100)
        >>> x = torch.randint(0, 100, (4, 10))  # (batch=4, seq_len=10)
        >>> mask = torch.ones(4, 10, dtype=torch.bool)
        >>> # Training:
        >>> z = model(x, mask)        # shape: (4, 32) — projected + normalized
        >>> # Inference:
        >>> emb = model.encode(x, mask)  # shape: (4, 64) — raw embeddings

    Args:
        vocab_size: Number of unique tokens (interaction types). Include +1 if 0 is padding.
        d_model: Embedding and model dimension. Default 64.
        n_heads: Number of attention heads. d_model must be divisible by this. Default 4.
        num_layers: Number of encoder blocks stacked. Default 2.
        ff_dim: Feed-forward hidden dimension. Default 128.
        proj_dim: Output dimension of the projection head (for contrastive loss). Default 32.
        max_len: Maximum sequence length for RoPE precomputation. Default 512.
        dropout: Dropout rate. Default 0.1.
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

        # Token embedding: maps integer token IDs to dense vectors
        # (batch, seq_len) → (batch, seq_len, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Precompute RoPE frequencies and register as buffer (moves with model to GPU)
        # Shape: (max_len, head_dim//2, 2) where head_dim = d_model // n_heads
        self.register_buffer("freqs", precompute_freqs(d_model // n_heads, max_len))

        # Stack of encoder blocks
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

        # Final layer norm (applied after all encoder blocks)
        self.norm = nn.LayerNorm(d_model)

        # Projection head: maps d_model → proj_dim for contrastive loss
        # Only used during training, discarded for clustering
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),   # d_model → d_model
            nn.ReLU(),                      # non-linearity
            nn.Linear(d_model, proj_dim),  # d_model → proj_dim
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode sequences into embeddings WITHOUT projection head.

        Use this AFTER training when you want to get embeddings for
        clustering, similarity search, or visualization (UMAP, etc.).
        Returns the richer d_model-dimensional representation.

        Args:
            x: Token IDs of shape (batch, seq_len).
            mask: Padding mask of shape (batch, seq_len).
                True = real token, False = padding.

        Returns:
            Embeddings of shape (batch, d_model) — one vector per sequence.
        """
        # Token embedding: (batch, seq_len) → (batch, seq_len, d_model)
        x = self.embedding(x)

        # Pass through each encoder block
        for layer in self.layers:
            x = layer(x, self.freqs, mask)
        # x shape: (batch, seq_len, d_model)

        # Final layer norm
        x = self.norm(x)
        # x shape: (batch, seq_len, d_model)

        # Mean pooling: average token embeddings to get one vector per sequence
        if mask is not None:
            # Zero out padding positions before summing
            # mask.unsqueeze(-1): (batch, seq_len) → (batch, seq_len, 1)
            # Broadcasting: (batch, seq_len, d_model) * (batch, seq_len, 1)
            # This sets padding token vectors to all zeros
            x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            # Numerator: sum over seq_len dim → (batch, d_model)
            # Denominator: count of real tokens per sample → (batch, 1)
            # Result: (batch, d_model) — mean of real tokens only
        else:
            # No mask: simple mean over all positions
            x = x.mean(dim=1)
            # (batch, seq_len, d_model) → (batch, d_model)

        return x
        # output shape: (batch, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass WITH projection head (for contrastive training ONLY).

        Use this DURING training with nt_xent_loss. The projection head
        maps embeddings to a smaller space that the contrastive loss
        optimizes over. Do NOT use this for getting final embeddings —
        use encode() instead.

        Args:
            x: Token IDs of shape (batch, seq_len).
            mask: Padding mask of shape (batch, seq_len).

        Returns:
            L2-normalized embeddings of shape (batch, proj_dim).
        """
        # Get raw embeddings: (batch, d_model)
        x = self.encode(x, mask)

        # Project to smaller space: (batch, d_model) → (batch, proj_dim)
        x = self.proj_head(x)

        # L2 normalize so cosine similarity = dot product
        # Each vector has unit length (norm = 1)
        return F.normalize(x, dim=-1)
        # output shape: (batch, proj_dim)
