import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RoPE ──────────────────────────────────────────────────────────────────────
# identical to RoPE in encoder_block.py — only used in MaskedSelfAttention (self-attention), not CrossAttention

class RoPE(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        theta = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_len).float()
        angles = torch.outer(positions, theta)
        self.register_buffer("freqs", torch.stack([angles.cos(), angles.sin()], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, n_heads, head_dim)
        seq_len = x.shape[1]
        freqs = self.freqs[:seq_len].unsqueeze(0).unsqueeze(2)
        x_pairs = x.unflatten(-1, (-1, 2))
        x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
        cos, sin = freqs[..., 0], freqs[..., 1]
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        return torch.stack([out0, out1], dim=-1).flatten(-2)


# ── Masked Self-Attention ──────────────────────────────────────────────────────
# same structure as MultiHeadAttention in encoder_block.py — adds a causal mask on top of the padding mask

class MaskedSelfAttention(nn.Module):
    """
    First sub-layer of the decoder.

    Same as encoder self-attention but with a causal mask applied —
    each token can only attend to itself and tokens before it.
    This prevents the decoder from "cheating" by looking at future tokens
    during training.

    Example: when predicting token at position 3,
    it can see positions 0, 1, 2 but NOT 4, 5, 6 ...
    """
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

        # all three projections come from x (the decoder input itself)
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)

        # apply RoPE to Q and K to encode position info
        q = self.rope(q)
        k = self.rope(k)

        # move n_heads next to batch so matmul runs all heads in parallel
        # (batch, seq_len, n_heads, head_dim) → (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # how much should each token attend to every other token?
        # (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # causal mask: build a (seq_len, seq_len) upper triangle of True values
        # True = future position → set to -inf so softmax gives it ~0 weight
        causal_mask = torch.ones(seq_len, seq_len, device=x.device).triu(diagonal=1).bool()
        # [None, None, :, :] adds batch and n_heads dims so it broadcasts over scores (batch, n_heads, seq_len, seq_len)
        scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        # padding mask: True means this token is just padding, not real content → also -inf
        # [:, None, None, :] expands (batch, seq_len) → (batch, 1, 1, seq_len) to broadcast over n_heads and tgt_len
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))

        # convert scores to probabilities — each row sums to 1
        attn = scores.softmax(dim=-1)

        # weighted sum of V: (batch, n_heads, seq_len, head_dim)
        out = torch.matmul(attn, v)
        # merge heads back: (batch, n_heads, seq_len, head_dim) → (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        return self.out_proj(out)


# ── Cross-Attention ────────────────────────────────────────────────────────────
# same projections as MultiHeadAttention in encoder_block.py — only difference is k_proj/v_proj receive enc_out not x

class CrossAttention(nn.Module):
    """
    Second sub-layer of the decoder.

    This is where the decoder "looks at" the encoder output.
    - Q comes from the decoder (what the decoder is currently building)
    - K and V come from the encoder output (the full encoded source sequence)

    This lets every decoder token attend to every encoder token,
    so the decoder can decide which parts of the input to focus on
    when generating each output token.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Q comes from decoder, K and V come from encoder
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # no RoPE here — Q and K come from different sequences
        # (decoder vs encoder) so relative positions don't apply the same way

    def forward(
        self,
        x: torch.Tensor,                          # decoder states:  (batch, tgt_len, d_model)
        enc_out: torch.Tensor,                    # encoder output:  (batch, src_len, d_model)
        enc_pad_mask: torch.Tensor | None = None  # padding mask for encoder input
    ) -> torch.Tensor:
        batch, tgt_len, d_model = x.shape
        src_len = enc_out.shape[1]

        # Q comes from decoder (x), K and V come from encoder (enc_out)
        # this is the key difference from self-attention — k_proj and v_proj receive enc_out not x
        q = self.q_proj(x).view(batch, tgt_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(enc_out).view(batch, src_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(enc_out).view(batch, src_len, self.n_heads, self.head_dim).transpose(1, 2)
        # q: (batch, n_heads, tgt_len, head_dim)  ← from decoder
        # k: (batch, n_heads, src_len, head_dim)  ← from encoder
        # v: (batch, n_heads, src_len, head_dim)  ← from encoder

        # how much does each decoder token want to attend to each encoder token?
        # result is (batch, n_heads, tgt_len, src_len) — note tgt vs src, not square like self-attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # mask out padding positions in the encoder input so decoder doesn't attend to them
        if enc_pad_mask is not None:
            scores = scores.masked_fill(enc_pad_mask[:, None, None, :], float("-inf"))

        attn = scores.softmax(dim=-1)

        # weighted sum of encoder V using decoder's attention weights
        # (batch, n_heads, tgt_len, head_dim) → (batch, tgt_len, d_model)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, tgt_len, d_model)

        return self.out_proj(out)


# ── FeedForward ────────────────────────────────────────────────────────────────
# identical to FeedForward in encoder_block.py — same class, same logic

class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        # expand: (batch, seq_len, d_model) → (batch, seq_len, d_model * expansion)
        self.ff1 = nn.Linear(d_model, d_model * expansion)
        # contract: (batch, seq_len, d_model * expansion) → (batch, seq_len, d_model)
        self.ff2 = nn.Linear(d_model * expansion, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff2(F.relu(self.ff1(x)))


# ── Decoder Block ──────────────────────────────────────────────────────────────
# extends EncoderBlock in encoder_block.py: EncoderBlock has 2 sub-layers (self-attn + FFN), DecoderBlock adds a 3rd (cross-attn)
# enc_out fed into cross_attn here is the output of EncoderBlock.forward

class DecoderBlock(nn.Module):
    """
    One full decoder layer from the original "Attention is All You Need" paper.

    Three sub-layers:
    1. Masked Self-Attention  — decoder attends to its own previous tokens only
    2. Cross-Attention        — decoder attends to the full encoder output
    3. FeedForward            — position-wise transformation

    Each sub-layer is wrapped with a residual connection and LayerNorm:
        output = LayerNorm(x + sublayer(x))
    """
    def __init__(self, d_model: int, n_heads: int, max_len: int, expansion: int = 4):
        super().__init__()
        self.masked_attn = MaskedSelfAttention(d_model, n_heads, max_len)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = CrossAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, expansion)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,                              # decoder input:  (batch, tgt_len, d_model)
        enc_out: torch.Tensor,                        # encoder output: (batch, src_len, d_model)
        tgt_pad_mask: torch.Tensor | None = None,     # padding mask for decoder input
        src_pad_mask: torch.Tensor | None = None      # padding mask for encoder input
    ) -> torch.Tensor:
        # 1. masked self-attention: decoder looks at its own past tokens only
        x = self.norm1(x + self.masked_attn(x, tgt_pad_mask))

        # 2. cross-attention: decoder looks at encoder output
        x = self.norm2(x + self.cross_attn(x, enc_out, src_pad_mask))

        # 3. FFN: position-wise transformation
        x = self.norm3(x + self.ffn(x))

        return x
