import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Event Embedding ────────────────────────────────────────────────────────────
# each purchase event has 3 features: policy (categorical), age (continuous), price (continuous)
# we project all three to d_model and sum them into one vector per event

class EventEmbedding(nn.Module):
    def __init__(self, num_policies: int, d_model: int):
        super().__init__()
        # policy is categorical — lookup table maps policy id → d_model vector
        self.policy_emb = nn.Embedding(num_embeddings=num_policies, embedding_dim=d_model)
        # age and price are continuous scalars — project each to d_model
        self.age_proj   = nn.Linear(1, d_model)
        self.price_proj = nn.Linear(1, d_model)

    def forward(
        self,
        policy: torch.Tensor,  # (batch, seq_len)         — integer policy ids
        age:    torch.Tensor,  # (batch, seq_len, 1)      — continuous age values
        price:  torch.Tensor,  # (batch, seq_len, 1)      — continuous price values
    ) -> torch.Tensor:
        # sum all three projections — each contributes equally to the event vector
        # output: (batch, seq_len, d_model)
        return self.policy_emb(policy) + self.age_proj(age) + self.price_proj(price)


# ── RoPE ──────────────────────────────────────────────────────────────────────
# identical to gpt_decoder_block.py — encodes relative position within the purchase sequence

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
# identical to gpt_decoder_block.py — causal mask ensures each event only sees past events

class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
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

        # causal mask: position i cannot attend to position j > i
        causal_mask = torch.ones(seq_len, seq_len, device=x.device).triu(diagonal=1).bool()
        scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))

        attn = scores.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        return self.out_proj(out)


# ── FeedForward ────────────────────────────────────────────────────────────────
# identical to gpt_decoder_block.py

class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.ff1 = nn.Linear(d_model, d_model * expansion)
        self.ff2 = nn.Linear(d_model * expansion, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff2(F.relu(self.ff1(x)))


# ── Insurance Decoder Block ────────────────────────────────────────────────────
# same structure as GPTDecoderBlock in gpt_decoder_block.py
# adds EventEmbedding at the input and three prediction heads at the output:
#   - policy head  → next policy (classification)
#   - age head     → next age at purchase (regression)
#   - price head   → next price at purchase (regression)

class InsuranceDecoderBlock(nn.Module):
    def __init__(self, num_policies: int, d_model: int, n_heads: int, max_len: int, expansion: int = 4):
        super().__init__()
        # input: fuse policy + age + price into one event vector
        self.event_emb = EventEmbedding(num_policies, d_model)

        # transformer block: same as GPTDecoderBlock
        self.attn  = MaskedSelfAttention(d_model, n_heads, max_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn   = FeedForward(d_model, expansion)
        self.norm2 = nn.LayerNorm(d_model)

        # output: three separate heads, one per feature to predict
        self.policy_head = nn.Linear(d_model, num_policies)  # logits over all policies
        self.age_head    = nn.Linear(d_model, 1)             # scalar age prediction
        self.price_head  = nn.Linear(d_model, 1)             # scalar price prediction

    def forward(
        self,
        policy:   torch.Tensor,                    # (batch, seq_len)     — policy ids
        age:      torch.Tensor,                    # (batch, seq_len, 1)  — age values
        price:    torch.Tensor,                    # (batch, seq_len, 1)  — price values
        pad_mask: torch.Tensor | None = None       # (batch, seq_len)     — True = padding
    ):
        # fuse the three input features into one vector per event
        x = self.event_emb(policy, age, price)    # (batch, seq_len, d_model)

        # decoder: each event attends to all previous events only (causal mask)
        x = self.norm1(x + self.attn(x, pad_mask))
        x = self.norm2(x + self.ffn(x))
        # x: (batch, seq_len, d_model)
        # position i holds a representation built from events 0..i
        # so predicting from position i gives the forecast for event i+1

        # three prediction heads applied to every position in the sequence
        policy_logits = self.policy_head(x)       # (batch, seq_len, num_policies)
        age_pred      = self.age_head(x)           # (batch, seq_len, 1)
        price_pred    = self.price_head(x)         # (batch, seq_len, 1)

        return policy_logits, age_pred, price_pred
