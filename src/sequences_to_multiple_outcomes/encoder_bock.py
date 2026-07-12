import torch
import torch.nn as nn
from sequences_to_multiple_outcomes.attention import MultiHeadAttention
from sequences_to_multiple_outcomes.embedding_layers import CategoricalEmbedding, ContinuousEmbedding
from sequences_to_multiple_outcomes.positional_embedding_layers import RoPE, RoPEStrategy
from sequences_to_multiple_outcomes.feedforward import FeedForward
from typing import Callable, Any

class EncoderBlock(nn.Module):
    def __init__(
            self,
            position_emb: Callable[[torch.Tensor], torch.Tensor],
            d_model: int,
            n_heads: int,
            expansion: int = 4,
    ):
        super().__init__()
        self.attention_head = MultiHeadAttention(
            position_emb,
            d_model=d_model,
            n_heads=n_heads
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, expansion)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # X in X + something is just a residual
        X = self.norm1(X + self.attention_head(X, mask))
        X = self.norm2(X + self.ffn(X))
        return X

class EncoderCLS(nn.Module):
    def __init__(
            self,
            fusion: Callable[[list[torch.Tensor], list[torch.Tensor]], torch.Tensor],
            position_emb: Callable[[torch.Tensor], torch.Tensor],
            d_model: int,
            n_heads: int, 
            num_layers: int = 2,
            expansion: int = 4,
        ):
        super().__init__()
        self.fusion = fusion
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.layers = nn.ModuleList([
            # since we implement CLS vector, we need to add max_len+1
            EncoderBlock(position_emb, d_model, n_heads, expansion) for _ in range(num_layers)
        ])

    def forward(self, categorical_features:list[torch.Tensor], continuous_features:list[torch.Tensor], pad_mask: torch.Tensor):
        # pad_mask: True = real token, False = padding -- produced once by collate_fn, threaded through unchanged.
        batch = pad_mask.shape[0]
        cls_col = torch.ones(batch, 1, dtype=torch.bool, device=pad_mask.device)   # CLS is always real -> True
        # concat: (batch, seq_len) -> (batch, cls_col+seq_len)
        pad_mask = torch.cat([cls_col, pad_mask], dim=1)
        x = self.fusion(categorical_features, continuous_features)
        # from (1, 1, d_model) to (batch, 1, d_model)
        cls = self.cls.expand(batch, -1, -1)
        # concat: (batch, seq_len, d_model) -> (batch, cls+seq_len, d_model)
        x = torch.cat([cls, x], dim=1)

        for layer in self.layers:
            x = layer(x, pad_mask)

        return x[:, 0]

