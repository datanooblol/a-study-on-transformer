import torch.nn as nn
import torch.nn.functional as F
import torch
from enum import StrEnum

class CategoricalEmbedding(nn.Module):
    def __init__(self, vocab_size:int, d_model:int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, X)->torch.Tensor:
        return self.emb(X)

class ContinuousEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.emb = nn.Linear(1, d_model)
    
    def forward(self, X)->torch.Tensor:
        """We must convert X from 1D array to 2D for the sequence embedding
        Examples:
        >>> X = torch.tensor([1,2,3])
        >>> X.shape
        torch.Size([3])
        >>> X = X.unsqueeze(-1)
        >>> X.shape
        torch.Size([3, 1])
        """
        return self.emb(X)

class FusionStrategy(StrEnum):
    SUM = "sum"
    CONCAT = "concat"
    SUM_CONCAT = "sum_concat"

class FusionEmbedding(nn.Module):
    def __init__(
            self,
            categorical_vocab_sizes: list[int],
            num_continuous: int,
            d_model: int,
            strategy: FusionStrategy = FusionStrategy.SUM,
    ):
        super().__init__()
        self.strategy = strategy
        self.num_categorical = len(categorical_vocab_sizes)
        self.num_continuous = num_continuous
        self.categorical_emb = nn.ModuleList([
            CategoricalEmbedding(vocab_size=vocab_size, d_model=d_model) for vocab_size in categorical_vocab_sizes
        ])
        self.continuous_emb = nn.ModuleList([
            ContinuousEmbedding(d_model=d_model) for _ in range(num_continuous)
        ])

        if strategy == FusionStrategy.CONCAT:
            concat_dim = d_model * (self.num_categorical + self.num_continuous)
            self.linear_proj = nn.Linear(concat_dim, d_model)

        elif strategy == FusionStrategy.SUM_CONCAT:
            concat_dim = d_model*2
            self.linear_proj = nn.Linear(concat_dim, d_model)
        
        else:
            self.project_down = None

    def forward(self, categorical_features: list[torch.Tensor], continuous_features: list[torch.Tensor]) -> torch.Tensor:
        assert len(categorical_features) == self.num_categorical
        assert len(continuous_features) == self.num_continuous

        cat_vecs = [
            emb(feat) for emb, feat in zip(self.categorical_emb, categorical_features)
        ]
        cont_vecs = [
            emb(feat) for emb, feat in zip(self.continuous_emb, continuous_features)
        ]

        if self.strategy == FusionStrategy.SUM:
            return torch.stack(cat_vecs, dim=0).sum(dim=0) + torch.stack(cont_vecs, dim=0).sum(dim=0)

        if self.strategy == FusionStrategy.CONCAT:
            # cat_vecs + cont_vecs is just a list operation [] + []
            fused = torch.cat(cat_vecs+cont_vecs, dim=-1)
            return self.linear_proj(fused)
        
        fused = torch.cat([
            torch.stack(cat_vecs, dim=0).sum(dim=0), 
            torch.stack(cont_vecs, dim=0).sum(dim=0)
        ], dim=-1)
        return self.linear_proj(fused)
