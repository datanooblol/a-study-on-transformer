import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.expand = nn.Linear(d_model, d_model * expansion)
        self.collapse = nn.Linear(d_model * expansion, d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.collapse(F.gelu(self.expand(X)))