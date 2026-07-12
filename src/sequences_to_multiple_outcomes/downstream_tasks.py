import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskModel(nn.Module):
    """
    Args:
        backbone (nn.Module) : shared transformer backbone
        d_model (int) : embedding dimensions
        vocab_size (int) : actual vocab size including PAD and other special tokens
    """
    def __init__(self, backbone, d_model: int, vocab_size: int):
        super().__init__()
        self.backbone = backbone
        self.item_head = nn.Linear(d_model, vocab_size)
        self.age_head = nn.Linear(d_model, 1)
        self.price_head = nn.Linear(d_model, 1)

    def forward(
            self,
            categorical_features: list[torch.Tensor],
            continuous_features: list[torch.Tensor],
            pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # pad_mask: True = real token, False = padding
        representations = self.backbone(categorical_features, continuous_features, pad_mask)
        ages_norm = continuous_features[0]
        lengths = pad_mask.sum(dim=1)
        batch_idx = torch.arange(ages_norm.shape[0], device=ages_norm.device)
        current_age = ages_norm[batch_idx, lengths-1, 0]
        delta_age = F.softplus(self.age_head(representations).squeeze(-1))
        age_pred = (current_age + delta_age).unsqueeze(-1)
        return (
            self.item_head(representations),
            age_pred,
            self.price_head(representations),
        )
