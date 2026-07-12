import torch
import torch.nn as nn
from enum import StrEnum

class TaskType(StrEnum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, task_types: list[TaskType]):
        super().__init__()
        self.task_types = task_types
        self.log_vars = nn.Parameter(torch.zeros(len(task_types)))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        # conment out because using tensor this way will not garantee it works in the same computation graph
        # total = torch.tensor(0.0)
        total = torch.zeros(1, device=self.log_vars.device)
        for i, (loss, task_type) in enumerate(zip(losses, self.task_types)):
            precision = torch.exp(-self.log_vars[i]) 
            if task_type == TaskType.CLASSIFICATION:
                total += (precision * loss) + self.log_vars[i]
            else:
                total += (precision * loss) + (0.5 * self.log_vars[i])
            
        return total

    def precisions(self):
        return {i: torch.exp(-self.log_vars[i]).item() for i in range(len(self.task_types))}
        