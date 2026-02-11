from __future__ import annotations
import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, weights: list[float] | None = None):
        super().__init__()
        self.register_buffer("w", torch.tensor(weights, dtype=torch.float32) if weights else None)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: [B,C,H,W]
        diff2 = (pred - target) ** 2
        if self.w is None:
            return diff2.mean()
        w = self.w.view(1, -1, 1, 1)
        return (diff2 * w).mean()
