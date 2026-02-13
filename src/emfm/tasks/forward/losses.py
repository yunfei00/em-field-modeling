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


class EHBalancedMSELoss(nn.Module):
    """Balance E/H contributions even when their amplitudes are very different."""

    def __init__(
        self,
        *,
        e_weight: float = 1.0,
        h_weight: float = 1.0,
        weights: list[float] | None = None,
    ):
        super().__init__()
        self.e_weight = float(e_weight)
        self.h_weight = float(h_weight)
        self.register_buffer("w", torch.tensor(weights, dtype=torch.float32) if weights else None)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: [B,C,H,W], Y channel order: 6xE then 6xH
        diff2 = (pred - target) ** 2
        if self.w is not None:
            diff2 = diff2 * self.w.view(1, -1, 1, 1)

        e_loss = diff2[:, :6].mean()
        h_loss = diff2[:, 6:].mean()
        denom = self.e_weight + self.h_weight
        if denom <= 0:
            raise ValueError("e_weight + h_weight must be positive")
        return (self.e_weight * e_loss + self.h_weight * h_loss) / denom
