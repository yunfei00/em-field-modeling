from __future__ import annotations
import torch

@torch.no_grad()
def rmse_per_channel(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    # [B,C,H,W]
    diff2 = (pred - target) ** 2
    mse_c = diff2.mean(dim=(0,2,3))
    rmse_c = torch.sqrt(mse_c).detach().cpu().tolist()
    return rmse_c
