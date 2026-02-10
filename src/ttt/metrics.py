from __future__ import annotations

import torch

def _to_2d(t: torch.Tensor) -> torch.Tensor:
    # (B, D) keeps; (B, ...) flatten to (B, D)
    if t.ndim == 1:
        return t.unsqueeze(1)
    if t.ndim == 2:
        return t
    return t.view(t.shape[0], -1)

@torch.no_grad()
def batch_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> dict:
    """
    Compute metrics for one batch. Returns sums and counts so caller can aggregate.
    Metrics are computed over batch dimension and per-output-dim (flattened).
    """
    pred2 = _to_2d(pred)
    tgt2 = _to_2d(target)

    err = pred2 - tgt2
    se = err.pow(2)  # (B, D)

    # MSE/RMSE
    mse_dim_sum = se.mean(dim=0)  # (D,)
    rmse_dim = torch.sqrt(mse_dim_sum + eps)  # (D,)
    rmse_mean = rmse_dim.mean()

    # Relative error (L2) per dim: ||e|| / (||y|| + eps)
    # computed over batch for each dim (treat each dim independently)
    num = torch.sqrt(se.mean(dim=0) + eps)
    den = torch.sqrt((tgt2.pow(2)).mean(dim=0) + eps)
    rel_dim = num / den
    rel_mean = rel_dim.mean()

    return {
        "rmse_mean": float(rmse_mean.item()),
        "rel_mean": float(rel_mean.item()),
        "rmse_dim": rmse_dim.detach().cpu().tolist(),
        "rel_dim": rel_dim.detach().cpu().tolist(),
        "out_dim": int(pred2.shape[1]),
    }

def pick_score(metrics: dict, metric_name: str) -> float:
    """
    Return a scalar score to track for best checkpoint.
    Lower is better (rmse/rel are losses).
    """
    if metric_name not in metrics:
        raise KeyError(f"metric '{metric_name}' not found in metrics keys: {list(metrics.keys())}")
    v = metrics[metric_name]
    if not isinstance(v, (int, float)):
        raise TypeError(f"metric '{metric_name}' must be scalar, got {type(v)}")
    return float(v)