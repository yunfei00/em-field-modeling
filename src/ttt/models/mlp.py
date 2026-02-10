import torch
from torch import nn

from ttt.registry import register_model

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@register_model("mlp")
def build_mlp(cfg: dict) -> nn.Module:
    d_in = int(cfg["data"]["input_dim"])
    d_out = int(cfg["data"]["output_dim"])
    hidden = int(cfg["model"]["hidden_dim"])
    return MLP(d_in, hidden, d_out)