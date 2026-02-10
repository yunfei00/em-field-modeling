from .mlp import MLP

def build_model(cfg: dict):
    name = cfg["model"]["name"]
    if name == "mlp":
        d_in = cfg["data"]["input_dim"]
        d_out = cfg["data"]["output_dim"]
        hidden = cfg["model"]["hidden_dim"]
        return MLP(d_in, hidden, d_out)
    raise ValueError(f"Unknown model: {name}")