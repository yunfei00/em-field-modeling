from .dummy import build_dataloaders

def build_data(cfg: dict):
    name = cfg["data"]["name"]
    if name == "dummy":
        return build_dataloaders(cfg)
    raise ValueError(f"Unknown dataset: {name}")