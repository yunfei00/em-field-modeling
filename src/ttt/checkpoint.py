import os
import torch

def save_checkpoint(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location)

def latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    if not files:
        return None
    files.sort()
    return os.path.join(ckpt_dir, files[-1])