import os
import torch

def save_checkpoint(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, map_location="cpu", weights_only: bool = False):
    """
    Load a training checkpoint.

    weights_only=False is REQUIRED for resume training because checkpoints
    may contain optimizer, scaler, rng_state, etc.

    For pure inference weights, set weights_only=True.
    """
    return torch.load(
        path,
        map_location=map_location,
        weights_only=weights_only,
    )

def latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    if not files:
        return None
    files.sort()
    return os.path.join(ckpt_dir, files[-1])