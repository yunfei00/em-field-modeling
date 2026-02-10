from __future__ import annotations

from typing import Callable, Dict, Any

# Factories:
# - model factory: (cfg: dict) -> torch.nn.Module
# - dataset factory: (cfg: dict) -> (dl_train, dl_val)
MODELS: Dict[str, Callable[[dict], Any]] = {}
DATASETS: Dict[str, Callable[[dict], Any]] = {}

def register_model(name: str):
    """
    Usage:
      from ttt.registry import register_model

      @register_model("mlp")
      def build_mlp(cfg): ...
    """
    def deco(fn: Callable[[dict], Any]):
        if name in MODELS:
            raise KeyError(f"Model '{name}' already registered")
        MODELS[name] = fn
        return fn
    return deco

def register_dataset(name: str):
    """
    Usage:
      from ttt.registry import register_dataset

      @register_dataset("dummy")
      def build_dummy(cfg): ...
    """
    def deco(fn: Callable[[dict], Any]):
        if name in DATASETS:
            raise KeyError(f"Dataset '{name}' already registered")
        DATASETS[name] = fn
        return fn
    return deco

def get_model_factory(name: str):
    if name not in MODELS:
        raise KeyError(f"Unknown model: {name}. Available: {sorted(MODELS.keys())}")
    return MODELS[name]

def get_dataset_factory(name: str):
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset: {name}. Available: {sorted(DATASETS.keys())}")
    return DATASETS[name]