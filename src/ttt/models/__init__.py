# Import modules so their @register_* decorators execute at import time.
# Add new models by creating a new file in this folder and importing it here.
from . import mlp  # noqa: F401

from ttt.registry import get_model_factory

def build_model(cfg: dict):
    name = cfg["model"]["name"]
    factory = get_model_factory(name)
    return factory(cfg)