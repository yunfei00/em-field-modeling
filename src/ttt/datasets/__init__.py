# Import modules so their @register_* decorators execute at import time.
# Add new datasets by creating a new file in this folder and importing it here.
from . import dummy  # noqa: F401

from ttt.registry import get_dataset_factory

def build_data(cfg: dict):
    name = cfg["data"]["name"]
    factory = get_dataset_factory(name)
    return factory(cfg)