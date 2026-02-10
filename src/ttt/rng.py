from __future__ import annotations

import random
import numpy as np
import torch

def get_rng_state() -> dict:
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
        "cuda_random": None,
    }
    if torch.cuda.is_available():
        try:
            state["cuda_random"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["cuda_random"] = None
    return state

def set_rng_state(state: dict) -> None:
    if not state:
        return
    try:
        if "python_random" in state and state["python_random"] is not None:
            random.setstate(state["python_random"])
        if "numpy_random" in state and state["numpy_random"] is not None:
            np.random.set_state(state["numpy_random"])
        if "torch_random" in state and state["torch_random"] is not None:
            torch.set_rng_state(state["torch_random"])
        if torch.cuda.is_available() and state.get("cuda_random") is not None:
            torch.cuda.set_rng_state_all(state["cuda_random"])
    except Exception:
        # best-effort restore; ignore if environment differs
        pass