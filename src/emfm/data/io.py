from __future__ import annotations
from pathlib import Path
import numpy as np

def load_npz_sample(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Expect:
      x: [Cin, 11, 11] float32
      y: [Cout, 51, 51] float32
      meta: optional dict-like (may be absent)
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    meta = {}
    if "meta" in data:
        m = data["meta"]
        # meta may be an object array
        try:
            meta = m.item() if hasattr(m, "item") else dict(m)
        except Exception:
            meta = {}
    meta["__file__"] = str(path)
    return x, y, meta
