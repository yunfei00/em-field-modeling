from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

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


def load_csv_case_sample(case_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load one sample from CSV case layout:
      <root>/cases/<id>/source_H.csv
      <root>/cases/<id>/target_E.csv
      <root>/cases/<id>/target_H.csv

    Returns:
      x: [4, 11, 11] float32
      y: [12, 51, 51] float32
      meta: dict
    """
    cdir = Path(case_dir)
    source_csv = cdir / "source_H.csv"
    target_e_csv = cdir / "target_E.csv"
    target_h_csv = cdir / "target_H.csv"

    df_src = pd.read_csv(source_csv)
    x = np.stack(
        [
            df_src["Hx_re"].values.reshape(11, 11),
            df_src["Hx_im"].values.reshape(11, 11),
            df_src["Hy_re"].values.reshape(11, 11),
            df_src["Hy_im"].values.reshape(11, 11),
        ],
        axis=0,
    ).astype(np.float32)

    df_e = pd.read_csv(target_e_csv)
    df_h = pd.read_csv(target_h_csv)

    def _pack(df: pd.DataFrame, keys: list[str]) -> list[np.ndarray]:
        return [df[k].values.reshape(51, 51) for k in keys]

    y = np.stack(
        _pack(df_e, ["Ex_re", "Ex_im"])
        + _pack(df_e, ["Ey_re", "Ey_im"])
        + _pack(df_e, ["Ez_re", "Ez_im"])
        + _pack(df_h, ["Hx_re", "Hx_im"])
        + _pack(df_h, ["Hy_re", "Hy_im"])
        + _pack(df_h, ["Hz_re", "Hz_im"]),
        axis=0,
    ).astype(np.float32)

    meta = {
        "__case_dir__": str(cdir),
        "__source_file__": str(source_csv),
        "__target_e_file__": str(target_e_csv),
        "__target_h_file__": str(target_h_csv),
    }
    return x, y, meta
