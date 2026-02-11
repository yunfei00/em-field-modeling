# src/emfm/nf/io.py
import numpy as np
import torch
import pandas as pd


def load_source_H(csv_path: str) -> torch.Tensor:
    """
    return: [4, 11, 11]  (Hx_re, Hx_im, Hy_re, Hy_im)
    """
    df = pd.read_csv(csv_path)

    hx_re = df["Hx_re"].values.reshape(11, 11)
    hx_im = df["Hx_im"].values.reshape(11, 11)
    hy_re = df["Hy_re"].values.reshape(11, 11)
    hy_im = df["Hy_im"].values.reshape(11, 11)

    x = np.stack([hx_re, hx_im, hy_re, hy_im], axis=0)
    return torch.from_numpy(x).float()


def load_target_EH(e_csv: str, h_csv: str) -> torch.Tensor:
    """
    return: [12, 51, 51]  (Ex,Ey,Ez,Hx,Hy,Hz all re+im)
    """
    dfE = pd.read_csv(e_csv)
    dfH = pd.read_csv(h_csv)

    def split(df, keys):
        return [df[k].values.reshape(51, 51) for k in keys]

    Ex = split(dfE, ["Ex_re", "Ex_im"])
    Ey = split(dfE, ["Ey_re", "Ey_im"])
    Ez = split(dfE, ["Ez_re", "Ez_im"])

    Hx = split(dfH, ["Hx_re", "Hx_im"])
    Hy = split(dfH, ["Hy_re", "Hy_im"])
    Hz = split(dfH, ["Hz_re", "Hz_im"])

    y = np.stack(
        Ex + Ey + Ez + Hx + Hy + Hz,
        axis=0
    )
    return torch.from_numpy(y).float()
