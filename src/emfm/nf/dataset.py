import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ttt.registry import register_dataset


def _load_source_H(csv_path: str) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    x = np.stack([
        df["Hx_re"].values.reshape(11, 11),
        df["Hx_im"].values.reshape(11, 11),
        df["Hy_re"].values.reshape(11, 11),
        df["Hy_im"].values.reshape(11, 11),
    ], axis=0)
    return torch.from_numpy(x).float()


def _load_target_EH(e_csv: str, h_csv: str) -> torch.Tensor:
    dfE = pd.read_csv(e_csv)
    dfH = pd.read_csv(h_csv)

    def pack(df, keys):
        return [df[k].values.reshape(51, 51) for k in keys]

    y = np.stack(
        pack(dfE, ["Ex_re","Ex_im"]) +
        pack(dfE, ["Ey_re","Ey_im"]) +
        pack(dfE, ["Ez_re","Ez_im"]) +
        pack(dfH, ["Hx_re","Hx_im"]) +
        pack(dfH, ["Hy_re","Hy_im"]) +
        pack(dfH, ["Hz_re","Hz_im"]),
        axis=0
    )
    return torch.from_numpy(y).float()


class NearFieldInversionDataset(Dataset):
    def __init__(self, root: str, split_file: str):
        self.root = root
        with open(split_file, "r", encoding="utf-8") as f:
            self.case_ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        cid = self.case_ids[idx]
        cdir = os.path.join(self.root, "cases", cid)
        x = _load_source_H(os.path.join(cdir, "source_H.csv"))
        y = _load_target_EH(
            os.path.join(cdir, "target_E.csv"),
            os.path.join(cdir, "target_H.csv"),
        )
        return x, y


@register_dataset("nf_inversion_v1")
def build_nf_inversion_v1(cfg: dict):
    dcfg = cfg["data"]
    tcfg = cfg["train"]

    ds_train = NearFieldInversionDataset(dcfg["root"], dcfg["train_split"])
    ds_val = NearFieldInversionDataset(dcfg["root"], dcfg["val_split"])

    dl_train = DataLoader(
        ds_train,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        num_workers=int(tcfg.get("num_workers", 4)),
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=int(tcfg["batch_size"]),
        shuffle=False,
        num_workers=int(tcfg.get("num_workers", 4)),
        pin_memory=True,
    )
    return dl_train, dl_val
