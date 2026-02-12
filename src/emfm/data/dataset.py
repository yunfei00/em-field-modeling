from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .io import load_npz_sample, load_csv_case_sample

@dataclass
class ForwardSample:
    x: torch.Tensor
    y: torch.Tensor
    meta: dict

class ForwardDataset(Dataset):
    def __init__(self, root: str | Path, ids: list[str]):
        self.root = Path(root)
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> ForwardSample:
        sid = self.ids[idx]
        # Prefer npz layout first: <root>/<sid>.npz
        npz_path = self.root / f"{sid}.npz"
        if npz_path.exists():
            x, y, meta = load_npz_sample(npz_path)
        else:
            # Fallback to csv case layout: <root>/cases/<sid>/*.csv
            case_dir = self.root / "cases" / sid
            x, y, meta = load_csv_case_sample(case_dir)

        x_t = torch.from_numpy(np.ascontiguousarray(x))  # [Cin,11,11]
        y_t = torch.from_numpy(np.ascontiguousarray(y))  # [Cout,51,51]
        meta["id"] = sid
        return ForwardSample(x=x_t, y=y_t, meta=meta)
