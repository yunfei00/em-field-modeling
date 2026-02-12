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


def collate_forward_samples(batch: list[ForwardSample]) -> ForwardSample:
    """Collate ForwardSample batches for torch DataLoader.

    PyTorch default collate may reject custom dataclass objects depending on
    version. Keeping this explicit ensures both outer and inner trainers can
    always build mini-batches from ``ForwardDataset`` safely.
    """
    if not batch:
        raise ValueError("Empty batch is not supported")

    x = torch.stack([sample.x for sample in batch], dim=0)
    y = torch.stack([sample.y for sample in batch], dim=0)

    # Keep per-sample metadata as a plain list to avoid lossy merging.
    meta = [sample.meta for sample in batch]
    return ForwardSample(x=x, y=y, meta={"items": meta})

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
