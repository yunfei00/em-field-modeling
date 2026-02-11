import torch
from torch.utils.data import Dataset, DataLoader

from ttt.registry import register_dataset


class DummyNFInversionDataset(Dataset):
    """
    返回 shape 与真实任务一致：
      x: [4,11,11]
      y: [12,51,51]
    """
    def __init__(self, n: int, seed: int = 1234):
        self.n = int(n)
        g = torch.Generator().manual_seed(int(seed))
        # 预生成一些固定随机样本，保证可复现、训练可跑
        self.x = torch.randn(self.n, 4, 11, 11, generator=g)
        # 用一个“可学习”的映射生成 y：先上采样再线性组合，避免完全随机导致 loss 不下降
        up = torch.nn.functional.interpolate(self.x, size=(51, 51), mode="bilinear", align_corners=False)
        # [n,4,51,51] -> [n,12,51,51]
        W = torch.randn(12, 4, generator=g)
        self.y = torch.einsum("oc,nchw->nohw", W, up)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


@register_dataset("nf_inversion_dummy")
def build_nf_inversion_dummy(cfg: dict):
    dcfg = cfg["data"]
    tcfg = cfg["train"]

    n_train = int(dcfg.get("n_train", 256))
    n_val = int(dcfg.get("n_val", 64))
    seed = int(cfg.get("seed", 1234))

    ds_train = DummyNFInversionDataset(n_train, seed=seed)
    ds_val = DummyNFInversionDataset(n_val, seed=seed + 1)

    pin_memory = (cfg.get("device", "cpu") == "cuda")

    dl_train = DataLoader(
        ds_train,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        num_workers=int(tcfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=int(tcfg["batch_size"]),
        shuffle=False,
        num_workers=int(tcfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )
    return dl_train, dl_val
