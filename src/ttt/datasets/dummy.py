import torch
from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    def __init__(self, n: int, in_dim: int, out_dim: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, in_dim, generator=g)
        w = torch.randn(in_dim, out_dim, generator=g)
        self.y = self.x @ w + 0.1 * torch.randn(n, out_dim, generator=g)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def build_dataloaders(cfg: dict):
    bs = cfg["data"]["batch_size"]
    in_dim = cfg["data"]["input_dim"]
    out_dim = cfg["data"]["output_dim"]
    train_size = cfg["data"]["train_size"]
    val_size = cfg["data"]["val_size"]
    seed = cfg["seed"]

    train_ds = DummyDataset(train_size, in_dim, out_dim, seed=seed)
    val_ds = DummyDataset(val_size, in_dim, out_dim, seed=seed + 1)

    dl_train = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=cfg["num_workers"])
    dl_val = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=cfg["num_workers"])
    return dl_train, dl_val