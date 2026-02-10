import os
import math
import torch
from torch import nn
from tqdm import tqdm

from .utils import ensure_dir
from .checkpoint import save_checkpoint

def pick_device(device_cfg: str):
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate(model: nn.Module, dl, device):
    model.eval()
    mse = 0.0
    n = 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = torch.mean((pred - y) ** 2).item()
        mse += loss * x.shape[0]
        n += x.shape[0]
    return mse / max(n, 1)

def train(cfg: dict, model: nn.Module, dl_train, dl_val, resume_state: dict | None = None):
    device = pick_device(cfg["device"])
    model = model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
    )

    start_epoch = 0
    best_val = math.inf

    if resume_state:
        model.load_state_dict(resume_state["model"])
        opt.load_state_dict(resume_state["optim"])
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        best_val = float(resume_state.get("best_val", best_val))

    run_dir = os.path.join(cfg["ckpt"]["dir"], cfg["ckpt"]["exp_name"])
    ensure_dir(run_dir)

    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        model.train()
        pbar = tqdm(dl_train, desc=f"epoch {epoch}", leave=False)
        for step, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = torch.mean((pred - y) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if (step + 1) % int(cfg["train"]["log_every"]) == 0:
                pbar.set_postfix({"loss": float(loss.item())})

        val_mse = evaluate(model, dl_val, device)

        # save last
        last_path = os.path.join(run_dir, f"last_epoch_{epoch:04d}.pth")
        save_checkpoint(last_path, {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": opt.state_dict(),
            "best_val": best_val,
            "cfg": cfg,
        })

        # save best
        if cfg["ckpt"]["save_best"] and val_mse < best_val:
            best_val = val_mse
            best_path = os.path.join(run_dir, "best.pth")
            save_checkpoint(best_path, {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": opt.state_dict(),
                "best_val": best_val,
                "cfg": cfg,
            })

        print(f"[epoch {epoch}] val_mse={val_mse:.6g} best={best_val:.6g}")

    return {"best_val": best_val, "run_dir": run_dir}