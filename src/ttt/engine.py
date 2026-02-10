import os
import math
import torch
from torch import nn
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .logging_utils import JsonlLogger

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

def train(
    cfg: dict,
    model: nn.Module,
    dl_train,
    dl_val,
    run_dir: str,
    exp_name: str,
    run_id: str,
    resume_state: dict | None = None,
):
    """
    run_dir: runs/<exp_name>/<run_id>
    Saves:
      - last.pth / best.pth
      - metrics.jsonl
    """
    device = pick_device(cfg["device"])
    model = model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
    )

    start_epoch = 0
    best_val = math.inf
    global_step = 0

    if resume_state:
        model.load_state_dict(resume_state["model"])
        opt.load_state_dict(resume_state["optim"])
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        best_val = float(resume_state.get("best_val", best_val))
        global_step = int(resume_state.get("global_step", 0))

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    jlog = JsonlLogger(metrics_path)

    total_epochs = int(cfg["train"]["epochs"])
    log_every = int(cfg["train"]["log_every"])
    eval_every = int(cfg["train"]["eval_every"])

    for epoch in range(start_epoch, total_epochs):
        model.train()
        pbar = tqdm(dl_train, desc=f"[{exp_name}/{run_id}] epoch {epoch}", leave=False)

        running = 0.0
        seen = 0

        for step, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = torch.mean((pred - y) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            running += float(loss.item()) * x.shape[0]
            seen += x.shape[0]

            if (step + 1) % log_every == 0:
                avg = running / max(seen, 1)
                pbar.set_postfix({"loss": avg})
                jlog.log({
                    "type": "train_step",
                    "exp_name": exp_name,
                    "run_id": run_id,
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": avg,
                })

        # periodic eval
        val_mse = None
        if (epoch + 1) % eval_every == 0:
            val_mse = evaluate(model, dl_val, device)

        # save last
        last_path = os.path.join(run_dir, "last.pth")
        save_checkpoint(last_path, {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optim": opt.state_dict(),
            "best_val": best_val,
            "cfg": cfg,
            "exp_name": exp_name,
            "run_id": run_id,
        })

        # save best
        if cfg["ckpt"]["save_best"] and (val_mse is not None) and val_mse < best_val:
            best_val = val_mse
            best_path = os.path.join(run_dir, "best.pth")
            save_checkpoint(best_path, {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optim": opt.state_dict(),
                "best_val": best_val,
                "cfg": cfg,
                "exp_name": exp_name,
                "run_id": run_id,
            })

        # epoch log
        jlog.log({
            "type": "epoch",
            "exp_name": exp_name,
            "run_id": run_id,
            "epoch": epoch,
            "global_step": global_step,
            "train_loss_epoch": (running / max(seen, 1)) if seen else None,
            "val_mse": val_mse,
            "best_val": best_val,
        })

        if val_mse is not None:
            print(f"[{exp_name}/{run_id}] epoch={epoch} val_mse={val_mse:.6g} best={best_val:.6g}")
        else:
            print(f"[{exp_name}/{run_id}] epoch={epoch} (no eval) best={best_val:.6g}")

    return {"best_val": best_val, "run_dir": run_dir, "exp_name": exp_name, "run_id": run_id}