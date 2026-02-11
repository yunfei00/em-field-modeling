from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from emfm.utils.seed import set_seed
from emfm.data.dataset import ForwardDataset
from emfm.models.forward_unet import ForwardUNetLite
from .losses import WeightedMSELoss
from .metrics import rmse_per_channel

def parse_ids(txt: str) -> list[str]:
    return [line.strip() for line in Path(txt).read_text(encoding="utf-8").splitlines() if line.strip()]

def save_ckpt(path: Path, model, optim, epoch: int, best_val: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_val": best_val,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
    }, path)

def load_ckpt(path: Path, model, optim):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    return ckpt["epoch"], ckpt.get("best_val", 1e30)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--train_ids", required=True)
    ap.add_argument("--val_ids", required=True)
    ap.add_argument("--run_dir", default="runs/forward_baseline")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(args.run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    train_ids = parse_ids(args.train_ids)
    val_ids = parse_ids(args.val_ids)

    ds_tr = ForwardDataset(args.data_root, train_ids)
    ds_va = ForwardDataset(args.data_root, val_ids)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = ForwardUNetLite(cin=4, cout=12).to(device)
    loss_fn = WeightedMSELoss(weights=None)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_val = 1e30
    last_ckpt = run_dir / "checkpoints" / "last.pth"
    if args.resume and last_ckpt.exists():
        start_epoch, best_val = load_ckpt(last_ckpt, model, optim)
        start_epoch += 1

    log_path = run_dir / "artifacts" / "metrics.jsonl"
    def log(obj: dict):
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # store resolved config-ish info
    (run_dir / "artifacts" / "run_args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        tr_loss = 0.0
        for batch in dl_tr:
            x = batch.x.to(device, non_blocking=True)
            y = batch.y.to(device, non_blocking=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            tr_loss += loss.item()

        tr_loss /= max(1, len(dl_tr))

        model.eval()
        va_loss = 0.0
        last_rmse = None
        with torch.no_grad():
            for batch in dl_va:
                x = batch.x.to(device, non_blocking=True)
                y = batch.y.to(device, non_blocking=True)
                pred = model(x)
                loss = loss_fn(pred, y)
                va_loss += loss.item()
                last_rmse = rmse_per_channel(pred, y)
        va_loss /= max(1, len(dl_va))

        is_best = va_loss < best_val
        if is_best:
            best_val = va_loss
            save_ckpt(run_dir / "checkpoints" / "best.pth", model, optim, epoch, best_val)
        save_ckpt(last_ckpt, model, optim, epoch, best_val)

        log({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "best_val": best_val,
            "rmse_last_batch": last_rmse,
            "is_best": is_best,
        })

        print(f"[E{epoch}] train={tr_loss:.6e} val={va_loss:.6e} best={best_val:.6e} best?={is_best}")

if __name__ == "__main__":
    main()
