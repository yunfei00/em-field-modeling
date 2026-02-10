import os
import math
import torch
from torch import nn
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .logging_utils import JsonlLogger
from .metrics import batch_metrics, pick_score

def pick_device(device_cfg: str):
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate(model: nn.Module, dl, device, eps: float = 1e-12):
    """
    Aggregate metrics over the whole dataloader.
    For simplicity we average batch-level metrics weighted by batch size.
    """
    model.eval()
    total = 0
    agg_rmse_mean = 0.0
    agg_rel_mean = 0.0
    rmse_dim_sum = None
    rel_dim_sum = None

    for x, y in dl:
        bs = x.shape[0]
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        m = batch_metrics(pred, y, eps=eps)

        total += bs
        agg_rmse_mean += m["rmse_mean"] * bs
        agg_rel_mean += m["rel_mean"] * bs

        # dim-wise lists
        rmse_dim = torch.tensor(m["rmse_dim"], dtype=torch.float64)
        rel_dim = torch.tensor(m["rel_dim"], dtype=torch.float64)
        rmse_dim_sum = rmse_dim if rmse_dim_sum is None else (rmse_dim_sum + rmse_dim * bs)
        rel_dim_sum = rel_dim if rel_dim_sum is None else (rel_dim_sum + rel_dim * bs)

    if total == 0:
        return {"rmse_mean": math.inf, "rel_mean": math.inf, "rmse_dim": [], "rel_dim": [], "out_dim": 0}

    rmse_dim_avg = (rmse_dim_sum / total).tolist() if rmse_dim_sum is not None else []
    rel_dim_avg = (rel_dim_sum / total).tolist() if rel_dim_sum is not None else []

    return {
        "rmse_mean": agg_rmse_mean / total,
        "rel_mean": agg_rel_mean / total,
        "rmse_dim": rmse_dim_avg,
        "rel_dim": rel_dim_avg,
        "out_dim": len(rmse_dim_avg),
    }

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
      - config snapshot is handled outside (train.py)
    """
    device = pick_device(cfg["device"])
    model = model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
    )

    # metrics config
    metrics_cfg = cfg.get("metrics", {})
    track = metrics_cfg.get("track", "rmse_mean")        # scalar key in evaluate() output
    eps = float(metrics_cfg.get("eps", 1e-12))
    lower_is_better = bool(metrics_cfg.get("lower_is_better", True))

    start_epoch = 0
    best_score = math.inf if lower_is_better else -math.inf
    best_metrics = None
    global_step = 0

    if resume_state:
        model.load_state_dict(resume_state["model"])
        opt.load_state_dict(resume_state["optim"])
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        global_step = int(resume_state.get("global_step", 0))
        best_score = float(resume_state.get("best_score", best_score))
        best_metrics = resume_state.get("best_metrics", best_metrics)

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    jlog = JsonlLogger(metrics_path)

    total_epochs = int(cfg["train"]["epochs"])
    log_every = int(cfg["train"]["log_every"])
    eval_every = int(cfg["train"]["eval_every"])

    def is_better(curr: float, best: float) -> bool:
        return (curr < best) if lower_is_better else (curr > best)

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
        val_metrics = None
        score = None
        if (epoch + 1) % eval_every == 0:
            val_metrics = evaluate(model, dl_val, device, eps=eps)
            score = pick_score(val_metrics, track)

        # save last
        last_path = os.path.join(run_dir, "last.pth")
        save_checkpoint(last_path, {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optim": opt.state_dict(),
            "best_score": best_score,
            "best_metrics": best_metrics,
            "cfg": cfg,
            "exp_name": exp_name,
            "run_id": run_id,
            "track_metric": track,
            "lower_is_better": lower_is_better,
        })

        # save best
        if cfg["ckpt"]["save_best"] and (score is not None) and is_better(score, best_score):
            best_score = score
            best_metrics = val_metrics
            best_path = os.path.join(run_dir, "best.pth")
            save_checkpoint(best_path, {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optim": opt.state_dict(),
                "best_score": best_score,
                "best_metrics": best_metrics,
                "cfg": cfg,
                "exp_name": exp_name,
                "run_id": run_id,
                "track_metric": track,
                "lower_is_better": lower_is_better,
            })

        # epoch log (jsonl)
        jlog.log({
            "type": "epoch",
            "exp_name": exp_name,
            "run_id": run_id,
            "epoch": epoch,
            "global_step": global_step,
            "train_loss_epoch": (running / max(seen, 1)) if seen else None,
            "val_metrics": val_metrics,
            "track_metric": track,
            "score": score,
            "best_score": best_score,
        })

        if val_metrics is not None:
            print(f"[{exp_name}/{run_id}] epoch={epoch} {track}={score:.6g} best={best_score:.6g}")
        else:
            print(f"[{exp_name}/{run_id}] epoch={epoch} (no eval) best={best_score:.6g}")

    return {"best_score": best_score, "best_metrics": best_metrics, "run_dir": run_dir, "exp_name": exp_name, "run_id": run_id}