import os
import math
import torch
from torch import nn
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .logging_utils import JsonlLogger
from .metrics import batch_metrics, pick_score
from .rng import get_rng_state, set_rng_state
from .scheduler import build_scheduler


def pick_device(device_cfg: str):
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_objects(cfg: dict, device: torch.device):
    amp_cfg = cfg.get("amp", {}) or {}
    enabled = bool(amp_cfg.get("enabled", False))
    if device.type != "cuda" or not torch.cuda.is_available():
        enabled = False

    dtype_name = str(amp_cfg.get("dtype", "fp16")).lower()
    if dtype_name in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    try:
        autocast = torch.amp.autocast
        GradScaler = torch.amp.GradScaler
    except Exception:
        autocast = torch.cuda.amp.autocast
        GradScaler = torch.cuda.amp.GradScaler

    scaler = GradScaler(enabled=enabled)
    ctx = autocast(device_type="cuda", dtype=amp_dtype, enabled=enabled)
    return enabled, ctx, scaler


@torch.no_grad()
def evaluate(model: nn.Module, dl, device, eps: float = 1e-12):
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


def _get_lr(optimizer: torch.optim.Optimizer) -> float:
    # assume one param group or take the first
    return float(optimizer.param_groups[0].get("lr", 0.0))


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
    device = pick_device(cfg["device"])
    model = model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
    )

    # Build scheduler (may be None)
    # For OneCycle, we need steps_per_epoch; inject if user didn't provide
    if cfg.get("scheduler", {}) and str(cfg["scheduler"].get("name", "")).lower() == "onecycle":
        cfg.setdefault("scheduler", {})
        cfg["scheduler"]["steps_per_epoch"] = int(cfg["scheduler"].get("steps_per_epoch", len(dl_train)))
    sched = build_scheduler(cfg, opt)

    # metrics config
    metrics_cfg = cfg.get("metrics", {}) or {}
    track = metrics_cfg.get("track", "rmse_mean")
    eps = float(metrics_cfg.get("eps", 1e-12))
    lower_is_better = bool(metrics_cfg.get("lower_is_better", True))

    start_epoch = 0
    best_score = math.inf if lower_is_better else -math.inf
    best_metrics = None
    global_step = 0

    # AMP objects
    use_amp, autocast_ctx, scaler = _amp_objects(cfg, device)

    # Resume
    if resume_state:
        set_rng_state(resume_state.get("rng_state"))

        model.load_state_dict(resume_state["model"])
        opt.load_state_dict(resume_state["optim"])

        # Optional: keep checkpoint optimizer moments, but overwrite lr by latest config.
        if bool(cfg.get("optim", {}).get("resume_new_lr", False)):
            new_lr = float(cfg["optim"]["lr"])
            for g in opt.param_groups:
                g["lr"] = new_lr

        start_epoch = int(resume_state.get("epoch", 0)) + 1
        global_step = int(resume_state.get("global_step", 0))
        best_score = float(resume_state.get("best_score", best_score))
        best_metrics = resume_state.get("best_metrics", best_metrics)

        # Restore scaler
        if scaler is not None and resume_state.get("scaler") is not None:
            try:
                scaler.load_state_dict(resume_state["scaler"])
            except Exception:
                pass

        # Restore scheduler
        resume_reset_sched = bool(cfg.get("scheduler", {}).get("resume_reset", False))
        if (not resume_reset_sched) and sched is not None and resume_state.get("scheduler") is not None:
            try:
                sched.load_state_dict(resume_state["scheduler"])
            except Exception:
                # best-effort
                pass

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

            with autocast_ctx:
                pred = model(x)
                loss = torch.mean((pred - y) ** 2)

            opt.zero_grad(set_to_none=True)

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            global_step += 1
            running += float(loss.item()) * x.shape[0]
            seen += x.shape[0]

            # Step-based scheduler (OneCycle) should step each optimizer step
            if sched is not None and isinstance(sched, torch.optim.lr_scheduler.OneCycleLR):
                sched.step()

            if (step + 1) % log_every == 0:
                avg = running / max(seen, 1)
                lr = _get_lr(opt)
                pbar.set_postfix({"loss": avg, "lr": lr, "amp": int(use_amp)})
                jlog.log({
                    "type": "train_step",
                    "exp_name": exp_name,
                    "run_id": run_id,
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": avg,
                    "lr": lr,
                    "amp": bool(use_amp),
                })

        # Epoch-based scheduler steps here (StepLR/Cosine)
        if sched is not None and not isinstance(sched, torch.optim.lr_scheduler.OneCycleLR):
            sched.step()

        # periodic eval
        val_metrics = None
        score = None
        if (epoch + 1) % eval_every == 0:
            val_metrics = evaluate(model, dl_val, device, eps=eps)
            score = pick_score(val_metrics, track)

        lr_epoch = _get_lr(opt)

        ckpt_payload = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optim": opt.state_dict(),
            "scheduler": (sched.state_dict() if sched is not None else None),
            "best_score": best_score,
            "best_metrics": best_metrics,
            "cfg": cfg,
            "exp_name": exp_name,
            "run_id": run_id,
            "track_metric": track,
            "lower_is_better": lower_is_better,
            "amp_enabled": bool(use_amp),
            "scaler": (scaler.state_dict() if (scaler is not None and use_amp) else None),
            "rng_state": get_rng_state(),
        }

        # save last
        last_path = os.path.join(run_dir, "last.pth")
        save_checkpoint(last_path, ckpt_payload)

        # save best
        if cfg["ckpt"]["save_best"] and (score is not None) and is_better(score, best_score):
            best_score = score
            best_metrics = val_metrics
            ckpt_payload["best_score"] = best_score
            ckpt_payload["best_metrics"] = best_metrics
            best_path = os.path.join(run_dir, "best.pth")
            save_checkpoint(best_path, ckpt_payload)

        # epoch log
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
            "lr": lr_epoch,
            "scheduler": (cfg.get("scheduler", {}) or {}).get("name", "none"),
            "amp": bool(use_amp),
        })

        if val_metrics is not None:
            print(f"[{exp_name}/{run_id}] epoch={epoch} {track}={score:.6g} best={best_score:.6g} lr={lr_epoch:.6g} amp={int(use_amp)}")
        else:
            print(f"[{exp_name}/{run_id}] epoch={epoch} (no eval) best={best_score:.6g} lr={lr_epoch:.6g} amp={int(use_amp)}")

    return {"best_score": best_score, "best_metrics": best_metrics, "run_dir": run_dir, "exp_name": exp_name, "run_id": run_id}