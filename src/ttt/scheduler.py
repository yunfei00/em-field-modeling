from __future__ import annotations

import math
from typing import Optional
import torch


def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Scheduler factory controlled by cfg["scheduler"].

    Supported:
      - none (default)
      - step: StepLR
      - cosine: CosineAnnealingLR
      - onecycle: OneCycleLR (steps-based)
    """
    scfg = cfg.get("scheduler", None)
    if not scfg:
        return None

    name = str(scfg.get("name", "none")).lower()
    if name in ("none", "", "null"):
        return None

    if name == "step":
        step_size = int(scfg.get("step_size", 10))
        gamma = float(scfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name == "cosine":
        # epoch-based cosine
        t_max = int(scfg.get("t_max", cfg["train"]["epochs"]))
        eta_min = float(scfg.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    if name == "onecycle":
        # OneCycle is step-based; caller must pass steps_per_epoch in cfg
        max_lr = float(scfg.get("max_lr", cfg["optim"]["lr"]))
        epochs = int(cfg["train"]["epochs"])
        steps_per_epoch = int(scfg.get("steps_per_epoch", 0))
        if steps_per_epoch <= 0:
            raise ValueError("scheduler.onecycle requires scheduler.steps_per_epoch > 0")
        pct_start = float(scfg.get("pct_start", 0.3))
        div_factor = float(scfg.get("div_factor", 25.0))
        final_div_factor = float(scfg.get("final_div_factor", 1e4))
        anneal_strategy = str(scfg.get("anneal_strategy", "cos")).lower()  # 'cos' or 'linear'

        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            anneal_strategy=anneal_strategy,
        )

    raise ValueError(f"Unknown scheduler name: {name}")