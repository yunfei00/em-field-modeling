import emfm.nf  # 触发 nf dataset / model 注册

import argparse
import yaml
import torch

from ttt.datasets import build_data
from ttt.models import build_model
from ttt.engine import evaluate
from ttt.checkpoint import load_checkpoint
from ttt.metrics import pick_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--track", default=None, help="override cfg.metrics.track")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    _, dl_val = build_data(cfg)
    model = build_model(cfg)

    state = load_checkpoint(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metrics_cfg = cfg.get("metrics", {})
    track = args.track or metrics_cfg.get("track", "rmse_mean")
    eps = float(metrics_cfg.get("eps", 1e-12))

    val_metrics = evaluate(model, dl_val, device, eps=eps)
    score = pick_score(val_metrics, track)

    print(f"{track}={score:.6g}")
    print(f"rmse_mean={val_metrics['rmse_mean']:.6g} rel_mean={val_metrics['rel_mean']:.6g}")
    # dim-wise metrics can be long; print only first few
    rmse_dim = val_metrics.get("rmse_dim", [])
    rel_dim = val_metrics.get("rel_dim", [])
    if rmse_dim:
        print(f"rmse_dim[:8]={rmse_dim[:8]}")
    if rel_dim:
        print(f"rel_dim[:8]={rel_dim[:8]}")

if __name__ == "__main__":
    main()
