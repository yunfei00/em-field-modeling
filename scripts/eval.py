import argparse
import yaml
import torch

from ttt.datasets import build_data
from ttt.models import build_model
from ttt.engine import evaluate
from ttt.checkpoint import load_checkpoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    dl_train, dl_val = build_data(cfg)
    model = build_model(cfg)

    state = load_checkpoint(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    val_mse = evaluate(model, dl_val, device)
    print(f"val_mse={val_mse:.6g}")

if __name__ == "__main__":
    main()