import argparse
import yaml

from ttt.utils import set_seed
from ttt.datasets import build_data
from ttt.models import build_model
from ttt.engine import train
from ttt.checkpoint import load_checkpoint, latest_checkpoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(int(cfg["seed"]))

    dl_train, dl_val = build_data(cfg)
    model = build_model(cfg)

    resume_state = None
    if args.resume:
        ckpt_dir = f'{cfg["ckpt"]["dir"]}/{cfg["ckpt"]["exp_name"]}'
        p = latest_checkpoint(ckpt_dir)
        if p:
            resume_state = load_checkpoint(p)
            print(f"[resume] loaded: {p}")
        else:
            print("[resume] no checkpoint found, start fresh")

    train(cfg, model, dl_train, dl_val, resume_state=resume_state)

if __name__ == "__main__":
    main()