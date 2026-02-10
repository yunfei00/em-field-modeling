import argparse
import os
import yaml

from ttt.utils import set_seed
from ttt.datasets import build_data
from ttt.models import build_model
from ttt.engine import train
from ttt.checkpoint import load_checkpoint
from ttt.experiment import resolve_run_dir, dump_resolved_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")

    # experiment identity
    ap.add_argument("--exp_name", default=None, help="experiment name under runs/<exp_name>/")
    ap.add_argument("--run_id", default=None, help="run id under runs/<exp_name>/<run_id>/ (default: timestamp)")

    # resume options
    ap.add_argument("--resume", action="store_true", help="resume from runs/<exp_name>/<run_id>/last.pth")
    ap.add_argument("--resume_run_id", default=None, help="resume from runs/<exp_name>/<resume_run_id>/last.pth")
    ap.add_argument("--resume_from", default=None, help="resume from a checkpoint path, e.g. runs/x/y/last.pth")

    # AMP overrides (optional)
    ap.add_argument("--amp", action="store_true", help="force enable AMP (CUDA only)")
    ap.add_argument("--no-amp", action="store_true", help="force disable AMP")

    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(int(cfg["seed"]))

    # Ensure amp section exists
    cfg.setdefault("amp", {})
    if args.amp and args.no_amp:
        raise SystemExit("Cannot set both --amp and --no-amp")
    if args.amp:
        cfg["amp"]["enabled"] = True
    if args.no_amp:
        cfg["amp"]["enabled"] = False

    # Decide run_dir (new run by default)
    run_dir, exp_name, run_id = resolve_run_dir(cfg, exp_name=args.exp_name, run_id=args.run_id)

    # Save resolved config snapshot into run_dir
    dump_resolved_config(cfg, run_dir)

    dl_train, dl_val = build_data(cfg)
    model = build_model(cfg)

    resume_state = None
    ckpt_path = None

    # Priority: resume_from > resume_run_id > resume flag
    if args.resume_from:
        ckpt_path = args.resume_from
    elif args.resume_run_id:
        ckpt_path = os.path.join(cfg["ckpt"]["dir"], exp_name, args.resume_run_id, "last.pth")
    elif args.resume:
        ckpt_path = os.path.join(run_dir, "last.pth")

    if ckpt_path and os.path.exists(ckpt_path):
        resume_state = load_checkpoint(ckpt_path, map_location="cpu")
        # If resuming, keep the same exp_name/run_id
        exp_name = resume_state.get("exp_name", exp_name)
        run_id = resume_state.get("run_id", run_id)
        run_dir = os.path.join(cfg["ckpt"]["dir"], exp_name, run_id)
        dump_resolved_config(cfg, run_dir)
        print(f"[resume] loaded: {ckpt_path}")
    elif ckpt_path:
        print(f"[resume] checkpoint not found: {ckpt_path} (start fresh)")

    train(
        cfg=cfg,
        model=model,
        dl_train=dl_train,
        dl_val=dl_val,
        run_dir=run_dir,
        exp_name=exp_name,
        run_id=run_id,
        resume_state=resume_state,
    )

    print(f"[done] run_dir={run_dir}")

if __name__ == "__main__":
    main()