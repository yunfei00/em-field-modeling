import emfm.nf  # 触发 nf dataset / model 注册

import argparse
import os
import yaml

from ttt.utils import set_seed
from ttt.datasets import build_data
from ttt.models import build_model
from ttt.engine import train
from ttt.checkpoint import load_checkpoint
from ttt.experiment import resolve_run_dir, dump_resolved_config


def _default_line_presets() -> dict:
    return {
        "forward": {"exp_name": "em_forward", "run_id": "main"},
        "inverse": {"exp_name": "em_inverse", "run_id": "main"},
        # aliases
        "em_forward": {"exp_name": "em_forward", "run_id": "main"},
        "em_inverse": {"exp_name": "em_inverse", "run_id": "main"},
    }


def apply_line_preset(cfg: dict, line_name: str | None) -> tuple[str | None, str | None]:
    """
    Resolve (exp_name, run_id) by a short line name.
    Priority: ckpt.line_presets in config > built-in defaults.
    """
    if not line_name:
        return None, None

    presets = _default_line_presets()
    presets.update(cfg.get("ckpt", {}).get("line_presets", {}))
    picked = presets.get(line_name)
    if not picked:
        valid = ", ".join(sorted(presets.keys()))
        raise SystemExit(f"Unknown --line '{line_name}'. Available: {valid}")

    return picked.get("exp_name"), picked.get("run_id")


def normalize_legacy_config(cfg: dict) -> None:
    """
    Backward-compatible normalization for older YAML layouts.

    Legacy inversion configs used `train.lr` (and optionally `train.weight_decay`)
    instead of an `optim` block. Keep accepting that format so users can continue
    editing existing YAML files.
    """
    tcfg = cfg.get("train", {})
    ocfg = cfg.setdefault("optim", {})

    if "lr" not in ocfg and "lr" in tcfg:
        ocfg["lr"] = float(tcfg["lr"])
    if "weight_decay" not in ocfg and "weight_decay" in tcfg:
        ocfg["weight_decay"] = float(tcfg["weight_decay"])

    ocfg.setdefault("name", "adamw")
    ocfg.setdefault("lr", 1e-3)
    ocfg.setdefault("weight_decay", 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--line", default=None, help="shortcut for exp/run, e.g. forward or inverse")

    # experiment identity
    ap.add_argument("--exp_name", default=None, help="experiment name under runs/<exp_name>/")
    ap.add_argument("--run_id", default=None, help="run id under runs/<exp_name>/<run_id>/ (default: timestamp)")

    # resume options
    ap.add_argument("--resume", action="store_true", help="resume from runs/<exp_name>/<run_id>/last.pth")
    ap.add_argument("--resume_run_id", default=None, help="resume from runs/<exp_name>/<resume_run_id>/last.pth")
    ap.add_argument("--resume_from", default=None, help="resume from a checkpoint path, e.g. runs/x/y/last.pth")
    ap.add_argument(
        "--resume_best",
        action="store_true",
        help="when using --resume/--resume_run_id, load best.pth instead of last.pth",
    )

    # AMP overrides (optional)
    ap.add_argument("--amp", action="store_true", help="force enable AMP (CUDA only)")
    ap.add_argument("--lr", type=float, default=None, help="override optim.lr in config")
    ap.add_argument(
        "--resume_new_lr",
        action="store_true",
        help="when resuming, keep checkpoint model/optimizer states but overwrite optimizer lr with current config/--lr",
    )
    ap.add_argument(
        "--resume_reset_scheduler",
        action="store_true",
        help="when resuming, do not load scheduler state from checkpoint (rebuild from current config)",
    )
    ap.add_argument("--no-amp", action="store_true", help="force disable AMP")

    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    normalize_legacy_config(cfg)
    set_seed(int(cfg["seed"]))

    line_exp_name, line_run_id = apply_line_preset(cfg, args.line)
    exp_name = args.exp_name or line_exp_name
    run_id = args.run_id or line_run_id

    # Ensure amp section exists
    cfg.setdefault("amp", {})
    if args.amp and args.no_amp:
        raise SystemExit("Cannot set both --amp and --no-amp")
    if args.amp:
        cfg["amp"]["enabled"] = True
    if args.no_amp:
        cfg["amp"]["enabled"] = False

    if args.lr is not None:
        cfg.setdefault("optim", {})["lr"] = float(args.lr)

    # By default, when user explicitly passes --lr for resume training,
    # optimizer lr should follow latest CLI/config instead of checkpoint value.
    if args.resume_new_lr or args.lr is not None:
        cfg.setdefault("optim", {})["resume_new_lr"] = True

    if args.resume_reset_scheduler:
        cfg.setdefault("scheduler", {})["resume_reset"] = True

    # Decide run_dir (new run by default)
    run_dir, exp_name, run_id = resolve_run_dir(cfg, exp_name=exp_name, run_id=run_id)

    # Save resolved config snapshot into run_dir
    dump_resolved_config(cfg, run_dir)

    dl_train, dl_val = build_data(cfg)
    model = build_model(cfg)

    resume_state = None
    ckpt_path = None

    # Priority: resume_from > resume_run_id > resume flag
    resume_ckpt_name = "best.pth" if args.resume_best else "last.pth"
    if args.resume_from:
        ckpt_path = args.resume_from
    elif args.resume_run_id:
        ckpt_path = os.path.join(cfg["ckpt"]["dir"], exp_name, args.resume_run_id, resume_ckpt_name)
    elif args.resume:
        ckpt_path = os.path.join(run_dir, resume_ckpt_name)

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
