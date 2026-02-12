from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml
from emfm.utils.seed import set_seed
from emfm.data.dataset import ForwardDataset
from emfm.models.forward_unet import ForwardUNetLite
from emfm.tasks.forward.losses import WeightedMSELoss
from emfm.tasks.forward.metrics import rmse_per_channel


def _as_int(cfg: dict, key: str, default: int) -> int:
    return int(cfg.get(key, default))


def _as_float(cfg: dict, key: str, default: float) -> float:
    return float(cfg.get(key, default))


def _as_bool(cfg: dict, key: str, default: bool) -> bool:
    v = cfg.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Invalid boolean value for {key}: {v!r}")




def _cfg_get(cfg: dict, key: str, default=None):
    """Read config values from unified nested YAML with flat-key fallback."""
    if key in cfg:
        return cfg.get(key, default)

    nested_map = {
        "data_root": (("data", "root"), ("data", "data_root")),
        "train_ids": (("data", "train_split"), ("data", "train_ids")),
        "val_ids": (("data", "val_split"), ("data", "val_ids")),
        "run_dir": ("ckpt", "run_dir"),
        "epochs": ("train", "epochs"),
        "batch_size": ("train", "batch_size"),
        "resume": ("train", "resume"),
        "lr": ("optim", "lr"),
        "num_workers": ("train", "num_workers"),
        "normalize_y": ("loss", "normalize_y"),
        "norm_max_batches": ("loss", "norm_max_batches"),
        "norm_eps": ("loss", "norm_eps"),
        "auto_channel_weight": ("loss", "auto_channel_weight"),
        "auto_weight_max_batches": ("loss", "auto_weight_max_batches"),
        "e_weight_multiplier": ("loss", "e_weight_multiplier"),
        "h_weight_multiplier": ("loss", "h_weight_multiplier"),
    }

    paths = nested_map.get(key)
    if not paths:
        return default

    if isinstance(paths[0], str):
        paths = (paths,)

    for path in paths:
        cur = cfg
        missing = False
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                missing = True
                break
            cur = cur[p]
        if not missing:
            return cur
    return default

def build_channel_weights(
    dl_tr: DataLoader,
    device: torch.device,
    *,
    max_batches: int = 64,
    eps: float = 1e-8,
    e_multiplier: float = 1.0,
    h_multiplier: float = 1.0,
) -> list[float]:
    """Estimate per-channel weights from target RMS to reduce E/H scale imbalance."""
    sum_sq = None
    n_elem = 0
    for i, batch in enumerate(dl_tr):
        if i >= max_batches:
            break
        y = batch.y.to(device, non_blocking=True)
        b, _, h, w = y.shape
        ch_sq = (y**2).sum(dim=(0, 2, 3))
        if sum_sq is None:
            sum_sq = ch_sq
        else:
            sum_sq += ch_sq
        n_elem += b * h * w

    if sum_sq is None or n_elem == 0:
        return [1.0] * 12

    rms = torch.sqrt(sum_sq / float(n_elem))
    inv_var = 1.0 / (rms**2 + eps)

    # Y channel order: [Ex,Ey,Ez,Hx,Hy,Hz] and each has re/im, so 0-5 is E, 6-11 is H.
    inv_var[:6] *= e_multiplier
    inv_var[6:] *= h_multiplier

    # Normalize to keep average loss scale stable.
    inv_var = inv_var / inv_var.mean()
    return inv_var.detach().cpu().tolist()


def estimate_target_norm_stats(
    dl_tr: DataLoader,
    device: torch.device,
    *,
    max_batches: int = 64,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Estimate per-channel mean/std from training targets for explicit normalization."""
    sum_y = None
    sum_y2 = None
    n_elem = 0
    for i, batch in enumerate(dl_tr):
        if i >= max_batches:
            break
        y = batch.y.to(device, non_blocking=True)
        b, _, h, w = y.shape
        ch_sum = y.sum(dim=(0, 2, 3))
        ch_sum2 = (y**2).sum(dim=(0, 2, 3))
        if sum_y is None:
            sum_y = ch_sum
            sum_y2 = ch_sum2
        else:
            sum_y += ch_sum
            sum_y2 += ch_sum2
        n_elem += b * h * w

    if sum_y is None or sum_y2 is None or n_elem == 0:
        return None, None

    mean = sum_y / float(n_elem)
    var = sum_y2 / float(n_elem) - mean**2
    std = torch.sqrt(torch.clamp(var, min=eps))
    return mean.detach(), std.detach()


def normalize_targets(pred: torch.Tensor, y: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = y_mean.view(1, -1, 1, 1)
    std = y_std.view(1, -1, 1, 1)
    return (pred - mean) / std, (y - mean) / std


def parse_ids(txt: str) -> list[str]:
    path = Path(txt)
    if path.suffix.lower() != ".csv":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            for key in ("id", "case_id", "sid", "sample_id"):
                if key in reader.fieldnames:
                    return [row[key].strip() for row in reader if row.get(key, "").strip()]

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        return [row[0].strip() for row in reader if row and row[0].strip()]


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
    ap.add_argument("--config", default=None, help="optional YAML config for this task")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--train_ids", default=None)
    ap.add_argument("--train_split", default=None)
    ap.add_argument("--val_ids", default=None)
    ap.add_argument("--val_split", default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")

    # Legacy loss re-weighting option.
    ap.add_argument("--auto_channel_weight", action="store_true")
    ap.add_argument("--auto_weight_max_batches", type=int, default=None)
    ap.add_argument("--e_weight_multiplier", type=float, default=None)
    ap.add_argument("--h_weight_multiplier", type=float, default=None)

    # Recommended explicit target normalization.
    ap.add_argument("--normalize_y", action="store_true")
    ap.add_argument("--norm_max_batches", type=int, default=None)
    ap.add_argument("--norm_eps", type=float, default=None)
    args = ap.parse_args()

    cfg = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}

    data_root = args.data_root or args.root or _cfg_get(cfg, "data_root")
    train_ids = args.train_ids or args.train_split or _cfg_get(cfg, "train_ids")
    val_ids = args.val_ids or args.val_split or _cfg_get(cfg, "val_ids")
    run_dir_str = args.run_dir or _cfg_get(cfg, "run_dir", "runs/forward_baseline")
    epochs = args.epochs if args.epochs is not None else int(_cfg_get(cfg, "epochs", 50))
    batch_size = args.batch_size if args.batch_size is not None else int(_cfg_get(cfg, "batch_size", 16))
    lr = args.lr if args.lr is not None else float(_cfg_get(cfg, "lr", 1e-3))
    seed = args.seed if args.seed is not None else _as_int(cfg, "seed", 42)
    auto_weight_max_batches = (
        args.auto_weight_max_batches
        if args.auto_weight_max_batches is not None
        else int(_cfg_get(cfg, "auto_weight_max_batches", 64))
    )
    e_weight_multiplier = (
        args.e_weight_multiplier if args.e_weight_multiplier is not None else float(_cfg_get(cfg, "e_weight_multiplier", 1.0))
    )
    h_weight_multiplier = (
        args.h_weight_multiplier if args.h_weight_multiplier is not None else float(_cfg_get(cfg, "h_weight_multiplier", 1.0))
    )
    norm_max_batches = args.norm_max_batches if args.norm_max_batches is not None else int(_cfg_get(cfg, "norm_max_batches", 64))
    norm_eps = args.norm_eps if args.norm_eps is not None else float(_cfg_get(cfg, "norm_eps", 1e-6))
    normalize_y = args.normalize_y or _as_bool({"normalize_y": _cfg_get(cfg, "normalize_y", False)}, "normalize_y", False)
    auto_channel_weight = args.auto_channel_weight or _as_bool({"auto_channel_weight": _cfg_get(cfg, "auto_channel_weight", False)}, "auto_channel_weight", False)
    resume = args.resume or _as_bool({"resume": _cfg_get(cfg, "resume", False)}, "resume", False)

    missing = [k for k, v in (("data_root", data_root), ("train_ids", train_ids), ("val_ids", val_ids)) if not v]
    if missing:
        raise SystemExit(f"Missing required argument(s): {', '.join(missing)}. Provide them via CLI or --config YAML.")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(run_dir_str)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    train_ids = parse_ids(train_ids)
    val_ids = parse_ids(val_ids)

    ds_tr = ForwardDataset(data_root, train_ids)
    ds_va = ForwardDataset(data_root, val_ids)
    num_workers = int(_cfg_get(cfg, "num_workers", 4))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = ForwardUNetLite(cin=4, cout=12).to(device)

    y_mean = None
    y_std = None
    if normalize_y:
        y_mean, y_std = estimate_target_norm_stats(
            dl_tr,
            device,
            max_batches=norm_max_batches,
            eps=norm_eps,
        )
        if y_mean is None or y_std is None:
            print("[init] failed to estimate y normalization stats, fallback to raw loss")
        else:
            print(f"[init] y_mean: {y_mean.detach().cpu().tolist()}")
            print(f"[init] y_std: {y_std.detach().cpu().tolist()}")

    channel_weights = None
    if auto_channel_weight:
        channel_weights = build_channel_weights(
            dl_tr,
            device,
            max_batches=auto_weight_max_batches,
            e_multiplier=e_weight_multiplier,
            h_multiplier=h_weight_multiplier,
        )
        print(f"[init] auto channel weights: {channel_weights}")

    loss_fn = WeightedMSELoss(weights=channel_weights)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    start_epoch = 0
    best_val = 1e30
    last_ckpt = run_dir / "checkpoints" / "last.pth"
    if resume and last_ckpt.exists():
        start_epoch, best_val = load_ckpt(last_ckpt, model, optim)
        start_epoch += 1

    log_path = run_dir / "artifacts" / "metrics.jsonl"

    def log(obj: dict):
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # store resolved config-ish info
    resolved_args = {
        "config": args.config,
        "data_root": data_root,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "run_dir": run_dir_str,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "resume": resume,
        "auto_channel_weight": auto_channel_weight,
        "auto_weight_max_batches": auto_weight_max_batches,
        "e_weight_multiplier": e_weight_multiplier,
        "h_weight_multiplier": h_weight_multiplier,
        "normalize_y": normalize_y,
        "norm_max_batches": norm_max_batches,
        "norm_eps": norm_eps,
    }
    (run_dir / "artifacts" / "run_args.json").write_text(
        json.dumps(resolved_args, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if channel_weights is not None:
        (run_dir / "artifacts" / "channel_weights.json").write_text(
            json.dumps(channel_weights, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if y_mean is not None and y_std is not None:
        (run_dir / "artifacts" / "y_norm_stats.json").write_text(
            json.dumps(
                {
                    "mean": y_mean.detach().cpu().tolist(),
                    "std": y_std.detach().cpu().tolist(),
                    "eps": norm_eps,
                    "max_batches": norm_max_batches,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    for epoch in range(start_epoch, epochs):
        model.train()
        tr_loss = 0.0
        for batch in dl_tr:
            x = batch.x.to(device, non_blocking=True)
            y = batch.y.to(device, non_blocking=True)
            pred = model(x)
            if y_mean is not None and y_std is not None:
                pred_loss, y_loss = normalize_targets(pred, y, y_mean, y_std)
            else:
                pred_loss, y_loss = pred, y
            loss = loss_fn(pred_loss, y_loss)
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
                if y_mean is not None and y_std is not None:
                    pred_loss, y_loss = normalize_targets(pred, y, y_mean, y_std)
                else:
                    pred_loss, y_loss = pred, y
                loss = loss_fn(pred_loss, y_loss)
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
