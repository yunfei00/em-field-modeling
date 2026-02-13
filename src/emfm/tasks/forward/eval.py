from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from emfm.data.dataset import ForwardDataset, collate_forward_samples
from emfm.models.forward_unet import ForwardUNetLite
from emfm.tasks.forward.train import _cfg_get, parse_ids


def _load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_state = ckpt.get("model")
    if model_state is None:
        raise SystemExit(f"Invalid checkpoint (missing 'model' key): {ckpt_path}")

    model = ForwardUNetLite(cin=4, cout=12).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model


def _summary_from_rmse(rmse_c: torch.Tensor) -> dict:
    e_rmse = rmse_c[:6]
    h_rmse = rmse_c[6:]
    return {
        "rmse_mean": float(rmse_c.mean().item()),
        "rmse_e_mean": float(e_rmse.mean().item()),
        "rmse_h_mean": float(h_rmse.mean().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate forward EM model on validation split.")
    ap.add_argument("--config", default="configs/forward/forward_train.yaml")
    ap.add_argument("--ckpt", required=True, help="checkpoint path, e.g. runs/forward_norm/best.pth")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--val_ids", default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--out_json", default=None, help="optional output json path")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}

    data_root = args.data_root or _cfg_get(cfg, "data_root")
    val_ids_file = args.val_ids or _cfg_get(cfg, "val_ids")
    batch_size = args.batch_size if args.batch_size is not None else int(_cfg_get(cfg, "batch_size", 16))
    num_workers = args.num_workers if args.num_workers is not None else int(_cfg_get(cfg, "num_workers", 4))

    missing = [k for k, v in (("data_root", data_root), ("val_ids", val_ids_file)) if not v]
    if missing:
        raise SystemExit(f"Missing required argument(s): {', '.join(missing)}")

    val_ids = parse_ids(val_ids_file)
    ds_val = ForwardDataset(data_root, val_ids)
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_forward_samples,
    )

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_from_ckpt(ckpt_path, device)

    sum_sq = torch.zeros(12, dtype=torch.float64)
    n_elem = 0
    val_loss_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dl_val:
            x = batch.x.to(device, non_blocking=True)
            y = batch.y.to(device, non_blocking=True)
            pred = model(x)

            diff = pred - y
            val_loss_sum += float((diff.pow(2)).mean().item())
            n_batches += 1

            sum_sq += diff.pow(2).sum(dim=(0, 2, 3)).detach().cpu().to(torch.float64)
            n_elem += int(y.shape[0] * y.shape[2] * y.shape[3])

    if n_elem == 0:
        raise SystemExit("Validation split is empty.")

    rmse_c = torch.sqrt(sum_sq / float(n_elem)).to(torch.float32)
    summary = _summary_from_rmse(rmse_c)

    metrics = {
        "ckpt": str(ckpt_path),
        "num_samples": len(ds_val),
        "num_batches": n_batches,
        "val_mse": val_loss_sum / max(1, n_batches),
        "rmse_per_channel": [float(v) for v in rmse_c.tolist()],
        **summary,
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved={out}")


if __name__ == "__main__":
    main()
