#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{i}: {e}") from e
    return rows


def _find_metrics_file(run_dir: Path) -> Path:
    candidates = [
        run_dir / "artifacts" / "metrics.jsonl",  # forward task
        run_dir / "metrics.jsonl",  # unified trainer
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"metrics.jsonl not found in {run_dir}. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def _parse_records(records: list[dict], run_name: str) -> pd.DataFrame:
    out: list[dict] = []

    # format 1: forward task logs (epoch-level)
    for rec in records:
        if "train_loss" in rec:
            out.append({"run": run_name, "step": int(rec.get("epoch", 0)), "split": "train", "loss": float(rec["train_loss"])})
        if "val_loss" in rec:
            out.append({"run": run_name, "step": int(rec.get("epoch", 0)), "split": "val", "loss": float(rec["val_loss"])})

    # format 2: unified trainer logs
    for rec in records:
        typ = rec.get("type", "")
        if typ == "train_step" and ("loss" in rec):
            out.append({
                "run": run_name,
                "step": int(rec.get("global_step", rec.get("epoch", 0))),
                "split": "train",
                "loss": float(rec["loss"]),
            })
        elif typ == "epoch" and ("train_loss_epoch" in rec) and (rec["train_loss_epoch"] is not None):
            out.append({
                "run": run_name,
                "step": int(rec.get("epoch", 0)),
                "split": "train_epoch",
                "loss": float(rec["train_loss_epoch"]),
            })

    if not out:
        raise ValueError(
            "No recognized loss records found. Supported keys: "
            "{train_loss,val_loss} (forward) or {type=train_step,loss} (unified)."
        )

    df = pd.DataFrame(out).sort_values(["run", "split", "step"]).reset_index(drop=True)
    return df


def _moving_average(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss curves from metrics.jsonl (supports forward/unified trainers).")
    parser.add_argument("--run_dir", nargs="+", required=True, help="One or multiple run directories.")
    parser.add_argument("--out_png", default="loss_curve.png", help="Output curve image path.")
    parser.add_argument("--out_csv", default="loss_long.csv", help="Output tidy CSV path.")
    parser.add_argument("--smooth", type=int, default=1, help="Moving average window for smoothing.")
    parser.add_argument("--title", default="Training Loss Curves")
    args = parser.parse_args()

    all_frames: list[pd.DataFrame] = []
    for run in args.run_dir:
        run_dir = Path(run)
        metrics_file = _find_metrics_file(run_dir)
        recs = _read_jsonl(metrics_file)
        all_frames.append(_parse_records(recs, run_name=run_dir.name))

    df = pd.concat(all_frames, ignore_index=True)
    df["loss_smooth"] = (
        df.groupby(["run", "split"], group_keys=False)["loss"]
        .apply(lambda s: _moving_average(s, window=args.smooth))
        .astype(float)
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib is unavailable ({e}); skip plotting, CSV is still generated.")
        return

    plt.figure(figsize=(11, 6), dpi=130)
    for (run, split), g in df.groupby(["run", "split"]):
        plt.plot(g["step"], g["loss_smooth"], label=f"{run}/{split}", linewidth=1.8)

    plt.title(args.title)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    print(f"[ok] saved curve: {out_png}")
    print(f"[ok] saved table: {out_csv}")


if __name__ == "__main__":
    main()
