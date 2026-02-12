import emfm.nf  # 触发 nf dataset / model 注册

import argparse
from pathlib import Path

import torch
import yaml

from ttt.checkpoint import load_checkpoint
from ttt.infer import (
    collect_input_files,
    infer_files,
    make_preview,
    parse_shape_text,
    save_outputs,
    save_outputs_as_nf_target_csv,
)
from ttt.models import build_model


def main():
    ap = argparse.ArgumentParser(description="Run model inference for one file or a folder of files.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True, help="checkpoint path, e.g. runs/<exp>/<run>/best.pth")
    ap.add_argument("--input", required=True, help="input file(.npy/.npz) or folder containing .npy/.npz")
    ap.add_argument("--input_shape", default=None, help="optional expected sample shape, e.g. '16' or '3,11,11'")
    ap.add_argument(
        "--out",
        default=None,
        help=(
            "optional output path. If --input is file, --out should be .npy/.txt file (or folder with --out_format nf_target_csv). "
            "If --input is folder, --out should be an output folder."
        ),
    )
    ap.add_argument(
        "--out_format",
        default="tensor",
        choices=["tensor", "nf_target_csv"],
        help="output format: tensor(.npy/.txt) or near-field target csv files(target_E.csv + target_H.csv)",
    )
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    model = build_model(cfg)
    state = load_checkpoint(str(ckpt_path), map_location="cpu")
    if "model" not in state:
        raise SystemExit(f"Invalid checkpoint (missing 'model' key): {ckpt_path}")
    model.load_state_dict(state["model"])

    input_path = Path(args.input)
    input_shape = parse_shape_text(args.input_shape)
    files = collect_input_files(input_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = infer_files(model, files, device=device, input_shape=input_shape)

    is_input_dir = input_path.is_dir()
    out_path = Path(args.out) if args.out else None
    if args.out_format == "tensor":
        if is_input_dir and out_path and out_path.suffix:
            raise SystemExit("When --input is a folder, --out must be a folder path (not a file).")
        if (not is_input_dir) and out_path and (not out_path.suffix):
            raise SystemExit("When --input is a file, --out must be a file path ending with .npy/.txt")
    else:
        if out_path is not None and out_path.suffix:
            raise SystemExit("With --out_format nf_target_csv, --out must be a folder path.")

    for idx, (src_file, x, y) in enumerate(results):
        print(f"[{idx+1}/{len(results)}] file={src_file} input_shape={tuple(x.shape)} output_shape={tuple(y.shape)}")
        for line in make_preview(y):
            print(line)

        if out_path is None:
            continue

        if args.out_format == "nf_target_csv":
            if is_input_dir:
                save_dir = out_path / src_file.stem
            else:
                save_dir = out_path
            saved_pairs = save_outputs_as_nf_target_csv(y, save_dir)
            for e_path, h_path in saved_pairs:
                print(f"saved={e_path}")
                print(f"saved={h_path}")
        else:
            if is_input_dir:
                save_file = out_path / f"{src_file.stem}_pred.npy"
            else:
                save_file = out_path
            save_outputs(y, save_file)
            print(f"saved={save_file}")


if __name__ == "__main__":
    main()
