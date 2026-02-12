import emfm.nf  # 触发 nf dataset / model 注册

import argparse
from pathlib import Path

import torch
import yaml

from ttt.checkpoint import load_checkpoint
from ttt.infer import collect_input_files, infer_files, make_preview, parse_shape_text, save_nf_target_csv, save_outputs
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
            "optional output path. If model output is NF [N,12,51,51], --out should be an output folder for target_E/target_H CSVs. "
            "Otherwise if --input is file, --out should be .npy/.txt file; folder input requires output folder."
        ),
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
    if is_input_dir and out_path and out_path.suffix:
        raise SystemExit("When --input is a folder, --out must be a folder path (not a file).")

    for idx, (src_file, x, y) in enumerate(results):
        print(f"[{idx+1}/{len(results)}] file={src_file} input_shape={tuple(x.shape)} output_shape={tuple(y.shape)}")
        for line in make_preview(y):
            print(line)

        if out_path is None:
            continue

        is_nf = (y.ndim == 4 and tuple(y.shape[1:]) == (12, 51, 51))
        if is_nf:
            if out_path.suffix:
                raise SystemExit("For NF output [N,12,51,51], --out must be a folder path (to write target_E.csv/target_H.csv).")
            save_dir = (out_path / src_file.stem) if is_input_dir else out_path
            saved = save_nf_target_csv(y, save_dir)
            print("saved=" + ", ".join(str(s) for s in saved))
            continue

        if is_input_dir:
            save_file = out_path / f"{src_file.stem}_pred.npy"
        else:
            if not out_path.suffix:
                raise SystemExit("When --input is a file for non-NF output, --out must be a file path ending with .npy/.txt")
            save_file = out_path
        save_outputs(y, save_file)
        print(f"saved={save_file}")


if __name__ == "__main__":
    main()
