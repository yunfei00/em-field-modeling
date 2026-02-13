from __future__ import annotations

import argparse
from pathlib import Path

import torch

from emfm.models.forward_unet import ForwardUNetLite
from ttt.infer import (
    collect_input_files,
    infer_files,
    make_preview,
    parse_shape_text,
    save_outputs,
    save_outputs_as_nf_target_csv,
)


def _load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_state = ckpt.get("model")
    if model_state is None:
        raise SystemExit(f"Invalid checkpoint (missing 'model' key): {ckpt_path}")

    model = ForwardUNetLite(cin=4, cout=12).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Run inference for forward EM model.")
    ap.add_argument("--ckpt", required=True, help="checkpoint path, e.g. runs/forward_norm/best.pth")
    ap.add_argument("--input", required=True, help="input .npy/.npz/.csv file or folder")
    ap.add_argument(
        "--input_shape",
        default="4,11,11",
        help="expected sample shape, default '4,11,11' for source_H",
    )
    ap.add_argument("--out", default=None, help="optional output file/folder")
    ap.add_argument(
        "--out_format",
        default="tensor",
        choices=["tensor", "nf_target_csv"],
        help="output format",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    input_path = Path(args.input)
    files = collect_input_files(input_path)
    input_shape = parse_shape_text(args.input_shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_from_ckpt(ckpt_path, device)

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
        print(f"[{idx + 1}/{len(results)}] file={src_file} input_shape={tuple(x.shape)} output_shape={tuple(y.shape)}")
        for line in make_preview(y):
            print(line)

        if out_path is None:
            continue

        if args.out_format == "nf_target_csv":
            save_dir = (out_path / src_file.stem) if is_input_dir else out_path
            saved_pairs = save_outputs_as_nf_target_csv(y, save_dir)
            for e_path, h_path in saved_pairs:
                print(f"saved={e_path}")
                print(f"saved={h_path}")
        else:
            save_file = (out_path / f"{src_file.stem}_pred.npy") if is_input_dir else out_path
            save_outputs(y, save_file)
            print(f"saved={save_file}")


if __name__ == "__main__":
    main()
