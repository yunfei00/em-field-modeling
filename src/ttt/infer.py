from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch


def parse_shape_text(shape_text: str | None) -> tuple[int, ...] | None:
    if not shape_text:
        return None
    parts = [p.strip() for p in shape_text.split(",") if p.strip()]
    if not parts:
        return None
    shape = tuple(int(p) for p in parts)
    if any(v <= 0 for v in shape):
        raise ValueError("All shape values must be > 0")
    return shape


def collect_input_files(input_path: str | Path) -> list[Path]:
    """Collect input files from one file or one folder."""
    p = Path(input_path)
    if not p.exists():
        raise ValueError(f"Input path does not exist: {p}")

    if p.is_file():
        files = [p]
    else:
        files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in {".npy", ".npz", ".csv"}])
        if not files:
            raise ValueError(f"No .npy/.npz/.csv files found in folder: {p}")
    return files


def _load_nf_source_csv(path: Path, input_shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Load near-field source CSV.

    Returns one of:
    - [P*4] when `input_shape` is 1D
    - [P, 4] when `input_shape` is 2D
    - [4, H, W] when `input_shape` is 3D (or when `input_shape` is omitted)
    """
    df = pd.read_csv(path)
    expected_cols = ("Hx_re", "Hx_im", "Hy_re", "Hy_im")
    missing_cols = [k for k in expected_cols if k not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV file {path} is missing columns: {missing_cols}. "
            f"Expected training format columns: {list(expected_cols)}"
        )

    values_2d = df[list(expected_cols)].to_numpy(dtype=np.float32)

    # Default to training shape [4, H, W] so model input is [B,4,11,11] after batching.
    if input_shape is None:
        side = int(round(np.sqrt(len(df))))
        if side * side != len(df):
            raise ValueError(
                f"CSV file {path} has {len(df)} rows; cannot infer square grid for default 3D input"
            )
        return np.stack([df[k].values.reshape(side, side) for k in expected_cols], axis=0).astype(np.float32)

    if len(input_shape) == 1:
        return values_2d.reshape(-1)

    if len(input_shape) == 2:
        return values_2d

    if len(input_shape) != 3:
        raise ValueError(f"Unsupported input_shape rank for CSV {path}: {input_shape}")

    # 3D shape requested: build [4, H, W] using x/y coordinates when available.
    if {"x", "y"}.issubset(df.columns):
        xs = np.sort(df["x"].unique())
        ys = np.sort(df["y"].unique())
        expected_len = len(xs) * len(ys)
        if len(df) != expected_len:
            raise ValueError(
                f"CSV file {path} has {len(df)} rows, expected {expected_len} rows from x/y grid"
            )
        ordered = df.sort_values(["y", "x"], kind="mergesort")
        arr = np.stack(
            [ordered[k].to_numpy(dtype=np.float32).reshape(len(ys), len(xs)) for k in expected_cols],
            axis=0,
        )
        return arr

    # Fallback for legacy CSVs without x/y: infer square grid.
    side = int(round(np.sqrt(len(df))))
    if side * side != len(df):
        raise ValueError(
            f"CSV file {path} has {len(df)} rows and no x/y columns; cannot infer 2D grid for 3D input"
        )
    return np.stack([df[k].values.reshape(side, side) for k in expected_cols], axis=0).astype(np.float32)


def _pick_npz_array(npz_obj: np.lib.npyio.NpzFile) -> np.ndarray:
    if "x" in npz_obj.files:
        return np.asarray(npz_obj["x"], dtype=np.float32)
    if not npz_obj.files:
        raise ValueError("Empty .npz file")
    return np.asarray(npz_obj[npz_obj.files[0]], dtype=np.float32)


def load_input_file(path: str | Path, input_shape: tuple[int, ...] | None = None) -> torch.Tensor:
    """Load one input file into tensor [N, ...].

    Supports `.npy`, `.npz`, and near-field training-format `.csv`.
    - If data shape is [...], it is treated as one sample and promoted to [1, ...]
    - If data shape is [N, ...], it is treated as batch.
    """
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(p)
    elif p.suffix.lower() == ".npz":
        arr = _pick_npz_array(np.load(p))
    elif p.suffix.lower() == ".csv":
        arr = _load_nf_source_csv(p, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported input file: {p}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError(f"Invalid scalar input in file: {p}")

    if input_shape is not None:
        if arr.ndim == len(input_shape):
            if tuple(arr.shape) != input_shape:
                raise ValueError(f"Shape mismatch in {p}: got {tuple(arr.shape)}, expected {input_shape}")
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim == len(input_shape) + 1:
            if tuple(arr.shape[1:]) != input_shape:
                raise ValueError(
                    f"Batch shape mismatch in {p}: got {tuple(arr.shape[1:])}, expected {input_shape}"
                )
        else:
            raise ValueError(
                f"Ndim mismatch in {p}: got ndim={arr.ndim}, expected {len(input_shape)} or {len(input_shape) + 1}"
            )
    else:
        # no explicit shape: [D] and [C,H,W] are treated as one sample.
        if arr.ndim in {1, 3}:
            arr = np.expand_dims(arr, axis=0)

    return torch.from_numpy(arr)


def run_inference(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        y = model(x.to(device))
    return y.detach().cpu()


def make_preview(outputs: torch.Tensor, max_rows: int = 5) -> Sequence[str]:
    rows = outputs.detach().cpu().numpy().tolist()
    return [f"[{i}] {row}" for i, row in enumerate(rows[:max_rows])]


def infer_files(
    model: torch.nn.Module,
    files: Iterable[Path],
    device: torch.device,
    input_shape: tuple[int, ...] | None = None,
) -> list[tuple[Path, torch.Tensor, torch.Tensor]]:
    results: list[tuple[Path, torch.Tensor, torch.Tensor]] = []
    for file in files:
        x = load_input_file(file, input_shape=input_shape)
        y = run_inference(model, x, device)
        results.append((file, x, y))
    return results


def _make_target_grid_51() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.arange(-25, 26, dtype=np.int32)
    xx, yy = np.meshgrid(axis, axis)
    zz = np.full_like(xx, 5)
    return xx, yy, zz


def save_outputs_as_nf_target_csv(outputs: torch.Tensor, out_dir: str | Path) -> list[tuple[Path, Path]]:
    """Save model outputs as training-format near-field target CSV files.

    Expected output shape is [B, 12, 51, 51], channel order:
    (Ex_re, Ex_im, Ey_re, Ey_im, Ez_re, Ez_im, Hx_re, Hx_im, Hy_re, Hy_im, Hz_re, Hz_im)
    """
    arr = outputs.detach().cpu().numpy()
    if arr.ndim != 4 or arr.shape[1:] != (12, 51, 51):
        raise ValueError(
            "save_outputs_as_nf_target_csv expects outputs shape [B,12,51,51], "
            f"got {tuple(arr.shape)}"
        )

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    xx, yy, zz = _make_target_grid_51()
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)
    flat_z = zz.reshape(-1)

    saved_paths: list[tuple[Path, Path]] = []
    for i in range(arr.shape[0]):
        case_dir = out_root if arr.shape[0] == 1 else out_root / f"sample_{i:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        sample = arr[i]
        df_e = pd.DataFrame(
            {
                "x": flat_x,
                "y": flat_y,
                "z": flat_z,
                "Ex_re": sample[0].reshape(-1),
                "Ex_im": sample[1].reshape(-1),
                "Ey_re": sample[2].reshape(-1),
                "Ey_im": sample[3].reshape(-1),
                "Ez_re": sample[4].reshape(-1),
                "Ez_im": sample[5].reshape(-1),
            }
        )
        df_h = pd.DataFrame(
            {
                "x": flat_x,
                "y": flat_y,
                "z": flat_z,
                "Hx_re": sample[6].reshape(-1),
                "Hx_im": sample[7].reshape(-1),
                "Hy_re": sample[8].reshape(-1),
                "Hy_im": sample[9].reshape(-1),
                "Hz_re": sample[10].reshape(-1),
                "Hz_im": sample[11].reshape(-1),
            }
        )

        e_path = case_dir / "target_E.csv"
        h_path = case_dir / "target_H.csv"
        df_e.to_csv(e_path, index=False)
        df_h.to_csv(h_path, index=False)
        saved_paths.append((e_path, h_path))

    return saved_paths


def save_outputs(outputs: torch.Tensor, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".npy":
        np.save(p, outputs.detach().cpu().numpy())
    elif p.suffix.lower() == ".txt":
        np.savetxt(p, outputs.detach().cpu().reshape(outputs.shape[0], -1).numpy(), fmt="%.8g")
    else:
        raise ValueError("Unsupported output format. Use .npy or .txt")
