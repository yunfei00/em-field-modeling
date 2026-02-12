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
    """Load near-field source CSV to [4, H, W], aligned with training loader."""
    df = pd.read_csv(path)
    expected_cols = ("Hx_re", "Hx_im", "Hy_re", "Hy_im")
    missing_cols = [k for k in expected_cols if k not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV file {path} is missing columns: {missing_cols}. "
            f"Expected training format columns: {list(expected_cols)}"
        )

    # Prefer explicit x/y coordinates when available.
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
    else:
        # Legacy fallback without x/y: infer square grid.
        side = int(round(np.sqrt(len(df))))
        if side * side != len(df):
            raise ValueError(
                f"CSV file {path} has {len(df)} rows and no x/y columns; cannot infer 2D grid"
            )
        arr = np.stack([df[k].values.reshape(side, side) for k in expected_cols], axis=0).astype(np.float32)

    # For NF inversion models, training uses [4, 11, 11].
    if arr.shape != (4, 11, 11):
        raise ValueError(
            f"CSV file {path} parsed shape {arr.shape}, expected (4, 11, 11) to match training/inference model input"
        )

    if input_shape is not None and tuple(input_shape) != (4, 11, 11):
        raise ValueError(
            f"For CSV near-field source input, expected input_shape=(4, 11, 11), got {input_shape}"
        )

    return arr


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
        # no explicit shape:
        # - 1D is treated as one sample
        # - CSV near-field source [4,11,11] is one sample and should be promoted to [1,4,11,11]
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)
        elif p.suffix.lower() == ".csv" and arr.ndim == 3:
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



def _make_xy_grid(x_min: int, x_max: int, y_min: int, y_max: int, z: int) -> pd.DataFrame:
    rows = []
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            rows.append({"x": x, "y": y, "z": z})
    return pd.DataFrame(rows)


def save_nf_target_csv(outputs: torch.Tensor, out_dir: str | Path) -> list[Path]:
    """Save NF inversion outputs to training-style target_E.csv/target_H.csv files.

    Expected output shape: [N, 12, 51, 51].
    """
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    arr = outputs.detach().cpu().numpy()
    if arr.ndim != 4 or tuple(arr.shape[1:]) != (12, 51, 51):
        raise ValueError(f"NF CSV export expects outputs shape [N,12,51,51], got {tuple(arr.shape)}")

    flat = arr.reshape(arr.shape[0], 12, -1)
    grid = _make_xy_grid(-25, 25, -25, 25, z=5)

    saved: list[Path] = []
    for i in range(arr.shape[0]):
        sample_dir = p if arr.shape[0] == 1 else (p / f"sample_{i:04d}")
        sample_dir.mkdir(parents=True, exist_ok=True)

        e_df = grid.copy()
        e_df["Ex_re"] = flat[i, 0]
        e_df["Ex_im"] = flat[i, 1]
        e_df["Ey_re"] = flat[i, 2]
        e_df["Ey_im"] = flat[i, 3]
        e_df["Ez_re"] = flat[i, 4]
        e_df["Ez_im"] = flat[i, 5]

        h_df = grid.copy()
        h_df["Hx_re"] = flat[i, 6]
        h_df["Hx_im"] = flat[i, 7]
        h_df["Hy_re"] = flat[i, 8]
        h_df["Hy_im"] = flat[i, 9]
        h_df["Hz_re"] = flat[i, 10]
        h_df["Hz_im"] = flat[i, 11]

        e_path = sample_dir / "target_E.csv"
        h_path = sample_dir / "target_H.csv"
        e_df.to_csv(e_path, index=False)
        h_df.to_csv(h_path, index=False)
        saved.extend([e_path, h_path])

    return saved

def save_outputs(outputs: torch.Tensor, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".npy":
        np.save(p, outputs.detach().cpu().numpy())
    elif p.suffix.lower() == ".txt":
        np.savetxt(p, outputs.detach().cpu().reshape(outputs.shape[0], -1).numpy(), fmt="%.8g")
    else:
        raise ValueError("Unsupported output format. Use .npy or .txt")
