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


def _load_nf_source_csv(path: Path) -> np.ndarray:
    """Load training-format near-field source CSV to [4, 11, 11]."""
    df = pd.read_csv(path)
    expected_cols = ("Hx_re", "Hx_im", "Hy_re", "Hy_im")
    missing_cols = [k for k in expected_cols if k not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV file {path} is missing columns: {missing_cols}. "
            f"Expected training format columns: {list(expected_cols)}"
        )
    expected_len = 11 * 11
    if len(df) != expected_len:
        raise ValueError(
            f"CSV file {path} has {len(df)} rows, expected {expected_len} rows for 11x11 source input"
        )

    arr = np.stack([df[k].values.reshape(11, 11) for k in expected_cols], axis=0).astype(np.float32)
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
        arr = _load_nf_source_csv(p)
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
        # no explicit shape: 1D is treated as one sample; >=2D is treated as already batched
        if arr.ndim == 1:
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


def save_outputs(outputs: torch.Tensor, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".npy":
        np.save(p, outputs.detach().cpu().numpy())
    elif p.suffix.lower() == ".txt":
        np.savetxt(p, outputs.detach().cpu().reshape(outputs.shape[0], -1).numpy(), fmt="%.8g")
    else:
        raise ValueError("Unsupported output format. Use .npy or .txt")
