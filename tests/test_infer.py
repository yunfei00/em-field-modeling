from pathlib import Path

import numpy as np
import torch

from ttt.infer import collect_input_files, load_input_file, parse_shape_text, run_inference


def test_parse_shape_text():
    assert parse_shape_text("16") == (16,)
    assert parse_shape_text("3,11,11") == (3, 11, 11)
    assert parse_shape_text(None) is None


def test_collect_input_files_file_and_dir(tmp_path: Path):
    f1 = tmp_path / "a.npy"
    f2 = tmp_path / "b.npz"
    np.save(f1, np.zeros((16,), dtype=np.float32))
    np.savez(f2, x=np.zeros((16,), dtype=np.float32))

    only = collect_input_files(f1)
    assert only == [f1]

    many = collect_input_files(tmp_path)
    assert many == [f1, f2]


def test_load_input_file_single_and_batch(tmp_path: Path):
    s = tmp_path / "s.npy"
    b = tmp_path / "b.npy"
    np.save(s, np.arange(4, dtype=np.float32))
    np.save(b, np.arange(12, dtype=np.float32).reshape(3, 4))

    xs = load_input_file(s, input_shape=(4,))
    xb = load_input_file(b, input_shape=(4,))

    assert tuple(xs.shape) == (1, 4)
    assert tuple(xb.shape) == (3, 4)


def test_load_input_file_csv_training_format(tmp_path: Path):
    csv_path = tmp_path / "source_H.csv"
    base = np.arange(121, dtype=np.float32)
    import pandas as pd

    pd.DataFrame(
        {
            "Hx_re": base,
            "Hx_im": base + 1,
            "Hy_re": base + 2,
            "Hy_im": base + 3,
        }
    ).to_csv(csv_path, index=False)

    x = load_input_file(csv_path, input_shape=(4, 11, 11))
    assert tuple(x.shape) == (1, 4, 11, 11)


def test_run_inference_batch_and_single():
    model = torch.nn.Linear(4, 2)
    device = torch.device("cpu")

    single = torch.randn(1, 4)
    batch = torch.randn(5, 4)

    y1 = run_inference(model, single, device)
    y2 = run_inference(model, batch, device)

    assert tuple(y1.shape) == (1, 2)
    assert tuple(y2.shape) == (5, 2)
