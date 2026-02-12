from pathlib import Path

import numpy as np
import pytest
import torch

from ttt.infer import collect_input_files, load_input_file, parse_shape_text, run_inference, save_nf_target_csv


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
    import pandas as pd

    rows = []
    for y in range(-5, 6):
        for x in range(-5, 6):
            rows.append({
                "x": x,
                "y": y,
                "z": 1,
                "Hx_re": float(x + y),
                "Hx_im": float(x - y),
                "Hy_re": float(x * y),
                "Hy_im": float(x * x + y * y),
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    x = load_input_file(csv_path)
    assert tuple(x.shape) == (1, 4, 11, 11)


def test_load_input_file_csv_invalid_input_shape_for_nf(tmp_path: Path):
    csv_path = tmp_path / "source_H.csv"
    import pandas as pd

    base = np.arange(121, dtype=np.float32)
    pd.DataFrame(
        {
            "Hx_re": base,
            "Hx_im": base + 1,
            "Hy_re": base + 2,
            "Hy_im": base + 3,
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match=r"expected input_shape=\(4, 11, 11\)"):
        load_input_file(csv_path, input_shape=(121, 4))


def test_run_inference_batch_and_single():
    model = torch.nn.Linear(4, 2)
    device = torch.device("cpu")

    single = torch.randn(1, 4)
    batch = torch.randn(5, 4)

    y1 = run_inference(model, single, device)
    y2 = run_inference(model, batch, device)

    assert tuple(y1.shape) == (1, 2)
    assert tuple(y2.shape) == (5, 2)


def test_save_nf_target_csv(tmp_path: Path):
    y = torch.arange(12 * 51 * 51, dtype=torch.float32).reshape(1, 12, 51, 51)
    saved = save_nf_target_csv(y, tmp_path)

    assert len(saved) == 2
    e_path = tmp_path / "target_E.csv"
    h_path = tmp_path / "target_H.csv"
    assert e_path.exists()
    assert h_path.exists()

    import pandas as pd

    e = pd.read_csv(e_path)
    h = pd.read_csv(h_path)

    assert list(e.columns) == ["x", "y", "z", "Ex_re", "Ex_im", "Ey_re", "Ey_im", "Ez_re", "Ez_im"]
    assert list(h.columns) == ["x", "y", "z", "Hx_re", "Hx_im", "Hy_re", "Hy_im", "Hz_re", "Hz_im"]
    assert len(e) == 51 * 51
    assert len(h) == 51 * 51
    assert int(e.iloc[0]["x"]) == -25 and int(e.iloc[0]["y"]) == -25 and int(e.iloc[0]["z"]) == 5
    assert int(e.iloc[-1]["x"]) == 25 and int(e.iloc[-1]["y"]) == 25 and int(e.iloc[-1]["z"]) == 5
