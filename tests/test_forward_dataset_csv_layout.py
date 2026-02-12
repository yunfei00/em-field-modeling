from pathlib import Path

import numpy as np
import pandas as pd

from emfm.data.dataset import ForwardDataset


def _write_source_h_csv(path: Path):
    n = 11 * 11
    df = pd.DataFrame(
        {
            "Hx_re": np.arange(n, dtype=np.float32),
            "Hx_im": np.arange(n, dtype=np.float32) + 1,
            "Hy_re": np.arange(n, dtype=np.float32) + 2,
            "Hy_im": np.arange(n, dtype=np.float32) + 3,
        }
    )
    df.to_csv(path, index=False)


def _write_target_csv(path: Path, prefixes: tuple[str, str, str]):
    n = 51 * 51
    cols = {}
    for i, p in enumerate(prefixes):
        cols[f"{p}_re"] = np.arange(n, dtype=np.float32) + i
        cols[f"{p}_im"] = np.arange(n, dtype=np.float32) + i + 0.5
    pd.DataFrame(cols).to_csv(path, index=False)


def test_forward_dataset_supports_csv_case_layout(tmp_path: Path):
    sid = "000001"
    cdir = tmp_path / "cases" / sid
    cdir.mkdir(parents=True)

    _write_source_h_csv(cdir / "source_H.csv")
    _write_target_csv(cdir / "target_E.csv", ("Ex", "Ey", "Ez"))
    _write_target_csv(cdir / "target_H.csv", ("Hx", "Hy", "Hz"))

    ds = ForwardDataset(tmp_path, [sid])
    sample = ds[0]
    x, y = sample

    assert tuple(x.shape) == (4, 11, 11)
    assert tuple(y.shape) == (12, 51, 51)


def test_forward_dataset_prefers_npz_when_both_exist(tmp_path: Path):
    sid = "000002"

    # npz sample
    x = np.zeros((4, 11, 11), dtype=np.float32)
    y = np.ones((12, 51, 51), dtype=np.float32)
    np.savez(tmp_path / f"{sid}.npz", x=x, y=y)

    # csv sample (different values) should be ignored because npz is preferred
    cdir = tmp_path / "cases" / sid
    cdir.mkdir(parents=True)
    _write_source_h_csv(cdir / "source_H.csv")
    _write_target_csv(cdir / "target_E.csv", ("Ex", "Ey", "Ez"))
    _write_target_csv(cdir / "target_H.csv", ("Hx", "Hy", "Hz"))

    ds = ForwardDataset(tmp_path, [sid])
    x_t, y_t = ds[0]

    assert float(x_t.sum()) == 0.0
    assert float(y_t.sum()) > 0.0
