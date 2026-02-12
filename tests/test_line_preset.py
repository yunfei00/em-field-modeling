import os
import subprocess
import sys
from pathlib import Path

import yaml


def test_train_line_forward_preset(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"

    with open(repo_root / "configs/inversion/nf_inversion_dummy_ci.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["ckpt"]["dir"] = str(tmp_path / "runs")
    cfg_path = tmp_path / "nf_line.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_path)

    train = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(cfg_path),
            "--line",
            "forward",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert train.returncode == 0, train.stderr + "\n" + train.stdout

    ckpt = tmp_path / "runs" / "em_forward" / "main" / "last.pth"
    assert ckpt.exists(), f"checkpoint not found: {ckpt}"
