import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml


def test_resume_uses_new_lr_override(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"

    with open(repo_root / "configs/inversion/nf_inversion_dummy_ci.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["ckpt"]["dir"] = str(tmp_path / "runs")
    cfg["train"]["epochs"] = 1
    cfg_path = tmp_path / "nf_resume_lr.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_path)

    exp_name = "nf_resume_lr"
    run_id = "case_001"

    # First stage training
    train1 = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(cfg_path),
            "--exp_name",
            exp_name,
            "--run_id",
            run_id,
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert train1.returncode == 0, train1.stderr + "\n" + train1.stdout

    # Resume with a new lr and run one more epoch.
    cfg["train"]["epochs"] = 2
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    new_lr = 1e-4
    train2 = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(cfg_path),
            "--exp_name",
            exp_name,
            "--run_id",
            run_id,
            "--resume",
            "--lr",
            str(new_lr),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert train2.returncode == 0, train2.stderr + "\n" + train2.stdout

    ckpt = tmp_path / "runs" / exp_name / run_id / "last.pth"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    loaded_lr = float(state["optim"]["param_groups"][0]["lr"])
    assert abs(loaded_lr - new_lr) < 1e-12
