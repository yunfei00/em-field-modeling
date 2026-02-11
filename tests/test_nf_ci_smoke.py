import os
import subprocess
import sys
from pathlib import Path

import yaml


def test_nf_train_and_eval_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"

    with open(repo_root / "configs/inversion/nf_inversion_dummy_ci.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["ckpt"]["dir"] = str(tmp_path / "runs")
    cfg_path = tmp_path / "nf_smoke.yaml"
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
            "--exp_name",
            "nf_smoke",
            "--run_id",
            "case_001",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert train.returncode == 0, train.stderr + "\n" + train.stdout

    ckpt = tmp_path / "runs" / "nf_smoke" / "case_001" / "last.pth"
    assert ckpt.exists(), f"checkpoint not found: {ckpt}"

    eval_run = subprocess.run(
        [
            sys.executable,
            "scripts/eval.py",
            "--config",
            str(cfg_path),
            "--ckpt",
            str(ckpt),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert eval_run.returncode == 0, eval_run.stderr + "\n" + eval_run.stdout
    assert "rmse_mean=" in eval_run.stdout
