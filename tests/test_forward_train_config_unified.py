from emfm.tasks.forward.train import _cfg_get, load_ckpt, parse_ids, resolve_resume_ckpt, resolve_run_dir

import torch


def test_load_ckpt_can_override_optimizer_lr(tmp_path):
    model = torch.nn.Linear(2, 1)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    ckpt_path = tmp_path / "last.pth"
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "epoch": 1, "best_val": 0.1}, ckpt_path)

    new_optim = torch.optim.AdamW(model.parameters(), lr=5e-4)
    load_ckpt(ckpt_path, model, new_optim, resume_new_lr=True, target_lr=5e-4)

    assert abs(float(new_optim.param_groups[0]["lr"]) - 5e-4) < 1e-12


def test_forward_cfg_nested_layout_preferred():
    cfg = {
        "data": {
            "data_root": "data/em",
            "train_ids": "splits/train.txt",
            "val_ids": "splits/val.txt",
        },
        "train": {"epochs": 50, "batch_size": 16, "resume": True, "num_workers": 2},
        "optim": {"lr": 1e-3},
        "loss": {"normalize_y": True, "norm_max_batches": 128, "norm_eps": 1e-6},
        "ckpt": {"run_dir": "runs/forward_norm"},
    }

    assert _cfg_get(cfg, "data_root") == "data/em"
    assert _cfg_get(cfg, "train_ids") == "splits/train.txt"
    assert _cfg_get(cfg, "val_ids") == "splits/val.txt"
    assert _cfg_get(cfg, "epochs") == 50
    assert _cfg_get(cfg, "batch_size") == 16
    assert _cfg_get(cfg, "resume") is True
    assert _cfg_get(cfg, "num_workers") == 2
    assert _cfg_get(cfg, "lr") == 1e-3
    assert _cfg_get(cfg, "normalize_y") is True
    assert _cfg_get(cfg, "run_dir") == "runs/forward_norm"


def test_forward_cfg_flat_layout_still_supported():
    cfg = {
        "data_root": "data/em",
        "train_ids": "splits/train.txt",
        "val_ids": "splits/val.txt",
        "epochs": 50,
        "batch_size": 16,
        "resume": False,
        "num_workers": 4,
        "lr": 1e-3,
        "normalize_y": False,
        "run_dir": "runs/forward_norm",
    }

    assert _cfg_get(cfg, "data_root") == "data/em"
    assert _cfg_get(cfg, "train_ids") == "splits/train.txt"
    assert _cfg_get(cfg, "val_ids") == "splits/val.txt"
    assert _cfg_get(cfg, "epochs") == 50
    assert _cfg_get(cfg, "batch_size") == 16
    assert _cfg_get(cfg, "resume") is False
    assert _cfg_get(cfg, "num_workers") == 4
    assert _cfg_get(cfg, "lr") == 1e-3
    assert _cfg_get(cfg, "normalize_y") is False
    assert _cfg_get(cfg, "run_dir") == "runs/forward_norm"


def test_forward_cfg_nested_new_data_keys_supported():
    cfg = {
        "data": {
            "root": "data/em_v2",
            "train_split": "splits/train.csv",
            "val_split": "splits/val.csv",
        }
    }

    assert _cfg_get(cfg, "data_root") == "data/em_v2"
    assert _cfg_get(cfg, "train_ids") == "splits/train.csv"
    assert _cfg_get(cfg, "val_ids") == "splits/val.csv"


def test_forward_cfg_resume_options_supported():
    cfg = {"train": {"resume_best": True, "resume_new_lr": True}}

    assert _cfg_get(cfg, "resume_best") is True
    assert _cfg_get(cfg, "resume_new_lr") is True


def test_parse_ids_supports_csv_with_id_column(tmp_path):
    split_csv = tmp_path / "train_split.csv"
    split_csv.write_text("id,weight\n000001,1\n000002,1\n", encoding="utf-8")

    assert parse_ids(str(split_csv)) == ["000001", "000002"]


def test_resolve_resume_ckpt_prefers_outer_layout(tmp_path):
    run_dir = tmp_path / "run"
    inner = run_dir / "checkpoints" / "last.pth"
    outer = run_dir / "last.pth"
    inner.parent.mkdir(parents=True)
    inner.write_bytes(b"x")
    outer.write_bytes(b"y")

    assert resolve_resume_ckpt(run_dir) == outer


def test_resolve_resume_ckpt_supports_best_layout(tmp_path):
    run_dir = tmp_path / "run"
    inner = run_dir / "checkpoints" / "best.pth"
    outer = run_dir / "best.pth"
    inner.parent.mkdir(parents=True)
    inner.write_bytes(b"x")
    outer.write_bytes(b"y")

    assert resolve_resume_ckpt(run_dir, resume_best=True) == outer


def test_resolve_resume_ckpt_falls_back_to_inner_layout(tmp_path):
    run_dir = tmp_path / "run"
    inner = run_dir / "checkpoints" / "last.pth"
    inner.parent.mkdir(parents=True)
    inner.write_bytes(b"y")

    assert resolve_resume_ckpt(run_dir) == inner


def test_resolve_run_dir_supports_exp_name_run_id():
    resolved = resolve_run_dir(
        run_dir=None,
        ckpt_dir="runs",
        exp_name="em_forward",
        run_id="main",
    )

    assert resolved == "runs/em_forward/main"


def test_resolve_run_dir_prefers_explicit_run_dir():
    resolved = resolve_run_dir(
        run_dir="runs/custom/path",
        ckpt_dir="runs",
        exp_name="em_forward",
        run_id="main",
    )

    assert resolved == "runs/custom/path"
