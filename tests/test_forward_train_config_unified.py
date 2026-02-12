from emfm.tasks.forward.train import _cfg_get


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
