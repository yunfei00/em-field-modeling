from ttt.config_compat import normalize_legacy_nf_data_keys


def test_nf_legacy_top_level_keys_normalized():
    cfg = {
        "data": {"name": "nf_inversion_v1"},
        "train": {"batch_size": 2},
        "data_root": "data/nf_dataset_v1",
        "train_ids": "splits/nf_v1/train.txt",
        "val_IDs": "splits/nf_v1/val.txt",
        "num_workers": 3,
    }

    normalize_legacy_nf_data_keys(cfg)

    assert cfg["data"]["root"] == "data/nf_dataset_v1"
    assert cfg["data"]["train_split"] == "splits/nf_v1/train.txt"
    assert cfg["data"]["val_split"] == "splits/nf_v1/val.txt"
    assert cfg["train"]["num_workers"] == 3


def test_nf_existing_new_keys_take_priority_over_legacy_aliases():
    cfg = {
        "data": {
            "name": "nf_inversion_v1",
            "root": "new_root",
            "train_split": "new_train",
            "val_split": "new_val",
            "data_root": "old_root",
            "train_ids": "old_train",
            "val_ids": "old_val",
        },
        "train": {},
    }

    normalize_legacy_nf_data_keys(cfg)

    assert cfg["data"]["root"] == "new_root"
    assert cfg["data"]["train_split"] == "new_train"
    assert cfg["data"]["val_split"] == "new_val"
