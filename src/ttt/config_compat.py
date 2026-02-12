def _first_present(mapping: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def normalize_legacy_nf_data_keys(cfg: dict) -> None:
    """
    Normalize legacy near-field inversion config keys in-place.

    Supported aliases include both top-level and nested `data.*` keys:
    - data_root -> data.root
    - train_ids/train_IDs -> data.train_split
    - val_ids/val_IDs -> data.val_split
    """
    dcfg = cfg.setdefault("data", {})
    tcfg = cfg.setdefault("train", {})

    root = _first_present(dcfg, ("root", "data_root"))
    if root is None:
        root = _first_present(cfg, ("data_root", "root"))
    if root is not None and "root" not in dcfg:
        dcfg["root"] = root

    train_split = _first_present(dcfg, ("train_split", "train_ids", "train_IDs"))
    if train_split is None:
        train_split = _first_present(cfg, ("train_ids", "train_IDs", "train_split"))
    if train_split is not None and "train_split" not in dcfg:
        dcfg["train_split"] = train_split

    val_split = _first_present(dcfg, ("val_split", "val_ids", "val_IDs"))
    if val_split is None:
        val_split = _first_present(cfg, ("val_ids", "val_IDs", "val_split"))
    if val_split is not None and "val_split" not in dcfg:
        dcfg["val_split"] = val_split

    if "num_workers" not in tcfg and "num_workers" in cfg:
        tcfg["num_workers"] = cfg["num_workers"]
    if "batch_size" not in tcfg and "batch_size" in cfg:
        tcfg["batch_size"] = cfg["batch_size"]

