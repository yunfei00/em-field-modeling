import os
import copy
from datetime import datetime

from .utils import ensure_dir

def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def resolve_run_dir(cfg: dict, exp_name: str | None = None, run_id: str | None = None) -> tuple[str, str, str]:
    """
    returns: (run_dir, exp_name, run_id)
    run_dir = <ckpt.dir>/<exp_name>/<run_id>
    """
    cfg = copy.deepcopy(cfg)
    base_dir = cfg["ckpt"]["dir"]
    exp = exp_name or cfg["ckpt"].get("exp_name", "default")
    rid = run_id or cfg["ckpt"].get("run_id") or make_run_id()
    run_dir = os.path.join(base_dir, exp, rid)
    ensure_dir(run_dir)
    return run_dir, exp, rid

def dump_resolved_config(cfg: dict, run_dir: str) -> str:
    """
    Save a snapshot of the actual config used in this run.
    """
    import yaml
    p = os.path.join(run_dir, "config.resolved.yaml")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return p
