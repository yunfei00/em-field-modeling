from ttt.registry import register_model
from .models.unetish_inverter import UNetishInverter


@register_model("nf_unetish")
def build_nf_unetish(cfg: dict):
    mcfg = cfg["model"]
    return UNetishInverter(
        in_ch=mcfg.get("in_ch", 4),
        out_ch=mcfg.get("out_ch", 12),
        base=mcfg.get("base", mcfg.get("base_ch", 64)),
        n_res=mcfg.get("n_res", 6),
        add_xy_positional=mcfg.get("add_xy_positional", True),
    )
