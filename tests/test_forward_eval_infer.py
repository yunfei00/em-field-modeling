import torch

from emfm.tasks.forward.eval import _summary_from_rmse
from emfm.tasks.forward.infer import _load_model_from_ckpt


def test_summary_from_rmse_splits_e_h():
    rmse = torch.arange(1, 13, dtype=torch.float32)
    out = _summary_from_rmse(rmse)

    assert out["rmse_mean"] == torch.mean(rmse).item()
    assert out["rmse_e_mean"] == torch.mean(rmse[:6]).item()
    assert out["rmse_h_mean"] == torch.mean(rmse[6:]).item()


def test_load_model_from_ckpt_requires_model_key(tmp_path):
    ckpt = tmp_path / "bad.pth"
    torch.save({"epoch": 1}, ckpt)

    try:
        _load_model_from_ckpt(ckpt, torch.device("cpu"))
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "missing 'model' key" in str(e)
