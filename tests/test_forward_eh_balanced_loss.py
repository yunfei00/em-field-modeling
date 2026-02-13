import torch

from emfm.tasks.forward.losses import EHBalancedMSELoss


def test_eh_balanced_loss_equal_group_weighting():
    pred = torch.zeros(1, 12, 1, 1)
    target = torch.zeros(1, 12, 1, 1)
    target[:, :6] = 10.0
    target[:, 6:] = 1.0

    loss = EHBalancedMSELoss(e_weight=1.0, h_weight=1.0)(pred, target)

    # E loss = 100, H loss = 1, balanced mean should be (100 + 1) / 2
    assert torch.isclose(loss, torch.tensor(50.5), atol=1e-6)


def test_eh_balanced_loss_with_group_weights():
    pred = torch.zeros(1, 12, 1, 1)
    target = torch.zeros(1, 12, 1, 1)
    target[:, :6] = 2.0
    target[:, 6:] = 1.0

    loss = EHBalancedMSELoss(e_weight=1.0, h_weight=3.0)(pred, target)

    # E loss = 4, H loss = 1 -> (1*4 + 3*1) / 4 = 1.75
    assert torch.isclose(loss, torch.tensor(1.75), atol=1e-6)
