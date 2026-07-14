from __future__ import annotations

import torch

from rc_irstd.losses.risk_aware import RiskAwareDetectorLoss


def _mean_logit_loss(logits, target):
    del target
    return logits.mean()


def test_auxiliary_base_loss_matches_weighted_reference_average() -> None:
    criterion = RiskAwareDetectorLoss(
        base_loss=_mean_logit_loss,
        lambda_tail=0.0,
        lambda_miss=0.0,
        auxiliary_weight=1.0,
    )
    final = torch.full((1, 1, 8, 8), 1.0)
    auxiliary = [
        torch.full((1, 1, 4, 4), 2.0),
        torch.full((1, 1, 2, 2), 4.0),
    ]
    target = torch.zeros_like(final)
    result = criterion(
        final,
        target,
        torch.zeros(1, dtype=torch.long),
        auxiliary_logits=auxiliary,
    )
    assert torch.isclose(result["base"], torch.tensor((1.0 + 2.0 + 4.0) / 3.0))
    assert torch.isclose(result["total"], result["base"])


def test_auxiliary_weight_zero_uses_final_map_only() -> None:
    criterion = RiskAwareDetectorLoss(
        base_loss=_mean_logit_loss,
        lambda_tail=0.0,
        lambda_miss=0.0,
        auxiliary_weight=0.0,
    )
    final = torch.full((1, 1, 8, 8), 1.5)
    target = torch.zeros_like(final)
    result = criterion(
        final,
        target,
        torch.zeros(1, dtype=torch.long),
        auxiliary_logits=[torch.full((1, 1, 4, 4), 10.0)],
    )
    assert torch.isclose(result["base"], torch.tensor(1.5))
