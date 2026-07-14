from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from losses.sls import SLSIoULoss as RobustSLSIoULoss
from scripts import train_multisource_tail as stage1_trainer


def test_empty_masks_with_extreme_logits_are_finite_and_differentiable_on_cpu() -> None:
    logits = torch.stack(
        (
            torch.full((1, 4, 4), -1000.0),
            torch.full((1, 4, 4), 1000.0),
        )
    ).requires_grad_(True)
    masks = torch.zeros_like(logits)

    loss = RobustSLSIoULoss()(logits, masks, warm_epoch=0, epoch=1)

    assert loss.device.type == "cpu"
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_mixed_empty_and_nonempty_targets_are_finite_on_cpu() -> None:
    logits = torch.cat(
        (torch.full((1, 1, 8, 8), -1000.0), torch.zeros((1, 1, 8, 8)))
    ).requires_grad_(True)
    masks = torch.zeros_like(logits)
    masks[1, 0, 3:5, 4:6] = 1.0

    loss = RobustSLSIoULoss()(logits, masks, warm_epoch=1, epoch=2)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_five_stage1_output_heads_have_finite_nonzero_gradients_on_cpu() -> None:
    masks = torch.zeros((2, 1, 16, 16))
    masks[1, 0, 6:10, 7:11] = 1.0
    final_logits = torch.zeros((2, 1, 16, 16), requires_grad=True)
    auxiliary_logits = [
        torch.zeros((2, 1, 16, 16), requires_grad=True),
        torch.zeros((2, 1, 8, 8), requires_grad=True),
        torch.zeros((2, 1, 4, 4), requires_grad=True),
        torch.zeros((2, 1, 2, 2), requires_grad=True),
    ]

    loss = stage1_trainer.multiscale_sls_loss(
        RobustSLSIoULoss(),
        final_logits,
        auxiliary_logits,
        masks,
        warm_epoch=1,
        epoch=2,
    )

    assert torch.isfinite(loss)
    loss.backward()
    for prediction in [final_logits, *auxiliary_logits]:
        assert prediction.grad is not None
        assert torch.isfinite(prediction.grad).all()
        assert torch.count_nonzero(prediction.grad) > 0


def test_stage1_entrypoint_binds_and_identifies_robust_sls_for_d0_d3() -> None:
    assert stage1_trainer.SLSIoULoss is RobustSLSIoULoss
    identity = stage1_trainer.stage1_segmentation_loss_implementation()
    assert identity == {
        "qualified_name": "losses.sls.SLSIoULoss",
        "implementation_revision": "empty-mask-safe-epsilon-v1",
        "eps": 1e-8,
        "multiscale_reduction": "mean_final_plus_four_auxiliary_heads",
        "paired_stage1_variants": ["D0", "D3"],
    }
