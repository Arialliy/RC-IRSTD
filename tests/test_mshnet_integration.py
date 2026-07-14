from __future__ import annotations

from pathlib import Path

import torch

from rc_irstd.losses.risk_aware import RiskAwareDetectorLoss
from rc_irstd.losses.sls import SLSIoULoss
from rc_irstd.models.detector_adapter import DetectorAdapter, build_detector
from rc_irstd.models.mshnet import DeterministicGlobalMaxPool2d, MSHNet


def test_global_max_pool_matches_adaptive_pool_forward_and_tie_gradient() -> None:
    expected_input = torch.tensor(
        [[[[1.0, 2.0, 2.0], [0.0, 2.0, 1.0]]]], requires_grad=True
    )
    actual_input = expected_input.detach().clone().requires_grad_(True)
    expected = torch.nn.functional.adaptive_max_pool2d(expected_input, 1)
    actual = DeterministicGlobalMaxPool2d()(actual_input)
    assert torch.equal(actual, expected)
    expected.sum().backward()
    actual.sum().backward()
    assert torch.equal(actual_input.grad, expected_input.grad)


def test_bundled_mshnet_forward_backward_and_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = MSHNet(
        input_channels=3,
        channels=(2, 4, 8, 16, 32),
        blocks=(1, 1, 1, 1),
    )
    adapter = DetectorAdapter(model, "mshnet")
    images = torch.randn(2, 3, 32, 32)
    masks = torch.zeros(2, 1, 32, 32)
    masks[0, 0, 8:10, 9:11] = 1.0
    masks[1, 0, 20:23, 21:24] = 1.0
    labels = torch.zeros_like(masks, dtype=torch.long)
    labels[0, 0, 8:10, 9:11] = 1
    labels[1, 0, 20:23, 21:24] = 1
    domains = torch.tensor([0, 1], dtype=torch.long)

    output = adapter(images, training_tag=True)
    assert output.logits.shape == (2, 1, 32, 32)
    assert len(output.auxiliary_logits) == 4
    assert [tuple(x.shape[-2:]) for x in output.auxiliary_logits] == [
        (32, 32),
        (16, 16),
        (8, 8),
        (4, 4),
    ]

    criterion = RiskAwareDetectorLoss(
        base_loss=SLSIoULoss(),
        lambda_tail=0.05,
        lambda_miss=0.05,
        auxiliary_weight=1.0,
    )
    losses = criterion(
        output.logits,
        masks,
        domains,
        auxiliary_logits=output.auxiliary_logits,
        component_labels=labels,
        warm_epoch=0,
        epoch=1,
    )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    checkpoint = tmp_path / "mshnet.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint)
    # Verify the public/default architecture's checkpoint path separately using
    # its own state dict.  This catches adapter prefix and round-trip issues.
    default = build_detector("mshnet", device="cpu")
    default_checkpoint = tmp_path / "mshnet_default.pt"
    torch.save({"model_state": default.model.state_dict()}, default_checkpoint)
    reloaded = build_detector("mshnet", checkpoint=default_checkpoint, device="cpu")
    assert set(default.model.state_dict()) == set(reloaded.model.state_dict())
