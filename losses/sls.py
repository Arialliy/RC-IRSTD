from __future__ import annotations

"""Numerically robust SLS-IoU for the strict flat Stage-1 runtime."""

import torch
from torch import nn
import torch.nn.functional as F


def location_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if probability.shape != target.shape:
        raise ValueError("probability and target must have identical shapes")
    batch, _, height, width = probability.shape
    dtype = probability.dtype
    device = probability.device
    x_index = (
        torch.arange(width, device=device, dtype=dtype)[None, None, :]
        .expand(1, height, width)
        / max(width, 1)
    )
    y_index = (
        torch.arange(height, device=device, dtype=dtype)[None, :, None]
        .expand(1, height, width)
        / max(height, 1)
    )
    smooth = torch.finfo(dtype).eps
    losses: list[torch.Tensor] = []
    for index in range(batch):
        pred_map = probability[index]
        target_map = target[index]
        pred_centerx = (x_index * pred_map).mean()
        pred_centery = (y_index * pred_map).mean()
        target_centerx = (x_index * target_map).mean()
        target_centery = (y_index * target_map).mean()
        pred_angle = torch.atan2(pred_centery, pred_centerx + smooth)
        target_angle = torch.atan2(target_centery, target_centerx + smooth)
        angle_loss = (4.0 / (torch.pi**2)) * (pred_angle - target_angle).square()
        pred_length = torch.sqrt(pred_centerx.square() + pred_centery.square() + smooth)
        target_length = torch.sqrt(
            target_centerx.square() + target_centery.square() + smooth
        )
        length_similarity = torch.minimum(pred_length, target_length) / (
            torch.maximum(pred_length, target_length) + smooth
        )
        losses.append(1.0 - length_similarity + angle_loss)
    return torch.stack(losses).mean() if losses else probability.sum() * 0.0


class SLSIoULoss(nn.Module):
    """Empty-mask-safe Scale and Location Sensitive IoU loss."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        pred_log: torch.Tensor,
        target: torch.Tensor,
        warm_epoch: int = 1,
        epoch: int = 1,
        with_shape: bool = True,
    ) -> torch.Tensor:
        target = target.to(dtype=pred_log.dtype)
        if target.shape[-2:] != pred_log.shape[-2:]:
            target = F.interpolate(target, size=pred_log.shape[-2:], mode="nearest")
        probability = torch.sigmoid(pred_log)
        intersection_sum = (probability * target).sum(dim=(1, 2, 3))
        pred_sum = probability.sum(dim=(1, 2, 3))
        target_sum = target.sum(dim=(1, 2, 3))
        denominator = pred_sum + target_sum - intersection_sum
        iou = (intersection_sum + self.eps) / (denominator + self.eps)
        if epoch <= warm_epoch:
            return 1.0 - iou.mean()

        distance = ((pred_sum - target_sum) / 2.0).square()
        alpha = (torch.minimum(pred_sum, target_sum) + distance + self.eps) / (
            torch.maximum(pred_sum, target_sum) + distance + self.eps
        )
        loss = 1.0 - (alpha * iou).mean()
        if with_shape:
            # Location has no physical meaning when an image contains no target.
            nonempty = target_sum > 0
            if torch.any(nonempty):
                loss = loss + location_loss(probability[nonempty], target[nonempty])
        return loss


__all__ = ["SLSIoULoss", "location_loss"]
