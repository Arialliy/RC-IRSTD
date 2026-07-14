from __future__ import annotations

from collections.abc import Callable
import inspect

import numpy as np
import torch
from scipy import ndimage
from torch import nn
import torch.nn.functional as F

from rc_irstd.losses.cvar import smooth_worst_group, upper_cvar


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def fallback_segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + soft_dice_loss(logits, target)


def differentiable_local_peak_mask(
    probability: torch.Tensor,
    kernel_size: int = 5,
) -> torch.Tensor:
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size must be a positive odd integer")
    pooled = F.max_pool2d(probability, kernel_size, stride=1, padding=kernel_size // 2)
    return probability >= pooled


def background_peak_cvar_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    domain_ids: torch.Tensor,
    quantile: float = 0.95,
    kernel_size: int = 5,
    exclusion_radius: int = 2,
    gamma: float = 10.0,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    if exclusion_radius > 0:
        kernel = 2 * exclusion_radius + 1
        excluded = F.max_pool2d(target, kernel, stride=1, padding=exclusion_radius) > 0
    else:
        excluded = target > 0
    peak_mask = differentiable_local_peak_mask(probability, kernel_size) & (~excluded)

    domain_risks: list[torch.Tensor] = []
    for domain in torch.unique(domain_ids):
        member_indices = torch.nonzero(domain_ids == domain, as_tuple=False).flatten()
        image_risks: list[torch.Tensor] = []
        for image_index in member_indices:
            values = probability[image_index][peak_mask[image_index]]
            if values.numel() == 0:
                # Fall back to valid background values for this image only. This
                # keeps candidate-rich images from dominating the entire domain.
                values = probability[image_index][~excluded[image_index]]
            if values.numel() == 0:
                image_risks.append(probability[image_index].sum() * 0.0)
            else:
                image_risks.append(upper_cvar(values, quantile))
        domain_risks.append(torch.stack(image_risks).mean())
    return smooth_worst_group(torch.stack(domain_risks), gamma=gamma)


def _component_scores(
    probability: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    component_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    if component_labels is not None:
        labels_batch = component_labels.to(device=probability.device, dtype=torch.long)
        if labels_batch.shape != target.shape:
            labels_batch = F.interpolate(
                labels_batch.to(torch.float32),
                size=target.shape[-2:],
                mode="nearest",
            ).to(torch.long)
    else:
        masks = target.detach().cpu().numpy() > 0.5
        label_arrays = [ndimage.label(masks[index, 0])[0] for index in range(target.shape[0])]
        labels_batch = torch.from_numpy(np.stack(label_arrays)[:, None]).to(
            probability.device, dtype=torch.long
        )

    for batch_index in range(target.shape[0]):
        labels = labels_batch[batch_index, 0]
        component_ids = torch.unique(labels)
        component_ids = component_ids[component_ids > 0]
        for component_id in component_ids:
            component = labels == component_id
            values = probability[batch_index, 0][component]
            if values.numel() == 0:
                continue
            if temperature <= 0:
                scores.append(values.max())
            else:
                # Normalised LSE pooling remains within a small additive constant
                # of the maximum and sends gradients to the whole target.
                pooled = torch.logsumexp(values * temperature, dim=0) / temperature
                pooled = pooled - torch.log(
                    torch.tensor(float(values.numel()), device=values.device)
                ) / temperature
                scores.append(pooled)
    if not scores:
        return probability.new_empty((0,))
    return torch.stack(scores)


def hard_target_miss_cvar_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    quantile: float = 0.8,
    temperature: float = 10.0,
    component_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    scores = _component_scores(
        probability,
        target,
        temperature,
        component_labels=component_labels,
    )
    if scores.numel() == 0:
        return probability.sum() * 0.0
    return upper_cvar(1.0 - scores, quantile)


class RiskAwareDetectorLoss(nn.Module):
    """SLS/base loss plus worst-domain false-peak and hard-target tails."""

    def __init__(
        self,
        base_loss: Callable[..., torch.Tensor] | None = None,
        lambda_tail: float = 0.1,
        lambda_miss: float = 0.1,
        tail_quantile: float = 0.95,
        miss_quantile: float = 0.8,
        peak_kernel: int = 5,
        exclusion_radius: int = 2,
        worst_gamma: float = 10.0,
        auxiliary_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss or fallback_segmentation_loss
        self.lambda_tail = lambda_tail
        self.lambda_miss = lambda_miss
        self.tail_quantile = tail_quantile
        self.miss_quantile = miss_quantile
        self.peak_kernel = peak_kernel
        self.exclusion_radius = exclusion_radius
        self.worst_gamma = worst_gamma
        self.auxiliary_weight = float(auxiliary_weight)
        try:
            signature = inspect.signature(self.base_loss.forward)  # type: ignore[attr-defined]
        except (AttributeError, TypeError, ValueError):
            signature = inspect.signature(self.base_loss)
        parameter_names = set(signature.parameters)
        self._base_accepts_schedule = "warm_epoch" in parameter_names or len(parameter_names) >= 4

    def _call_base(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        warm_epoch: int,
        epoch: int,
    ) -> torch.Tensor:
        if self._base_accepts_schedule:
            return self.base_loss(logits, target, warm_epoch, epoch)
        return self.base_loss(logits, target)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        domain_ids: torch.Tensor,
        auxiliary_logits: list[torch.Tensor] | None = None,
        component_labels: torch.Tensor | None = None,
        warm_epoch: int = 0,
        epoch: int = 0,
    ) -> dict[str, torch.Tensor]:
        final_base = self._call_base(logits, target, warm_epoch, epoch)
        # The reference MSHNet path supervises the final map and every
        # multi-scale auxiliary map, then averages those SLS terms. Adaptive
        # max pooling preserves tiny positive masks when matching an auxiliary
        # resolution and is equivalent to the original repeated max-pooling
        # path for the usual integer scale factors.
        if auxiliary_logits and self.auxiliary_weight > 0.0:
            auxiliary_terms: list[torch.Tensor] = []
            for auxiliary in auxiliary_logits:
                scaled_target = F.adaptive_max_pool2d(
                    target, output_size=auxiliary.shape[-2:]
                )
                auxiliary_terms.append(
                    self._call_base(auxiliary, scaled_target, warm_epoch, epoch)
                )
            auxiliary_sum = torch.stack(auxiliary_terms).sum()
            denominator = 1.0 + self.auxiliary_weight * len(auxiliary_terms)
            base = (
                final_base + self.auxiliary_weight * auxiliary_sum
            ) / denominator
        else:
            base = final_base

        if self.lambda_tail != 0.0:
            tail = background_peak_cvar_loss(
                logits,
                target,
                domain_ids,
                quantile=self.tail_quantile,
                kernel_size=self.peak_kernel,
                exclusion_radius=self.exclusion_radius,
                gamma=self.worst_gamma,
            )
        else:
            tail = logits.sum() * 0.0
        if self.lambda_miss != 0.0:
            miss = hard_target_miss_cvar_loss(
                logits,
                target,
                quantile=self.miss_quantile,
                component_labels=component_labels,
            )
        else:
            miss = logits.sum() * 0.0
        total = base + self.lambda_tail * tail + self.lambda_miss * miss
        return {"total": total, "base": base, "tail": tail, "miss": miss}
