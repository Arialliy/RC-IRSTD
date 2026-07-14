from __future__ import annotations

"""Domain-level target/background tail separation for RC-IRSTD.

The proposed detector objective is deliberately defined in *logit-difference*
space.  It therefore cannot reduce the loss by shifting every detector logit
up or down together.  Background images are weighted equally inside each
domain, target instances are weighted equally inside each domain, and domains
are combined with a normalised smooth maximum.
"""

from collections.abc import Callable
from dataclasses import dataclass
import inspect
import math

import numpy as np
import torch
from scipy import ndimage
from torch import nn
import torch.nn.functional as F


def _validate_fraction(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def top_fraction_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    """Mean of the largest ``fraction`` values, with at least one element."""
    fraction = _validate_fraction(fraction, "fraction")
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return values.sum() * 0.0
    count = max(1, int(math.ceil(fraction * flat.numel())))
    return torch.topk(flat, k=count, largest=True, sorted=False).values.mean()


def bottom_fraction_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    """Mean of the smallest ``fraction`` values, with at least one element."""
    fraction = _validate_fraction(fraction, "fraction")
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return values.sum() * 0.0
    count = max(1, int(math.ceil(fraction * flat.numel())))
    return torch.topk(flat, k=count, largest=False, sorted=False).values.mean()


def normalised_smooth_max(values: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
    """Log-mean-exp aggregation that is invariant to the number of domains."""
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return values.sum() * 0.0
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return (torch.logsumexp(float(gamma) * flat, dim=0) - math.log(flat.numel())) / float(
        gamma
    )


def dilate_target_mask(target: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate GT masks so near-target responses are not treated as background."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    binary = target > 0.5
    if radius == 0:
        return binary
    kernel = 2 * int(radius) + 1
    return F.max_pool2d(binary.to(torch.float32), kernel, stride=1, padding=radius) > 0


def _collapse_peak_plateaus(
    candidate_mask: torch.Tensor,
    detached_logits: torch.Tensor,
) -> torch.Tensor:
    """Keep one deterministic point per connected local-maximum plateau.

    Candidate selection is intentionally detached; selected logit values still
    receive gradients.  The lexicographically first maximum is retained when a
    plateau contains tied values.  This avoids the all-equal initial-logit case
    being miscounted as tens of thousands of background candidates.
    """

    if candidate_mask.ndim != 4 or candidate_mask.shape[1] != 1:
        raise ValueError("candidate_mask must have shape [B, 1, H, W]")
    output = torch.zeros_like(candidate_mask, dtype=torch.bool)
    masks_np = candidate_mask.detach().cpu().numpy().astype(bool)
    values_np = detached_logits.detach().cpu().numpy().astype(np.float64)
    for batch_index in range(candidate_mask.shape[0]):
        mask = masks_np[batch_index, 0]
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        for component_id in range(1, count + 1):
            coordinates = np.argwhere(labels == component_id)
            if coordinates.size == 0:
                continue
            values = values_np[batch_index, 0, coordinates[:, 0], coordinates[:, 1]]
            best_value = np.max(values)
            best = coordinates[values == best_value]
            order = np.lexsort((best[:, 1], best[:, 0]))
            y, x = (int(v) for v in best[order[0]])
            output[batch_index, 0, y, x] = True
    return output


def background_local_peak_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 5,
    exclusion_radius: int = 2,
    collapse_plateaus: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-rule local background peaks and the valid background mask."""

    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits must have shape [B, 1, H, W]")
    if target.shape != logits.shape:
        raise ValueError("target must have the same shape as logits")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    excluded = dilate_target_mask(target, exclusion_radius)
    valid_background = ~excluded
    detached = logits.detach()
    # Exclude target/near-target logits *before* local max pooling.  Pooling
    # the unmasked detector map first lets a strong target suppress an
    # otherwise valid neighbouring background maximum, which changes R_d^-.
    background_logits = torch.where(
        valid_background,
        detached,
        torch.full_like(detached, -torch.inf),
    )
    pooled = F.max_pool2d(
        background_logits,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    candidates = (
        (background_logits == pooled)
        & valid_background
        & torch.isfinite(detached)
    )
    if collapse_plateaus:
        candidates = _collapse_peak_plateaus(candidates, detached)
    return candidates, valid_background


def _component_label_tensor(
    target: torch.Tensor,
    component_labels: torch.Tensor | None,
) -> torch.Tensor:
    if component_labels is not None:
        labels = component_labels.to(device=target.device, dtype=torch.long)
        if labels.shape != target.shape:
            labels = F.interpolate(
                labels.to(torch.float32), size=target.shape[-2:], mode="nearest"
            ).to(torch.long)
        return labels
    masks = (target.detach().cpu().numpy() > 0.5).astype(np.uint8)
    labels_np = [
        ndimage.label(masks[index, 0], structure=np.ones((3, 3), dtype=np.uint8))[0]
        for index in range(target.shape[0])
    ]
    return torch.from_numpy(np.stack(labels_np)[:, None]).to(target.device, torch.long)


def object_top_fraction_scores(
    logits: torch.Tensor,
    target: torch.Tensor,
    top_fraction: float = 0.25,
    component_labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one differentiable logit score and image index per GT object."""

    top_fraction = _validate_fraction(top_fraction, "top_fraction")
    labels_batch = _component_label_tensor(target, component_labels)
    scores: list[torch.Tensor] = []
    image_indices: list[int] = []
    for batch_index in range(logits.shape[0]):
        labels = labels_batch[batch_index, 0]
        component_ids = torch.unique(labels)
        component_ids = component_ids[component_ids > 0]
        for component_id in component_ids:
            values = logits[batch_index, 0][labels == component_id]
            if values.numel() == 0:
                continue
            scores.append(top_fraction_mean(values, top_fraction))
            image_indices.append(batch_index)
    if not scores:
        return logits.new_empty((0,)), torch.empty(
            (0,), device=logits.device, dtype=torch.long
        )
    return torch.stack(scores), torch.tensor(
        image_indices, device=logits.device, dtype=torch.long
    )


@dataclass(frozen=True)
class DomainTailSeparationOutput:
    loss: torch.Tensor
    domain_background_tail: torch.Tensor
    domain_target_tail: torch.Tensor
    domain_gap: torch.Tensor
    domain_ids: torch.Tensor
    valid_domain_mask: torch.Tensor
    image_background_tail: torch.Tensor
    object_scores: torch.Tensor


def domain_tail_separation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    domain_ids: torch.Tensor,
    *,
    margin: float = 1.0,
    background_tail_fraction: float = 0.05,
    object_top_fraction: float = 0.25,
    hard_object_fraction: float = 0.25,
    peak_kernel_size: int = 5,
    exclusion_radius: int = 2,
    worst_gamma: float = 10.0,
    component_labels: torch.Tensor | None = None,
) -> DomainTailSeparationOutput:
    """Compute the true domain-level two-tail hinge in logit space.

    ``R_d^-`` is formed by first computing an upper background-peak tail for
    every image and then averaging images in domain ``d``.  ``R_d^+`` is the
    lower tail of one score per GT object, where each object score is the mean
    of its highest logits.  The hinge is applied *after* these two domain tails
    have been formed.
    """

    background_tail_fraction = _validate_fraction(
        background_tail_fraction, "background_tail_fraction"
    )
    hard_object_fraction = _validate_fraction(hard_object_fraction, "hard_object_fraction")
    if domain_ids.ndim != 1 or domain_ids.shape[0] != logits.shape[0]:
        raise ValueError("domain_ids must have shape [B]")

    peak_mask, valid_background = background_local_peak_mask(
        logits,
        target,
        kernel_size=peak_kernel_size,
        exclusion_radius=exclusion_radius,
        collapse_plateaus=True,
    )
    image_background_tail: list[torch.Tensor] = []
    for image_index in range(logits.shape[0]):
        values = logits[image_index][peak_mask[image_index]]
        if values.numel() == 0:
            # A fully excluded image is the only case where this remains empty.
            values = logits[image_index][valid_background[image_index]]
        image_background_tail.append(
            top_fraction_mean(values, background_tail_fraction)
            if values.numel()
            else logits[image_index].sum() * 0.0
        )
    image_background = torch.stack(image_background_tail)

    object_scores, object_image_indices = object_top_fraction_scores(
        logits,
        target,
        top_fraction=object_top_fraction,
        component_labels=component_labels,
    )
    unique_domains = torch.unique(domain_ids, sorted=True)
    background_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    gaps: list[torch.Tensor] = []
    valid_rows: list[bool] = []
    for domain in unique_domains:
        image_members = domain_ids == domain
        background_tail = image_background[image_members].mean()
        background_rows.append(background_tail)

        if object_scores.numel() > 0:
            object_domains = domain_ids[object_image_indices]
            domain_object_scores = object_scores[object_domains == domain]
        else:
            domain_object_scores = object_scores
        if domain_object_scores.numel() == 0:
            # No positive tail exists for this domain in the current batch.  The
            # background images still contribute diagnostics, but an undefined
            # positive-vs-negative hinge is excluded rather than fabricated.
            target_tail = logits.sum() * 0.0
            gap = logits.sum() * 0.0
            valid_rows.append(False)
        else:
            target_tail = bottom_fraction_mean(domain_object_scores, hard_object_fraction)
            gap = F.relu(float(margin) + background_tail - target_tail)
            valid_rows.append(True)
        target_rows.append(target_tail)
        gaps.append(gap)

    background_tensor = torch.stack(background_rows)
    target_tensor = torch.stack(target_rows)
    gap_tensor = torch.stack(gaps)
    valid_mask = torch.tensor(valid_rows, device=logits.device, dtype=torch.bool)
    if valid_mask.any():
        loss = normalised_smooth_max(gap_tensor[valid_mask], gamma=worst_gamma)
    else:
        loss = logits.sum() * 0.0
    return DomainTailSeparationOutput(
        loss=loss,
        domain_background_tail=background_tensor,
        domain_target_tail=target_tensor,
        domain_gap=gap_tensor,
        domain_ids=unique_domains,
        valid_domain_mask=valid_mask,
        image_background_tail=image_background,
        object_scores=object_scores,
    )


def risk_ramp_weight(epoch: int, start_epoch: int, ramp_epochs: int) -> float:
    """Delayed linear risk ramp used to avoid optimising random-logit tails."""

    if start_epoch < 0 or ramp_epochs < 0:
        raise ValueError("start_epoch and ramp_epochs must be non-negative")
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs == 0:
        return 1.0
    return float(min(1.0, (epoch - start_epoch + 1) / ramp_epochs))


class DomainTailSeparationDetectorLoss(nn.Module):
    """Segmentation loss plus the final RC-IRSTD domain-tail separation term."""

    def __init__(
        self,
        base_loss: Callable[..., torch.Tensor],
        *,
        lambda_sep: float = 0.2,
        margin: float = 1.0,
        background_tail_fraction: float = 0.05,
        object_top_fraction: float = 0.25,
        hard_object_fraction: float = 0.25,
        peak_kernel_size: int = 5,
        exclusion_radius: int = 2,
        worst_gamma: float = 10.0,
        risk_start_epoch: int = 5,
        risk_ramp_epochs: int = 10,
        auxiliary_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if lambda_sep < 0:
            raise ValueError("lambda_sep must be non-negative")
        self.base_loss = base_loss
        self.lambda_sep = float(lambda_sep)
        self.margin = float(margin)
        self.background_tail_fraction = background_tail_fraction
        self.object_top_fraction = object_top_fraction
        self.hard_object_fraction = hard_object_fraction
        self.peak_kernel_size = int(peak_kernel_size)
        self.exclusion_radius = int(exclusion_radius)
        self.worst_gamma = float(worst_gamma)
        self.risk_start_epoch = int(risk_start_epoch)
        self.risk_ramp_epochs = int(risk_ramp_epochs)
        self.auxiliary_weight = float(auxiliary_weight)
        try:
            signature = inspect.signature(self.base_loss.forward)  # type: ignore[attr-defined]
        except (AttributeError, TypeError, ValueError):
            signature = inspect.signature(self.base_loss)
        names = set(signature.parameters)
        self._base_accepts_schedule = "warm_epoch" in names or len(names) >= 4

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
        *,
        auxiliary_logits: list[torch.Tensor] | None = None,
        component_labels: torch.Tensor | None = None,
        warm_epoch: int = 0,
        epoch: int = 0,
    ) -> dict[str, torch.Tensor]:
        final_base = self._call_base(logits, target, warm_epoch, epoch)
        if auxiliary_logits and self.auxiliary_weight > 0:
            aux_terms: list[torch.Tensor] = []
            for auxiliary in auxiliary_logits:
                scaled_target = F.adaptive_max_pool2d(
                    target, output_size=auxiliary.shape[-2:]
                )
                aux_terms.append(
                    self._call_base(auxiliary, scaled_target, warm_epoch, epoch)
                )
            aux_sum = torch.stack(aux_terms).sum()
            denominator = 1.0 + self.auxiliary_weight * len(aux_terms)
            base = (final_base + self.auxiliary_weight * aux_sum) / denominator
        else:
            base = final_base

        schedule = risk_ramp_weight(epoch, self.risk_start_epoch, self.risk_ramp_epochs)
        if self.lambda_sep > 0.0 and schedule > 0.0:
            separation = domain_tail_separation_loss(
                logits,
                target,
                domain_ids,
                margin=self.margin,
                background_tail_fraction=self.background_tail_fraction,
                object_top_fraction=self.object_top_fraction,
                hard_object_fraction=self.hard_object_fraction,
                peak_kernel_size=self.peak_kernel_size,
                exclusion_radius=self.exclusion_radius,
                worst_gamma=self.worst_gamma,
                component_labels=component_labels,
            )
            weighted_separation = self.lambda_sep * schedule * separation.loss
            valid = separation.valid_domain_mask
            if valid.any():
                background_mean = separation.domain_background_tail[valid].mean()
                target_mean = separation.domain_target_tail[valid].mean()
                gap_mean = separation.domain_gap[valid].mean()
            else:
                zero = logits.sum() * 0.0
                background_mean = zero
                target_mean = zero
                gap_mean = zero
            separation_loss = separation.loss
            valid_count = float(valid.sum().item())
        else:
            zero = logits.sum() * 0.0
            separation_loss = zero
            weighted_separation = zero
            background_mean = zero
            target_mean = zero
            gap_mean = zero
            valid_count = 0.0
        total = base + weighted_separation
        return {
            "total": total,
            "base": base,
            "separation": separation_loss,
            "weighted_separation": weighted_separation,
            "risk_weight": logits.new_tensor(schedule),
            "background_tail": background_mean,
            "target_tail": target_mean,
            "margin_gap": gap_mean,
            "valid_domains": logits.new_tensor(valid_count),
        }
