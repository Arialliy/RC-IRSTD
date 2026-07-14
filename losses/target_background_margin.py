"""Shift-invariant target--background tail-margin risk.

The reduction contract is deliberately fixed: local/background and object
tails are paired within each image, image risks are averaged with equal weight
within each represented source domain, and the trainer applies a normalized
smooth worst-domain aggregation.  No pixel count or domain batch size can
silently reweight the objective.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F

from losses.hard_target_loss import object_top_fraction_logits
from losses.local_peak_cvar import (
    aggregate_image_risks_by_domain,
    local_background_peak_logits,
    top_fraction_mean,
)


def _check_fraction(value: float, name: str) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}")


def image_target_background_margin_risks(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    background_q: float = 0.01,
    target_q: float = 0.2,
    object_pixel_fraction: float = 0.25,
    margin: float = 1.0,
    kernel_size: int = 3,
    plateau_atol: float = 0.0,
) -> torch.Tensor:
    """Return one target--background tail-margin violation per image.

    For an image ``i``, the background summary is the mean of its largest
    ``background_q`` deterministic local-peak logits.  The target summary is
    the mean of its lowest ``target_q`` object logits, where each object logit
    is itself the mean of its top foreground-pixel fraction.  The risk is::

        relu(margin + background_tail_logit - hard_target_logit)

    A common shift of every logit cancels exactly.  If an image has no target
    or no background candidate, its risk is a graph-connected zero instead of
    a fabricated target/background score.
    """

    _check_fraction(background_q, "background_q")
    _check_fraction(target_q, "target_q")
    _check_fraction(object_pixel_fraction, "object_pixel_fraction")
    if margin < 0.0 or not math.isfinite(margin):
        raise ValueError(f"margin must be finite and non-negative, got {margin}")

    background_by_image = local_background_peak_logits(
        logits,
        masks,
        kernel_size=kernel_size,
        plateau_atol=plateau_atol,
    )
    targets_by_image = object_top_fraction_logits(
        logits,
        masks,
        object_pixel_fraction=object_pixel_fraction,
    )
    image_risks = []
    for image_index, (background, targets) in enumerate(
        zip(background_by_image, targets_by_image)
    ):
        if background.numel() == 0 or targets.numel() == 0:
            image_risks.append(logits[image_index].sum() * 0.0)
            continue
        background_tail = top_fraction_mean(background, background_q)
        # Lowest target scores are the largest values after negation.
        hard_target = -top_fraction_mean(-targets, target_q)
        image_risks.append(F.relu(margin + background_tail - hard_target))
    return torch.stack(image_risks)


def domain_target_background_margin_risks(
    logits: torch.Tensor,
    masks: torch.Tensor,
    domain_ids: torch.Tensor,
    *,
    background_q: float = 0.01,
    target_q: float = 0.2,
    object_pixel_fraction: float = 0.25,
    margin: float = 1.0,
    kernel_size: int = 3,
    plateau_atol: float = 0.0,
    return_domain_ids: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Reduce image margin risks by an equal image mean in each domain."""

    image_risks = image_target_background_margin_risks(
        logits,
        masks,
        background_q=background_q,
        target_q=target_q,
        object_pixel_fraction=object_pixel_fraction,
        margin=margin,
        kernel_size=kernel_size,
        plateau_atol=plateau_atol,
    )
    return aggregate_image_risks_by_domain(
        image_risks,
        domain_ids,
        return_domain_ids=return_domain_ids,
    )
