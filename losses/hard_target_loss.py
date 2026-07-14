"""Object-aware hard-miss CVaR for infrared small targets."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from skimage import measure

from losses.local_peak_cvar import top_fraction_mean


def _check_fraction(value: float, name: str) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}")


def _normalise_inputs(
    logits: torch.Tensor,
    masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(
            "logits must have shape [batch, 1, height, width], got "
            f"{tuple(logits.shape)}"
        )
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    if masks.shape != logits.shape:
        raise ValueError(
            "masks and logits must have the same shape, got "
            f"{tuple(masks.shape)} and {tuple(logits.shape)}"
        )
    return logits, masks.to(device=logits.device, dtype=logits.dtype)


def _component_flat_indices(mask: torch.Tensor) -> List[np.ndarray]:
    """Return flat indices of 8-connected foreground objects on the CPU."""

    binary = (mask.detach().to(device="cpu").numpy() >= 0.5).astype(np.uint8)
    labels = measure.label(binary, connectivity=2)
    return [
        np.flatnonzero(labels.reshape(-1) == component_id)
        for component_id in range(1, int(labels.max()) + 1)
    ]


def _object_top_fraction_values(
    logits: torch.Tensor,
    masks: torch.Tensor,
    object_pixel_fraction: float,
    *,
    logit_space: bool,
) -> List[torch.Tensor]:
    _check_fraction(object_pixel_fraction, "object_pixel_fraction")
    logits, masks = _normalise_inputs(logits, masks)
    values = logits if logit_space else torch.sigmoid(logits)
    scores_by_image: List[torch.Tensor] = []

    for image_index in range(values.shape[0]):
        flat_values = values[image_index, 0].reshape(-1)
        object_scores = []
        for indices in _component_flat_indices(masks[image_index, 0]):
            torch_indices = torch.as_tensor(
                indices,
                device=logits.device,
                dtype=torch.long,
            )
            pixels = flat_values.index_select(0, torch_indices)
            object_scores.append(top_fraction_mean(pixels, object_pixel_fraction))

        if object_scores:
            scores_by_image.append(torch.stack(object_scores))
        else:
            # A length-zero view tied to logits gives downstream callers a
            # differentiable zero path without inventing a fake target.
            scores_by_image.append(flat_values[:0])

    return scores_by_image


def object_top_fraction_scores(
    logits: torch.Tensor,
    masks: torch.Tensor,
    object_pixel_fraction: float = 0.25,
) -> List[torch.Tensor]:
    """Return one differentiable probability score per 8-connected GT object.

    Each object's score is the mean of its highest predicted foreground
    probabilities.  Connected-component discovery is discrete and runs on a
    detached mask, while probability gathering remains in the autograd graph.
    The outer list follows batch order; entries can be empty for target-free
    images.
    """

    return _object_top_fraction_values(
        logits,
        masks,
        object_pixel_fraction,
        logit_space=False,
    )


def object_top_fraction_logits(
    logits: torch.Tensor,
    masks: torch.Tensor,
    object_pixel_fraction: float = 0.25,
) -> List[torch.Tensor]:
    """Return object scores in logit space for shift-invariant margins.

    Ranking the pixels in logit space selects the same top fraction as ranking
    probabilities because sigmoid is monotone.  Averaging logits, rather than
    probabilities, makes a target--background difference invariant to a
    common logit shift.
    """

    return _object_top_fraction_values(
        logits,
        masks,
        object_pixel_fraction,
        logit_space=True,
    )


def hard_target_miss_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    q: float = 0.2,
    object_pixel_fraction: float = 0.25,
) -> torch.Tensor:
    """CVaR over the hardest missed GT objects in the batch.

    Images without targets contribute no artificial object.  If the complete
    batch is target-free, the returned scalar is graph-connected zero and
    backpropagation is safe.
    """

    _check_fraction(q, "q")
    _check_fraction(object_pixel_fraction, "object_pixel_fraction")
    logits, masks = _normalise_inputs(logits, masks)
    scores_by_image = object_top_fraction_scores(
        logits,
        masks,
        object_pixel_fraction=object_pixel_fraction,
    )
    non_empty = [scores for scores in scores_by_image if scores.numel() > 0]
    if not non_empty:
        return logits.sum() * 0.0

    object_scores = torch.cat(non_empty)
    miss_scores = 1.0 - object_scores
    return top_fraction_mean(miss_scores, q)


def image_hard_target_miss_risks(
    logits: torch.Tensor,
    masks: torch.Tensor,
    q: float = 0.2,
    object_pixel_fraction: float = 0.25,
) -> torch.Tensor:
    """Return one hard-object miss risk per image for diagnostics."""

    _check_fraction(q, "q")
    _check_fraction(object_pixel_fraction, "object_pixel_fraction")
    logits, masks = _normalise_inputs(logits, masks)
    scores_by_image = object_top_fraction_scores(
        logits,
        masks,
        object_pixel_fraction=object_pixel_fraction,
    )
    risks = []
    for image_index, object_scores in enumerate(scores_by_image):
        if object_scores.numel() == 0:
            risks.append(logits[image_index].sum() * 0.0)
        else:
            risks.append(top_fraction_mean(1.0 - object_scores, q))
    return torch.stack(risks)
