"""Tail-risk losses over background local maxima.

The aggregation order is intentional: pixels are reduced to a tail risk for
each image first, and image risks are then averaged within each source domain.
This prevents large images or domains with more images in a batch from
silently receiving more weight.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F


def _check_fraction(value: float, name: str) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}")


def _normalise_inputs(
    logits: torch.Tensor,
    masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 4:
        raise ValueError(f"logits must be BCHW, got shape {tuple(logits.shape)}")
    if logits.shape[1] != 1:
        raise ValueError(
            "local-peak risk currently supports one foreground channel, "
            f"got {logits.shape[1]}"
        )
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    if masks.shape != logits.shape:
        raise ValueError(
            "masks and logits must have the same shape, got "
            f"{tuple(masks.shape)} and {tuple(logits.shape)}"
        )
    return logits, masks.to(device=logits.device, dtype=logits.dtype)


def top_fraction_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    """Return the mean of the largest ``fraction`` of a tensor.

    An empty tensor returns ``values.sum()``.  Unlike constructing a fresh
    scalar zero, that expression remains attached to the caller's autograd
    graph and therefore produces a well-defined zero gradient.
    """

    _check_fraction(fraction, "fraction")
    values = values.reshape(-1)
    if values.numel() == 0:
        return values.sum()

    k = max(1, int(math.ceil(fraction * values.numel())))
    return torch.topk(values, k=k, largest=True, sorted=False).values.mean()


def _plateau_representatives(
    candidate_mask: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """Choose deterministic representatives instead of retaining a plateau.

    A plain equality-to-max-pool test marks every pixel of a constant plateau
    as a peak.  Here a row-major rank is used only to break such ties; gathered
    probability values, not the ranks, remain the differentiable loss inputs.
    On an ordinary rectangular plateau this keeps exactly one representative.
    """

    _, _, height, width = candidate_mask.shape
    rank = torch.arange(
        height * width,
        device=candidate_mask.device,
        dtype=torch.float32,
    ).reshape(1, 1, height, width)
    ranked_candidates = torch.where(
        candidate_mask,
        rank,
        torch.full_like(rank, -1.0),
    )
    local_rank_max = F.max_pool2d(
        ranked_candidates,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    return candidate_mask & (ranked_candidates == local_rank_max)


def local_background_peak_scores(
    logits: torch.Tensor,
    masks: torch.Tensor,
    kernel_size: int = 3,
    min_score: float = 0.05,
    plateau_atol: float = 0.0,
) -> List[torch.Tensor]:
    """Extract differentiable background peak probabilities per image.

    Foreground pixels never participate in max pooling. ``min_score`` avoids
    treating a low, constant background as a large collection of candidates.
    Exact plateaus are deterministically reduced by
    :func:`_plateau_representatives`.
    """

    logits, masks = _normalise_inputs(logits, masks)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError(f"min_score must be in [0, 1], got {min_score}")
    if plateau_atol < 0.0:
        raise ValueError(f"plateau_atol must be non-negative, got {plateau_atol}")

    probabilities = torch.sigmoid(logits)
    background_mask = masks < 0.5
    minus_inf = torch.full_like(probabilities, -torch.inf)
    background_scores = torch.where(background_mask, probabilities, minus_inf)
    pooled = F.max_pool2d(
        background_scores,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    if plateau_atol == 0.0:
        reaches_local_max = background_scores == pooled
    else:
        reaches_local_max = torch.isclose(
            background_scores,
            pooled,
            rtol=0.0,
            atol=plateau_atol,
        )
    candidates = background_mask & (probabilities >= min_score) & reaches_local_max
    representatives = _plateau_representatives(candidates, kernel_size)

    return [
        probabilities[index, 0][representatives[index, 0]]
        for index in range(probabilities.shape[0])
    ]


def local_background_peak_logits(
    logits: torch.Tensor,
    masks: torch.Tensor,
    kernel_size: int = 3,
    plateau_atol: float = 0.0,
) -> List[torch.Tensor]:
    """Extract deterministic background local maxima in logit space.

    This is the shift-invariant counterpart of
    :func:`local_background_peak_scores` for target--background margin losses.
    It deliberately has no absolute score threshold: adding a common constant
    to every logit therefore changes neither the candidate set nor the
    returned target--background differences.  Empty-background images return
    a length-zero view that remains attached to ``logits``.
    """

    logits, masks = _normalise_inputs(logits, masks)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    if plateau_atol < 0.0:
        raise ValueError(f"plateau_atol must be non-negative, got {plateau_atol}")

    background_mask = masks < 0.5
    minus_inf = torch.full_like(logits, -torch.inf)
    background_logits = torch.where(background_mask, logits, minus_inf)
    pooled = F.max_pool2d(
        background_logits,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    if plateau_atol == 0.0:
        reaches_local_max = background_logits == pooled
    else:
        reaches_local_max = torch.isclose(
            background_logits,
            pooled,
            rtol=0.0,
            atol=plateau_atol,
        )
    candidates = background_mask & reaches_local_max
    representatives = _plateau_representatives(candidates, kernel_size)
    return [
        logits[index, 0][representatives[index, 0]]
        for index in range(logits.shape[0])
    ]


def image_tail_risks(
    logits: torch.Tensor,
    masks: torch.Tensor,
    q: float = 0.01,
    kernel_size: int = 3,
    min_score: float = 0.05,
    plateau_atol: float = 0.0,
) -> torch.Tensor:
    """Compute local-peak CVaR independently for every image."""

    _check_fraction(q, "q")
    peaks = local_background_peak_scores(
        logits,
        masks,
        kernel_size=kernel_size,
        min_score=min_score,
        plateau_atol=plateau_atol,
    )
    return torch.stack([top_fraction_mean(values, q) for values in peaks])


def image_background_pixel_tail_risks(
    logits: torch.Tensor,
    masks: torch.Tensor,
    q: float = 0.01,
) -> torch.Tensor:
    """Pixel-top-k baseline with the same image-first aggregation contract."""

    _check_fraction(q, "q")
    logits, masks = _normalise_inputs(logits, masks)
    probabilities = torch.sigmoid(logits)
    background_mask = masks < 0.5
    values = [
        probabilities[index][background_mask[index]]
        for index in range(probabilities.shape[0])
    ]
    return torch.stack([top_fraction_mean(item, q) for item in values])


def aggregate_image_risks_by_domain(
    risks: torch.Tensor,
    domain_ids: torch.Tensor,
    return_domain_ids: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Average one scalar risk per image within each represented domain."""

    if risks.ndim != 1:
        raise ValueError(f"risks must be one-dimensional, got {tuple(risks.shape)}")
    domain_ids = torch.as_tensor(domain_ids, device=risks.device).reshape(-1).long()
    if domain_ids.numel() != risks.numel():
        raise ValueError(
            "domain_ids must contain one entry per image, got "
            f"{domain_ids.numel()} ids for {risks.numel()} risks"
        )
    if risks.numel() == 0:
        raise ValueError("at least one image risk is required")

    unique_ids = torch.unique(domain_ids, sorted=True)
    domain_risks = torch.stack(
        [risks[domain_ids == domain_id].mean() for domain_id in unique_ids]
    )
    if return_domain_ids:
        return domain_risks, unique_ids
    return domain_risks


def domain_tail_risks(
    logits: torch.Tensor,
    masks: torch.Tensor,
    domain_ids: torch.Tensor,
    q: float = 0.01,
    kernel_size: int = 3,
    min_score: float = 0.05,
    plateau_atol: float = 0.0,
    return_domain_ids: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Local-peak CVaR reduced per image and then averaged per domain."""

    risks = image_tail_risks(
        logits,
        masks,
        q=q,
        kernel_size=kernel_size,
        min_score=min_score,
        plateau_atol=plateau_atol,
    )
    return aggregate_image_risks_by_domain(risks, domain_ids, return_domain_ids)


def domain_pixel_tail_risks(
    logits: torch.Tensor,
    masks: torch.Tensor,
    domain_ids: torch.Tensor,
    q: float = 0.01,
    return_domain_ids: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Pixel-top-k baseline reduced per image and then averaged per domain."""

    risks = image_background_pixel_tail_risks(logits, masks, q=q)
    return aggregate_image_risks_by_domain(risks, domain_ids, return_domain_ids)
