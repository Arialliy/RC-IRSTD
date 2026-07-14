"""Shift-invariant target--background tail-margin risks.

The original image-paired margin API is retained as a legacy ablation.  The
final detector objective is :func:`domain_tail_separation_loss`: background
tails are formed per image and averaged per domain, target scores are pooled
over objects in each domain, and the hinge is applied only after both domain
tails exist.  This distinction matters because averaging image-level hinges
is not equivalent to a hinge between domain-level tails.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure

from losses.hard_target_loss import object_top_fraction_logits
from losses.local_peak_cvar import (
    aggregate_image_risks_by_domain,
    local_background_peak_logits,
    top_fraction_mean,
)
from losses.smooth_worst_domain import smooth_worst_domain


def _check_fraction(value: float, name: str) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}")


@dataclass(frozen=True)
class _LegacyImageMarginComponents:
    image_background_tail: torch.Tensor
    image_target_tail: torch.Tensor
    image_raw_gap: torch.Tensor
    image_violation: torch.Tensor
    image_valid_mask: torch.Tensor
    image_background_candidate_count: torch.Tensor
    object_scores: torch.Tensor
    object_image_indices: torch.Tensor


def _legacy_image_margin_components(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    background_q: float,
    target_q: float,
    object_pixel_fraction: float,
    margin: float,
    kernel_size: int,
    plateau_atol: float,
) -> _LegacyImageMarginComponents:
    """Form the historical image-paired tails without changing its semantics."""

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
    graph_zero = logits.sum() * 0.0
    background_rows: List[torch.Tensor] = []
    target_rows: List[torch.Tensor] = []
    raw_gap_rows: List[torch.Tensor] = []
    violation_rows: List[torch.Tensor] = []
    valid_rows: List[bool] = []
    object_score_rows: List[torch.Tensor] = []
    object_image_rows: List[torch.Tensor] = []
    for image_index, (background, targets) in enumerate(
        zip(background_by_image, targets_by_image)
    ):
        background_tail = (
            top_fraction_mean(background, background_q)
            if background.numel() > 0
            else graph_zero
        )
        target_tail = (
            -top_fraction_mean(-targets, target_q)
            if targets.numel() > 0
            else graph_zero
        )
        valid = background.numel() > 0 and targets.numel() > 0
        if valid:
            raw_gap = target_tail - background_tail
            violation = F.relu(margin - raw_gap)
        else:
            raw_gap = graph_zero
            violation = graph_zero
        background_rows.append(background_tail)
        target_rows.append(target_tail)
        raw_gap_rows.append(raw_gap)
        violation_rows.append(violation)
        valid_rows.append(valid)
        if targets.numel() > 0:
            object_score_rows.append(targets)
            object_image_rows.append(
                torch.full(
                    (targets.numel(),),
                    image_index,
                    device=logits.device,
                    dtype=torch.long,
                )
            )

    if object_score_rows:
        object_scores = torch.cat(object_score_rows)
        object_image_indices = torch.cat(object_image_rows)
    else:
        object_scores = logits.reshape(-1)[:0]
        object_image_indices = torch.empty(
            (0,), device=logits.device, dtype=torch.long
        )
    return _LegacyImageMarginComponents(
        image_background_tail=torch.stack(background_rows),
        image_target_tail=torch.stack(target_rows),
        image_raw_gap=torch.stack(raw_gap_rows),
        image_violation=torch.stack(violation_rows),
        image_valid_mask=torch.tensor(
            valid_rows, device=logits.device, dtype=torch.bool
        ),
        image_background_candidate_count=torch.tensor(
            [values.numel() for values in background_by_image],
            device=logits.device,
            dtype=torch.long,
        ),
        object_scores=object_scores,
        object_image_indices=object_image_indices,
    )


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

    components = _legacy_image_margin_components(
        logits,
        masks,
        background_q=background_q,
        target_q=target_q,
        object_pixel_fraction=object_pixel_fraction,
        margin=margin,
        kernel_size=kernel_size,
        plateau_atol=plateau_atol,
    )
    return components.image_violation


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


def bottom_fraction_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    """Return the mean of the smallest ``fraction`` of ``values``.

    The empty case stays connected to the input graph.  Callers must still
    distinguish an undefined target tail from a real zero score.
    """

    _check_fraction(fraction, "fraction")
    values = values.reshape(-1)
    if values.numel() == 0:
        return values.sum()
    count = max(1, int(math.ceil(fraction * values.numel())))
    return torch.topk(values, k=count, largest=False, sorted=False).values.mean()


def dilate_target_mask(masks: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Return a boolean GT exclusion mask dilated by ``radius`` pixels."""

    if isinstance(radius, bool) or int(radius) != radius or radius < 0:
        raise ValueError(f"radius must be a non-negative integer, got {radius}")
    radius = int(radius)
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(
            "masks must have shape [batch, 1, height, width], got "
            f"{tuple(masks.shape)}"
        )
    binary = masks > 0.5
    if radius == 0:
        return binary
    kernel_size = 2 * radius + 1
    return F.max_pool2d(
        binary.to(dtype=torch.float32),
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    ) > 0.0


def _collapse_peak_plateaus(
    candidate_mask: torch.Tensor,
    detached_logits: torch.Tensor,
) -> torch.Tensor:
    """Keep one deterministic representative per 8-connected plateau.

    Candidate discovery is deliberately detached.  The returned boolean mask
    is used to gather values from the original ``logits`` tensor, so selected
    peak values retain gradients.  A plateau keeps its highest value and then
    its lexicographically first coordinate, making CPU/GPU runs agree even at
    constant-logit initialization.
    """

    if candidate_mask.ndim != 4 or candidate_mask.shape[1] != 1:
        raise ValueError(
            "candidate_mask must have shape [batch, 1, height, width]"
        )
    if detached_logits.shape != candidate_mask.shape:
        raise ValueError("detached_logits and candidate_mask must have equal shapes")

    output = torch.zeros_like(candidate_mask, dtype=torch.bool)
    candidates_numpy = candidate_mask.detach().cpu().numpy().astype(np.uint8)
    logits_numpy = detached_logits.detach().cpu().numpy().astype(np.float64)
    for image_index in range(candidate_mask.shape[0]):
        labels = measure.label(candidates_numpy[image_index, 0], connectivity=2)
        for component_id in range(1, int(labels.max()) + 1):
            coordinates = np.argwhere(labels == component_id)
            if coordinates.size == 0:
                continue
            values = logits_numpy[
                image_index,
                0,
                coordinates[:, 0],
                coordinates[:, 1],
            ]
            best_value = np.max(values)
            best = coordinates[values == best_value]
            order = np.lexsort((best[:, 1], best[:, 0]))
            row, column = (int(value) for value in best[order[0]])
            output[image_index, 0, row, column] = True
    return output


def background_local_peak_mask(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    kernel_size: int = 3,
    exclusion_radius: int = 2,
    plateau_atol: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic local-background peaks and valid-background mask.

    Dilated target pixels are excluded before max pooling, so a strong target
    response cannot suppress a nearby valid background maximum.  There is no
    absolute logit cutoff: a common logit shift leaves both candidate
    selection and the resulting separation loss unchanged.
    """

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
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}"
        )
    if plateau_atol < 0.0 or not math.isfinite(plateau_atol):
        raise ValueError(
            f"plateau_atol must be finite and non-negative, got {plateau_atol}"
        )

    masks = masks.to(device=logits.device, dtype=logits.dtype)
    excluded = dilate_target_mask(masks, exclusion_radius)
    valid_background = ~excluded
    detached = logits.detach()
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
    if plateau_atol == 0.0:
        reaches_local_max = background_logits == pooled
    else:
        reaches_local_max = torch.isclose(
            background_logits,
            pooled,
            rtol=0.0,
            atol=plateau_atol,
        )
    candidates = valid_background & torch.isfinite(detached) & reaches_local_max
    return _collapse_peak_plateaus(candidates, detached), valid_background


@dataclass(frozen=True)
class DomainTailSeparationOutput:
    """Structured diagnostics for the final domain-level detector loss."""

    loss: torch.Tensor
    domain_background_tail: torch.Tensor
    domain_target_tail: torch.Tensor
    domain_raw_gap: torch.Tensor
    domain_violation: torch.Tensor
    # Backward-compatible alias for the historical, ambiguously named
    # violation field. New code should use ``domain_violation``.
    domain_gap: torch.Tensor
    domain_ids: torch.Tensor
    valid_domain_mask: torch.Tensor
    image_background_tail: torch.Tensor
    object_scores: torch.Tensor
    domain_background_candidate_mean: torch.Tensor
    domain_object_count: torch.Tensor


def legacy_image_margin_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    domain_ids: torch.Tensor,
    *,
    margin: float = 1.0,
    background_tail_fraction: float = 0.01,
    object_top_fraction: float = 0.25,
    hard_object_fraction: float = 0.2,
    peak_kernel_size: int = 3,
    worst_gamma: float = 10.0,
    plateau_atol: float = 0.0,
) -> DomainTailSeparationOutput:
    """Trainable legacy ablation: image hinge first, then domain mean.

    Invalid image pairs (no object or no background candidate) retain the
    established legacy zero-risk behavior and therefore remain in the domain
    mean as zeros. Tail and raw-gap diagnostics, however, are averaged only
    over valid image pairs so an absent target is never reported as a real
    zero-valued target tail.
    """

    if worst_gamma <= 0.0 or not math.isfinite(worst_gamma):
        raise ValueError(
            f"worst_gamma must be finite and positive, got {worst_gamma}"
        )
    components = _legacy_image_margin_components(
        logits,
        masks,
        background_q=background_tail_fraction,
        target_q=hard_object_fraction,
        object_pixel_fraction=object_top_fraction,
        margin=margin,
        kernel_size=peak_kernel_size,
        plateau_atol=plateau_atol,
    )
    domain_ids = torch.as_tensor(domain_ids, device=logits.device).reshape(-1).long()
    if domain_ids.numel() != logits.shape[0]:
        raise ValueError(
            "domain_ids must contain one entry per image, got "
            f"{domain_ids.numel()} ids for {logits.shape[0]} images"
        )
    if domain_ids.numel() == 0:
        raise ValueError("at least one image is required")

    unique_domains = torch.unique(domain_ids, sorted=True)
    graph_zero = logits.sum() * 0.0
    background_rows: List[torch.Tensor] = []
    target_rows: List[torch.Tensor] = []
    raw_gap_rows: List[torch.Tensor] = []
    violation_rows: List[torch.Tensor] = []
    candidate_rows: List[torch.Tensor] = []
    object_count_rows: List[torch.Tensor] = []
    valid_rows: List[bool] = []
    object_domains = domain_ids.index_select(
        0, components.object_image_indices
    )
    for domain_id in unique_domains:
        image_mask = domain_ids == domain_id
        valid_mask = image_mask & components.image_valid_mask
        valid = bool(valid_mask.any())
        if valid:
            background_tail = components.image_background_tail[valid_mask].mean()
            target_tail = components.image_target_tail[valid_mask].mean()
            raw_gap = components.image_raw_gap[valid_mask].mean()
        else:
            background_tail = graph_zero
            target_tail = graph_zero
            raw_gap = graph_zero
        # This all-image mean exactly preserves
        # ``domain_target_background_margin_risks`` semantics.
        violation = components.image_violation[image_mask].mean()
        background_rows.append(background_tail)
        target_rows.append(target_tail)
        raw_gap_rows.append(raw_gap)
        violation_rows.append(violation)
        candidate_rows.append(
            components.image_background_candidate_count[image_mask]
            .to(dtype=logits.dtype)
            .mean()
        )
        object_count_rows.append((object_domains == domain_id).sum())
        valid_rows.append(valid)

    domain_violation = torch.stack(violation_rows)
    return DomainTailSeparationOutput(
        loss=smooth_worst_domain(domain_violation, gamma=worst_gamma),
        domain_background_tail=torch.stack(background_rows),
        domain_target_tail=torch.stack(target_rows),
        domain_raw_gap=torch.stack(raw_gap_rows),
        domain_violation=domain_violation,
        domain_gap=domain_violation,
        domain_ids=unique_domains,
        valid_domain_mask=torch.tensor(
            valid_rows, device=logits.device, dtype=torch.bool
        ),
        image_background_tail=components.image_background_tail,
        object_scores=components.object_scores,
        domain_background_candidate_mean=torch.stack(candidate_rows),
        domain_object_count=torch.stack(object_count_rows),
    )


def domain_tail_separation_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    domain_ids: torch.Tensor,
    *,
    margin: float = 1.0,
    background_tail_fraction: float = 0.01,
    object_top_fraction: float = 0.25,
    hard_object_fraction: float = 0.2,
    peak_kernel_size: int = 3,
    exclusion_radius: int = 2,
    worst_gamma: float = 10.0,
    plateau_atol: float = 0.0,
    trainable_tail: str = "both",
) -> DomainTailSeparationOutput:
    """Compute the final domain-level target/background two-tail hinge.

    For each image, the upper tail of deterministic background-peak logits is
    computed first.  These image summaries are averaged with equal weight in
    domain ``d`` to form :math:`R_d^-`; target-free images therefore still
    contribute background evidence.  Every GT object contributes one top-pixel
    logit score, whose lower domain tail forms :math:`R_d^+`.  Only then is the
    domain hinge evaluated::

        raw_gap_d = R_d^+ - R_d^-
        violation_d = relu(margin - raw_gap_d)

    ``trainable_tail`` freezes the ablation gradient route without changing
    the forward hinge value: ``background`` detaches :math:`R_d^+`, ``target``
    detaches :math:`R_d^-`, and ``both`` is the full objective.  This makes the
    D1/D2 single-tail ablations differ from D3 only in which branch is allowed
    to update the detector.

    Domains without any target object have a defined background diagnostic but
    no fabricated positive tail, so they are excluded from the normalized
    smooth maximum.  If every represented domain is target-free, ``loss`` is a
    graph-connected zero.
    """

    _check_fraction(background_tail_fraction, "background_tail_fraction")
    _check_fraction(object_top_fraction, "object_top_fraction")
    _check_fraction(hard_object_fraction, "hard_object_fraction")
    if margin < 0.0 or not math.isfinite(margin):
        raise ValueError(f"margin must be finite and non-negative, got {margin}")
    if worst_gamma <= 0.0 or not math.isfinite(worst_gamma):
        raise ValueError(
            f"worst_gamma must be finite and positive, got {worst_gamma}"
        )
    if trainable_tail not in {"background", "target", "both"}:
        raise ValueError(
            "trainable_tail must be one of background, target, both; "
            f"got {trainable_tail!r}"
        )
    if logits.shape[0] == 0:
        raise ValueError("at least one image is required")
    domain_ids = torch.as_tensor(domain_ids, device=logits.device).reshape(-1).long()
    if domain_ids.numel() != logits.shape[0]:
        raise ValueError(
            "domain_ids must contain one entry per image, got "
            f"{domain_ids.numel()} ids for {logits.shape[0]} images"
        )

    peak_mask, valid_background = background_local_peak_mask(
        logits,
        masks,
        kernel_size=peak_kernel_size,
        exclusion_radius=exclusion_radius,
        plateau_atol=plateau_atol,
    )
    image_background_rows: List[torch.Tensor] = []
    for image_index in range(logits.shape[0]):
        values = logits[image_index][peak_mask[image_index]]
        # This fallback is reachable only if the finite valid background has no
        # selected peak (for example, an unusual non-finite score map).
        if values.numel() == 0:
            valid_values = logits[image_index][valid_background[image_index]]
            values = valid_values[torch.isfinite(valid_values)]
        if values.numel() == 0:
            image_background_rows.append(logits[image_index].sum() * 0.0)
        else:
            image_background_rows.append(
                top_fraction_mean(values, background_tail_fraction)
            )
    image_background_tail = torch.stack(image_background_rows)
    image_background_candidate_count = peak_mask.flatten(1).sum(dim=1)

    scores_by_image = object_top_fraction_logits(
        logits,
        masks,
        object_pixel_fraction=object_top_fraction,
    )
    non_empty_scores = [scores for scores in scores_by_image if scores.numel() > 0]
    if non_empty_scores:
        object_scores = torch.cat(non_empty_scores)
        object_image_indices = torch.cat(
            [
                torch.full(
                    (scores.numel(),),
                    image_index,
                    device=logits.device,
                    dtype=torch.long,
                )
                for image_index, scores in enumerate(scores_by_image)
                if scores.numel() > 0
            ]
        )
        object_domains = domain_ids.index_select(0, object_image_indices)
    else:
        object_scores = logits.reshape(-1)[:0]
        object_domains = domain_ids[:0]

    unique_domains = torch.unique(domain_ids, sorted=True)
    background_rows: List[torch.Tensor] = []
    target_rows: List[torch.Tensor] = []
    gap_rows: List[torch.Tensor] = []
    raw_gap_rows: List[torch.Tensor] = []
    candidate_rows: List[torch.Tensor] = []
    object_count_rows: List[torch.Tensor] = []
    valid_rows: List[bool] = []
    graph_zero = logits.sum() * 0.0
    for domain_id in unique_domains:
        background_tail = image_background_tail[domain_ids == domain_id].mean()
        domain_object_scores = object_scores[object_domains == domain_id]
        background_rows.append(background_tail)
        candidate_rows.append(
            image_background_candidate_count[domain_ids == domain_id]
            .to(dtype=logits.dtype)
            .mean()
        )
        object_count_rows.append((object_domains == domain_id).sum())
        if domain_object_scores.numel() == 0:
            target_tail = graph_zero
            raw_gap = graph_zero
            gap = graph_zero
            valid_rows.append(False)
        else:
            target_tail = bottom_fraction_mean(
                domain_object_scores,
                hard_object_fraction,
            )
            raw_gap = target_tail - background_tail
            if trainable_tail == "background":
                hinge_gap = target_tail.detach() - background_tail
            elif trainable_tail == "target":
                hinge_gap = target_tail - background_tail.detach()
            else:
                hinge_gap = raw_gap
            gap = F.relu(margin - hinge_gap)
            valid_rows.append(True)
        target_rows.append(target_tail)
        raw_gap_rows.append(raw_gap)
        gap_rows.append(gap)

    domain_background_tail = torch.stack(background_rows)
    domain_target_tail = torch.stack(target_rows)
    domain_raw_gap = torch.stack(raw_gap_rows)
    domain_gap = torch.stack(gap_rows)
    valid_domain_mask = torch.tensor(
        valid_rows,
        device=logits.device,
        dtype=torch.bool,
    )
    if bool(valid_domain_mask.any()):
        loss = smooth_worst_domain(
            domain_gap[valid_domain_mask],
            gamma=worst_gamma,
        )
    else:
        loss = graph_zero

    return DomainTailSeparationOutput(
        loss=loss,
        domain_background_tail=domain_background_tail,
        domain_target_tail=domain_target_tail,
        domain_raw_gap=domain_raw_gap,
        domain_violation=domain_gap,
        domain_gap=domain_gap,
        domain_ids=unique_domains,
        valid_domain_mask=valid_domain_mask,
        image_background_tail=image_background_tail,
        object_scores=object_scores,
        domain_background_candidate_mean=torch.stack(candidate_rows),
        domain_object_count=torch.stack(object_count_rows),
    )
