from __future__ import annotations

"""Query-risk-aligned objective for the no-reject inverse-risk calibrator."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RiskAlignedLossOutput:
    total: torch.Tensor
    violation: torch.Tensor
    utility: torch.Tensor
    oracle: torch.Tensor
    smoothness: torch.Tensor
    surrogate_pixel_risk: torch.Tensor
    surrogate_pd: torch.Tensor


def surrogate_query_pixel_risk(
    threshold_logit: torch.Tensor,
    background_logits: torch.Tensor,
    background_valid: torch.Tensor,
    background_fraction: torch.Tensor,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Differentiable query pixel false-alarm rate.

    Background logits may be a deterministic uniform sample.  Multiplication by
    ``background_fraction`` restores the denominator ``|Omega_Q|`` from the
    paper definition instead of normalising by background pixels only.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if threshold_logit.ndim != 2:
        raise ValueError("threshold_logit must have shape [B, J]")
    if background_logits.ndim != 2 or background_valid.shape != background_logits.shape:
        raise ValueError("background arrays must have shape [B, M]")
    if background_logits.shape[0] != threshold_logit.shape[0]:
        raise ValueError("batch dimensions disagree")
    if background_fraction.shape != (threshold_logit.shape[0],):
        raise ValueError("background_fraction must have shape [B]")
    valid = background_valid.to(dtype=threshold_logit.dtype)
    active = torch.sigmoid(
        (background_logits[:, None, :] - threshold_logit[:, :, None]) / float(temperature)
    )
    denominator = valid.sum(dim=1).clamp_min(1.0)[:, None]
    conditional_background_rate = (active * valid[:, None, :]).sum(dim=2) / denominator
    has_background = background_valid.any(dim=1, keepdim=True).to(threshold_logit.dtype)
    return (
        conditional_background_rate
        * background_fraction[:, None]
        * has_background
    )


def surrogate_query_pd(
    threshold_logit: torch.Tensor,
    object_scores: torch.Tensor,
    object_valid: torch.Tensor,
    *,
    temperature: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable object detection rate and episode validity mask."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if object_scores.ndim != 2 or object_valid.shape != object_scores.shape:
        raise ValueError("object arrays must have shape [B, O]")
    if object_scores.shape[0] != threshold_logit.shape[0]:
        raise ValueError("batch dimensions disagree")
    valid = object_valid.to(dtype=threshold_logit.dtype)
    detected = torch.sigmoid(
        (object_scores[:, None, :] - threshold_logit[:, :, None]) / float(temperature)
    )
    count = valid.sum(dim=1)
    pd = (detected * valid[:, None, :]).sum(dim=2) / count.clamp_min(1.0)[:, None]
    episode_valid = count > 0
    return pd, episode_valid


def threshold_curve_smoothness(
    threshold_logit: torch.Tensor,
    budgets: torch.Tensor,
) -> torch.Tensor:
    """Squared second derivative over log10 budget; zero for fewer than 3 points."""

    if threshold_logit.shape[1] < 3:
        return threshold_logit.sum() * 0.0
    if budgets.ndim == 1:
        x = torch.log10(budgets).unsqueeze(0).expand(threshold_logit.shape[0], -1)
    elif budgets.shape == threshold_logit.shape:
        x = torch.log10(budgets)
    else:
        raise ValueError("budgets must have shape [J] or [B, J]")
    dx = x[:, 1:] - x[:, :-1]
    slopes = (threshold_logit[:, 1:] - threshold_logit[:, :-1]) / dx
    midpoint_dx = 0.5 * (dx[:, 1:] + dx[:, :-1])
    second = (slopes[:, 1:] - slopes[:, :-1]) / midpoint_dx
    return second.square().mean()


def risk_aligned_calibrator_loss(
    threshold_logit: torch.Tensor,
    budgets: torch.Tensor,
    oracle_threshold_logit: torch.Tensor,
    background_logits: torch.Tensor,
    background_valid: torch.Tensor,
    background_fraction: torch.Tensor,
    object_scores: torch.Tensor,
    object_valid: torch.Tensor,
    *,
    lambda_violation: float = 4.0,
    lambda_utility: float = 1.0,
    lambda_oracle: float = 0.10,
    lambda_smoothness: float = 0.01,
    pixel_temperature: float = 0.10,
    object_temperature: float = 0.20,
    epsilon: float = 1e-8,
    huber_delta: float = 1.0,
) -> RiskAlignedLossOutput:
    """Optimise budget violation and detection utility on the independent query."""

    if any(value < 0 for value in (lambda_violation, lambda_utility, lambda_oracle, lambda_smoothness)):
        raise ValueError("loss weights must be non-negative")
    if threshold_logit.shape != oracle_threshold_logit.shape:
        raise ValueError("predicted and oracle threshold-logit curves must share shape")
    if budgets.ndim == 1:
        budget_matrix = budgets.unsqueeze(0).expand_as(threshold_logit)
    elif budgets.shape == threshold_logit.shape:
        budget_matrix = budgets
    else:
        raise ValueError("budgets must have shape [J] or [B, J]")
    if torch.any(budget_matrix <= 0):
        raise ValueError("budgets must be positive")

    pixel_risk = surrogate_query_pixel_risk(
        threshold_logit,
        background_logits,
        background_valid,
        background_fraction,
        temperature=pixel_temperature,
    )
    pd, object_episode_valid = surrogate_query_pd(
        threshold_logit,
        object_scores,
        object_valid,
        temperature=object_temperature,
    )
    log_ratio = torch.log((pixel_risk + epsilon) / (budget_matrix + epsilon))
    violation = F.relu(log_ratio).square().mean()

    if object_episode_valid.any():
        utility = (1.0 - pd[object_episode_valid]).mean()
    else:
        utility = threshold_logit.sum() * 0.0
    oracle = F.huber_loss(
        threshold_logit,
        oracle_threshold_logit,
        reduction="mean",
        delta=huber_delta,
    )
    smoothness = threshold_curve_smoothness(threshold_logit, budgets)
    total = (
        float(lambda_violation) * violation
        + float(lambda_utility) * utility
        + float(lambda_oracle) * oracle
        + float(lambda_smoothness) * smoothness
    )
    return RiskAlignedLossOutput(
        total=total,
        violation=violation,
        utility=utility,
        oracle=oracle,
        smoothness=smoothness,
        surrogate_pixel_risk=pixel_risk,
        surrogate_pd=pd,
    )
