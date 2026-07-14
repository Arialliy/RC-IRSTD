from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float = 0.9,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")
    error = target - prediction
    loss = torch.maximum(quantile * error, (quantile - 1.0) * error)
    if weight is not None:
        broadcast = torch.broadcast_to(weight, loss.shape)
        denominator = broadcast.sum().clamp_min(1e-12)
        return (loss * broadcast).sum() / denominator
    return loss.mean()


def budget_focused_weight(
    target_log_risk: torch.Tensor,
    budget: float,
    base_weight: float = 1.0,
    focus_weight: float = 4.0,
    log_scale: float = 1.0,
    empty_action_weight: float = 0.1,
) -> torch.Tensor:
    """Emphasise risk-curve points near the deployment budget crossing."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    if base_weight < 0 or focus_weight < 0 or log_scale <= 0:
        raise ValueError("Invalid budget weighting parameters")
    log_budget = math.log10(float(budget))
    weight = base_weight + focus_weight * torch.exp(
        -torch.abs(target_log_risk - log_budget) / log_scale
    )
    if weight.shape[-1] > 0:
        weight = weight.clone()
        weight[..., -1] = weight[..., -1] * float(empty_action_weight)
    return weight


def crossing_loss(
    prediction_log_risk: torch.Tensor,
    target_log_risk: torch.Tensor,
    budget: float,
    temperature: float = 0.25,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary loss for predicting which thresholds satisfy a risk budget."""
    if prediction_log_risk.shape != target_log_risk.shape:
        raise ValueError("Prediction and target curves must have the same shape")
    if budget <= 0 or temperature <= 0:
        raise ValueError("budget and temperature must be positive")
    log_budget = math.log10(float(budget))
    safe_target = (target_log_risk <= log_budget).to(prediction_log_risk.dtype)
    safe_logits = (log_budget - prediction_log_risk) / temperature
    loss = F.binary_cross_entropy_with_logits(safe_logits, safe_target, reduction="none")
    if weight is not None:
        broadcast = torch.broadcast_to(weight, loss.shape)
        return (loss * broadcast).sum() / broadcast.sum().clamp_min(1e-12)
    return loss.mean()
