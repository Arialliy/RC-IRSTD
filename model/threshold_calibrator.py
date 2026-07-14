"""Small budget-aware threshold and rejection calibrator."""

from __future__ import annotations

import torch
import torch.nn as nn


VALID_THRESHOLD_TRANSFORMS = ("identity", "logit", "tail")


def transform_threshold(
    threshold: torch.Tensor,
    transform: str = "identity",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply a monotone target transform while keeping episode targets raw.

    ``tail`` is ``-log(1-threshold)``.  Unlike ``log(1-threshold)``, it is
    increasing, so an underestimated transformed value still means an unsafe
    underestimated threshold and receives the asymmetric penalty.
    """

    if transform not in VALID_THRESHOLD_TRANSFORMS:
        raise ValueError(f"unknown threshold transform: {transform!r}")
    if transform == "identity":
        return threshold
    clipped = threshold.clamp(min=eps, max=1.0 - eps)
    if transform == "logit":
        return torch.logit(clipped)
    return -torch.log1p(-clipped)


def asymmetric_threshold_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    under_weight: float = 4.0,
    *,
    transform: str = "identity",
    sample_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Squared threshold error with extra cost for unsafe underestimation."""

    if under_weight < 1.0:
        raise ValueError("under_weight must be at least 1")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    pred_transformed = transform_threshold(prediction, transform)
    target_transformed = transform_threshold(target, transform)
    error = pred_transformed - target_transformed
    asymmetric_weight = torch.where(
        prediction < target,
        torch.full_like(error, float(under_weight)),
        torch.ones_like(error),
    )
    loss = asymmetric_weight * error.square()
    if sample_weight is not None:
        loss = loss * sample_weight.to(dtype=loss.dtype, device=loss.device)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        if sample_weight is None:
            return loss.mean()
        denominator = sample_weight.to(dtype=loss.dtype, device=loss.device).sum().clamp_min(1.0)
        return loss.sum() / denominator
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


class ThresholdCalibrator(nn.Module):
    """Predict a raw threshold in ``[0, 1]`` and a rejection logit."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.threshold_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.reject_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"features must have shape [batch, {self.input_dim}], got {tuple(features.shape)}"
            )
        hidden = self.encoder(features)
        threshold = self.threshold_head(hidden).squeeze(-1)
        reject_logit = self.reject_head(hidden).squeeze(-1)
        return threshold, reject_logit

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
        reject_probability: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        if not 0.0 <= reject_probability <= 1.0:
            raise ValueError("reject_probability must lie in [0, 1]")
        threshold, reject_logit = self(features)
        probability = torch.sigmoid(reject_logit)
        return {
            "threshold": threshold,
            "reject_probability": probability,
            "reject": probability >= reject_probability,
        }
