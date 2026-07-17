"""Direct no-reject Stage-2 pixel-threshold baseline.

The primary T8 model predicts a structurally monotone inverse-risk curve.  T6
is the deliberately simpler control: one shared hidden block followed by one
independent logit per frozen pixel budget.  It does not sort or project its
outputs because doing so would erase the mechanism being tested by T7/T8.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from model.monotone_pixel_calibrator import NoRejectPixelRiskCalibratorOutput


DIRECT_NO_REJECT_MODEL_ID = "direct_no_reject_pixel_calibrator"
DIRECT_NO_REJECT_BUDGET_SCOPE = "pixel_risk_full_grid_no_component_no_reject"


class DirectNoRejectPixelCalibrator(nn.Module):
    """Predict an independent threshold at every trained pixel-risk budget.

    ``hidden_dims`` is intentionally an explicit one-element sequence.  W08
    freezes it to ``[32]`` for a 93 -> 32 -> 3 model with 3,107 trainable
    parameters.  Requests outside, between, or in a different order from the
    trained grid are rejected: a direct head has no preregistered interpolation
    semantics.
    """

    budget_scope = DIRECT_NO_REJECT_BUDGET_SCOPE
    supports_component_budget = False
    supports_reject = False
    supports_complete_budget_curve = True
    supports_query_risk_aligned_loss = False
    structural_monotonicity = False
    training_pipeline_integrated = True

    def __init__(
        self,
        context_feature_dim: int,
        pixel_budget_grid: Sequence[float],
        hidden_dims: Sequence[int] = (32,),
        dropout: float = 0.1,
        min_logit: float = -10.0,
        max_logit: float = 18.0,
    ) -> None:
        super().__init__()
        if (
            isinstance(context_feature_dim, bool)
            or not isinstance(context_feature_dim, int)
            or context_feature_dim <= 0
        ):
            raise ValueError("context_feature_dim must be a positive integer")
        if isinstance(dropout, bool):
            raise TypeError("dropout must be a real number, not bool")
        dropout = float(dropout)
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite and lie in [0, 1)")
        if isinstance(min_logit, bool) or isinstance(max_logit, bool):
            raise TypeError("logit bounds must be real numbers, not bool")
        min_logit = float(min_logit)
        max_logit = float(max_logit)
        if not math.isfinite(min_logit) or not math.isfinite(max_logit):
            raise ValueError("logit bounds must be finite")
        if min_logit >= max_logit:
            raise ValueError("min_logit must be below max_logit")

        if isinstance(hidden_dims, (str, bytes)):
            raise TypeError("hidden_dims must be a one-element integer sequence")
        widths = list(hidden_dims)
        if len(widths) != 1:
            raise ValueError("DirectNoRejectPixelCalibrator requires one hidden layer")
        width = widths[0]
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("hidden_dims must contain one positive integer")

        if isinstance(pixel_budget_grid, (str, bytes)):
            raise TypeError("pixel_budget_grid must be a numeric sequence")
        try:
            budgets = tuple(float(value) for value in pixel_budget_grid)
        except (TypeError, ValueError) as error:
            raise ValueError("pixel_budget_grid must be a numeric sequence") from error
        if not budgets:
            raise ValueError("pixel_budget_grid must not be empty")
        if not all(math.isfinite(value) and value > 0.0 for value in budgets):
            raise ValueError("pixel_budget_grid must contain finite positive values")
        if not all(loose > strict for loose, strict in zip(budgets, budgets[1:])):
            raise ValueError("pixel_budget_grid must be strictly descending")

        self.context_feature_dim = context_feature_dim
        self.hidden_dims = (width,)
        self.dropout = dropout
        self.min_logit = min_logit
        self.max_logit = max_logit
        self.register_buffer(
            "pixel_budget_grid", torch.tensor(budgets, dtype=torch.float64)
        )
        self.encoder = nn.Sequential(
            nn.Linear(context_feature_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.threshold_head = nn.Linear(width, len(budgets))

    @property
    def num_pixel_budgets(self) -> int:
        return int(self.pixel_budget_grid.numel())

    def _exact_grid_request(
        self, request: torch.Tensor, *, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if not isinstance(request, torch.Tensor):
            raise TypeError("pixel_budgets must be a torch.Tensor")
        normalized = request.to(device=device, dtype=torch.float64)
        if normalized.ndim == 1:
            normalized = normalized.reshape(1, -1).expand(batch_size, -1)
        elif normalized.ndim == 2 and normalized.shape[0] == 1:
            normalized = normalized.expand(batch_size, -1)
        elif normalized.ndim != 2 or normalized.shape[0] != batch_size:
            raise ValueError("pixel_budgets must have shape [J], [1,J], or [B,J]")
        expected = self.pixel_budget_grid.to(device=device).reshape(1, -1)
        if normalized.shape[1] != self.num_pixel_budgets or not torch.equal(
            normalized, expected.expand(batch_size, -1)
        ):
            raise ValueError(
                "direct calibrator accepts only its complete trained budget grid"
            )
        return normalized

    def forward(
        self,
        context_features: torch.Tensor,
        *,
        pixel_budgets: torch.Tensor | None = None,
    ) -> NoRejectPixelRiskCalibratorOutput:
        if not isinstance(context_features, torch.Tensor):
            raise TypeError("context_features must be a torch.Tensor")
        if context_features.ndim != 2 or context_features.shape[1] != self.context_feature_dim:
            raise ValueError(
                "context_features must have shape "
                f"[B, {self.context_feature_dim}], got {tuple(context_features.shape)}"
            )
        if not context_features.is_floating_point():
            raise TypeError("context_features must be floating point")
        if not bool(torch.isfinite(context_features).all().item()):
            raise ValueError("context_features must be finite")
        raw_logits = self.threshold_head(self.encoder(context_features))
        if not bool(torch.isfinite(raw_logits).all().item()):
            raise FloatingPointError("direct calibrator produced non-finite raw logits")
        logits = self.min_logit + torch.sigmoid(raw_logits) * (
            self.max_logit - self.min_logit
        )
        if not bool(
            torch.isfinite(logits).all().item()
            and (logits >= self.min_logit).all().item()
            and (logits <= self.max_logit).all().item()
        ):
            raise FloatingPointError("direct calibrator broke its bounded-logit contract")
        thresholds = torch.sigmoid(logits)
        if pixel_budgets is None:
            return NoRejectPixelRiskCalibratorOutput(
                pixel_budget_grid=self.pixel_budget_grid,
                grid_logits=logits,
                grid_thresholds=thresholds,
            )
        normalized = self._exact_grid_request(
            pixel_budgets,
            batch_size=context_features.shape[0],
            device=context_features.device,
        )
        return NoRejectPixelRiskCalibratorOutput(
            pixel_budget_grid=self.pixel_budget_grid,
            grid_logits=logits,
            grid_thresholds=thresholds,
            requested_pixel_budgets=normalized,
            requested_logits=logits,
            requested_thresholds=thresholds,
        )

    def export_config(self) -> dict[str, object]:
        return {
            "context_feature_dim": self.context_feature_dim,
            "pixel_budget_grid": self.pixel_budget_grid.detach().cpu().tolist(),
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "min_logit": self.min_logit,
            "max_logit": self.max_logit,
        }

    def capability_contract(self) -> dict[str, object]:
        return {
            "stage": "stage2_direct_no_reject_baseline",
            "model_id": DIRECT_NO_REJECT_MODEL_ID,
            "budget_scope": self.budget_scope,
            "budget_axis": "pixel_false_alarm_rate",
            "supports_component_budget": self.supports_component_budget,
            "supports_reject": self.supports_reject,
            "supports_complete_budget_curve": self.supports_complete_budget_curve,
            "supports_query_risk_aligned_loss": self.supports_query_risk_aligned_loss,
            "structural_monotonicity": self.structural_monotonicity,
            "curve_output_shape": "[batch,J]",
            "training_pipeline_integrated": self.training_pipeline_integrated,
        }


__all__ = [
    "DIRECT_NO_REJECT_BUDGET_SCOPE",
    "DIRECT_NO_REJECT_MODEL_ID",
    "DirectNoRejectPixelCalibrator",
]
