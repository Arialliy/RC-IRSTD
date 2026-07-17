"""Monotone threshold calibrators for one-dimensional pixel-risk budgets.

This module is deliberately separate from :mod:`model.threshold_calibrator`.
The direct and reject-aware models remain compatibility baselines.  The final
``MonotoneNoRejectPixelRiskCalibrator`` maps unlabeled context statistics to a
complete inverse pixel-risk curve and is integrated with the v5 training and
online pipelines.  Component budgets are not treated as a monotone axis.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rc.schema import BudgetSpec


PIXEL_BUDGET_ONLY_SCOPE = "pixel_budget_only_no_component_no_reject"
PIXEL_RISK_NO_REJECT_SCOPE = "pixel_risk_full_curve_no_component_no_reject"
PIXEL_RISK_WITH_REJECT_SCOPE = "pixel_budget_only_no_component_with_reject"


def pixel_budget_from_spec(budgets: BudgetSpec) -> float:
    """Extract a pixel-only budget, rejecting unsupported schema semantics.

    Silently dropping an active component budget would change the feasible
    operating set defined by ``BudgetSpec``.  Callers adapting schema-v3
    episodes must therefore pass exactly ``active=(True, False)``.  Rejection
    is an oracle target on ``RCEpisode`` and is outside this model entirely.
    """

    if not isinstance(budgets, BudgetSpec):
        raise TypeError("budgets must be an rc.schema.BudgetSpec")
    if budgets.active != (True, False):
        raise ValueError(
            "MonotonePixelBudgetCalibrator requires a pixel-only BudgetSpec "
            "with active=(True, False); component budgets are not supported"
        )
    return float(budgets.values[0])


@dataclass(frozen=True)
class PixelBudgetCalibratorOutput:
    """Threshold curve output with intentionally no component/reject fields."""

    pixel_budget_grid: torch.Tensor
    grid_logits: torch.Tensor
    grid_thresholds: torch.Tensor
    requested_pixel_budgets: torch.Tensor | None = None
    requested_logits: torch.Tensor | None = None
    requested_thresholds: torch.Tensor | None = None


@dataclass(frozen=True)
class NoRejectPixelRiskCalibratorOutput(PixelBudgetCalibratorOutput):
    """Complete ``[B, J]`` inverse-risk curve with no abstention output.

    This deliberately inherits only threshold-curve fields.  In particular,
    there is no reject logit, reject probability, or component-risk output
    that a caller could accidentally use as part of the primary Stage-2
    method.
    """


@dataclass(frozen=True)
class PixelRiskCalibratorOutput:
    """Monotone pixel-risk curve plus an ordered abstention curve."""

    pixel_budget_grid: torch.Tensor
    grid_logits: torch.Tensor
    grid_thresholds: torch.Tensor
    grid_reject_logits: torch.Tensor
    grid_reject_probabilities: torch.Tensor
    requested_pixel_budgets: torch.Tensor | None = None
    requested_logits: torch.Tensor | None = None
    requested_thresholds: torch.Tensor | None = None
    requested_reject_logits: torch.Tensor | None = None
    requested_reject_probabilities: torch.Tensor | None = None


class MonotonePixelBudgetCalibrator(nn.Module):
    """Predict a strictly ordered inverse-risk curve for pixel budgets.

    ``pixel_budget_grid`` is ordered from loose to strict, so it must be
    positive and strictly descending (for example ``[1e-4, 1e-5, 1e-6]``).
    The corresponding threshold logits are constructed to be strictly
    increasing: a stricter pixel false-alarm budget can never receive a lower
    threshold on the trained grid.

    ``context_features`` must contain only budget-invariant, unlabeled context
    statistics.  In particular, callers must not pass
    ``RCEpisode.encoded_features()``, which appends pixel/component budget
    values and activity bits.  Feature standardisation and grouping multiple
    scalar schema-v3 episodes onto a common pixel grid are intentionally left
    to a future training adapter.

    This foundation does not support component budgets or rejection.  Keep
    ``ThresholdCalibrator`` for the current dual-budget + reject baseline.
    """

    budget_scope = PIXEL_BUDGET_ONLY_SCOPE
    supports_component_budget = False
    supports_reject = False
    training_pipeline_integrated = False

    def __init__(
        self,
        context_feature_dim: int,
        pixel_budget_grid: Sequence[float],
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        min_logit: float = -10.0,
        max_logit: float = 18.0,
        minimum_logit_gap: float = 1e-3,
    ) -> None:
        super().__init__()
        if (
            isinstance(context_feature_dim, bool)
            or not isinstance(context_feature_dim, int)
            or context_feature_dim <= 0
        ):
            raise ValueError("context_feature_dim must be a positive integer")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        try:
            budgets = tuple(float(value) for value in pixel_budget_grid)
        except (TypeError, ValueError) as error:
            raise ValueError("pixel_budget_grid must be a sequence of numbers") from error
        if len(budgets) < 2:
            raise ValueError("pixel_budget_grid must contain at least two values")
        if not all(math.isfinite(value) and value > 0.0 for value in budgets):
            raise ValueError("pixel_budget_grid values must be finite and positive")
        if not all(loose > strict for loose, strict in zip(budgets, budgets[1:])):
            raise ValueError(
                "pixel_budget_grid must be strictly descending from loose to strict"
            )

        widths: list[int] = []
        for value in hidden_dims:
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError("hidden_dims must contain only positive integers")
            widths.append(int(value))
        min_logit = float(min_logit)
        max_logit = float(max_logit)
        minimum_logit_gap = float(minimum_logit_gap)
        if not math.isfinite(min_logit) or not math.isfinite(max_logit):
            raise ValueError("logit bounds must be finite")
        if min_logit >= max_logit:
            raise ValueError("min_logit must be lower than max_logit")
        if not math.isfinite(minimum_logit_gap) or minimum_logit_gap <= 0.0:
            raise ValueError("minimum_logit_gap must be finite and positive")
        span = max_logit - min_logit
        if len(budgets) * minimum_logit_gap >= span:
            raise ValueError(
                "minimum_logit_gap leaves no free span for the pixel budget grid"
            )

        self.context_feature_dim = int(context_feature_dim)
        self.hidden_dims = tuple(widths)
        self.dropout = float(dropout)
        self.min_logit = min_logit
        self.max_logit = max_logit
        self.minimum_logit_gap = minimum_logit_gap
        # Float64 preserves the declared budget order before interpolation;
        # model logits retain the model/feature dtype.
        budget_tensor = torch.tensor(budgets, dtype=torch.float64)
        self.register_buffer("pixel_budget_grid", budget_tensor)
        self.register_buffer("log_pixel_budget_grid", torch.log10(budget_tensor))

        layers: list[nn.Module] = []
        previous = self.context_feature_dim
        for width in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                ]
            )
            previous = width
        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()
        # J+1 intervals reserve room below/above the J predicted grid points.
        self.spacing_head = nn.Linear(previous, self.num_pixel_budgets + 1)

    @property
    def num_pixel_budgets(self) -> int:
        return int(self.pixel_budget_grid.numel())

    def _ordered_values(
        self,
        hidden: torch.Tensor,
        *,
        head: nn.Linear,
        minimum: float,
        maximum: float,
        minimum_gap: float,
        value_name: str,
    ) -> torch.Tensor:
        """Map positive interval masses to strictly ordered bounded values."""

        raw_intervals = head(hidden)
        # Extreme-tail thresholds routinely occupy logits where float32
        # sigmoid collapses distinct ordered values to exactly 1.0.  Keep the
        # curve construction/interpolation in float64; gradients still flow
        # through the cast to the float32 encoder parameters.
        calculation_dtype = torch.float64
        interval_weights = F.softmax(
            raw_intervals.to(dtype=calculation_dtype), dim=1
        )
        cumulative_weight = torch.cumsum(interval_weights[:, :-1], dim=1)
        grid_index = torch.arange(
            1,
            self.num_pixel_budgets + 1,
            device=hidden.device,
            dtype=calculation_dtype,
        )[None, :]
        free_span = maximum - minimum - self.num_pixel_budgets * minimum_gap
        values = (
            minimum
            + grid_index * minimum_gap
            + cumulative_weight * free_span
        )
        differences = values[:, 1:] - values[:, :-1]
        if not bool(torch.isfinite(values).all().item()):
            raise FloatingPointError(f"non-finite monotone {value_name}")
        if not bool((differences > 0.0).all().item()):
            raise FloatingPointError(
                f"numeric precision broke strict {value_name} ordering"
            )
        return values

    def _ordered_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._ordered_values(
            hidden,
            head=self.spacing_head,
            minimum=self.min_logit,
            maximum=self.max_logit,
            minimum_gap=self.minimum_logit_gap,
            value_name="pixel-budget logits",
        )

    def _prepare_pixel_budget_requests(
        self,
        pixel_budgets: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(pixel_budgets, torch.Tensor):
            raise TypeError("pixel_budgets must be a torch.Tensor")
        request = pixel_budgets.to(device=device, dtype=torch.float64)
        if request.ndim == 0:
            request = request.reshape(1, 1).expand(batch_size, 1)
        elif request.ndim == 1:
            # A rank-one request is always a shared query grid.  Per-sample
            # budgets must be explicit [B, 1], avoiding candidate API ambiguity
            # when the query count happens to equal the batch size.
            if request.numel() == 0:
                raise ValueError("pixel_budgets must not be empty")
            request = request.reshape(1, -1).expand(batch_size, -1)
        elif request.ndim == 2:
            if request.shape[1] == 0 or request.shape[0] not in (1, batch_size):
                raise ValueError(
                    "pixel_budgets must have shape [Q], [1, Q], or [B, Q]"
                )
            if request.shape[0] == 1:
                request = request.expand(batch_size, -1)
        else:
            raise ValueError("pixel_budgets must be a scalar, [Q], [1, Q], or [B, Q]")

        if not bool(torch.isfinite(request).all().item()) or not bool(
            (request > 0.0).all().item()
        ):
            raise ValueError("requested pixel budgets must be finite and positive")
        lower = self.pixel_budget_grid[-1].to(device=device)
        upper = self.pixel_budget_grid[0].to(device=device)
        # Float32 literals can land a few ulps outside a float64 grid endpoint.
        tolerance = (
            torch.finfo(pixel_budgets.dtype).eps * 16.0
            if pixel_budgets.is_floating_point()
            else 0.0
        )
        outside = (request < lower * (1.0 - tolerance)) | (
            request > upper * (1.0 + tolerance)
        )
        if bool(outside.any().item()):
            raise ValueError(
                "requested pixel budgets must stay inside the trained pixel grid "
                f"[{float(lower):.6g}, {float(upper):.6g}]; extrapolation is disabled"
            )
        return request.clamp(min=lower, max=upper)

    def interpolate_logits(
        self,
        grid_logits: torch.Tensor,
        *,
        pixel_budgets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate in log10(pixel-budget) space without extrapolation."""

        if grid_logits.ndim != 2 or grid_logits.shape[1] != self.num_pixel_budgets:
            raise ValueError(
                "grid_logits must have shape "
                f"[B, {self.num_pixel_budgets}], got {tuple(grid_logits.shape)}"
            )
        if not bool(torch.isfinite(grid_logits).all().item()) or not bool(
            (grid_logits[:, 1:] > grid_logits[:, :-1]).all().item()
        ):
            raise ValueError("grid_logits must be finite and strictly ordered")

        request = self._prepare_pixel_budget_requests(
            pixel_budgets,
            batch_size=grid_logits.shape[0],
            device=grid_logits.device,
        )
        ascending_log_budgets = torch.flip(
            self.log_pixel_budget_grid.to(device=grid_logits.device), dims=[0]
        )
        descending_logits = torch.flip(grid_logits, dims=[1])
        query = torch.log10(request)
        right = torch.searchsorted(
            ascending_log_budgets, query, right=True
        ).clamp(1, self.num_pixel_budgets - 1)
        left = right - 1
        x0 = ascending_log_budgets[left]
        x1 = ascending_log_budgets[right]
        batch_index = torch.arange(
            grid_logits.shape[0], device=grid_logits.device
        )[:, None]
        y0 = descending_logits[batch_index, left]
        y1 = descending_logits[batch_index, right]
        weight = ((query - x0) / (x1 - x0)).to(dtype=grid_logits.dtype)
        requested_logits = y0 + weight * (y1 - y0)
        return requested_logits, request

    def forward(
        self,
        context_features: torch.Tensor,
        *,
        pixel_budgets: torch.Tensor | None = None,
    ) -> PixelBudgetCalibratorOutput:
        if (
            context_features.ndim != 2
            or context_features.shape[1] != self.context_feature_dim
        ):
            raise ValueError(
                "context_features must have shape "
                f"[B, {self.context_feature_dim}], got {tuple(context_features.shape)}"
            )
        if not context_features.is_floating_point():
            raise TypeError("context_features must be a floating-point tensor")
        if not bool(torch.isfinite(context_features).all().item()):
            raise ValueError("context_features must be finite")
        hidden = self.encoder(context_features)
        grid_logits = self._ordered_logits(hidden)
        grid_thresholds = torch.sigmoid(grid_logits)
        if not bool(
            (grid_thresholds[:, 1:] > grid_thresholds[:, :-1]).all().item()
        ):
            raise FloatingPointError(
                "sigmoid precision broke strict pixel-budget threshold ordering; "
                "reduce max_logit or increase minimum_logit_gap"
            )
        if pixel_budgets is None:
            return PixelBudgetCalibratorOutput(
                pixel_budget_grid=self.pixel_budget_grid,
                grid_logits=grid_logits,
                grid_thresholds=grid_thresholds,
            )

        requested_logits, normalized_requests = self.interpolate_logits(
            grid_logits, pixel_budgets=pixel_budgets
        )
        return PixelBudgetCalibratorOutput(
            pixel_budget_grid=self.pixel_budget_grid,
            grid_logits=grid_logits,
            grid_thresholds=grid_thresholds,
            requested_pixel_budgets=normalized_requests,
            requested_logits=requested_logits,
            requested_thresholds=torch.sigmoid(requested_logits),
        )

    def export_config(self) -> dict[str, object]:
        """Return only constructor arguments for reproducible reconstruction."""

        return {
            "context_feature_dim": self.context_feature_dim,
            "pixel_budget_grid": self.pixel_budget_grid.detach().cpu().tolist(),
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "min_logit": self.min_logit,
            "max_logit": self.max_logit,
            "minimum_logit_gap": self.minimum_logit_gap,
        }

    def capability_contract(self) -> dict[str, object]:
        """Declare the intentionally narrow, non-integrated model scope."""

        return {
            "budget_scope": self.budget_scope,
            "supports_component_budget": self.supports_component_budget,
            "supports_reject": self.supports_reject,
            "training_pipeline_integrated": self.training_pipeline_integrated,
        }


class MonotoneNoRejectPixelRiskCalibrator(MonotonePixelBudgetCalibrator):
    """Primary Stage-2 model: a complete no-Reject pixel-risk curve.

    The architecture is the same tested bounded-spacing construction as
    :class:`MonotonePixelBudgetCalibrator`, but this class makes the intended
    paper capability explicit: one context produces all ``J`` operating
    points, query labels are consumed only by the separate meta-training loss,
    and deployment emits thresholds rather than an abstention decision.

    This class is the checkpoint-v6 Stage-2 target.  Legacy reject-aware APIs
    remain separate baselines and retain their original contracts.
    """

    budget_scope = PIXEL_RISK_NO_REJECT_SCOPE
    supports_component_budget = False
    supports_reject = False
    supports_complete_budget_curve = True
    supports_query_risk_aligned_loss = True
    training_pipeline_integrated = True

    def forward(
        self,
        context_features: torch.Tensor,
        *,
        pixel_budgets: torch.Tensor | None = None,
    ) -> NoRejectPixelRiskCalibratorOutput:
        output = super().forward(
            context_features,
            pixel_budgets=pixel_budgets,
        )
        return NoRejectPixelRiskCalibratorOutput(
            pixel_budget_grid=output.pixel_budget_grid,
            grid_logits=output.grid_logits,
            grid_thresholds=output.grid_thresholds,
            requested_pixel_budgets=output.requested_pixel_budgets,
            requested_logits=output.requested_logits,
            requested_thresholds=output.requested_thresholds,
        )

    def capability_contract(self) -> dict[str, object]:
        return {
            "stage": "stage2_final_no_reject",
            "budget_scope": self.budget_scope,
            "budget_axis": "pixel_false_alarm_rate",
            "supports_component_budget": self.supports_component_budget,
            "supports_reject": self.supports_reject,
            "supports_complete_budget_curve": self.supports_complete_budget_curve,
            "curve_output_shape": "[batch,J]",
            "supports_query_risk_aligned_loss": (
                self.supports_query_risk_aligned_loss
            ),
            "training_objective": (
                "query_violation_plus_utility_plus_oracle_logit_plus_"
                "curve_smoothness_plus_exact_suffix_coverage"
            ),
            "training_pipeline_integrated": self.training_pipeline_integrated,
            "curve_compute_dtype": "float64",
            "budget_interpolation": "piecewise_linear_log10_no_extrapolation",
            "deployment_output": "threshold_curve_no_reject",
            "risk_guarantee": "empirical_not_certified",
            "component_budget_reason": (
                "connected-component false-alarm counts are not monotone in threshold"
            ),
        }


class MonotonePixelRiskCalibrator(MonotonePixelBudgetCalibrator):
    """Integrated pixel-risk inverse curve with monotone abstention scores.

    The primary threshold curve is structurally non-decreasing as the pixel
    false-alarm budget tightens.  The rejection probability is constrained in
    the same direction: tightening a budget cannot make abstention less
    likely.  Connected-component budgets are intentionally rejected because
    component counts can increase when thresholding fragments a region and
    therefore do not define the monotone inverse-risk relation assumed here.
    """

    budget_scope = PIXEL_RISK_WITH_REJECT_SCOPE
    supports_component_budget = False
    supports_reject = True
    training_pipeline_integrated = True

    def __init__(
        self,
        context_feature_dim: int,
        pixel_budget_grid: Sequence[float],
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        min_logit: float = -10.0,
        max_logit: float = 18.0,
        minimum_logit_gap: float = 1e-3,
        reject_min_logit: float = -12.0,
        reject_max_logit: float = 12.0,
        minimum_reject_logit_gap: float = 1e-3,
    ) -> None:
        super().__init__(
            context_feature_dim=context_feature_dim,
            pixel_budget_grid=pixel_budget_grid,
            hidden_dims=hidden_dims,
            dropout=dropout,
            min_logit=min_logit,
            max_logit=max_logit,
            minimum_logit_gap=minimum_logit_gap,
        )
        reject_min_logit = float(reject_min_logit)
        reject_max_logit = float(reject_max_logit)
        minimum_reject_logit_gap = float(minimum_reject_logit_gap)
        if not math.isfinite(reject_min_logit) or not math.isfinite(
            reject_max_logit
        ):
            raise ValueError("reject logit bounds must be finite")
        if reject_min_logit >= reject_max_logit:
            raise ValueError("reject_min_logit must be lower than reject_max_logit")
        if (
            not math.isfinite(minimum_reject_logit_gap)
            or minimum_reject_logit_gap <= 0.0
        ):
            raise ValueError(
                "minimum_reject_logit_gap must be finite and positive"
            )
        if (
            self.num_pixel_budgets * minimum_reject_logit_gap
            >= reject_max_logit - reject_min_logit
        ):
            raise ValueError(
                "minimum_reject_logit_gap leaves no free reject-logit span"
            )
        self.reject_min_logit = reject_min_logit
        self.reject_max_logit = reject_max_logit
        self.minimum_reject_logit_gap = minimum_reject_logit_gap
        encoder_width = self.hidden_dims[-1] if self.hidden_dims else self.context_feature_dim
        self.reject_spacing_head = nn.Linear(
            encoder_width, self.num_pixel_budgets + 1
        )

    def _ordered_reject_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._ordered_values(
            hidden,
            head=self.reject_spacing_head,
            minimum=self.reject_min_logit,
            maximum=self.reject_max_logit,
            minimum_gap=self.minimum_reject_logit_gap,
            value_name="pixel-budget reject logits",
        )

    def forward(
        self,
        context_features: torch.Tensor,
        *,
        pixel_budgets: torch.Tensor | None = None,
    ) -> PixelRiskCalibratorOutput:
        if (
            context_features.ndim != 2
            or context_features.shape[1] != self.context_feature_dim
        ):
            raise ValueError(
                "context_features must have shape "
                f"[B, {self.context_feature_dim}], got {tuple(context_features.shape)}"
            )
        if not context_features.is_floating_point():
            raise TypeError("context_features must be a floating-point tensor")
        if not bool(torch.isfinite(context_features).all().item()):
            raise ValueError("context_features must be finite")
        hidden = self.encoder(context_features)
        grid_logits = self._ordered_logits(hidden)
        grid_reject_logits = self._ordered_reject_logits(hidden)
        grid_thresholds = torch.sigmoid(grid_logits)
        grid_reject_probabilities = torch.sigmoid(grid_reject_logits)
        for values, name in (
            (grid_thresholds, "threshold"),
            (grid_reject_probabilities, "reject probability"),
        ):
            if not bool((values[:, 1:] > values[:, :-1]).all().item()):
                raise FloatingPointError(
                    f"sigmoid precision broke strict pixel-budget {name} ordering"
                )
        if pixel_budgets is None:
            return PixelRiskCalibratorOutput(
                pixel_budget_grid=self.pixel_budget_grid,
                grid_logits=grid_logits,
                grid_thresholds=grid_thresholds,
                grid_reject_logits=grid_reject_logits,
                grid_reject_probabilities=grid_reject_probabilities,
            )
        requested_logits, normalized_requests = self.interpolate_logits(
            grid_logits, pixel_budgets=pixel_budgets
        )
        requested_reject_logits, reject_requests = self.interpolate_logits(
            grid_reject_logits, pixel_budgets=pixel_budgets
        )
        if not torch.equal(normalized_requests, reject_requests):
            raise RuntimeError("threshold/reject budget interpolation disagrees")
        return PixelRiskCalibratorOutput(
            pixel_budget_grid=self.pixel_budget_grid,
            grid_logits=grid_logits,
            grid_thresholds=grid_thresholds,
            grid_reject_logits=grid_reject_logits,
            grid_reject_probabilities=grid_reject_probabilities,
            requested_pixel_budgets=normalized_requests,
            requested_logits=requested_logits,
            requested_thresholds=torch.sigmoid(requested_logits),
            requested_reject_logits=requested_reject_logits,
            requested_reject_probabilities=torch.sigmoid(
                requested_reject_logits
            ),
        )

    def export_config(self) -> dict[str, object]:
        return {
            **super().export_config(),
            "reject_min_logit": self.reject_min_logit,
            "reject_max_logit": self.reject_max_logit,
            "minimum_reject_logit_gap": self.minimum_reject_logit_gap,
        }

    def capability_contract(self) -> dict[str, object]:
        return {
            "budget_scope": self.budget_scope,
            "supports_component_budget": self.supports_component_budget,
            "supports_reject": self.supports_reject,
            "training_pipeline_integrated": self.training_pipeline_integrated,
            "risk_aligned_query_loss": False,
            "training_objective": "asymmetric_oracle_threshold_plus_reject_bce",
            "curve_compute_dtype": "float64",
            "budget_interpolation": "piecewise_linear_log10_no_extrapolation",
            "risk_guarantee": "empirical_not_certified",
            "component_budget_reason": (
                "connected-component false-alarm counts are not monotone in threshold"
            ),
        }
